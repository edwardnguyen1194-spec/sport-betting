"""SelfReflection sub-agent — introspective weekly learner.

Where :mod:`skills_learner` scans the web for new techniques, and
:mod:`strategy_auditor` reviews bet outcomes, ``SelfReflection`` is
the mirror: each sub-agent periodically reviews ITS OWN past
decisions and proposes patches to its own prompt / token budget /
temperature / schema.

Design contract
---------------

1. **One agent at a time.** Each ``analyze()`` call targets a single
   sub-agent by name (``pick_reviewer``, ``news_triage``, etc.).
   Patches are scoped so a bad reflection on one agent doesn't
   leak into another.

2. **Evidence-based.** The reflection loop pulls the last N
   AgentLog entries for the target agent and, when available,
   correlates them with closed-bet outcomes. If a pick_reviewer
   approved 8 picks that lost, that is evidence worth flagging.

3. **Propose, don't apply.** SelfReflection NEVER mutates the
   target agent's config. It writes structured proposals to
   ``/data/sba/agent_prompt_proposals.json`` for Uncle (or the
   StrategyAuditor's weekly merge) to approve.

4. **Fail SAFE.** No client / HTTP error / parse error -> benign
   empty-proposal AgentDecision. The trading loop never blocks.

Output contract
---------------

``AgentDecision`` with:

* ``approved=True`` (always — this is a meta-review, not a bet gate).
* ``reasoning``: 200-400 word markdown audit starting with
  ``### SelfReflection audit for <agent> <YYYY-MM-DD>``.
* ``metadata["target_agent"]``: the agent under review.
* ``metadata["proposed_patches"]``: list of
  ``{"kind": "system_prompt"|"max_tokens"|"temperature"|"schema",
     "old": str, "new": str, "rationale": str}`` entries — at most 5.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import AgentDecision, AgentLog, BaseAgent


logger = logging.getLogger(__name__)


VALID_TARGETS = (
    "pick_reviewer",
    "news_triage",
    "post_mortem",
    "game_analyst",
    "opportunity_scout",
    "strategy_auditor",
)


VALID_PATCH_KINDS = ("system_prompt", "max_tokens", "temperature", "schema")


SYSTEM_PROMPT = """You are the SelfReflection meta-agent for Uncle Phung's
sports-betting AI. You review ONE sub-agent's last N decisions and propose
concrete, surgical patches to its own configuration.

Mindset
=======
- Be specific and evidence-based. Cite concrete log entries or recurring
  patterns. "pick_reviewer approved 8 -4.5 road favorites that lost" is
  actionable; "could be better" is noise.
- Be conservative. Propose AT MOST 5 patches per reflection. Zero is
  acceptable when the agent is performing well.
- Favor the smallest correct change. A 2-sentence system_prompt tweak
  beats a full rewrite. A +256 max_tokens bump beats a 4x increase.
- Stay in scope. You patch the target agent ONLY. Do not propose
  strategy-code changes or new sub-agents — that belongs to
  StrategyAuditor / SkillsLearner.

Patch kinds (each entry in ``proposed_patches`` must pick exactly one)
======================================================================
- ``system_prompt``: surgical addition/replacement. ``old`` is the
  exact substring of the current prompt being changed (or empty string
  to APPEND). ``new`` is the replacement. Keep each patch ≤ 400 chars.
- ``max_tokens``: ``old`` and ``new`` are integers as strings
  (e.g. "512" -> "768"). Only propose when evidence shows truncation
  or wasted tokens.
- ``temperature``: ``old``/``new`` are floats in [0.0, 1.0] as strings.
  Propose LOWER temp when outputs are inconsistent on similar inputs;
  HIGHER when outputs are repetitive/stuck.
- ``schema``: propose adding a new metadata key the agent should track
  (e.g. "add 'stale_line_flag' to pick_reviewer metadata"). ``old`` is
  the current schema keys (comma-joined), ``new`` is the key being
  added, rationale explains why.

Output format (STRICT JSON)
===========================
Return exactly this JSON shape — no prose outside the JSON, no
markdown fences, no extra keys:

{
  "approved": true,
  "reasoning": "<markdown string, 200-400 words, see template>",
  "metadata": {
    "target_agent": "<agent name>",
    "proposed_patches": [
      {
        "kind": "system_prompt",
        "old": "<exact substring to replace or empty to append>",
        "new": "<replacement text>",
        "rationale": "<1-2 sentences citing the pattern in the log>"
      }
    ]
  }
}

Markdown reasoning template
---------------------------
Start with ``### SelfReflection audit for <agent> <YYYY-MM-DD>`` using the
supplied ``target_agent`` and ``audit_date``. Then include:

    **Sample:** N decisions reviewed, date range.

    **Patterns noticed:**
    - <pattern 1 with concrete count>
    - <pattern 2 with concrete count>

    **Proposed patches:** <count, or "none — agent is performing well">
    - <patch kind>: <one-line justification>
    - ...

Be honest. If the log shows the agent is already well-calibrated, say so
and return an empty proposed_patches list.
"""


def _decisions_rollup(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the AgentLog entries into stats the model can reason over.

    We deliberately do NOT do complex outcome-correlation here — the
    model can chew on raw decision/context summaries. We just count
    approvals/vetoes, token averages, latency outliers, and pick the
    top few error strings so the payload stays compact.
    """
    if not entries:
        return {
            "count": 0,
            "approved": 0,
            "vetoed": 0,
            "errors": {},
            "avg_tokens_in": 0,
            "avg_tokens_out": 0,
            "max_tokens_out": 0,
            "avg_latency_ms": 0,
        }

    approved = 0
    vetoed = 0
    errors: Dict[str, int] = {}
    tokens_in_sum = 0
    tokens_out_sum = 0
    max_tokens_out = 0
    latency_sum = 0
    stake_mults: List[float] = []
    confidence_deltas: List[float] = []

    for e in entries:
        dec = e.get("decision") or {}
        if dec.get("approved", True):
            approved += 1
        else:
            vetoed += 1
        err = dec.get("error")
        if err:
            key = str(err).split(":", 1)[0][:60]
            errors[key] = errors.get(key, 0) + 1
        t_in = int(e.get("tokens_in") or 0)
        t_out = int(e.get("tokens_out") or 0)
        tokens_in_sum += t_in
        tokens_out_sum += t_out
        if t_out > max_tokens_out:
            max_tokens_out = t_out
        latency_sum += int(e.get("latency_ms") or 0)
        sm = dec.get("stake_multiplier")
        if isinstance(sm, (int, float)):
            stake_mults.append(float(sm))
        cd = dec.get("confidence_delta")
        if isinstance(cd, (int, float)):
            confidence_deltas.append(float(cd))

    n = len(entries)

    def _avg(xs: List[float]) -> Optional[float]:
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "count": n,
        "approved": approved,
        "vetoed": vetoed,
        "approval_rate": round(approved / n * 100, 1) if n else 0,
        "errors": errors,
        "avg_tokens_in": tokens_in_sum // n,
        "avg_tokens_out": tokens_out_sum // n,
        "max_tokens_out": max_tokens_out,
        "avg_latency_ms": latency_sum // n,
        "avg_stake_multiplier": _avg(stake_mults),
        "avg_confidence_delta": _avg(confidence_deltas),
    }


class SelfReflection(BaseAgent):
    """Introspective meta-agent — reviews its siblings' own logs."""

    name = "self_reflection"
    # Enough room for 200-400 word markdown + a few patches.
    max_tokens = 2048

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_user_message(self, context: Dict[str, Any]) -> str:
        target = context.get("target_agent") or "unknown"
        entries = context.get("recent_entries") or []
        audit_date = context.get("audit_date") or datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d")
        current_config = context.get("current_config") or {}

        # Trim each entry to what the model actually needs.
        trimmed: List[Dict[str, Any]] = []
        for e in entries[-100:]:  # hard cap for token safety
            dec = e.get("decision") or {}
            trimmed.append({
                "ts": e.get("ts"),
                "context_summary": str(e.get("context_summary") or "")[:160],
                "approved": dec.get("approved"),
                "confidence_delta": dec.get("confidence_delta"),
                "stake_multiplier": dec.get("stake_multiplier"),
                "reasoning": str(dec.get("reasoning") or "")[:200],
                "error": dec.get("error"),
                "tokens_out": e.get("tokens_out"),
                "latency_ms": e.get("latency_ms"),
            })

        rollup = _decisions_rollup(entries)

        payload = {
            "target_agent": target,
            "audit_date": audit_date,
            "rollup": rollup,
            "recent_entries": trimmed,
            "current_config": current_config,
        }
        body = json.dumps(payload, default=str, indent=2)
        return (
            f"Reflect on the last {len(entries)} decisions from "
            f"'{target}'. At most 5 patches. STRICT JSON only.\n\n"
            f"{body}\n\n"
            "Return JSON only."
        )

    def parse_response(
        self, text: str, context: Dict[str, Any]
    ) -> AgentDecision:
        """Override: preserve full markdown reasoning and cap patches."""
        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```", 2)[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError) as exc:
            return AgentDecision(
                approved=True, reasoning="", error=f"parse_error: {exc}"
            )

        metadata = data.get("metadata") or {}
        target = metadata.get("target_agent") or context.get("target_agent") or ""
        proposed = metadata.get("proposed_patches") or []
        if not isinstance(proposed, list):
            proposed = []

        clean_patches: List[Dict[str, str]] = []
        for entry in proposed[:5]:   # hard cap
            if not isinstance(entry, dict):
                continue
            kind = str(entry.get("kind", "")).strip()
            if kind not in VALID_PATCH_KINDS:
                # Drop anything the model invented.
                continue
            clean_patches.append({
                "kind": kind,
                "old": str(entry.get("old", ""))[:800],
                "new": str(entry.get("new", ""))[:800],
                "rationale": str(entry.get("rationale", ""))[:400],
            })

        metadata = {
            "target_agent": str(target or ""),
            "proposed_patches": clean_patches,
        }

        return AgentDecision(
            approved=bool(data.get("approved", True)),
            confidence_delta=0.0,
            stake_multiplier=1.0,
            # 200-400 word markdown easily fits; generous cap.
            reasoning=str(data.get("reasoning", "") or "")[:6000],
            metadata=metadata,
        )

    def _summarize_context(self, context: Dict[str, Any]) -> str:
        target = context.get("target_agent") or "?"
        n = len(context.get("recent_entries") or [])
        date = context.get("audit_date") or datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d")
        return f"reflect {target} n={n} {date}"


def run_self_reflection(
    agent: "SelfReflection | None",
    target_name: str,
    agent_log: AgentLog,
    n: int = 50,
    current_config: Optional[Dict[str, Any]] = None,
) -> AgentDecision:
    """Convenience wrapper used by daily_learner / dashboard endpoint.

    Parameters
    ----------
    agent
        A :class:`SelfReflection` instance or ``None`` (returns a
        benign no-op decision).
    target_name
        One of :data:`VALID_TARGETS`. Unknown names return a no-op
        so callers don't have to validate up-front.
    agent_log
        The shared :class:`AgentLog` — we call ``recent(n, agent=...)``.
    n
        How many recent entries to review (default 50).
    current_config
        Optional snapshot of the target's current max_tokens / temp /
        etc so the model can anchor proposals to real values.
    """
    if agent is None:
        return AgentDecision(
            reasoning="",
            metadata={"target_agent": target_name, "proposed_patches": []},
        )
    if target_name not in VALID_TARGETS:
        return AgentDecision(
            reasoning="",
            error=f"unknown_target: {target_name}",
            metadata={"target_agent": target_name, "proposed_patches": []},
        )

    try:
        entries = agent_log.recent(n=n, agent=target_name)
    except Exception as exc:
        logger.warning("run_self_reflection recent() failed: %s", exc)
        entries = []

    audit_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        return agent.analyze({
            "target_agent": target_name,
            "recent_entries": entries,
            "audit_date": audit_date,
            "current_config": current_config or {},
        })
    except Exception as exc:  # pragma: no cover — BaseAgent catches HTTP errors
        return AgentDecision(
            error=f"self_reflection_exception: {exc}",
            metadata={"target_agent": target_name, "proposed_patches": []},
        )


# ---------------------------------------------------------------------------
# Persistence: /data/sba/agent_prompt_proposals.json
# ---------------------------------------------------------------------------


def _proposals_path(data_dir: str) -> str:
    return os.path.join(data_dir, "agent_prompt_proposals.json")


def save_proposals(
    data_dir: str,
    target_agent: str,
    decision: AgentDecision,
    audit_date: Optional[str] = None,
) -> str:
    """Append one reflection's patches to the rolling proposals file.

    The file is a JSON object keyed by ``target_agent`` with a bounded
    list of entries (newest last). Each entry captures the audit
    date, reasoning markdown, and the list of patches.

    Returns the absolute path written (empty string on failure —
    never raises so the caller's weekly loop can continue).
    """
    os.makedirs(data_dir, exist_ok=True)
    path = _proposals_path(data_dir)
    data: Dict[str, List[Dict[str, Any]]] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                # Filter to the known shape — drop anything else.
                data = {
                    str(k): list(v)
                    for k, v in raw.items()
                    if isinstance(v, list)
                }
        except Exception as exc:
            logger.warning("save_proposals read failed: %s", exc)
            data = {}

    audit_date = audit_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = {
        "audit_date": audit_date,
        "reasoning": decision.reasoning or "",
        "proposed_patches": (decision.metadata or {}).get(
            "proposed_patches", []
        ),
        "error": decision.error,
    }
    bucket = data.setdefault(target_agent, [])
    bucket.append(entry)
    # Bound per-agent history at 52 entries (~1 year of weekly reflections).
    data[target_agent] = bucket[-52:]

    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("save_proposals write failed: %s", exc)
        return ""
    return path


def load_proposals(data_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Read the proposals JSON. Returns ``{}`` when the file is missing
    or unreadable — dashboard endpoint can treat missing = "no audits
    have fired yet" without special-casing.
    """
    path = _proposals_path(data_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            return {
                str(k): list(v)
                for k, v in raw.items()
                if isinstance(v, list)
            }
    except Exception as exc:
        logger.warning("load_proposals failed: %s", exc)
    return {}


def latest_proposals_by_agent(
    data_dir: str,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Return the most-recent reflection entry per target agent.

    Shape: ``{"pick_reviewer": {"audit_date": ..., "proposed_patches": [...]},
              "news_triage": None, ...}``. Missing agents are keyed to
    ``None`` so the dashboard can render a row per sub-agent.
    """
    store = load_proposals(data_dir)
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for name in VALID_TARGETS:
        entries = store.get(name) or []
        out[name] = entries[-1] if entries else None
    return out
