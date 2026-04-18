"""HooksDiscovery — propose Claude Code hooks for safety + productivity.

Claude Code supports pre/post hooks that run on tool calls (Edit, Write,
Bash). This sub-agent scans the sports-betting repo's risk surfaces and
proposes hooks that would make working on the codebase safer (blocking
ledger corruption, gating tests, guarding moneyline emissions) and more
productive (auto-format, pre-push deploy sanity, post-tool audit log).

Focus areas baked into the system prompt
----------------------------------------

1. **LEDGER SAFETY** — direct edits to ``/data/sba/ledger.json`` and
   ``clv_records.json`` must be blocked. These files are the book of
   record; a stray ``Edit`` could destroy the P&L audit trail.
2. **TEST GATING** — any change under
   ``src/sports_betting_agent/strategies/*.py`` should trigger pytest
   before the edit is treated as accepted.
3. **SPREAD+TOTAL GUARD** — the house rule is spreads + totals only.
   A new strategy file that emits a ``"moneyline"`` market slips that
   rule. A pre/post-Edit hook can grep for the literal and flag.
4. **DEPLOY SANITY** — before ``git push``, run pytest and refuse if
   MIDDLE-related code is uncommitted (that strategy is quarantined).
5. **AGENT LOG** — every post-tool hook appends a one-line entry to a
   human-readable log so Uncle can skim what the agent did today.

Contract
--------

* ``approved = True`` always — proposing hooks never gates a bet.
* ``reasoning`` — markdown summary of which hooks were proposed and why.
* ``metadata["hooks"]`` — list of ``{name, trigger, script, rationale,
  safety_level}`` entries. ``trigger`` is one of
  ``preEdit|postEdit|preBash|prePush``. ``safety_level`` is
  ``safe|moderate|destructive`` so Uncle can triage what's one-click
  adoptable vs. what needs review.

Outputs persisted
-----------------

* ``<data_dir>/hook_candidates.json`` — rolling bounded log of all
  proposals (most recent 100), surfaced at ``/api/hook-candidates``.
* ``<repo_root>/.claude/hooks_proposed.json`` — a settings.local.json
  -shaped template Uncle can copy/paste to activate any hook.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import AgentDecision, BaseAgent


logger = logging.getLogger(__name__)


_VALID_TRIGGERS = ("preEdit", "postEdit", "preBash", "prePush")
_VALID_SAFETY = ("safe", "moderate", "destructive")

# Bounded on-disk history of hook proposals. 100 entries keeps the file
# small enough for the dashboard to render inline without pagination.
_HOOK_CANDIDATES_MAX = 100


SYSTEM_PROMPT = """You are the HooksDiscovery agent for Uncle Phung's
sports-betting AI. Your job: propose Claude Code hooks (pre/post tool
hooks) that would make working on this codebase safer and more productive.

The repo ships a paper-betting engine (``paper_trader.py``), a CLV
tracker (``clv_tracker.py``), a family of strategies under
``src/sports_betting_agent/strategies/``, and a Flask dashboard. The
house rule is SPREADS + TOTALS ONLY — moneyline emissions are bugs.

Focus areas (in priority order)
===============================
1. LEDGER SAFETY. Block direct edits / writes to ``/data/sba/ledger.json``
   and ``/data/sba/clv_records.json``. Those are the book of record.
2. TEST GATING. When ``src/sports_betting_agent/strategies/*.py`` is
   edited, run pytest on the strategy tests before the edit is final.
3. SPREAD+TOTAL GUARD. Scan new or modified strategy files for the
   literal ``"moneyline"`` in a ``market=`` assignment. Flag loudly.
4. DEPLOY SANITY. Pre-push: run full pytest and refuse when the diff
   touches MIDDLE-related code (``MiddleDetectorStrategy`` is
   quarantined; commits reintroducing it need explicit sign-off).
5. AGENT LOG. Post-tool: append a one-line human-readable entry to
   ``/data/sba/agent_tool_log.txt`` so Uncle can skim today's edits.

Rules
=====
- Propose 4-8 hooks, prioritized by safety_level. Include at least
  one hook from each of the five focus areas above.
- Every hook entry MUST include:
    name, trigger (preEdit|postEdit|preBash|prePush),
    script (shell snippet, single line or short ``$'...'`` block),
    rationale, safety_level (safe|moderate|destructive).
- ``safety_level`` semantics:
    safe       = advisory / logging / grep flagging (no side effects).
    moderate   = runs tests or formatters (may fail the edit).
    destructive= blocks the call entirely (e.g. deny ledger writes).
- Prefer BLOCK over warn for ledger safety. Prefer warn over block
  for productivity hooks.
- scripts should exit non-zero to block. ``$CLAUDE_FILE_PATHS`` and
  ``$CLAUDE_TOOL_NAME`` are available to hooks.

Output (STRICT JSON, no prose, no markdown fences)
==================================================
{
  "approved": true,
  "reasoning": "<markdown summary, 4-8 bullets>",
  "metadata": {
    "hooks": [
      {
        "name": "<snake_case>",
        "trigger": "preEdit|postEdit|preBash|prePush",
        "script": "<shell one-liner>",
        "rationale": "<1-2 sentences>",
        "safety_level": "safe|moderate|destructive"
      }
    ]
  }
}
"""


# Deterministic fallback catalog. These are the hooks the agent would
# propose even if the LLM is unavailable (no API key, budget exhausted,
# parse error). They cover all five focus areas so the dashboard always
# has something concrete to show Uncle.
DEFAULT_HOOKS: List[Dict[str, str]] = [
    {
        "name": "ledger_write_guard",
        "trigger": "preEdit",
        "script": (
            "case \"$CLAUDE_FILE_PATHS\" in "
            "*data/sba/ledger.json*|*data/sba/clv_records.json*) "
            "echo 'BLOCK: ledger files are append-only. Use the "
            "PaperTrader API.' >&2; exit 2 ;; esac"
        ),
        "rationale": (
            "ledger.json and clv_records.json are the book of record. "
            "Direct edits would silently corrupt P&L and CLV. Block at "
            "the pre-Edit hook so the trader API is the only writer."
        ),
        "safety_level": "destructive",
    },
    {
        "name": "ledger_write_guard_write_tool",
        "trigger": "preEdit",
        "script": (
            "case \"$CLAUDE_FILE_PATHS\" in "
            "*data/sba/*.json) "
            "echo 'WARN: editing raw state file. Prefer the trader API.' "
            ">&2 ;; esac"
        ),
        "rationale": (
            "Belt-and-suspenders: warn on any edit into /data/sba/*.json "
            "(learnings, news cache, agent_log) that isn't the ledger "
            "itself. Keeps raw-state edits intentional."
        ),
        "safety_level": "moderate",
    },
    {
        "name": "strategy_pytest_gate",
        "trigger": "postEdit",
        "script": (
            "case \"$CLAUDE_FILE_PATHS\" in "
            "*src/sports_betting_agent/strategies/*.py) "
            "pytest -q src/sports_betting_agent/tests/ -k 'strategy or "
            "strategies' || exit 1 ;; esac"
        ),
        "rationale": (
            "Strategies drive real stakes. Re-run the strategy test "
            "slice on every strategy edit so a silent math bug can't "
            "linger until the next manual pytest."
        ),
        "safety_level": "moderate",
    },
    {
        "name": "moneyline_emission_guard",
        "trigger": "postEdit",
        "script": (
            "case \"$CLAUDE_FILE_PATHS\" in "
            "*src/sports_betting_agent/strategies/*.py) "
            "if grep -nE 'market\\s*=\\s*[\"']moneyline[\"']' "
            "\"$CLAUDE_FILE_PATHS\" >/dev/null; then "
            "echo 'BLOCK: strategy emits moneyline. Spread+Total only.' "
            ">&2; exit 2; fi ;; esac"
        ),
        "rationale": (
            "House rule: spreads and over/unders only. Any new or edited "
            "strategy file emitting market='moneyline' is a bug that "
            "would slip a banned bet into the ensemble. Block the edit."
        ),
        "safety_level": "destructive",
    },
    {
        "name": "prepush_pytest",
        "trigger": "prePush",
        "script": (
            "pytest -q src/sports_betting_agent/tests/ || "
            "{ echo 'BLOCK: pytest failing — refusing push.' >&2; exit 1; }"
        ),
        "rationale": (
            "A push with red tests is how sports-betting-ai-agent.fly.dev "
            "ends up stuck on a broken build. Gate every push on the "
            "local test suite."
        ),
        "safety_level": "moderate",
    },
    {
        "name": "prepush_middle_guard",
        "trigger": "prePush",
        "script": (
            "if git diff --cached --name-only | grep -qi 'middle'; then "
            "echo 'BLOCK: MIDDLE-related change detected in staged "
            "diff. MiddleDetectorStrategy is quarantined — confirm with "
            "Uncle before pushing.' >&2; exit 1; fi"
        ),
        "rationale": (
            "MiddleDetectorStrategy was disabled after the multi-agent "
            "audit found its middle-prob math 3-5× too loose. Any "
            "re-introduction needs explicit human sign-off, not a "
            "silent push."
        ),
        "safety_level": "destructive",
    },
    {
        "name": "agent_tool_log",
        "trigger": "postEdit",
        "script": (
            "mkdir -p data/sba && printf '%s  %s  %s\\n' "
            "\"$(date -u +%FT%TZ)\" \"$CLAUDE_TOOL_NAME\" "
            "\"$CLAUDE_FILE_PATHS\" >> data/sba/agent_tool_log.txt"
        ),
        "rationale": (
            "Human-readable append-only log of every edit the agent "
            "makes. Uncle can `tail -f data/sba/agent_tool_log.txt` "
            "during long sessions to see what changed, when."
        ),
        "safety_level": "safe",
    },
    {
        "name": "destructive_bash_warn",
        "trigger": "preBash",
        "script": (
            "case \"$CLAUDE_COMMAND\" in "
            "*'rm -rf '*|*'git reset --hard'*|*'git push --force'*) "
            "echo 'WARN: destructive command. Review before continuing.' "
            ">&2 ;; esac"
        ),
        "rationale": (
            "Pre-Bash advisory on rm -rf, hard resets, force pushes. "
            "Claude Code still executes but the warning surfaces in the "
            "transcript so Uncle can catch it."
        ),
        "safety_level": "safe",
    },
]


def _default_reasoning() -> str:
    """Markdown summary used when the LLM is unavailable."""
    return (
        "## Hook proposals (fallback catalog)\n\n"
        "LLM was unavailable — returning the deterministic baseline:\n\n"
        "- **ledger_write_guard / ledger_write_guard_write_tool** — "
        "block direct edits to /data/sba/ledger.json + clv_records.json, "
        "warn on other /data/sba/*.json.\n"
        "- **strategy_pytest_gate** — run strategy tests after any "
        "strategies/*.py edit.\n"
        "- **moneyline_emission_guard** — block any strategy file that "
        "emits a moneyline market.\n"
        "- **prepush_pytest / prepush_middle_guard** — gate pushes on "
        "green tests and warn on MIDDLE-related staged changes.\n"
        "- **agent_tool_log** — append every edit to a human-readable "
        "log in data/sba/agent_tool_log.txt.\n"
        "- **destructive_bash_warn** — advisory on rm -rf / hard "
        "resets / force pushes."
    )


def _normalise_hook(raw: Any) -> Optional[Dict[str, str]]:
    """Coerce a model-returned hook dict into the canonical shape.

    Drops entries that are missing required fields or use an unknown
    trigger / safety_level so downstream consumers never have to guard
    against malformed metadata.
    """
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name", "")).strip()
    trigger = str(raw.get("trigger", "")).strip()
    script = str(raw.get("script", "")).strip()
    rationale = str(raw.get("rationale", "")).strip()
    safety = str(raw.get("safety_level", "")).strip().lower()
    if not name or not trigger or not script:
        return None
    if trigger not in _VALID_TRIGGERS:
        return None
    if safety not in _VALID_SAFETY:
        safety = "moderate"
    return {
        "name": name[:64],
        "trigger": trigger,
        "script": script[:2000],
        "rationale": rationale[:400],
        "safety_level": safety,
    }


class HooksDiscovery(BaseAgent):
    """Propose Claude Code hooks for safety + productivity."""

    name = "hooks_discovery"
    # Mid-sized budget: we want 4-8 structured hook entries plus a
    # short markdown summary, nothing more.
    max_tokens = 2048

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_user_message(self, context: Dict[str, Any]) -> str:
        existing_hooks = context.get("existing_hooks") or []
        repo_layout = context.get("repo_layout") or {
            "ledger_files": [
                "/data/sba/ledger.json",
                "/data/sba/clv_records.json",
            ],
            "strategies_glob": "src/sports_betting_agent/strategies/*.py",
            "quarantined": ["MiddleDetectorStrategy"],
            "house_rule": "spreads and over/unders only",
        }
        payload = {
            "existing_hooks": existing_hooks,
            "repo_layout": repo_layout,
            "focus_areas": [
                "ledger_safety",
                "test_gating",
                "spread_total_guard",
                "deploy_sanity",
                "agent_log",
            ],
            "instructions": (
                "Propose 4-8 hooks covering every focus area. STRICT "
                "JSON only, schema in the system prompt."
            ),
        }
        return (
            "Propose Claude Code hooks for the sports-betting project.\n\n"
            + json.dumps(payload, indent=2)
            + "\n\nReturn JSON only."
        )

    def parse_response(
        self, text: str, context: Dict[str, Any]
    ) -> AgentDecision:
        """Custom parser: we care about metadata.hooks, not confidence."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            try:
                cleaned = cleaned.split("```", 2)[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            except IndexError:
                pass
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return AgentDecision(
                approved=True,
                reasoning="",
                error=f"parse_error: {exc}",
            )

        metadata = data.get("metadata") or {}
        raw_hooks = metadata.get("hooks") or []
        hooks: List[Dict[str, str]] = []
        if isinstance(raw_hooks, list):
            for entry in raw_hooks[:16]:  # hard cap — don't flood the log
                norm = _normalise_hook(entry)
                if norm:
                    hooks.append(norm)

        return AgentDecision(
            approved=True,
            confidence_delta=0.0,
            stake_multiplier=1.0,
            # Keep full markdown; the 500-char default would butcher it.
            reasoning=str(data.get("reasoning") or "")[:4000],
            metadata={"hooks": hooks},
        )

    def _summarize_context(self, context: Dict[str, Any]) -> str:
        existing = context.get("existing_hooks") or []
        return f"hooks_discovery: existing={len(existing)}"


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _persist_candidates(
    data_dir: str, entry: Dict[str, Any]
) -> str:
    """Append ``entry`` to ``<data_dir>/hook_candidates.json`` (bounded)."""
    path = os.path.join(data_dir, "hook_candidates.json")
    entries: List[Dict[str, Any]] = []
    if os.path.exists(path):
        try:
            with open(path) as fh:
                entries = json.load(fh).get("entries", [])
        except Exception:
            entries = []
    entries.append(entry)
    entries = entries[-_HOOK_CANDIDATES_MAX:]
    try:
        os.makedirs(data_dir, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"entries": entries}, fh, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("hooks_discovery: persist failed: %s", exc)
        return ""
    return path


def recent_hook_candidates(data_dir: str, n: int = 25) -> List[Dict[str, Any]]:
    """Read the most recent hook-candidate entries for dashboard display."""
    path = os.path.join(data_dir, "hook_candidates.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as fh:
            entries = json.load(fh).get("entries", [])
    except Exception:
        return []
    if n <= 0:
        return []
    return entries[-n:]


def _settings_local_shape(hooks: List[Dict[str, str]]) -> Dict[str, Any]:
    """Render hooks in the ``.claude/settings.local.json`` shape.

    Claude Code's settings file keys hooks under the top-level ``hooks``
    map by trigger name. We produce ordered lists keyed by trigger so
    Uncle can cherry-pick which hooks to paste into his real settings.
    """
    out: Dict[str, List[Dict[str, str]]] = {t: [] for t in _VALID_TRIGGERS}
    for h in hooks:
        trigger = h.get("trigger")
        if trigger not in out:
            continue
        out[trigger].append({
            "name": h["name"],
            "script": h["script"],
            "rationale": h.get("rationale", ""),
            "safety_level": h.get("safety_level", "moderate"),
        })
    # Drop empty buckets so the template is easy to skim.
    return {"hooks": {k: v for k, v in out.items() if v}}


def write_hooks_template(
    repo_root: str, hooks: List[Dict[str, str]]
) -> str:
    """Write ``<repo_root>/.claude/hooks_proposed.json`` for one-click adopt.

    Uncle copies the snippet(s) he likes into his real
    ``.claude/settings.local.json``. We never touch settings.local.json
    directly — that's the user's file, edits would be surprising.
    """
    target_dir = os.path.join(repo_root, ".claude")
    target = os.path.join(target_dir, "hooks_proposed.json")
    try:
        os.makedirs(target_dir, exist_ok=True)
        payload = {
            "_comment": (
                "Proposed Claude Code hooks from the HooksDiscovery "
                "sub-agent. Copy any entry below into the 'hooks' map "
                "of .claude/settings.local.json to activate. Safe hooks "
                "are advisory; destructive hooks block the tool call."
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **_settings_local_shape(hooks),
        }
        tmp = target + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, target)
    except Exception as exc:
        logger.warning("hooks_discovery: template write failed: %s", exc)
        return ""
    return target


# ---------------------------------------------------------------------------
# Public helper used by the dashboard + CLI
# ---------------------------------------------------------------------------


def discover_hooks(
    agent: "HooksDiscovery | None",
    existing_hooks: Optional[List[Dict[str, Any]]] = None,
    *,
    data_dir: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> AgentDecision:
    """Run the agent, persist the result, and emit a settings template.

    Fails safe: if ``agent`` is ``None`` or the agent errors, we return
    the deterministic ``DEFAULT_HOOKS`` catalog so callers always get a
    non-empty proposal list.

    Parameters
    ----------
    agent
        A live :class:`HooksDiscovery` or ``None``.
    existing_hooks
        What Uncle already has in ``.claude/settings.local.json``. Lets
        the model avoid re-proposing the same guard. Best-effort shape;
        missing keys are fine.
    data_dir
        Where to append the rolling ``hook_candidates.json`` log.
    repo_root
        Repo root for writing ``.claude/hooks_proposed.json``. If the
        directory does not exist we create it.
    """
    existing_hooks = existing_hooks or []

    decision: AgentDecision
    if agent is None:
        decision = AgentDecision(
            approved=True,
            reasoning=_default_reasoning(),
            metadata={"hooks": list(DEFAULT_HOOKS)},
        )
    else:
        try:
            decision = agent.analyze({"existing_hooks": existing_hooks})
        except Exception as exc:  # pragma: no cover — BaseAgent catches
            decision = AgentDecision(error=f"discover_exception: {exc}")
        # Fall back to the deterministic catalog on any failure path so
        # the dashboard always has something actionable to show.
        hooks = (decision.metadata or {}).get("hooks") or []
        if decision.error or not hooks:
            decision = AgentDecision(
                approved=True,
                reasoning=decision.reasoning or _default_reasoning(),
                metadata={"hooks": list(DEFAULT_HOOKS)},
                error=decision.error,
            )

    hooks_list = (decision.metadata or {}).get("hooks") or []

    # Persist rolling audit log.
    if data_dir:
        _persist_candidates(data_dir, {
            "ts": datetime.now(timezone.utc).isoformat(),
            "decision": asdict(decision),
            "existing_hook_count": len(existing_hooks),
        })

    # Render the one-click-adopt template.
    if repo_root:
        write_hooks_template(repo_root, hooks_list)

    return decision


__all__ = [
    "HooksDiscovery",
    "discover_hooks",
    "recent_hook_candidates",
    "write_hooks_template",
    "DEFAULT_HOOKS",
]
