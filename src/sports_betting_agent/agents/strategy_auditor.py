"""StrategyAuditor sub-agent — weekly deep review.

Where :mod:`pick_reviewer` is the per-bet sanity check and
:mod:`daily_learner` is the once-per-day auto-tuner, ``StrategyAuditor``
is the Monday-morning quant meeting: a broad, narrative review of the
last seven days of play, one strategy at a time, with explicit
threshold tune proposals the human (Uncle) can accept or reject.

Design contract
---------------

1. **CLV is the truth, ROI is variance.** The auditor prioritizes
   closing-line value over win-rate / ROI. A strategy that is
   routinely beating the close is long-run profitable even when last
   week's P&L looks ugly.

2. **Conservative.** Never propose more than 3 threshold changes in
   one review. One bad week is not a reason to gut the book.

3. **Fail SAFE.** No API key, HTTP error, parse error -> return a
   neutral approve-with-empty-proposals AgentDecision. The daily_learner
   keeps trading on the existing thresholds.

4. **Auditable.** The markdown review lives both in the AgentDecision
   and as a standalone ``/data/sba/weekly_audits/YYYY-MM-DD.md`` file
   so Uncle can skim a week's worth of audits without touching the JSON
   log.

Output contract
---------------

``AgentDecision`` with:

* ``approved=True`` (always — this is a review, not a bet gate).
* ``reasoning``: a 300-500 word markdown review structured as
  "Weekly Strategy Audit <date>" -> "Top performer" -> "Hidden drag"
  -> "Proposed tunes".
* ``metadata["proposed_changes"]``: list of
  ``{"file": str, "line": str, "diff": str}`` entries — at most 3.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .base import AgentDecision, BaseAgent


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a senior quant reviewing a sports-betting AI's
weekly performance. You are writing the Monday-morning strategy audit
the head of trading (Uncle) reads with coffee.

Mindset
=======
- CLV is the truth. ROI is variance. A strategy with +1.0% average CLV
  over 30 bets is long-run profitable even if last week's ROI was -8%.
  Prefer signal (CLV) over noise (ROI).
- Be conservative. One week is a small sample. Never propose more
  than THREE threshold changes in a single audit, and each proposed
  change must be justified by BOTH a persistent CLV signal AND an
  adequate sample size (>= 15 closed bets on that strategy).
- Approve on the big picture. You are not vetoing trades — you are
  tuning the dials for next week.
- Be specific. "public_fade threshold 0.65 -> 0.72" is actionable.
  "public_fade could be tighter" is not.

Output format (STRICT JSON)
===========================
Return exactly this JSON shape — no prose outside the JSON, no
markdown fences, no extra keys:

{
  "approved": true,
  "reasoning": "<markdown string, 300-500 words, see template>",
  "metadata": {
    "proposed_changes": [
      {
        "file": "src/sports_betting_agent/config.py",
        "line": "SBA_CONTRARIAN_PUBLIC_THRESHOLD",
        "diff": "raise from 0.65 to 0.72"
      }
    ]
  }
}

Markdown reasoning template
---------------------------
Must start with ``## Weekly Strategy Audit YYYY-MM-DD`` using the
supplied ``audit_date`` from the user message. Then include, in order:

    ### Top performer: <strategy_name> (ROI +X.X%, CLV +X.X%)
    <1-2 sentences why>

    ### Hidden drag: <strategy_name> (ROI -X.X%, CLV -X.X%)
    <1-2 sentences why CLV points to a real problem, not just bad luck>

    ### Proposed tunes:
    - <strategy>: <specific change and justification>
    - <strategy>: <specific change and justification>

Rules on proposed_changes
-------------------------
- At most THREE entries. Zero is acceptable when nothing meets the bar.
- Each entry's ``file`` is the full path from repo root (typically
  ``src/sports_betting_agent/config.py`` or a strategy file under
  ``src/sports_betting_agent/strategies/``).
- ``line`` is the config attribute or env var name being changed
  (e.g. ``SBA_CONTRARIAN_PUBLIC_THRESHOLD``, ``spread_value_min_edge``).
- ``diff`` is a short human-readable description of the change
  ("raise from 0.65 to 0.72", "tighten market_reg from 0.55 to 0.60").
- No invented files, no invented knobs. Only reference thresholds
  present in the supplied ``config_snapshot``.
"""


def _bet_to_plain(bet: Any) -> Dict[str, Any]:
    """Best-effort turn a Bet into a JSON-safe dict."""
    if bet is None:
        return {}
    if hasattr(bet, "to_dict"):
        try:
            return bet.to_dict()
        except Exception:
            pass
    if is_dataclass(bet):
        try:
            return asdict(bet)
        except Exception:
            pass
    if isinstance(bet, dict):
        return dict(bet)
    fields = (
        "id", "placed_at", "settled_at", "status", "result",
        "strategy", "sport", "market", "selection",
        "american", "decimal", "stake", "confidence", "edge",
    )
    return {f: getattr(bet, f, None) for f in fields}


def _last_7_days_bets(closed_bets: Iterable) -> List[Dict[str, Any]]:
    """Filter the closed-bet ledger down to the last 7 UTC days."""
    now = datetime.now(timezone.utc)
    cutoff_ts = now.timestamp() - 7 * 24 * 3600
    out: List[Dict[str, Any]] = []
    for bet in closed_bets:
        plain = _bet_to_plain(bet)
        settled_at = plain.get("settled_at") or plain.get("placed_at")
        if not settled_at:
            continue
        try:
            when = datetime.fromisoformat(str(settled_at).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if when.timestamp() < cutoff_ts:
            continue
        out.append(plain)
    return out


def _roi_by_strategy(bets: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute per-strategy W-L/ROI from the trimmed ledger slice."""
    out: Dict[str, Dict[str, float]] = {}
    for bet in bets:
        strat = bet.get("strategy") or "unknown"
        if not strat:
            continue
        # Ensemble co-credits both strategies via '+'-joined names.
        for name in str(strat).split("+"):
            key = name.strip() or "unknown"
            bucket = out.setdefault(key, {
                "bets": 0, "won": 0, "lost": 0, "push": 0, "void": 0,
                "stake": 0.0, "pnl": 0.0,
            })
            bucket["bets"] += 1
            status = bet.get("status", "open")
            if status == "won":
                bucket["won"] += 1
                stake = float(bet.get("stake") or 0.0)
                dec = float(bet.get("decimal") or 1.0)
                bucket["stake"] += stake
                bucket["pnl"] += stake * (dec - 1.0)
            elif status == "lost":
                bucket["lost"] += 1
                stake = float(bet.get("stake") or 0.0)
                bucket["stake"] += stake
                bucket["pnl"] -= stake
            elif status == "push":
                bucket["push"] += 1
            elif status == "void":
                bucket["void"] += 1
    for bucket in out.values():
        stake = bucket["stake"]
        bucket["roi_pct"] = round(bucket["pnl"] / stake * 100, 2) if stake > 0 else 0.0
        bucket["record"] = f"{int(bucket['won'])}-{int(bucket['lost'])}"
    return out


class StrategyAuditor(BaseAgent):
    """Monday-morning weekly deep review."""

    name = "strategy_auditor"
    # Haiku is fine — this is a once-a-week narrative, not latency-critical,
    # and the structured-JSON contract keeps the token footprint bounded.
    max_tokens = 2048

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_user_message(self, context: Dict[str, Any]) -> str:
        audit_date = context.get("audit_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        clv_by_strategy = context.get("clv_by_strategy") or {}
        roi_by_strategy = context.get("roi_by_strategy") or {}
        closed_bets_last_7_days = context.get("closed_bets_last_7_days") or []
        config_snapshot = context.get("config_snapshot") or {}

        # Trim the bet slice to the fields the model actually needs.
        trimmed_bets: List[Dict[str, Any]] = []
        for bet in closed_bets_last_7_days[:200]:  # hard cap for token safety
            trimmed_bets.append({
                "strategy": bet.get("strategy"),
                "sport": bet.get("sport"),
                "market": bet.get("market"),
                "status": bet.get("status"),
                "stake": bet.get("stake"),
                "decimal": bet.get("decimal"),
                "edge": bet.get("edge"),
                "confidence": bet.get("confidence"),
                "placed_at": bet.get("placed_at"),
            })

        payload = {
            "audit_date": audit_date,
            "window_days": 7,
            "clv_by_strategy": clv_by_strategy,
            "roi_by_strategy": roi_by_strategy,
            "closed_bets_last_7_days": trimmed_bets,
            "config_snapshot": config_snapshot,
        }
        body = json.dumps(payload, default=str, indent=2)
        return (
            "Write this week's strategy audit. Remember: CLV over ROI, "
            "at most 3 proposed changes, STRICT JSON only.\n\n"
            f"{body}\n\n"
            "Return JSON only."
        )

    def parse_response(
        self, text: str, context: Dict[str, Any]
    ) -> AgentDecision:
        """Override: we care about reasoning + proposed_changes, not
        confidence_delta / stake_multiplier (this agent doesn't gate a
        single bet). Also preserve the FULL markdown reasoning; the
        default 500-char truncate would destroy the weekly review.
        """
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
        proposed = metadata.get("proposed_changes") or []
        # Enforce the conservative cap even if the model ignored it.
        if isinstance(proposed, list):
            proposed = proposed[:3]
        else:
            proposed = []
        # Normalize each entry.
        clean_changes: List[Dict[str, str]] = []
        for entry in proposed:
            if not isinstance(entry, dict):
                continue
            clean_changes.append({
                "file": str(entry.get("file", "")),
                "line": str(entry.get("line", "")),
                "diff": str(entry.get("diff", "")),
            })
        metadata["proposed_changes"] = clean_changes

        return AgentDecision(
            approved=bool(data.get("approved", True)),
            confidence_delta=0.0,
            stake_multiplier=1.0,
            # Cap generously — a 500-word markdown review fits well
            # under 8k chars. This keeps the log file sane without
            # mangling the review.
            reasoning=str(data.get("reasoning", "") or "")[:8000],
            metadata=metadata,
        )

    def _summarize_context(self, context: Dict[str, Any]) -> str:
        date = context.get("audit_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        strat_count = len(context.get("clv_by_strategy") or {})
        bet_count = len(context.get("closed_bets_last_7_days") or [])
        return f"audit {date} strategies={strat_count} bets={bet_count}"


def _collect_config_snapshot(settings: Any) -> Dict[str, Any]:
    """Pick the tunable knobs out of a ``Settings`` instance. Kept small
    so the model only proposes changes to things we actually ship.
    """
    if settings is None:
        return {}
    keys = (
        "spread_value_min_edge",
        "total_value_min_edge",
        "contrarian_public_threshold",
        "contrarian_min_books",
        "min_confidence",
        "heavy_fav_min_american",
        "heavy_fav_max_american",
        "heavy_fav_min_books",
        "kelly_fraction",
        "max_bet_pct",
    )
    out: Dict[str, Any] = {}
    for k in keys:
        if hasattr(settings, k):
            try:
                out[k] = getattr(settings, k)
            except Exception:
                continue
    return out


def run_weekly_audit(
    agent: "StrategyAuditor | None",
    paper: Any,
    clv: Any,
    config_snapshot: Any,
) -> AgentDecision:
    """Convenience helper for the daily_learner wiring.

    Parameters
    ----------
    agent
        A :class:`StrategyAuditor` instance or ``None`` (returns a
        benign no-op decision so callers never have to special-case
        the missing-agent path).
    paper
        The :class:`PaperTrader` — we read ``paper.closed_bets`` to
        build the 7-day window.
    clv
        The :class:`CLVTracker` — we call ``clv.stats_by_strategy()``.
    config_snapshot
        Either a ``Settings`` instance (we extract the tunable keys)
        or a pre-built ``dict`` of knob -> value.

    Returns
    -------
    AgentDecision
        Always non-None. ``approved=True`` with empty reasoning when
        the agent is disabled, absent, or raises.
    """
    if agent is None:
        return AgentDecision(reasoning="", metadata={"proposed_changes": []})

    # Collect inputs. All three are best-effort — any single failure
    # degrades the audit gracefully rather than blowing up the caller.
    try:
        closed = list(getattr(paper, "closed_bets", []) or [])
    except Exception as exc:
        logger.warning("run_weekly_audit: reading closed_bets failed: %s", exc)
        closed = []
    bets_7d = _last_7_days_bets(closed)
    roi = _roi_by_strategy(bets_7d)

    try:
        clv_stats = clv.stats_by_strategy() if clv is not None else {}
    except Exception as exc:
        logger.warning("run_weekly_audit: stats_by_strategy failed: %s", exc)
        clv_stats = {}

    if isinstance(config_snapshot, dict):
        snapshot = dict(config_snapshot)
    else:
        snapshot = _collect_config_snapshot(config_snapshot)

    audit_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        return agent.analyze({
            "audit_date": audit_date,
            "clv_by_strategy": clv_stats,
            "roi_by_strategy": roi,
            "closed_bets_last_7_days": bets_7d,
            "config_snapshot": snapshot,
        })
    except Exception as exc:  # pragma: no cover — BaseAgent already catches HTTP errors
        return AgentDecision(error=f"weekly_audit_exception: {exc}")


def save_weekly_audit_markdown(
    data_dir: str,
    audit_date: str,
    markdown: str,
) -> str:
    """Persist the markdown review to ``<data_dir>/weekly_audits/YYYY-MM-DD.md``.

    Returns the absolute path written. Creates the directory on first
    call. Never raises — swallow-and-log so a disk hiccup doesn't
    bubble out of the once-a-week run.
    """
    out_dir = os.path.join(data_dir, "weekly_audits")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as exc:
        logger.warning("save_weekly_audit_markdown mkdir failed: %s", exc)
        return ""
    path = os.path.join(out_dir, f"{audit_date}.md")
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(markdown or "")
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("save_weekly_audit_markdown write failed: %s", exc)
        return ""
    return path


def list_recent_weekly_audits(data_dir: str, limit: int = 12) -> List[Dict[str, str]]:
    """Return the most recent weekly-audit markdown files, newest first.

    Each entry is ``{"date": "YYYY-MM-DD", "path": "<abs>", "markdown": "<body>"}``.
    """
    out_dir = os.path.join(data_dir, "weekly_audits")
    if not os.path.isdir(out_dir):
        return []
    try:
        files = [
            f for f in os.listdir(out_dir)
            if f.endswith(".md") and len(f) >= 13  # YYYY-MM-DD.md
        ]
    except Exception:
        return []
    files.sort(reverse=True)
    out: List[Dict[str, str]] = []
    for fname in files[:limit]:
        path = os.path.join(out_dir, fname)
        date = fname[:-3]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                body = fh.read()
        except Exception:
            body = ""
        out.append({"date": date, "path": path, "markdown": body})
    return out
