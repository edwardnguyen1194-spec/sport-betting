"""NewsTriage sub-agent.

Runs every trade cycle over OPEN bets. For each bet, it pulls the
recent headlines from :class:`NewsReader` (filtered to the home / away
teams involved) and asks Claude to classify whether a line-moving
story has broken since the bet was placed.

This agent **does not veto** — its job is to flag. The hardcoded
strategies remain in charge of entry/exit. A red alert on an open
bet is surfaced as a warning log + persisted to disk so Uncle can
investigate, but the bet itself is left alone.

Contract with the base framework
--------------------------------

- ``approved``: always ``True`` — this agent never vetoes.
- ``confidence_delta``: shaved down to ``[-0.05, 0]`` for
  bet-negative news, and up to ``[0, +0.02]`` for bet-positive news.
  These tighter bounds (vs. the ±0.10 the base allows) reflect that
  headline sentiment is noisy and we don't want one injury rumor
  swinging a bet hard.
- ``reasoning``: ``"News flag: <short description>"``.
- ``metadata``: ``{"alert_level": "red"|"yellow"|"green",
  "matched_headline": "..."}``.

Alert levels
~~~~~~~~~~~~

- ``red`` — confirmed injury / scratch / postponement that materially
  hurts the side we're on. Triggers a warning log.
- ``yellow`` — soft signal (questionable, doubtful, rain watch).
- ``green`` — no concerning news, or genuinely bet-positive headline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import AgentDecision, BaseAgent


logger = logging.getLogger(__name__)


# Keyword banks used both to pre-filter headlines (so we only spend
# tokens on stories that could plausibly move the line) and to give
# Claude a concrete vocabulary in the system prompt.
INJURY_KEYWORDS = (
    "injured",
    "out",
    "il",
    "ruled out",
    "scratched",
    "questionable",
    "doubtful",
    "dnp",
)
WEATHER_KEYWORDS = (
    "rain delay",
    "postponed",
    "suspended",
)


class NewsTriage(BaseAgent):
    """Sports news triage for live (open) bets."""

    name = "news_triage"
    # Tight max_tokens — triage responses should be a few dozen tokens
    # of JSON. Cheap by design because we call this per open bet per
    # cycle.
    max_tokens = 256

    # Per-agent clamps override the base ±0.10 default. The base
    # ``parse_response`` clamps to ±0.10 which is wider than we want
    # for noise-prone headline sentiment, so we re-clamp here.
    MIN_DELTA = -0.05
    MAX_DELTA = 0.02

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def system_prompt(self) -> str:
        return (
            "You are a sports news triage agent for live betting. "
            "Given an open bet and a short list of recent headlines "
            "about the teams involved, decide whether any headline "
            "is likely to move the line against or in favor of the "
            "bet.\n\n"
            "Attend especially to these injury keywords: "
            f"{', '.join(INJURY_KEYWORDS)}. "
            "Attend to these weather / logistics keywords: "
            f"{', '.join(WEATHER_KEYWORDS)}. "
            "A headline about a star player on the side of the bet "
            "being 'ruled out', 'scratched', or 'placed on IL' is "
            "bet-negative. A headline about a star player on the "
            "OPPOSING team being ruled out is bet-positive. A rain "
            "delay / postponement is bet-negative for totals.\n\n"
            "Output STRICT JSON (no prose, no code fences) with the "
            "following schema and nothing else:\n"
            "{\n"
            '  "approved": true,\n'
            '  "confidence_delta": <number between -0.05 and 0.02>,\n'
            '  "reasoning": "News flag: <short description>",\n'
            '  "metadata": {\n'
            '    "alert_level": "red" | "yellow" | "green",\n'
            '    "matched_headline": "<exact headline title or empty string>"\n'
            "  }\n"
            "}\n\n"
            "Rules:\n"
            "- approved is ALWAYS true. You never veto a bet.\n"
            "- confidence_delta is negative ONLY for bet-negative news "
            "(injury, scratch, postponement on our side). Bound: "
            "-0.05 to 0.\n"
            "- confidence_delta is positive ONLY for confirmed "
            "bet-positive news (opponent's star ruled out). Bound: "
            "0 to +0.02.\n"
            "- If no headline matches the bet, set confidence_delta "
            "to 0, alert_level to 'green', matched_headline to ''.\n"
            "- Keep reasoning under 140 chars."
        )

    def build_user_message(self, context: Dict[str, Any]) -> str:
        bet = context.get("bet")
        headlines = context.get("headlines", []) or []

        bet_block = {
            "selection": getattr(bet, "selection", ""),
            "market": getattr(bet, "market", ""),
            "line": getattr(bet, "line", None),
            "home_team": getattr(bet, "home_team", ""),
            "away_team": getattr(bet, "away_team", ""),
            "sport": getattr(bet, "sport", ""),
        }

        # Pass headlines as a compact list of {title, description}.
        slim = [
            {
                "title": (h.get("title") or "")[:200],
                "description": (h.get("description") or "")[:200],
            }
            for h in headlines[:20]
        ]

        return (
            "Open bet:\n"
            + json.dumps(bet_block, ensure_ascii=False)
            + "\n\nRecent headlines (team-filtered, newest first):\n"
            + json.dumps(slim, ensure_ascii=False)
            + "\n\nReturn the JSON described in the system prompt."
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def parse_response(
        self, text: str, context: Dict[str, Any]
    ) -> AgentDecision:
        decision = super().parse_response(text, context)
        # Re-clamp confidence_delta to our tighter news-triage window.
        decision.confidence_delta = self._clamp(
            decision.confidence_delta, self.MIN_DELTA, self.MAX_DELTA
        )
        # This agent never vetoes.
        decision.approved = True
        decision.stake_multiplier = 1.0
        # Normalise metadata — downstream code keys on these fields.
        meta = decision.metadata or {}
        alert = str(meta.get("alert_level", "green")).lower()
        if alert not in ("red", "yellow", "green"):
            alert = "green"
        meta["alert_level"] = alert
        meta.setdefault("matched_headline", "")
        decision.metadata = meta
        if not decision.reasoning:
            decision.reasoning = "News flag: none"
        return decision

    def _summarize_context(self, context: Dict[str, Any]) -> str:
        bet = context.get("bet")
        if bet is None:
            return super()._summarize_context(context)
        return (
            f"bet={getattr(bet, 'id', '?')} "
            f"{getattr(bet, 'selection', '?')} "
            f"{getattr(bet, 'market', '?')}"
        )


# ---------------------------------------------------------------------------
# Helper used by the dashboard trade-and-settle loop
# ---------------------------------------------------------------------------


def _filter_headlines_for_bet(bet, headlines: List[Dict]) -> List[Dict]:
    """Keep only headlines that mention the home or away team.

    Matching is case-insensitive and also tries the last word of the
    team name (e.g. "Lakers" from "LA Lakers") since ESPN often drops
    the city. Mirrors the logic in ``NewsReader.penalty_for_game``.
    """
    import re

    teams = [
        (getattr(bet, "home_team", "") or "").strip(),
        (getattr(bet, "away_team", "") or "").strip(),
    ]
    teams = [t for t in teams if t]
    if not teams:
        return []

    patterns = []
    for t in teams:
        variants = {t.lower()}
        parts = [p for p in t.split() if len(p) >= 3]
        if parts:
            variants.add(parts[-1].lower())
        for v in variants:
            patterns.append(re.compile(r"\b" + re.escape(v) + r"\b", re.IGNORECASE))

    filtered: List[Dict] = []
    for h in headlines:
        title = h.get("title") or ""
        desc = h.get("description") or ""
        hay = f"{title} {desc}"
        if any(p.search(hay) for p in patterns):
            filtered.append(h)
    return filtered


def triage_open_bets(
    triage: NewsTriage,
    open_bets,
    news_reader,
) -> Dict[str, AgentDecision]:
    """Run ``triage`` over each open bet, return decisions keyed by bet id.

    ``open_bets`` may be a dict (``PaperTrader.open_bets``) or any
    iterable of ``Bet`` objects. ``news_reader`` only needs to expose
    ``recent_headlines()`` — the call site passes the live
    :class:`NewsReader` instance. The agent's own ``analyze()`` handles
    enabled/budget/error cases; each decision is always returned, even
    if it carries an ``error`` field.
    """
    if isinstance(open_bets, dict):
        bets = list(open_bets.values())
    else:
        bets = list(open_bets)

    # One headline fetch per cycle is plenty — NewsReader caches 10m.
    try:
        all_headlines = news_reader.recent_headlines(n=20)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("news_triage: headline fetch failed: %s", exc)
        all_headlines = []

    results: Dict[str, AgentDecision] = {}
    for bet in bets:
        bet_id = getattr(bet, "id", None)
        if not bet_id:
            continue
        team_headlines = _filter_headlines_for_bet(bet, all_headlines)
        if not team_headlines:
            # Nothing to triage — emit a neutral green decision without
            # spending any tokens.
            results[bet_id] = AgentDecision(
                approved=True,
                confidence_delta=0.0,
                reasoning="News flag: none",
                metadata={"alert_level": "green", "matched_headline": ""},
            )
            continue
        try:
            decision = triage.analyze({"bet": bet, "headlines": team_headlines})
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("news_triage: analyze failed for %s: %s", bet_id, exc)
            decision = AgentDecision(
                approved=True,
                reasoning="News flag: triage error",
                metadata={"alert_level": "green", "matched_headline": ""},
                error=f"triage_error: {exc}",
            )
        results[bet_id] = decision
    return results


def persist_triage_audit(clv_tracker, decisions: Dict[str, AgentDecision]) -> None:
    """Append triage decisions to an audit file next to the CLV records.

    We keep this out of the CLVRecord dataclass (which is narrowly
    typed for closing-line tracking) and instead write a sidecar file
    ``news_triage_log.json`` in the same data dir. Bounded to the most
    recent 5000 entries so the log doesn't grow without limit.
    """
    if not decisions:
        return
    import os

    data_dir = getattr(clv_tracker, "data_dir", None)
    if not data_dir:
        return
    path = os.path.join(data_dir, "news_triage_log.json")
    entries: List[Dict[str, Any]] = []
    if os.path.exists(path):
        try:
            with open(path) as fh:
                entries = json.load(fh).get("entries", [])
        except Exception:
            entries = []
    now = datetime.now(timezone.utc).isoformat()
    for bet_id, decision in decisions.items():
        entries.append({
            "ts": now,
            "bet_id": bet_id,
            "decision": asdict(decision),
        })
    entries = entries[-5000:]
    os.makedirs(data_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"entries": entries}, fh, indent=2)
    os.replace(tmp, path)


__all__ = [
    "NewsTriage",
    "triage_open_bets",
    "persist_triage_audit",
    "INJURY_KEYWORDS",
    "WEATHER_KEYWORDS",
]
