"""PickReviewer sub-agent.

A sharp-bettor Claude sanity check that runs on a single
:class:`BetRecommendation` the moment before
``PaperTrader.place()`` books it.

Purpose
-------
The hardcoded strategies decide *what* to bet on. They have perfect
math but zero awareness of today's context — a pitcher scratch, a
coach-fired headline, a stale line the aggregator hasn't refreshed.
``PickReviewer`` reads the rec plus a small game-context bundle
(headlines, weather, elo diff) and returns an
:class:`AgentDecision` that can:

* **Veto** clear blunders (``approved=False``)
* **Trim** the stake when reasoning is shaky
  (``stake_multiplier`` in ``[0.5, 1.0]``)
* **Nudge** confidence up/down by ≤ 0.10 when context suggests the
  rec's conviction is mis-calibrated

The review is *polish* on top of the strategies, never a replacement.
Failures (no API key, HTTP error, parse error, daily budget
exhausted) fall back to a safe approve-with-no-changes via the
``BaseAgent`` contract — the paper trader never blocks on the agent.

Uncle's rule is non-negotiable: **SPREAD and OVER/UNDER ONLY**.
Anything else is already screened out upstream, but the system
prompt still tells Claude to veto on sight as a belt-and-suspenders
defense.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from .base import AgentDecision, BaseAgent


SYSTEM_PROMPT = """You are a sharp, experienced sports bettor doing a final
sanity check on ONE pick just before it is placed. You are reviewing, not
generating — a hardcoded math strategy produced the recommendation already
and it is your job to catch obvious blunders and tune conviction.

Absolute rules
==============
1. Uncle's rule: only SPREAD and OVER/UNDER (total) markets are allowed.
   If the rec's market is anything else (moneyline, prop, parlay, etc.),
   VETO it immediately (approved=false, stake_multiplier=0.0) and say
   so in reasoning.
2. Approve by default. Strategies earned their place by back-testing.
   Veto ONLY on a clear red flag:
     - bet contradicts obvious injury / lineup / weather news from the
       headlines provided (e.g. "betting Yankees -1.5 but their ace was
       just scratched")
     - the rec's line looks stale vs the reasoning's timestamp, or the
       reasoning explicitly says it's using an outdated model
     - the rec's confidence is implausibly high (>0.80) relative to the
       market price and edge — strategies occasionally over-fit
     - the rec's own ``reasoning`` text contradicts itself or the
       selection (e.g. says "total should GO OVER" but selection is
       Under)
3. Never invent picks and never raise confidence or stake above the
   original. You may only tighten, not inflate.
4. Output STRICT JSON, no prose, no markdown fences.

Output schema
=============
Return exactly this JSON shape (no extra keys, no comments):

{
  "approved": true | false,
  "confidence_delta": <number in [-0.10, 0.10]>,
  "stake_multiplier": <number in [0.0, 1.0]>,
  "reasoning": "<1-2 sentence justification, no line breaks>",
  "metadata": { "<optional diagnostic keys>": "<values>" }
}

Guidance on the numbers
-----------------------
* ``approved=true`` with ``stake_multiplier=1.0`` and
  ``confidence_delta=0.0`` is the default — use it whenever nothing
  in the context changes the call.
* ``stake_multiplier`` between 0.5 and 1.0 when the reasoning is
  thin or context is mildly concerning but not veto-worthy.
* ``confidence_delta`` of ±0.05 when a headline clearly supports or
  undercuts the thesis; up to ±0.10 only in strong cases.
* ``approved=false`` (or equivalently ``stake_multiplier=0.0``) only
  on the clear red flags in rule 2.

Be concise. Be skeptical but not paranoid. You are Uncle's last
check before money moves.
"""


def _rec_to_plain(rec: Any) -> Dict[str, Any]:
    """Best-effort turn a BetRecommendation into a JSON-safe dict."""
    if rec is None:
        return {}
    if hasattr(rec, "to_dict"):
        try:
            return rec.to_dict()
        except Exception:
            pass
    if is_dataclass(rec):
        try:
            return asdict(rec)
        except Exception:
            pass
    if isinstance(rec, dict):
        return dict(rec)
    # Fallback: pull the well-known fields.
    fields = (
        "selection", "market", "line", "american", "decimal", "book",
        "confidence", "edge", "strategy", "reasoning",
        "home_team", "away_team", "sport", "league", "game_key",
        "stake_fraction",
    )
    return {f: getattr(rec, f, None) for f in fields}


class PickReviewer(BaseAgent):
    """Reviews a single rec just before ``paper_trader.place()``."""

    name = "pick_reviewer"
    # Haiku is fast + cheap and this runs once per bet. Base class
    # picks ``claude-3-5-haiku-latest`` as the default model.
    max_tokens = 512

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_user_message(self, context: Dict[str, Any]) -> str:
        rec = context.get("rec")
        game_context = context.get("game_context") or {}

        rec_dict = _rec_to_plain(rec)
        # Trim giant fields that waste tokens.
        if isinstance(rec_dict.get("sources"), list):
            rec_dict["sources"] = rec_dict["sources"][:5]
        # ``meta`` can carry large aggregator payloads — keep it terse.
        meta = rec_dict.get("meta")
        if isinstance(meta, dict):
            keep = {k: meta[k] for k in ("line_observed_at",) if k in meta}
            rec_dict["meta"] = keep

        headlines = game_context.get("recent_headlines") or []
        if isinstance(headlines, list):
            headlines = headlines[:10]
        weather = game_context.get("weather")
        commence_time = game_context.get("commence_time")
        elo_diff = game_context.get("elo_diff")

        sport = (rec_dict.get("sport") or "").lower()
        include_weather = "baseball" in sport or sport.startswith("mlb")

        payload: Dict[str, Any] = {
            "rec": rec_dict,
            "headlines": headlines,
            "commence_time": commence_time,
            "elo_diff": elo_diff,
        }
        if include_weather and weather is not None:
            payload["weather"] = weather

        body = json.dumps(payload, default=str, indent=2)
        return (
            "Review this single pick. Remember: spread + over/under only, "
            "approve by default, veto on clear red flags, output the JSON "
            "schema exactly.\n\n"
            f"{body}\n\n"
            "Return JSON only."
        )

    # ``parse_response`` is inherited from BaseAgent; it already
    # clamps ``confidence_delta`` to ±0.10 and ``stake_multiplier``
    # to [0, 1], truncates reasoning, and returns a safe
    # approve-no-op on parse errors.


def review_rec(
    reviewer: "PickReviewer | None",
    rec: Any,
    game_context: Dict[str, Any] | None = None,
) -> AgentDecision:
    """Convenience wrapper callers use from ``paper_trader.place()``.

    Returns a benign approve-no-op :class:`AgentDecision` when the
    reviewer is absent or disabled, so callers never have to special-
    case the "no agent" path.
    """
    if reviewer is None:
        return AgentDecision(reasoning="")
    try:
        return reviewer.analyze({
            "rec": rec,
            "game_context": game_context or {},
        })
    except Exception as exc:  # pragma: no cover — BaseAgent already catches HTTP errors
        return AgentDecision(error=f"review_rec_exception: {exc}")
