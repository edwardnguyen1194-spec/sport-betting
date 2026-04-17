"""OpportunityScout — on-demand "best plays right now" sub-agent.

Uncle clicks a button on the dashboard. The scout reads ALL current
recommendations, surrounding game context, and fresh news headlines,
then hands back the top 3 bets available right now with confidence
scores and full written reasoning he can paste into Telegram.

This is an *informational* endpoint. It does NOT adjust any rec's
confidence, does NOT veto or resize bets, and does NOT feed back into
the paper trader. It only renders markdown the human reads. The
emitted ``AgentDecision`` reflects that:

* ``approved = True``
* ``confidence_delta = 0.0``
* ``stake_multiplier = 1.0``
* ``reasoning`` holds the markdown writeup.
* ``metadata["top_picks"]`` holds the structured picks so the
  dashboard can render them without re-parsing the markdown.

Constraints baked into the system prompt:

* Spread or total only — Uncle does not play moneyline.
* Explain the edge, the context, and *why this pick beats the others*
  on the board today.
* Output must be Telegram-friendly markdown (short lines, no tables).

Fails safe: any parse / HTTP / budget error returns a neutral
``AgentDecision`` with ``error`` set; the dashboard surfaces it as a
"scout unavailable" state instead of crashing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from .base import AgentDecision, BaseAgent


# Scout writes Uncle-facing prose, so give it more room than the
# ±10% review agents. 2048 is plenty for a 3-pick writeup.
_SCOUT_MAX_TOKENS = 2048

# We only send the top-N recs to Claude. More than this and the
# prompt gets noisy without adding signal; the strategies already
# rank picks by confidence before we get here.
_DEFAULT_MAX_RECS = 20


def _rec_to_summary(rec: Any) -> Dict[str, Any]:
    """Flatten a ``BetRecommendation`` (or dict) into the minimal
    fields the scout needs. Tolerates both the dataclass and plain
    dicts so tests / callers can pass either."""
    if is_dataclass(rec):
        d = asdict(rec)
    elif isinstance(rec, dict):
        d = dict(rec)
    else:
        # Duck-type: best-effort attribute read.
        d = {
            k: getattr(rec, k, None)
            for k in (
                "game_key", "sport", "league", "home_team", "away_team",
                "market", "selection", "line", "american", "decimal",
                "book", "strategy", "confidence", "edge", "reasoning",
            )
        }
    return {
        "game": f"{d.get('away_team', '?')} @ {d.get('home_team', '?')}",
        "sport": d.get("sport", ""),
        "league": d.get("league", ""),
        "market": d.get("market", ""),
        "selection": d.get("selection", ""),
        "line": d.get("line"),
        "american": d.get("american"),
        "book": d.get("book", ""),
        "strategy": d.get("strategy", ""),
        "confidence": round(float(d.get("confidence") or 0.0), 4),
        "edge": round(float(d.get("edge") or 0.0), 4),
        "reasoning": (d.get("reasoning") or "")[:240],
    }


def _filter_spread_total(recs: List[Any]) -> List[Any]:
    """Scout is spread/total only. Drop moneyline (and anything else)
    before the LLM sees them so the model can't be tempted."""
    keep = []
    for r in recs:
        market = (
            r.get("market") if isinstance(r, dict)
            else getattr(r, "market", "")
        )
        if (market or "").lower() in ("spread", "total"):
            keep.append(r)
    return keep


class OpportunityScout(BaseAgent):
    """Top-3 "best plays right now" concierge for Uncle."""

    name = "opportunity_scout"
    max_tokens = _SCOUT_MAX_TOKENS

    def system_prompt(self) -> str:
        return (
            "You are an expert sports handicapper briefing Uncle — a "
            "careful bettor who wants the THREE best plays available "
            "right now, nothing more.\n\n"
            "Hard rules:\n"
            "1. Spreads and totals ONLY. Never recommend a moneyline.\n"
            "2. You must pick FROM the candidate list the user gives "
            "you. Do not invent new bets, teams, or prices.\n"
            "3. Your job is selection + explanation, not numeric "
            "adjustment. Trust the confidence / edge numbers from the "
            "hardcoded strategies and use them to rank and defend "
            "your picks.\n"
            "4. For each pick, explain: the EDGE (why this is +EV), "
            "the CONTEXT (news, matchup, situational factors), and "
            "WHY THIS pick is better than the other candidates today.\n"
            "5. Output is Telegram-friendly markdown Uncle will paste "
            "into chat. Short lines. Use '**Pick N:** ...' headings. "
            "No tables. No emoji. No hedging language like 'could' "
            "or 'maybe' — be decisive.\n\n"
            "Return STRICT JSON with this shape and nothing else:\n"
            "{\n"
            '  "top_picks": [\n'
            '    {"rec_summary": "<game + market + selection + line + '
            'book + price>", "why": "<one-paragraph markdown '
            'reasoning>"},\n'
            "    ...\n"
            "  ],\n"
            '  "markdown": "<full markdown writeup, ready to paste>"\n'
            "}\n"
        )

    def build_user_message(self, context: Dict[str, Any]) -> str:
        recs: List[Any] = list(context.get("recs") or [])
        headlines: List[Any] = list(context.get("headlines") or [])
        max_picks: int = int(context.get("max_picks") or 3)

        # Spread/total filter is applied here (belt-and-suspenders with
        # the system prompt) so moneyline candidates never even reach
        # the model.
        recs = _filter_spread_total(recs)

        # Keep the top-N by confidence*edge so the LLM doesn't waste
        # budget on marginal picks.
        def _score(r: Any) -> float:
            d = (
                r if isinstance(r, dict)
                else {"confidence": getattr(r, "confidence", 0.0),
                      "edge": getattr(r, "edge", 0.0)}
            )
            return float(d.get("confidence") or 0.0) * max(
                float(d.get("edge") or 0.0), 0.0
            )

        recs = sorted(recs, key=_score, reverse=True)[:_DEFAULT_MAX_RECS]
        summaries = [_rec_to_summary(r) for r in recs]

        # Summary stats for the model: how many picks per sport, how
        # many per strategy, average edge. Helps it reason about
        # concentration and which signals are firing today.
        by_sport: Dict[str, int] = {}
        by_strat: Dict[str, int] = {}
        edges: List[float] = []
        for s in summaries:
            by_sport[s["sport"]] = by_sport.get(s["sport"], 0) + 1
            by_strat[s["strategy"]] = by_strat.get(s["strategy"], 0) + 1
            edges.append(s["edge"])
        avg_edge = round(sum(edges) / len(edges), 4) if edges else 0.0

        # News: keep titles only, capped, so we don't blow budget on
        # article bodies.
        news_lines: List[str] = []
        for h in headlines[:15]:
            if isinstance(h, dict):
                title = h.get("title") or h.get("headline") or ""
                sport = h.get("sport") or ""
            else:
                title = getattr(h, "title", "") or getattr(h, "headline", "")
                sport = getattr(h, "sport", "")
            if title:
                news_lines.append(f"- [{sport}] {title}"[:200])

        payload = {
            "max_picks": max_picks,
            "summary_stats": {
                "candidate_count": len(summaries),
                "by_sport": by_sport,
                "by_strategy": by_strat,
                "average_edge": avg_edge,
            },
            "candidates": summaries,
            "recent_headlines": news_lines,
        }
        return (
            "Uncle hit the SCOUT button. Pick the top "
            f"{max_picks} plays right now from the candidates below "
            "(spreads and totals only) and explain each one.\n\n"
            + json.dumps(payload, indent=2)
        )

    def parse_response(
        self, text: str, context: Dict[str, Any]
    ) -> AgentDecision:
        """Scout returns custom JSON — ``top_picks`` + ``markdown``.
        We do NOT touch confidence / stake; this is informational."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Strip fenced code if the model wraps its JSON.
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

        picks_raw = data.get("top_picks") or []
        picks: List[Dict[str, str]] = []
        for p in picks_raw:
            if not isinstance(p, dict):
                continue
            picks.append({
                "rec_summary": str(p.get("rec_summary", ""))[:240],
                "why": str(p.get("why", ""))[:1200],
            })

        markdown = str(data.get("markdown") or "").strip()
        if not markdown and picks:
            # Fall back to synthesizing markdown from the structured
            # picks so the dashboard always has something to render.
            lines = [
                "**Uncle — top plays right now**",
                "",
            ]
            for i, pk in enumerate(picks, 1):
                lines.append(f"**Pick {i}:** {pk['rec_summary']}")
                if pk["why"]:
                    lines.append(pk["why"])
                lines.append("")
            markdown = "\n".join(lines).strip()

        return AgentDecision(
            approved=True,
            confidence_delta=0.0,
            stake_multiplier=1.0,
            # ``reasoning`` field in the base is 500 chars — we override
            # the parse path here to keep the full markdown body so
            # Uncle gets the whole writeup, not a truncation.
            reasoning=markdown,
            metadata={"top_picks": picks},
        )

    def _summarize_context(self, context: Dict[str, Any]) -> str:
        recs = context.get("recs") or []
        headlines = context.get("headlines") or []
        return f"scout: {len(recs)} recs, {len(headlines)} headlines"


def scout_top_picks(
    agent: OpportunityScout,
    all_recs: List[Any],
    news_headlines: List[Any],
    max_picks: int = 3,
) -> AgentDecision:
    """Public helper — Uncle's dashboard button entry point.

    Parameters
    ----------
    agent
        A live ``OpportunityScout`` instance (shares the global
        ``AgentLog`` and API key with the other sub-agents).
    all_recs
        All current candidate recs. The agent itself filters to
        spread/total and keeps the top 20 by confidence * edge.
    news_headlines
        Recent headlines. Titles only are sent to the model.
    max_picks
        How many picks Uncle wants — defaults to 3.
    """
    return agent.analyze({
        "recs": all_recs,
        "headlines": news_headlines,
        "max_picks": max_picks,
    })


__all__ = ["OpportunityScout", "scout_top_picks"]
