"""GameAnalyst sub-agent.

Deep per-game contextual synthesis. For a single game, this agent
pulls together every non-odds signal we have — weather, umpire,
park, rolling form, Elo, travel/rest, news — and asks Claude to
identify the single strongest betting angle.

Unlike the PickReviewer (which scrutinizes an already-generated
rec), the GameAnalyst looks at the game holistically BEFORE the
ensemble finalizes its picks. Its output is a ``context score``
delivered as a small ``confidence_delta`` plus a ``metadata``
block of named angles that the paper trader, dashboard, and any
downstream agent can consume.

Design
------

* Runs only for a small number of top-N recs (N=5 by default,
  env ``SBA_GAME_ANALYST_TOP_N``). This keeps the token budget
  sane while still polishing the bets most likely to be placed.
* MLB-aware: when the game is MLB we gather weather run-delta,
  the assigned home-plate umpire, and the park factor. For
  other sports those fields are omitted from the prompt.
* Sport-agnostic context: rolling scored/allowed (from the
  TeamScoringTracker), Elo ratings, ESPN headline snippets, and
  the commence time / rest gap implied by UTC day.
* Fails open — if the LLM call errors or returns junk, the
  decision is a no-op so the hardcoded ensemble output stands.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentDecision


logger = logging.getLogger(__name__)


class GameAnalyst(BaseAgent):
    """Sharp pregame analyst — one game at a time."""

    name = "game_analyst"
    # Slightly larger budget than PickReviewer since we're asking
    # for a short structured write-up (angles + best_bet_hint).
    max_tokens = 700

    # --------------------------------------------------------------
    # Prompt construction
    # --------------------------------------------------------------

    def system_prompt(self) -> str:
        """Base prompt used by tests + the first call. During
        ``analyze()`` we layer the sport-specific expertise
        block on top via dynamic replacement below."""
        return self._base_system_prompt()

    def _base_system_prompt(self) -> str:
        return (
            "You are a sharp pregame sports analyst advising a betting "
            "syndicate. Synthesize the context the user provides — "
            "weather, ballpark, umpire, recent form, travel/rest, and "
            "injury news — into the SINGLE strongest betting angle for "
            "this game. Prefer totals (Over/Under) or point-spread "
            "takes; we do NOT bet moneylines or props.\n\n"
            "Rules:\n"
            "1. One strongest angle only — do not hedge across several.\n"
            "2. Quantify when you can (\"wind 18mph out to LF = ~+0.4 "
            "runs\", \"home team 0-5 ATS on current 10-day trip\").\n"
            "3. Be skeptical — if the context is thin or the book line "
            "already accounts for the angle, say so and return a "
            "neutral delta near zero.\n"
            "4. ``confidence_delta`` is capped by the caller to "
            "[-0.05, +0.05]. Use the full range only when you have a "
            "concrete, named reason. Default to 0.0 when unsure.\n"
            "5. Reference KEY NUMBERS when applicable — if the book total "
            "is at a historical cluster (e.g. NFL 41/44, MLB 8/9, "
            "NBA 225), say so explicitly.\n"
            "6. Call out REVERSE LINE MOVEMENT if visible in the book "
            "lines summary — lines moving against the public is the "
            "highest-quality sharp signal available.\n\n"
            "Output a single JSON object. Do not wrap it in code "
            "fences. Schema:\n"
            "{\n"
            "  \"approved\": true,\n"
            "  \"confidence_delta\": number in [-0.05, 0.05],\n"
            "  \"reasoning\": \"2-3 short sentences\",\n"
            "  \"metadata\": {\n"
            "    \"angles\": [\"short label\", ...],\n"
            "    \"best_bet_hint\": \"e.g. Under 8.5 or Home -1.5\"\n"
            "  }\n"
            "}"
        )

    def analyze(self, context):
        """Inject sport-specific expertise so the model gets key
        numbers, public bias, and edge factors for THIS sport
        before it reads the game context. Master-bettor mode."""
        try:
            from .betting_expertise import full_expertise_preamble
        except Exception:
            return super().analyze(context)

        sport = (context.get("sport") or "").lower()
        preamble = full_expertise_preamble(sport=sport)

        orig_fn = self.system_prompt
        upgraded = "\n\n".join([preamble, self._base_system_prompt()])
        self.system_prompt = lambda: upgraded  # type: ignore
        try:
            return super().analyze(context)
        finally:
            self.system_prompt = orig_fn  # type: ignore

    def build_user_message(self, context: Dict[str, Any]) -> str:
        # Context is a plain dict prepared by ``analyze_game``; keep
        # keys stable so Claude can rely on them. JSON-encode for a
        # compact, deterministic payload.
        payload = {
            "home_team": context.get("home_team"),
            "away_team": context.get("away_team"),
            "sport": context.get("sport"),
            "commence_time": context.get("commence_time"),
            "weather": context.get("weather"),
            "park_factor": context.get("park_factor"),
            "umpire": context.get("umpire"),
            "recent_headlines": context.get("recent_headlines", {}),
            "elo_ratings": context.get("elo_ratings", {}),
            "team_scoring": context.get("team_scoring", {}),
            "book_lines_summary": context.get("book_lines_summary", {}),
        }
        return (
            "Game context:\n"
            + json.dumps(payload, indent=2, default=str)
            + "\n\nIdentify the single strongest betting angle. "
            + "Return the JSON object described in the system prompt."
        )

    # --------------------------------------------------------------
    # Stricter parse — clamp delta to ±0.05 (half the base ±0.10).
    # --------------------------------------------------------------

    def parse_response(
        self, text: str, context: Dict[str, Any]
    ) -> AgentDecision:
        decision = super().parse_response(text, context)
        # Re-clamp the confidence delta to the tighter ±0.05 band
        # this agent is specified to emit.
        decision.confidence_delta = self._clamp(
            decision.confidence_delta, -0.05, 0.05
        )
        # Normalize metadata shape so downstream consumers (dashboard,
        # PickReviewer) can count on it.
        md = dict(decision.metadata or {})
        angles = md.get("angles") or []
        if not isinstance(angles, list):
            angles = [str(angles)]
        md["angles"] = [str(a)[:120] for a in angles][:5]
        hint = md.get("best_bet_hint")
        if hint is not None and not isinstance(hint, str):
            hint = str(hint)
        if hint:
            md["best_bet_hint"] = hint[:80]
        decision.metadata = md
        return decision

    def _summarize_context(self, context: Dict[str, Any]) -> str:
        return (
            f"{context.get('away_team', '?')} @ "
            f"{context.get('home_team', '?')} ({context.get('sport', '?')})"
        )


# -----------------------------------------------------------------
# analyze_game — gather context and call .analyze()
# -----------------------------------------------------------------


def _is_mlb(sport: str) -> bool:
    s = (sport or "").lower()
    return "mlb" in s or s == "baseball_mlb"


def _recent_form_for(scoring: Any, sport: str, team: str) -> Dict[str, Any]:
    """Summarize the last ~5 games of scored/allowed for a team.

    ``scoring`` is a duck-typed TeamScoringTracker (may be None).
    Returns a dict with counts, averages, and a compact "W-L last 5"
    derived from scored vs allowed. Keeps failures silent.
    """
    if scoring is None or not team:
        return {}
    try:
        rec = scoring.team(sport, team)
    except Exception:
        return {}
    if not rec:
        return {}

    # Concatenate venue splits in the order recorded, then take the
    # tail so "last N" reflects actual recency. Because the tracker
    # stores parallel scored/allowed lists keyed by venue, we
    # interleave them by index to keep pairs aligned.
    def _pairs(scored_key: str, allowed_key: str):
        s = rec.get(scored_key, []) or []
        a = rec.get(allowed_key, []) or []
        return list(zip(s, a))

    combined = _pairs("home_scored", "home_allowed") + _pairs(
        "away_scored", "away_allowed"
    )
    if not combined:
        return {}

    last_n = combined[-5:]
    wins = sum(1 for sc, al in last_n if sc > al)
    losses = sum(1 for sc, al in last_n if sc < al)
    pushes = len(last_n) - wins - losses
    total_games = len(combined)
    avg_scored = sum(sc for sc, _ in last_n) / len(last_n)
    avg_allowed = sum(al for _, al in last_n) / len(last_n)
    return {
        "games_tracked": total_games,
        "last_5_wl": f"{wins}-{losses}"
        + (f"-{pushes}" if pushes else ""),
        "last_5_avg_scored": round(avg_scored, 2),
        "last_5_avg_allowed": round(avg_allowed, 2),
    }


def _headlines_for(news_reader: Any, team: str, limit: int = 3) -> List[Dict[str, str]]:
    """Pull up to ``limit`` headline titles/descriptions mentioning team."""
    if news_reader is None or not team:
        return []
    try:
        heads = getattr(news_reader, "headlines", None) or []
    except Exception:
        return []
    needle = team.lower()
    # Prefer the last word of the team (e.g. "Lakers") as a looser
    # secondary match — mirrors NewsReader.penalty_for_game's heuristic.
    parts = [p for p in team.split() if len(p) >= 3]
    alt = parts[-1].lower() if parts else needle

    matched: List[Dict[str, str]] = []
    for h in heads:
        title = (h.get("title") or "").lower()
        desc = (h.get("description") or "").lower()
        if needle in title or needle in desc or alt in title or alt in desc:
            matched.append(
                {
                    "title": (h.get("title") or "")[:140],
                    "description": (h.get("description") or "")[:200],
                }
            )
            if len(matched) >= limit:
                break
    return matched


def _book_lines_summary(game: Any) -> Dict[str, Any]:
    """Compact summary of the best available spread/total lines."""
    out: Dict[str, Any] = {}
    try:
        lines = list(getattr(game, "lines", []) or [])
    except Exception:
        return out

    totals: List[float] = []
    home_spreads: List[float] = []
    for ln in lines:
        market = getattr(ln, "market", "")
        if market == "total" and ln.line is not None:
            totals.append(float(ln.line))
        elif market == "spread" and ln.line is not None:
            sel = (getattr(ln, "selection", "") or "").lower()
            home = (getattr(game, "home_team", "") or "").lower()
            if sel and sel == home:
                home_spreads.append(float(ln.line))
    if totals:
        out["posted_total_median"] = round(
            sorted(totals)[len(totals) // 2], 2
        )
    if home_spreads:
        out["home_spread_median"] = round(
            sorted(home_spreads)[len(home_spreads) // 2], 2
        )
    return out


def analyze_game(
    agent: GameAnalyst,
    game: Any,
    settings: Optional[Any] = None,
    news_reader: Optional[Any] = None,
    scoring: Optional[Any] = None,
    elo: Optional[Any] = None,
) -> AgentDecision:
    """Gather context for a single game and ask the GameAnalyst.

    Parameters
    ----------
    agent
        An instantiated ``GameAnalyst``.
    game
        A ``GameOdds`` (or any object exposing home_team, away_team,
        sport, commence_time, lines).
    settings, news_reader, scoring, elo
        Injected collaborators. Any may be ``None``; the relevant
        context section is simply omitted.
    """
    if agent is None or game is None:
        return AgentDecision(reasoning="no_input")

    home = getattr(game, "home_team", "") or ""
    away = getattr(game, "away_team", "") or ""
    sport = getattr(game, "sport", "") or ""
    commence_time = getattr(game, "commence_time", None)
    commence_iso: Optional[str] = None
    if commence_time is not None:
        try:
            commence_iso = commence_time.astimezone(timezone.utc).isoformat()
        except Exception:
            commence_iso = str(commence_time)

    context: Dict[str, Any] = {
        "home_team": home,
        "away_team": away,
        "sport": sport,
        "commence_time": commence_iso,
    }

    # MLB-only signals. Each guarded so a missing module or bad
    # data never stops the analysis.
    if _is_mlb(sport):
        try:
            from ..weather import total_adjustment as _weather_adj
            delta, reason = _weather_adj(home, commence_time)
            context["weather"] = {
                "total_adjustment": round(float(delta or 0.0), 2),
                "reason": reason,
            }
        except Exception as exc:
            logger.debug("game_analyst: weather lookup failed: %s", exc)

        try:
            from ..park_factors import park_factor as _park
            pf = _park(home)
            if pf is not None:
                context["park_factor"] = round(float(pf), 3)
        except Exception as exc:
            logger.debug("game_analyst: park factor failed: %s", exc)

        try:
            from ..mlb_umpires import run_adjustment as _ump
            u_delta, u_reason = _ump(home, away, commence_time)
            context["umpire"] = {
                "run_adjustment": round(float(u_delta or 0.0), 2),
                "name_and_tendency": u_reason,
            }
        except Exception as exc:
            logger.debug("game_analyst: umpire lookup failed: %s", exc)

    # Sport-agnostic signals.
    context["recent_headlines"] = {
        home: _headlines_for(news_reader, home),
        away: _headlines_for(news_reader, away),
    }

    if elo is not None:
        try:
            context["elo_ratings"] = {
                home: elo.get_rating(sport, home),
                away: elo.get_rating(sport, away),
            }
        except Exception as exc:
            logger.debug("game_analyst: elo lookup failed: %s", exc)

    context["team_scoring"] = {
        home: _recent_form_for(scoring, sport, home),
        away: _recent_form_for(scoring, sport, away),
    }

    context["book_lines_summary"] = _book_lines_summary(game)

    return agent.analyze(context)
