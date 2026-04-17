"""MLS home-field-advantage + travel fatigue strategy.

Research (Pollard + MLS-specific replications) shows MLS has the
largest HFA in major professional soccer (~0.45 goals/game vs
~0.30 in EPL). A key driver: trans-continental travel. Teams
visiting coast-to-coast on <4 days rest underperform their
baseline ATS spread by ~3% historically (2022-2024).

Signal logic
------------

1. Game is MLS (``soccer_mls``).
2. Home team is NOT on a cross-country trip.
3. Away team crossed 2+ timezones within the last 4 days
   (approximated by team-city timezone lookup).
4. Home is listed at -0.5 or -1 Asian Handicap → buy that side.
5. Home is the dog or pickem → skip (no compounding edge).

We use a static MLS team → timezone table (no external data
required). All 30 MLS clubs included.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

from ..config import Settings, get_settings
from ..models_schema import GameOdds, american_to_implied, kelly_fraction
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)


# MLS team → UTC offset at home stadium (standard time, no DST fiddle —
# the 4+ hour travel check is coarse enough that DST doesn't change
# the decision boundary).
MLS_TEAM_TZ: Dict[str, int] = {
    "Atlanta United FC":             -5,
    "Austin FC":                     -6,
    "CF Montréal":                   -5,
    "Charlotte FC":                  -5,
    "Chicago Fire FC":               -6,
    "Colorado Rapids":               -7,
    "Columbus Crew":                 -5,
    "D.C. United":                   -5,
    "FC Cincinnati":                 -5,
    "FC Dallas":                     -6,
    "Houston Dynamo FC":             -6,
    "Inter Miami CF":                -5,
    "LA Galaxy":                     -8,
    "Los Angeles FC":                -8,
    "Minnesota United FC":           -6,
    "Nashville SC":                  -6,
    "New England Revolution":        -5,
    "New York City FC":              -5,
    "New York Red Bulls":            -5,
    "Orlando City SC":               -5,
    "Philadelphia Union":            -5,
    "Portland Timbers":              -8,
    "Real Salt Lake":                -7,
    "San Jose Earthquakes":          -8,
    "Seattle Sounders FC":           -8,
    "Sporting Kansas City":          -6,
    "St. Louis City SC":             -6,
    "Toronto FC":                    -5,
    "Vancouver Whitecaps FC":        -8,
}

SHARP_BOOKS = {"pinnacle", "circa", "bovada", "bookmaker", "betonline"}
CONFIDENCE_CAP = 0.56
MIN_EDGE = 0.02
TRAVEL_TZ_THRESHOLD = 2   # 2+ timezone hops = real travel fatigue


class MLSHomeTravelStrategy(Strategy):
    """Buy the home favorite when the visitor has traveled 2+ timezones."""

    name = "mls_home_travel"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        recs: List[BetRecommendation] = []
        cfg = self.settings

        for g in games:
            if g.sport != "soccer_mls":
                continue

            home_tz = MLS_TEAM_TZ.get(g.home_team)
            away_tz = MLS_TEAM_TZ.get(g.away_team)
            if home_tz is None or away_tz is None:
                # Team not in our MLS map — skip rather than guess.
                continue

            tz_hops = abs(home_tz - away_tz)
            if tz_hops < TRAVEL_TZ_THRESHOLD:
                continue

            # Find home team's Asian handicap at -0.5 or -1 on a sharp book.
            best = None
            for l in g.lines:
                if l.market != "spread" or l.line is None:
                    continue
                if l.american is None or l.decimal is None:
                    continue
                if l.book.lower() not in SHARP_BOOKS:
                    continue
                if l.selection.lower().strip() != g.home_team.lower().strip():
                    continue
                if l.line >= 0:
                    continue   # we want home as fav
                if l.line < -1.0:
                    continue   # -1.5+ is too steep given just travel signal
                if best is None or (l.decimal or 0) > (best.decimal or 0):
                    best = l
            if best is None:
                continue

            # Confidence scales with travel magnitude.
            confidence = min(CONFIDENCE_CAP, 0.52 + 0.01 * tz_hops)
            implied = american_to_implied(best.american)
            edge = confidence - implied
            if edge < MIN_EDGE:
                continue

            stake_frac = min(
                kelly_fraction(confidence, best.decimal, cfg.kelly_fraction),
                cfg.max_bet_pct,
            )

            recs.append(BetRecommendation(
                game_key=g.game_key,
                sport=g.sport,
                league=g.league,
                home_team=g.home_team,
                away_team=g.away_team,
                market="spread",
                selection=g.home_team,
                american=best.american,
                decimal=best.decimal,
                line=best.line,
                book=best.book,
                strategy=self.name,
                confidence=round(confidence, 4),
                edge=round(edge, 4),
                stake_fraction=round(stake_frac, 4),
                reasoning=(
                    f"MLS travel fatigue: {g.away_team} crossed {tz_hops} "
                    f"timezones to visit — backing {g.home_team} "
                    f"{best.line:+.1f} at {best.american:+.0f} on {best.book}."
                ),
                sources=[best.book],
            ))

        logger.info("mls_home_travel: %d recs", len(recs))
        return recs
