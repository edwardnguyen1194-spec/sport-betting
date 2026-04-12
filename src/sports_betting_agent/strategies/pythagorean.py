"""Pythagorean Expected Wins Strategy.

From Joe Peta's "Trading Bases" — uses runs scored vs runs allowed
to calculate a team's TRUE win percentage. Teams whose actual record
differs significantly from Pythagorean are due for regression.

If a team is 20-10 but Pythagorean says they should be 16-14,
they're overvalued by the market. Fade them.

If a team is 10-20 but Pythagorean says they should be 14-16,
they're undervalued. Bet on them.

Uses MLB Stats API (free, no key) to get team standings with
runs scored and runs allowed.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

import requests

from ..config import Settings, get_settings
from ..models_schema import GameOdds, american_to_implied, kelly_fraction
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)

MLB_STANDINGS = "https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2026&standingsTypes=regularSeason&hydrate=team"


class PythagoreanStrategy(Strategy):
    """Bet on teams whose Pythagorean record disagrees with actual record."""
    name = "pythagorean"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._pyth_cache: Dict[str, float] = {}
        self._last_fetch = 0

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        recs: List[BetRecommendation] = []
        cfg = self.settings

        # Only works for MLB
        mlb_games = [g for g in games if g.sport == "baseball_mlb"]
        if not mlb_games:
            return recs

        # Fetch Pythagorean data
        import time
        now = time.time()
        if now - self._last_fetch > 3600:  # Cache for 1 hour
            self._fetch_pythagorean()
            self._last_fetch = now

        if not self._pyth_cache:
            return recs

        for game in mlb_games:
            for side in ("home", "away"):
                team = game.home_team if side == "home" else game.away_team

                # Find Pythagorean edge
                pyth_edge = self._pyth_cache.get(team.lower())
                if pyth_edge is None:
                    continue

                # Only bet on significantly undervalued teams (pyth_edge > 0.03)
                # pyth_edge > 0 means team is BETTER than their record shows
                if abs(pyth_edge) < 0.03:
                    continue

                ml_lines = [
                    l for l in game.lines
                    if l.market == "moneyline"
                    and l.selection.lower() == team.lower()
                    and l.american is not None
                ]
                if not ml_lines:
                    continue

                best = max(ml_lines, key=lambda l: l.decimal or 0.0)
                if best.decimal is None or best.american is None:
                    continue

                market_implied = american_to_implied(best.american)

                if pyth_edge > 0:
                    # Team is undervalued — their Pythagorean says they're better
                    confidence = min(market_implied + pyth_edge, 0.90)
                    edge = confidence - market_implied
                else:
                    continue  # Don't bet against teams, just find undervalued

                if edge < 0.03:
                    continue

                stake_frac = min(
                    kelly_fraction(confidence, best.decimal, cfg.kelly_fraction),
                    cfg.max_bet_pct,
                )

                recs.append(
                    BetRecommendation(
                        game_key=game.game_key,
                        sport=game.sport,
                        league=game.league,
                        home_team=game.home_team,
                        away_team=game.away_team,
                        market="moneyline",
                        selection=team,
                        american=best.american,
                        decimal=best.decimal,
                        book=best.book,
                        strategy=self.name,
                        confidence=round(confidence, 4),
                        edge=round(edge, 4),
                        stake_fraction=round(stake_frac, 4),
                        reasoning=(
                            f"Pythagorean: {team} undervalued by {pyth_edge:.1%}. "
                            f"Actual record worse than expected from runs scored/allowed. "
                            f"Regression to mean expected. (Joe Peta method)"
                        ),
                        sources=sorted({l.book for l in ml_lines}),
                    )
                )

        logger.info("pythagorean: %d recs from %d MLB games", len(recs), len(mlb_games))
        return recs

    def _fetch_pythagorean(self):
        """Fetch MLB standings and compute Pythagorean expected wins."""
        try:
            resp = requests.get(MLB_STANDINGS, timeout=5,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("pythagorean: failed to fetch standings: %s", exc)
            return

        self._pyth_cache.clear()
        exponent = 1.83  # Baseball Pythagorean exponent

        for record in data.get("records", []):
            for entry in record.get("teamRecords", []):
                team_name = entry.get("team", {}).get("name", "")
                if not team_name:
                    continue

                wins = entry.get("wins", 0)
                losses = entry.get("losses", 0)
                games = wins + losses
                if games < 10:
                    continue

                rs = entry.get("runsScored", 0)
                ra = entry.get("runsAllowed", 0)
                if rs <= 0 or ra <= 0:
                    continue

                # Pythagorean expected win%
                pyth_wpct = rs ** exponent / (rs ** exponent + ra ** exponent)
                actual_wpct = wins / games

                # Edge = how much better they SHOULD be than their record
                edge = pyth_wpct - actual_wpct
                self._pyth_cache[team_name.lower()] = edge

        logger.info("pythagorean: computed edges for %d teams", len(self._pyth_cache))
