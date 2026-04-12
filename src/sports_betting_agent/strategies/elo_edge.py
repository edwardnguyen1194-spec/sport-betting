"""Elo Edge Strategy.

Uses our own Elo power ratings to generate independent win
probabilities, then compares against market odds to find
mispriced games. This is the core of what separates sharp
bettors from the public — having your OWN line.

When our Elo says Team A has 65% chance but the market
implies 55%, that's a 10% edge worth betting.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from ..config import Settings, get_settings
from ..models_schema import GameOdds, american_to_implied, kelly_fraction
from ..power_ratings import EloRatings
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)


class EloEdgeStrategy(Strategy):
    """Find edges where our Elo disagrees with the market."""
    name = "elo_edge"

    def __init__(self, settings: Optional[Settings] = None,
                 elo: Optional[EloRatings] = None,
                 min_edge: float = 0.05) -> None:
        self.settings = settings or get_settings()
        self.elo = elo or EloRatings(self.settings.data_dir)
        self.min_edge = min_edge

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        recs: List[BetRecommendation] = []
        cfg = self.settings

        for game in games:
            # Get our Elo prediction
            elo_home = self.elo.predict_win_prob(game.sport, game.home_team, game.away_team)
            if elo_home is None:
                continue
            elo_away = 1.0 - elo_home

            # Check both sides
            for side, team, elo_prob in [
                ("home", game.home_team, elo_home),
                ("away", game.away_team, elo_away),
            ]:
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
                edge = elo_prob - market_implied

                if edge < self.min_edge:
                    continue

                # Get Elo ratings for context
                h_rating = self.elo.get_rating(game.sport, game.home_team) or 1500
                a_rating = self.elo.get_rating(game.sport, game.away_team) or 1500

                stake_frac = min(
                    kelly_fraction(elo_prob, best.decimal, cfg.kelly_fraction),
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
                        confidence=round(elo_prob, 4),
                        edge=round(edge, 4),
                        stake_fraction=round(stake_frac, 4),
                        reasoning=(
                            f"Elo edge: {team} rated {h_rating if side == 'home' else a_rating:.0f} "
                            f"vs {a_rating if side == 'home' else h_rating:.0f}. "
                            f"Our model: {elo_prob:.0%} win prob vs market {market_implied:.0%} "
                            f"(edge +{edge:.1%}). Best price {best.american:+.0f} at {best.book}."
                        ),
                        sources=sorted({l.book for l in ml_lines}),
                    )
                )

        logger.info("elo_edge: %d recs", len(recs))
        return recs
