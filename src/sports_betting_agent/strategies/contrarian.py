"""Contrarian betting strategy.

When the public heavily backs one side (>70% of bets) but the sharp
line hasn't moved to reflect that money, the sharp books are telling
you the public is wrong. Bet the unpopular side.

Requires public betting percentage data in game.meta (populated by
the Action Network fetcher).
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from ..config import Settings, get_settings
from ..models_schema import GameOdds, american_to_implied, kelly_fraction
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)

SHARP_BOOKS = {"pinnacle", "circa", "bovada", "betonline", "bookmaker"}


class ContrarianStrategy(Strategy):
    name = "contrarian"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.public_threshold = getattr(self.settings, "contrarian_public_threshold", 0.70)
        self.min_books = getattr(self.settings, "contrarian_min_books", 3)

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        recs: List[BetRecommendation] = []
        cfg = self.settings

        for game in games:
            # Check for public betting data
            public_home = game.meta.get("public_ml_home_pct")
            public_away = game.meta.get("public_ml_away_pct")

            if public_home is None and public_away is None:
                continue

            # Determine public side and contrarian side
            if public_home is not None and public_home > self.public_threshold:
                contrarian_side = "away"
                contrarian_team = game.away_team
                public_pct = public_home
            elif public_away is not None and public_away > self.public_threshold:
                contrarian_side = "home"
                contrarian_team = game.home_team
                public_pct = public_away
            else:
                continue  # No lopsided public action

            # Get moneyline lines for the contrarian side
            ml_lines = [
                l for l in game.lines
                if l.market == "moneyline"
                and l.selection.lower() == contrarian_team.lower()
                and l.american is not None
            ]
            if len(ml_lines) < self.min_books:
                continue

            # Find best price for contrarian side
            best = max(ml_lines, key=lambda l: l.decimal or 0.0)
            if best.decimal is None or best.american is None:
                continue

            # Base confidence from how lopsided the public is
            confidence = public_pct

            # Boost if sharp books agree with contrarian side
            sharp_lines = [l for l in ml_lines if l.book.lower() in SHARP_BOOKS]
            if sharp_lines:
                sharp_implied = sum(american_to_implied(l.american) for l in sharp_lines) / len(sharp_lines)
                if sharp_implied > 0.40:
                    confidence = min(confidence + 0.05, 0.95)

            implied_best = american_to_implied(best.american)
            edge = confidence - implied_best
            if edge < 0.02:
                continue

            stake_frac = min(
                kelly_fraction(confidence, best.decimal, cfg.kelly_fraction),
                cfg.max_bet_pct,
            )

            public_side_team = game.home_team if contrarian_side == "away" else game.away_team
            recs.append(
                BetRecommendation(
                    game_key=game.game_key,
                    sport=game.sport,
                    league=game.league,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    market="moneyline",
                    selection=contrarian_team,
                    american=best.american,
                    decimal=best.decimal,
                    book=best.book,
                    strategy=self.name,
                    confidence=round(confidence, 4),
                    edge=round(edge, 4),
                    stake_fraction=round(stake_frac, 4),
                    reasoning=(
                        f"Contrarian: {public_pct:.0%} public on {public_side_team} "
                        f"but sharps disagree. Betting {contrarian_team} at "
                        f"{best.american:+.0f} ({best.book}). "
                        f"Edge {edge:.1%} vs implied {implied_best:.1%}."
                    ),
                    sources=sorted({l.book for l in ml_lines}),
                )
            )

        logger.info("contrarian: %d recs", len(recs))
        return recs
