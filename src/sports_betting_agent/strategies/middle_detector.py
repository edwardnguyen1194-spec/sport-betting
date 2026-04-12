"""Middle and scalp opportunity detector.

From Stanford Wong's "Sharp Sports Betting": when different books
post different spread lines, you can sometimes bet both sides at a
profit (scalp) or create a "middle" where both bets win if the result
falls between the two lines.

Example: Book A has Team X -3 (-110), Book B has Team Y +4 (-110).
If Team X wins by exactly 3, you push one and win one.
If Team X wins by exactly 4, you push one and win one.
If Team X wins by 3.5 (between 3 and 4), you WIN BOTH.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from ..config import Settings, get_settings
from ..models_schema import GameOdds, OddsLine, american_to_implied, kelly_fraction
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)


class MiddleDetectorStrategy(Strategy):
    """Detect middle opportunities across books on spread markets."""
    name = "middle_detector"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        recs: List[BetRecommendation] = []
        cfg = self.settings

        for game in games:
            # Get all spread lines for each side
            home_spreads = [
                l for l in game.lines
                if l.market == "spread"
                and l.selection.lower() == game.home_team.lower()
                and l.line is not None and l.american is not None
            ]
            away_spreads = [
                l for l in game.lines
                if l.market == "spread"
                and l.selection.lower() == game.away_team.lower()
                and l.line is not None and l.american is not None
            ]

            if not home_spreads or not away_spreads:
                continue

            # Find the best line for each side (most favorable spread)
            # For home favorite: most negative spread at best juice
            # For away dog: most positive spread at best juice
            best_home = max(home_spreads, key=lambda l: l.decimal or 0)
            best_away = max(away_spreads, key=lambda l: l.decimal or 0)

            # Check for middle: home spread and away spread don't overlap
            # e.g., home -3 and away +4 → middle at 3-4
            home_line = best_home.line  # e.g., -3
            away_line = best_away.line  # e.g., +4

            if home_line is None or away_line is None:
                continue

            # Gap = away_line + home_line (both include sign)
            # If home = -3, away = +4: gap = +4 + (-3) = +1 (1 point middle)
            # If home = -3, away = +3: gap = 0 (no middle, just push)
            # If home = -3, away = +2.5: gap = -0.5 (no middle)
            gap = away_line + home_line
            if gap <= 0:
                continue  # No middle exists

            # Calculate if the middle is profitable
            # Cost: juice on both sides
            home_cost = 1.0  # risk 1 unit
            away_cost = 1.0
            home_payout = (best_home.decimal or 1) - 1  # net profit if home wins
            away_payout = (best_away.decimal or 1) - 1

            # Middle probability estimate: ~5% per point of gap for spreads
            middle_prob = min(gap * 0.05, 0.25)

            # EV calculation:
            # If middle hits: win both (+home_payout + away_payout)
            # If one side wins: win one, lose one (net = payout - 1)
            # Simplified: net_ev = middle_prob * (home_payout + away_payout) - (1 - middle_prob) * max_juice_loss
            juice_loss = max(1 - home_payout, 1 - away_payout, 0)
            net_ev = middle_prob * (home_payout + away_payout) - (1 - middle_prob) * juice_loss * 0.5

            if net_ev < 0.01:
                continue

            confidence = min(0.55 + middle_prob, 0.85)
            edge = net_ev

            recs.append(
                BetRecommendation(
                    game_key=game.game_key,
                    sport=game.sport,
                    league=game.league,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    market="spread",
                    selection=f"{game.home_team} / {game.away_team} MIDDLE",
                    american=best_home.american,
                    decimal=best_home.decimal or 1.0,
                    line=home_line,
                    book=f"{best_home.book} + {best_away.book}",
                    strategy=self.name,
                    confidence=round(confidence, 4),
                    edge=round(edge, 4),
                    stake_fraction=round(min(edge * 0.5, cfg.max_bet_pct), 4),
                    reasoning=(
                        f"Middle: {game.home_team} {home_line:+.1f} at {best_home.book} "
                        f"({best_home.american:+.0f}) + {game.away_team} {away_line:+.1f} at "
                        f"{best_away.book} ({best_away.american:+.0f}). "
                        f"Gap={gap:.1f}pts, middle prob ~{middle_prob:.0%}, EV={net_ev:.1%}."
                    ),
                    sources=sorted({best_home.book, best_away.book}),
                )
            )

        logger.info("middle_detector: %d opportunities", len(recs))
        return recs
