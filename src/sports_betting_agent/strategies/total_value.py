"""Total (Over/Under) value-bet detector.

Same principle as spread value but for total markets. Groups total
lines by (Over/Under, total_number) across books, finds the sharp
book's juice, removes vig, and flags soft books offering better
juice on the same total number.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from ..config import Settings, get_settings
from ..models_schema import GameOdds, OddsLine, american_to_implied, kelly_fraction, remove_vig_two_way
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)

SHARP_BOOKS = {"pinnacle", "circa", "bovada", "betonline", "bookmaker"}


class TotalValueStrategy(Strategy):
    name = "total_value"

    def __init__(self, settings: Optional[Settings] = None, min_edge: float = 0.025) -> None:
        self.settings = settings or get_settings()
        self.min_edge = getattr(self.settings, "total_value_min_edge", min_edge)

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        recs: List[BetRecommendation] = []
        cfg = self.settings

        for game in games:
            groups = game.total_lines_grouped()

            for (selection, total_num), lines in groups.items():
                sharp_lines = [l for l in lines if l.book.lower() in SHARP_BOOKS]
                if not sharp_lines:
                    continue

                # Find the opposite side at the same total number
                opposite_sel = "under" if selection == "over" else "over"
                opposite_key = (opposite_sel, total_num)
                if opposite_key not in groups:
                    continue

                opposite_lines = groups[opposite_key]
                opposite_sharp = [l for l in opposite_lines if l.book.lower() in SHARP_BOOKS]
                if not opposite_sharp:
                    continue

                # Remove vig from sharp book total juice
                sharp_implied = sum(american_to_implied(l.american) for l in sharp_lines) / len(sharp_lines)
                opp_implied = sum(american_to_implied(l.american) for l in opposite_sharp) / len(opposite_sharp)
                fair_this, fair_opp = remove_vig_two_way(sharp_implied, opp_implied)

                # Find best price at any book for this side+total
                best = max(lines, key=lambda l: l.decimal or 0.0)
                if best.decimal is None or best.american is None:
                    continue
                implied_best = american_to_implied(best.american)
                edge = fair_this - implied_best

                # Multi-book edge (classic value) vs single sharp-book confidence pick
                multi_book = len(lines) >= 2
                if multi_book:
                    if edge < self.min_edge:
                        continue
                else:
                    # Only Bovada (or other sharp) quotes — surface confident picks
                    # (fair-vig prob above 55%) since we can't compute cross-book edge.
                    if fair_this < 0.55:
                        continue

                display_sel = "Over" if selection == "over" else "Under"
                stake_frac = min(
                    kelly_fraction(fair_this, best.decimal, cfg.kelly_fraction),
                    cfg.max_bet_pct,
                )

                recs.append(
                    BetRecommendation(
                        game_key=game.game_key,
                        sport=game.sport,
                        league=game.league,
                        home_team=game.home_team,
                        away_team=game.away_team,
                        market="total",
                        selection=display_sel,
                        american=best.american,
                        decimal=best.decimal,
                        line=total_num,
                        book=best.book,
                        strategy=self.name,
                        confidence=round(fair_this, 4),
                        edge=round(edge, 4),
                        stake_fraction=round(stake_frac, 4),
                        reasoning=(
                            f"Total value: {display_sel} {total_num} in "
                            f"{game.away_team} @ {game.home_team} paying "
                            f"{best.american:+.0f} at {best.book}; "
                            f"sharp fair prob {fair_this:.1%} vs implied {implied_best:.1%} "
                            f"(edge +{edge:.1%})."
                        ),
                        sources=sorted({l.book for l in lines}),
                    )
                )

        logger.info("total_value: %d recs", len(recs))
        return recs
