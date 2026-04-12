"""Spread value-bet detector.

Same principle as the moneyline value-bet strategy but for spread
markets. Groups spread lines by (team, handicap) across books, finds
the sharp book's juice, removes vig, and flags soft books offering
better juice on the same spread number.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from ..config import Settings, get_settings
from ..models_schema import GameOdds, OddsLine, american_to_implied, kelly_fraction, remove_vig_two_way
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)

SHARP_BOOKS = {"pinnacle", "circa", "bovada", "betonline", "bookmaker"}


class SpreadValueStrategy(Strategy):
    name = "spread_value"

    def __init__(self, settings: Optional[Settings] = None, min_edge: float = 0.025) -> None:
        self.settings = settings or get_settings()
        self.min_edge = getattr(self.settings, "spread_value_min_edge", min_edge)

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        recs: List[BetRecommendation] = []
        cfg = self.settings

        for game in games:
            groups = game.spread_lines_grouped()

            for (selection, handicap), lines in groups.items():
                if len(lines) < 2:
                    continue

                # Find sharp and soft lines for this side
                sharp_lines = [l for l in lines if l.book.lower() in SHARP_BOOKS]
                if not sharp_lines:
                    continue

                # Find the opposite side at the opposite handicap
                opposite_handicap = -handicap
                opposite_key = None
                for key in groups:
                    if key[1] == opposite_handicap and key[0] != selection:
                        opposite_key = key
                        break
                if opposite_key is None:
                    continue

                opposite_lines = groups[opposite_key]
                opposite_sharp = [l for l in opposite_lines if l.book.lower() in SHARP_BOOKS]
                if not opposite_sharp:
                    continue

                # Remove vig from sharp book spread juice
                sharp_implied = sum(american_to_implied(l.american) for l in sharp_lines) / len(sharp_lines)
                opp_implied = sum(american_to_implied(l.american) for l in opposite_sharp) / len(opposite_sharp)
                fair_this, fair_opp = remove_vig_two_way(sharp_implied, opp_implied)

                # Find the best price at any book for this side+handicap
                best = max(lines, key=lambda l: l.decimal or 0.0)
                if best.decimal is None or best.american is None:
                    continue
                implied_best = american_to_implied(best.american)
                edge = fair_this - implied_best

                if edge < self.min_edge:
                    continue

                # Resolve team name from normalized selection
                team_name = _resolve_team(game, selection)
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
                        market="spread",
                        selection=team_name,
                        american=best.american,
                        decimal=best.decimal,
                        line=handicap,
                        book=best.book,
                        strategy=self.name,
                        confidence=round(fair_this, 4),
                        edge=round(edge, 4),
                        stake_fraction=round(stake_frac, 4),
                        reasoning=(
                            f"Spread value: {team_name} {handicap:+.1f} paying "
                            f"{best.american:+.0f} at {best.book}; "
                            f"sharp fair prob {fair_this:.1%} vs implied {implied_best:.1%} "
                            f"(edge +{edge:.1%})."
                        ),
                        sources=sorted({l.book for l in lines}),
                    )
                )

        logger.info("spread_value: %d recs", len(recs))
        return recs


def _resolve_team(game: GameOdds, normalized: str) -> str:
    """Return the original-cased team name from a normalized selection."""
    from ..models_schema import _norm
    for t in (game.home_team, game.away_team):
        if _norm(t) == normalized or t.lower().strip() == normalized:
            return t
    return normalized
