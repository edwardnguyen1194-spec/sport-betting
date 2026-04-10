"""Value-bet detector.

Uses the sharp books (Pinnacle / Circa / Bovada) as the "fair" line,
removes the vig, and flags any other book where the same side pays
materially better. This is classic +EV hunting.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from ..config import Settings, get_settings
from ..models_schema import GameOdds, OddsLine, american_to_implied, kelly_fraction, remove_vig_two_way
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)


SHARP_BOOKS = {"pinnacle", "circa", "bovada", "betonline", "bookmaker"}


class ValueBetStrategy(Strategy):
    name = "value_bets"

    def __init__(self, settings: Optional[Settings] = None, min_edge: float = 0.025) -> None:
        self.settings = settings or get_settings()
        self.min_edge = min_edge

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        recs: List[BetRecommendation] = []
        cfg = self.settings

        for game in games:
            fair_home, fair_away = self._fair_probs(game)
            if fair_home is None or fair_away is None:
                continue

            for side, team, fair_p in (
                ("home", game.home_team, fair_home),
                ("away", game.away_team, fair_away),
            ):
                lines = [
                    l for l in game.lines
                    if l.market == "moneyline" and l.selection.lower() == team.lower() and l.american is not None
                ]
                if not lines:
                    continue
                best = max(lines, key=lambda l: l.decimal or 0.0)
                if best.decimal is None or best.american is None:
                    continue
                implied = american_to_implied(best.american)
                edge = fair_p - implied
                if edge < self.min_edge:
                    continue

                stake_frac = min(
                    kelly_fraction(fair_p, best.decimal, cfg.kelly_fraction),
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
                        confidence=round(fair_p, 4),
                        edge=round(edge, 4),
                        stake_fraction=round(stake_frac, 4),
                        reasoning=(
                            f"Value: {team} paying {best.american:+.0f} at {best.book}; "
                            f"sharp no-vig fair prob {fair_p:.1%} vs implied {implied:.1%} "
                            f"(edge +{edge:.1%})."
                        ),
                        sources=sorted({l.book for l in lines}),
                    )
                )
        return recs

    # ------------------------------------------------------------------

    def _fair_probs(self, game: GameOdds) -> tuple[Optional[float], Optional[float]]:
        sharp_home = self._sharp_line(game, game.home_team)
        sharp_away = self._sharp_line(game, game.away_team)
        if sharp_home is None or sharp_away is None:
            return None, None
        p_home = american_to_implied(sharp_home.american or 0)
        p_away = american_to_implied(sharp_away.american or 0)
        return remove_vig_two_way(p_home, p_away)

    @staticmethod
    def _sharp_line(game: GameOdds, team: str) -> Optional[OddsLine]:
        for l in game.lines:
            if l.market != "moneyline":
                continue
            if l.selection.lower() != team.lower():
                continue
            if l.book.lower() in SHARP_BOOKS and l.american is not None:
                return l
        # Fall back to any available line.
        fallback = [l for l in game.lines if l.market == "moneyline" and l.selection.lower() == team.lower()]
        return fallback[0] if fallback else None
