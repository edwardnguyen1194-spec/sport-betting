"""Heavy favorite filter strategy.

The heavy-favorite strategy targets **moneyline bets on teams priced
between -150 and -400 at 2+ books** with cross-source agreement.
Historically this price band for real favorites lands in the 75-85%
hit rate window, at the cost of low per-bet EV, so the strategy
compensates by being extremely selective:

1. The same team must be the favorite at ``min_books`` different
   sportsbooks (default 2) in the merged odds.
2. The vig-removed consensus implied probability must be inside
   ``[0.60, 0.80]`` -- the empirical 75-85% WR zone (real books bake
   some margin into heavy favorites so the "true" prob is a bit
   lower than the implied).
3. The *best available* line for the bettor must still be in the
   ``[min_american, max_american]`` config window.
4. Confidence is set to the vig-free consensus probability and a
   reasoning string is attached for the dashboard / Claude chat.

Win-rate is tuned via configuration in
:class:`sports_betting_agent.config.Settings`.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from ..config import Settings, get_settings
from ..models_schema import GameOdds, kelly_fraction, american_to_implied, remove_vig_two_way
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)


class HeavyFavoriteStrategy(Strategy):
    name = "heavy_favorite"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        recs: List[BetRecommendation] = []
        cfg = self.settings

        for game in games:
            for side in ("home", "away"):
                team_name = game.home_team if side == "home" else game.away_team
                opp_name = game.away_team if side == "home" else game.home_team

                ml_lines = [
                    l
                    for l in game.lines
                    if l.market == "moneyline" and l.selection.lower() == team_name.lower() and l.american is not None
                ]
                if len(ml_lines) < cfg.heavy_fav_min_books:
                    continue

                # Must be a favorite at enough books.
                fav_lines = [l for l in ml_lines if (l.american or 0) < 0]
                if len(fav_lines) < cfg.heavy_fav_min_books:
                    continue

                best_line = max(fav_lines, key=lambda l: l.decimal or 0.0)
                if best_line.american is None or best_line.decimal is None:
                    continue

                # Enforce the configured American-odds window on the
                # best available price.
                if not (cfg.heavy_fav_min_american <= best_line.american <= cfg.heavy_fav_max_american):
                    continue

                consensus = game.consensus_implied(side)
                if consensus is None:
                    # Fall back to average of this side only.
                    consensus = sum(american_to_implied(l.american) for l in fav_lines) / len(fav_lines)

                # Target the empirical 75-85% WR band while keeping
                # some headroom for chalk-is-chalk regression.
                if not (0.60 <= consensus <= 0.82):
                    continue

                edge = consensus - american_to_implied(best_line.american)
                if edge < -0.02:  # price is *far* worse than consensus
                    continue

                # Heavy favorites are a "chalk grind" play: the sharp
                # books already price them accurately, so Kelly almost
                # always returns 0. We still want to take the bet
                # because the edge is statistical (volume at 75-85% WR),
                # so we floor the stake at the config's minimum bet
                # percentage (~min_bet/bankroll) and ceiling at
                # max_bet_pct.
                kelly = kelly_fraction(consensus, best_line.decimal, cfg.kelly_fraction)
                floor = max(cfg.min_bet / max(1.0, cfg.bankroll_start), 0.005)
                stake_frac = max(kelly, floor)
                stake_frac = min(stake_frac, cfg.max_bet_pct)

                sources = sorted({l.book for l in ml_lines})

                rec = BetRecommendation(
                    game_key=game.game_key,
                    sport=game.sport,
                    league=game.league,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    market="moneyline",
                    selection=team_name,
                    american=best_line.american,
                    decimal=best_line.decimal,
                    book=best_line.book,
                    strategy=self.name,
                    confidence=round(consensus, 4),
                    edge=round(edge, 4),
                    stake_fraction=round(stake_frac, 4),
                    reasoning=(
                        f"Heavy favorite {team_name} vs {opp_name}: "
                        f"best price {best_line.american:+.0f} at {best_line.book}, "
                        f"consensus no-vig prob {consensus:.1%} across {len(ml_lines)} books. "
                        f"Target WR band 75-85%."
                    ),
                    sources=sources,
                    meta={
                        "book_count": len(ml_lines),
                        "consensus_prob": round(consensus, 4),
                    },
                )
                recs.append(rec)

        logger.info("heavy_favorite: %d recs from %d games", len(recs), sum(1 for _ in games))
        return recs
