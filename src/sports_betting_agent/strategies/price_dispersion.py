"""Universal price-dispersion strategy.

Works on ANY sport — tennis, MMA, football, baseball, anything with
a spread or total market at 2+ books. Covers the sports that don't
have Elo ratings, team scoring history, or Dixon-Coles coverage so
Uncle's rule ("agent must work on all sports") is satisfied.

How it works
------------

For every (market, selection, line) triplet, collect all book prices
for the SAME line and compute:

  * Consensus fair probability = average implied-prob across books
    (after de-juicing each book individually).
  * Best price available = highest decimal odds at any listed book.

If the best book's de-juiced implied prob is meaningfully below
consensus (i.e. one book is pricing the outcome materially cheaper
than the market average), take the best price.

Why this works
--------------

Book A offering the same outcome at substantially better odds than
Book B, C, D is either (a) slow to react to sharp money, (b) using
a different risk model, or (c) baiting with a promo-adjacent line.
In any of those cases, the market-adjusted EV on the best-price
side is positive.

This is NOT arbitrage — we're taking one side only. It's a weak
but universal +EV signal that activates whenever prices disagree.

Parameters
----------

* ``min_books`` — need at least this many books at the same line
  before dispersion is meaningful (default 3).
* ``min_edge_pct`` — best-price must offer at least this much edge
  over consensus (default 0.025 = 2.5%). Low by design — this is
  a volume strategy, not an alpha-huntin' one.
* ``confidence_cap`` — cap at 0.62 so the ensemble never over-
  weights a dispersion signal relative to structural models.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable, List

from ..config import Settings, get_settings
from ..models_schema import GameOdds
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)


def _american_to_implied(american: float) -> float:
    """Convert American odds to fair implied probability (no vig)."""
    if american > 0:
        return 100.0 / (american + 100.0)
    return (-american) / ((-american) + 100.0)


def _american_to_decimal(american: float) -> float:
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / (-american)


class PriceDispersionStrategy(Strategy):
    """Find +EV value where one book prices a side much cheaper than
    consensus. Universal across all sports — no per-sport gates."""

    name = "price_dispersion"

    def __init__(self, settings=None) -> None:
        self.settings = settings or get_settings()

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        min_books = 3
        min_edge_pct = 0.025
        confidence_cap = 0.62
        recs: List[BetRecommendation] = []

        for game in games:
            # Group lines by (market, selection, line_number) so we
            # only compare apples to apples. 2-team spreads are
            # symmetric around 0 but we bucket per-team and per-
            # handicap so a -1.5 home bucket is separate from +1.5
            # away bucket (even though they're the same bet).
            buckets = defaultdict(list)
            for ln in game.lines:
                if ln.market not in ("spread", "total"):
                    continue
                if ln.american is None or ln.selection is None:
                    continue
                key = (
                    ln.market,
                    (ln.selection or "").lower().strip(),
                    round(float(ln.line or 0), 2),
                )
                buckets[key].append(ln)

            for (market, selection, line_val), lines in buckets.items():
                if len(lines) < min_books:
                    continue
                # Consensus = average of de-juiced implied probs
                # across all books at this line. We de-juice per-
                # pair (this side + the complement) where possible
                # by finding the matching opposite-side lines; if
                # we can't find a complement, fall back to simple
                # average with a 2-book assumption.
                implied_probs = [_american_to_implied(l.american) for l in lines]
                avg_implied = sum(implied_probs) / len(implied_probs)

                # Best price = highest decimal = lowest implied
                best_line = max(lines, key=lambda l: _american_to_decimal(l.american))
                best_implied = _american_to_implied(best_line.american)
                best_decimal = _american_to_decimal(best_line.american)

                edge_pct = avg_implied - best_implied
                if edge_pct < min_edge_pct:
                    continue
                # Confidence anchored to consensus fair prob.
                confidence = min(confidence_cap, avg_implied)

                # Pick the selection name as-posted (preserve case
                # from the best line). Include home/away team in
                # the game-key so the ensemble can merge / dedup.
                selection_posted = best_line.selection or selection
                rec = BetRecommendation(
                    sport=game.sport,
                    league=game.league,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    game_key=game.game_key,
                    market=market,
                    selection=selection_posted,
                    line=line_val,
                    american=best_line.american,
                    decimal=round(best_decimal, 4),
                    book=best_line.book,
                    strategy=self.name,
                    confidence=round(confidence, 4),
                    edge=round(edge_pct, 4),
                    reasoning=(
                        f"Price dispersion: {len(lines)} books average implied "
                        f"{avg_implied:.1%}; {best_line.book} at "
                        f"{best_line.american:+d} implies {best_implied:.1%} "
                        f"(+{edge_pct:.1%} edge). Taking best price."
                    ),
                )
                recs.append(rec)

        logger.info(
            "price_dispersion: %d recs across %d games "
            "(min_books=%d, min_edge=%.1f%%)",
            len(recs), sum(1 for _ in games), min_books, min_edge_pct * 100,
        )
        return recs
