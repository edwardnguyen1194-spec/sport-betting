"""Ensemble strategy runner.

Runs every strategy you hand it, merges duplicate recommendations on
the same game + selection, and blends their confidences. The output
is a single ranked list the paper trader or dashboard can consume.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

from ..config import Settings, get_settings
from ..models_schema import GameOdds
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)


class EnsembleStrategy(Strategy):
    name = "ensemble"

    def __init__(self, strategies: List[Strategy], settings: Optional[Settings] = None) -> None:
        self.strategies = strategies
        self.settings = settings or get_settings()

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        games = list(games)
        buckets: Dict[tuple, BetRecommendation] = {}

        for strat in self.strategies:
            try:
                for rec in strat.generate(games):
                    key = (rec.game_key, rec.market, rec.selection.lower())
                    if key not in buckets:
                        buckets[key] = rec
                        buckets[key].meta.setdefault("contributing_strategies", [])
                        buckets[key].meta["contributing_strategies"].append(strat.name)
                    else:
                        existing = buckets[key]
                        # Blend confidences and take the better price.
                        existing.confidence = round(
                            (existing.confidence + rec.confidence) / 2, 4
                        )
                        existing.edge = round(max(existing.edge, rec.edge), 4)
                        if (rec.decimal or 0.0) > (existing.decimal or 0.0):
                            existing.american = rec.american
                            existing.decimal = rec.decimal
                            existing.book = rec.book
                        strategies = existing.meta.setdefault("contributing_strategies", [])
                        if strat.name not in strategies:
                            strategies.append(strat.name)
                        existing.strategy = "+".join(strategies)
                        existing.reasoning += "\n" + rec.reasoning
            except Exception as exc:  # pragma: no cover
                logger.exception("ensemble: strategy %s failed: %s", strat.name, exc)

        recs = list(buckets.values())
        # Gate on the configured minimum confidence.
        min_conf = self.settings.min_confidence
        filtered = [r for r in recs if r.confidence >= min_conf]
        filtered.sort(key=lambda r: (r.confidence, r.edge), reverse=True)
        logger.info(
            "ensemble: %d recs from %d strategies (%d passed min_conf=%.2f)",
            len(recs),
            len(self.strategies),
            len(filtered),
            min_conf,
        )
        return filtered
