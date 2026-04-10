"""Odds aggregator.

Runs every enabled fetcher in parallel and merges their output into
a single, de-duplicated list of :class:`GameOdds`, with sources
tracked in each game's ``meta``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Type

from ..config import Settings, get_settings
from ..models_schema import GameOdds, merge_games
from .actionnetwork import ActionNetworkFetcher
from .base import BaseFetcher, FetcherError
from .bovada import BovadaFetcher
from .covers import CoversFetcher
from .espn import ESPNFetcher
from .sbr import SBRFetcher
from .scoresandodds import ScoresAndOddsFetcher
from .vegasinsider import VegasInsiderFetcher


logger = logging.getLogger(__name__)


_FETCHER_REGISTRY: Dict[str, Type[BaseFetcher]] = {
    "bovada": BovadaFetcher,
    "espn": ESPNFetcher,
    "scoresandodds": ScoresAndOddsFetcher,
    "vegasinsider": VegasInsiderFetcher,
    "covers": CoversFetcher,
    "actionnetwork": ActionNetworkFetcher,
    "sbr": SBRFetcher,
}


class OddsAggregator:
    """Pull + merge odds from every enabled source."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        fetchers: Optional[Dict[str, BaseFetcher]] = None,
    ) -> None:
        self.settings = settings or get_settings()
        if fetchers is not None:
            self.fetchers = fetchers
        else:
            self.fetchers = {
                name: cls(self.settings)
                for name, cls in _FETCHER_REGISTRY.items()
                if name in self.settings.enabled_sources
            }

    # ------------------------------------------------------------------

    def list_sources(self) -> List[str]:
        return sorted(self.fetchers.keys())

    def fetch_sport(self, sport_key: str) -> List[GameOdds]:
        """Fetch a single sport from every enabled source in parallel."""

        results: List[GameOdds] = []
        if not self.fetchers:
            return results

        with ThreadPoolExecutor(max_workers=max(4, len(self.fetchers))) as pool:
            future_map = {
                pool.submit(self._safe_fetch, fetcher, sport_key): name
                for name, fetcher in self.fetchers.items()
            }
            for fut in as_completed(future_map):
                name = future_map[fut]
                try:
                    results.extend(fut.result())
                except Exception as exc:  # pragma: no cover
                    logger.warning("aggregator: %s raised %s", name, exc)

        merged = merge_games(results)
        logger.info(
            "aggregator.fetch_sport(%s): %d raw -> %d merged games across %d sources",
            sport_key,
            len(results),
            len(merged),
            len(self.fetchers),
        )
        return merged

    def fetch_sports(self, sport_keys: Iterable[str]) -> List[GameOdds]:
        """Fetch and merge multiple sports."""

        all_games: List[GameOdds] = []
        for sport_key in sport_keys:
            all_games.extend(self.fetch_sport(sport_key))
        return all_games

    # ------------------------------------------------------------------

    @staticmethod
    def _safe_fetch(fetcher: BaseFetcher, sport_key: str) -> List[GameOdds]:
        try:
            return fetcher.fetch(sport_key)
        except FetcherError as exc:
            logger.info("%s: skip %s (%s)", fetcher.source_name, sport_key, exc)
            return []
        except Exception as exc:  # pragma: no cover
            logger.exception("%s crashed on %s: %s", fetcher.source_name, sport_key, exc)
            return []
