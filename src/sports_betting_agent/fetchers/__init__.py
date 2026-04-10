"""Odds source fetchers.

Each fetcher exposes a ``fetch(sport_key)`` method that returns a list
of :class:`sports_betting_agent.models_schema.GameOdds` objects.
"""

from .base import BaseFetcher, FetcherError
from .bovada import BovadaFetcher
from .espn import ESPNFetcher
from .scoresandodds import ScoresAndOddsFetcher
from .vegasinsider import VegasInsiderFetcher
from .covers import CoversFetcher
from .actionnetwork import ActionNetworkFetcher
from .sbr import SBRFetcher
from .aggregator import OddsAggregator

__all__ = [
    "BaseFetcher",
    "FetcherError",
    "BovadaFetcher",
    "ESPNFetcher",
    "ScoresAndOddsFetcher",
    "VegasInsiderFetcher",
    "CoversFetcher",
    "ActionNetworkFetcher",
    "SBRFetcher",
    "OddsAggregator",
]
