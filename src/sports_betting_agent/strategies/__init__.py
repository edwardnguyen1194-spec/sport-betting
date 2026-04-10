"""Betting strategies.

Each strategy consumes a list of :class:`GameOdds` and emits a list
of :class:`BetRecommendation` objects that downstream code (paper
trader, dashboard, Claude chat) can reason about uniformly.
"""

from .base import BetRecommendation, Strategy
from .heavy_favorite import HeavyFavoriteStrategy
from .value_bets import ValueBetStrategy
from .ensemble import EnsembleStrategy
from .player_props import PlayerPropsStrategy, PropMarket

__all__ = [
    "BetRecommendation",
    "Strategy",
    "HeavyFavoriteStrategy",
    "ValueBetStrategy",
    "EnsembleStrategy",
    "PlayerPropsStrategy",
    "PropMarket",
]
