"""Betting strategies.

Each strategy consumes a list of :class:`GameOdds` and emits a list
of :class:`BetRecommendation` objects that downstream code (paper
trader, dashboard, Claude chat) can reason about uniformly.
"""

from .base import BetRecommendation, Strategy
from .heavy_favorite import HeavyFavoriteStrategy
from .value_bets import ValueBetStrategy
from .spread_value import SpreadValueStrategy
from .total_value import TotalValueStrategy
from .total_projection import TotalProjectionStrategy
from .steam_follow import SteamFollowStrategy
from .public_fade import PublicFadeStrategy
from .contrarian import ContrarianStrategy
from .middle_detector import MiddleDetectorStrategy
from .situational import SituationalStrategy
from .elo_edge import EloEdgeStrategy
from .pythagorean import PythagoreanStrategy
from .ensemble import EnsembleStrategy
from .player_props import PlayerPropsStrategy, PropMarket

__all__ = [
    "BetRecommendation",
    "Strategy",
    "HeavyFavoriteStrategy",
    "ValueBetStrategy",
    "SpreadValueStrategy",
    "TotalValueStrategy",
    "TotalProjectionStrategy",
    "SteamFollowStrategy",
    "PublicFadeStrategy",
    "ContrarianStrategy",
    "MiddleDetectorStrategy",
    "SituationalStrategy",
    "EloEdgeStrategy",
    "PythagoreanStrategy",
    "EnsembleStrategy",
    "PlayerPropsStrategy",
    "PropMarket",
]
