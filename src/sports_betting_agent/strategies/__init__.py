"""Betting strategies.

Each strategy consumes a list of :class:`GameOdds` and emits a list
of :class:`BetRecommendation` objects that downstream code (paper
trader, dashboard, Claude chat) can reason about uniformly.
"""

from .base import BetRecommendation, Strategy
from .spread_value import SpreadValueStrategy
from .total_value import TotalValueStrategy
from .total_projection import TotalProjectionStrategy
from .steam_follow import SteamFollowStrategy
from .public_fade import PublicFadeStrategy
from .reverse_line_movement import ReverseLineMovementStrategy
from .nhl_goalie import NHLGoalieB2BStrategy
from .mls_travel import MLSHomeTravelStrategy
from .middle_detector import MiddleDetectorStrategy
from .situational import SituationalStrategy
from .elo_edge import EloEdgeStrategy
from .ensemble import EnsembleStrategy
from .player_props import PlayerPropsStrategy, PropMarket

__all__ = [
    "BetRecommendation",
    "Strategy",
    "SpreadValueStrategy",
    "TotalValueStrategy",
    "TotalProjectionStrategy",
    "SteamFollowStrategy",
    "PublicFadeStrategy",
    "ReverseLineMovementStrategy",
    "NHLGoalieB2BStrategy",
    "MLSHomeTravelStrategy",
    "MiddleDetectorStrategy",
    "SituationalStrategy",
    "EloEdgeStrategy",
    "EnsembleStrategy",
    "PlayerPropsStrategy",
    "PropMarket",
]
