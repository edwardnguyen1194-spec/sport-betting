"""Betting strategies — spread and total only, per Uncle's rule.

Each strategy consumes a list of :class:`GameOdds` and emits a list
of :class:`BetRecommendation` objects that downstream code (paper
trader, dashboard, Claude chat) can reason about uniformly.

Purged 2026-04-17 after a multi-agent audit:
  - middle_detector: empirical 1-pt NFL middle hits ~1.5%, our code
    used gap*0.05 = 3-5× inflation → Kelly blowup risk.
  - situational: 9 hand-picked additive buckets, favorites-only, cap
    0.92 — classic overfit-to-intuition.
  - nhl_goalie: B2B detection via "yesterday appears in today's feed"
    almost never fires; when it did, it was noise not a starter signal.
  - elo_edge: moneyline-only, violates "spread/total only" rule.
  - heavy_favorite / value_bets / contrarian / pythagorean: all
    moneyline, superseded by spread_value/total_value/projection.
  - player_props: props are neither spread nor total.
"""

from .base import BetRecommendation, Strategy
from .spread_value import SpreadValueStrategy
from .total_value import TotalValueStrategy
from .total_projection import TotalProjectionStrategy
from .steam_follow import SteamFollowStrategy
from .public_fade import PublicFadeStrategy
from .reverse_line_movement import ReverseLineMovementStrategy
from .mls_travel import MLSHomeTravelStrategy
from .elo_spread import EloSpreadStrategy
from .price_dispersion import PriceDispersionStrategy
from .ensemble import EnsembleStrategy

__all__ = [
    "BetRecommendation",
    "Strategy",
    "SpreadValueStrategy",
    "TotalValueStrategy",
    "TotalProjectionStrategy",
    "SteamFollowStrategy",
    "PublicFadeStrategy",
    "ReverseLineMovementStrategy",
    "MLSHomeTravelStrategy",
    "EloSpreadStrategy",
    "PriceDispersionStrategy",
    "EnsembleStrategy",
]
