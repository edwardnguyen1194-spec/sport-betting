"""Predictive models (player props, game outcomes, confidence).

The heavy lifter is :class:`LSTMPlayerPropsModel`, which wraps a
sequence model over player game logs to predict the probability
that a prop (points, hits, strikeouts, ...) lands over/under a
sportsbook's posted line.
"""

from .lstm_player_props import (
    LSTMPlayerPropsModel,
    LSTMPlayerPropsConfig,
    PlayerGameLog,
    PlayerPropsDataset,
)

__all__ = [
    "LSTMPlayerPropsModel",
    "LSTMPlayerPropsConfig",
    "PlayerGameLog",
    "PlayerPropsDataset",
]
