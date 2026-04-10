"""Smoke tests for the LSTM player-props model.

These are deliberately cheap: they verify the NumPy fallback (and, if
torch is installed, the torch path) can fit on synthetic data and
return a valid probability. Training accuracy is not asserted, only
contract.
"""

from __future__ import annotations

from sports_betting_agent.models.lstm_player_props import (
    LSTMPlayerPropsConfig,
    LSTMPlayerPropsModel,
    make_synthetic_log,
)


def test_model_fits_and_predicts():
    cfg = LSTMPlayerPropsConfig(feature_dim=8, sequence_length=5, epochs=3, batch_size=16)
    logs = [make_synthetic_log(player_id=f"p{i}", games=30, feature_dim=8, seed=i) for i in range(5)]
    model = LSTMPlayerPropsModel(cfg)
    info = model.fit(logs)
    assert info["backend"] in {"torch", "numpy"}

    prob = model.predict_over(logs[0], line=20.0)
    assert 0.0 <= prob <= 1.0


def test_predict_recommendation_requires_confidence():
    cfg = LSTMPlayerPropsConfig(feature_dim=6, sequence_length=4, epochs=2)
    logs = [make_synthetic_log(player_id="p", games=20, feature_dim=6, seed=3)]
    model = LSTMPlayerPropsModel(cfg)
    model.fit(logs)
    rec = model.predict_recommendation(logs[0], line=15.0, player_name="Demo")
    # May or may not clear the 60% threshold on synthetic data.
    assert rec is None or (0.6 <= rec["confidence"] <= 1.0)
