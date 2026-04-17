"""Tests for PaperTrader."""

from __future__ import annotations

import os
import tempfile

import pytest

from sports_betting_agent.config import Settings
from sports_betting_agent.paper_trader import PaperTrader
from sports_betting_agent.strategies.base import BetRecommendation


def make_rec(**overrides):
    base = {
        "game_key": "abc",
        "sport": "baseball_ncaa",
        "league": "ncaa",
        "home_team": "Duke",
        "away_team": "Virginia",
        # Uncle's rule: spread + total only. Moneyline picks are
        # blocked at the paper-trader layer.
        "market": "spread",
        "selection": "Duke",
        "american": -110.0,
        "decimal": 1.909,
        "line": -1.5,
        "book": "bovada",
        "strategy": "spread_value",
        "confidence": 0.72,
        "edge": 0.05,
        "stake_fraction": 0.02,
        "reasoning": "test",
    }
    base.update(overrides)
    return BetRecommendation(**base)


@pytest.fixture
def tmp_trader(tmp_path):
    settings = Settings(
        bankroll_start=10_000.0,
        min_bet=50.0,
        max_bet_pct=0.05,
        data_dir=str(tmp_path),
    )
    trader = PaperTrader(settings, state_path=str(tmp_path / "ledger.json"))
    return trader


def test_place_and_settle_won(tmp_trader):
    bet = tmp_trader.place(make_rec())
    assert bet is not None
    assert tmp_trader.bankroll < 10_000

    settled = tmp_trader.settle_bet(bet.id, "won")
    assert settled is not None
    assert settled.status == "won"
    assert tmp_trader.bankroll > 10_000


def test_place_and_settle_lost(tmp_trader):
    bet = tmp_trader.place(make_rec())
    bankroll_after_place = tmp_trader.bankroll
    settled = tmp_trader.settle_bet(bet.id, "lost")
    assert settled.status == "lost"
    assert tmp_trader.bankroll == pytest.approx(bankroll_after_place)


def test_push_refunds_stake(tmp_trader):
    bet = tmp_trader.place(make_rec())
    settled = tmp_trader.settle_bet(bet.id, "push")
    assert settled.status == "push"
    assert tmp_trader.bankroll == pytest.approx(10_000.0)


def test_duplicate_bets_skipped(tmp_trader):
    tmp_trader.place(make_rec())
    second = tmp_trader.place(make_rec())
    assert second is None


def test_stake_respects_min_bet(tmp_trader):
    bet = tmp_trader.place(make_rec(stake_fraction=0.00001))
    # Should be clamped to min_bet of $50.
    assert bet.stake >= 50.0


def test_persistence(tmp_trader):
    tmp_trader.place(make_rec())
    path = tmp_trader.state_path
    reloaded = PaperTrader(tmp_trader.settings, state_path=path)
    assert len(reloaded.open_bets) == 1
