"""Unit tests for odds-conversion helpers and GameOdds merging."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sports_betting_agent.models_schema import (
    GameOdds,
    OddsLine,
    american_to_decimal,
    american_to_implied,
    decimal_to_american,
    kelly_fraction,
    merge_games,
    remove_vig_two_way,
)


def test_american_to_decimal_positive():
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(100) == pytest.approx(2.0)


def test_american_to_decimal_negative():
    assert american_to_decimal(-200) == pytest.approx(1.5)
    assert american_to_decimal(-110) == pytest.approx(1.909, rel=1e-3)


def test_decimal_to_american_roundtrip():
    for american in (-400, -200, -110, 100, 150, 300):
        assert decimal_to_american(american_to_decimal(american)) == american


def test_implied_and_vig_removal():
    p_home = american_to_implied(-150)
    p_away = american_to_implied(+130)
    assert p_home > 0.5
    assert p_away < 0.5
    fh, fa = remove_vig_two_way(p_home, p_away)
    assert fh + fa == pytest.approx(1.0)


def test_kelly_fraction_positive_edge():
    stake = kelly_fraction(0.65, 2.0, fraction=1.0)
    # b=1, p=0.65, q=0.35 -> edge = (1*0.65 - 0.35)/1 = 0.30
    assert stake == pytest.approx(0.30, rel=1e-3)


def test_kelly_fraction_negative_edge_is_zero():
    assert kelly_fraction(0.40, 2.0) == 0.0


def test_oddsline_populates_both_representations():
    line = OddsLine(book="bovada", market="moneyline", selection="Duke", american=-200)
    assert line.decimal == pytest.approx(1.5)
    line2 = OddsLine(book="pinnacle", market="moneyline", selection="Duke", decimal=2.5)
    assert line2.american == 150


def test_game_key_stable_across_sources():
    t = datetime(2026, 4, 10, 23, 0, tzinfo=timezone.utc)
    a = GameOdds(sport="baseball_ncaa", league="ncaa", home_team="Duke Blue Devils", away_team="Virginia Cavaliers", commence_time=t)
    b = GameOdds(sport="baseball_ncaa", league="ncaa", home_team="Duke", away_team="Virginia", commence_time=t)
    assert a.game_key == b.game_key


def test_merge_games_combines_sources():
    t = datetime(2026, 4, 10, 23, 0, tzinfo=timezone.utc)
    g1 = GameOdds(
        sport="baseball_ncaa", league="ncaa", home_team="Duke", away_team="Virginia",
        commence_time=t, source="bovada",
        lines=[OddsLine(book="bovada", market="moneyline", selection="Duke", american=-200)],
    )
    g2 = GameOdds(
        sport="baseball_ncaa", league="ncaa", home_team="Duke", away_team="Virginia",
        commence_time=t, source="espn",
        lines=[OddsLine(book="espn", market="moneyline", selection="Duke", american=-210)],
    )
    merged = merge_games([g1, g2])
    assert len(merged) == 1
    assert len(merged[0].lines) == 2
    assert set(merged[0].meta["sources"]) == {"bovada", "espn"}


def test_best_moneyline_picks_higher_decimal():
    game = GameOdds(
        sport="baseball_ncaa", league="ncaa", home_team="Duke", away_team="Virginia",
        lines=[
            OddsLine(book="b1", market="moneyline", selection="Duke", american=-210),
            OddsLine(book="b2", market="moneyline", selection="Duke", american=-200),
            OddsLine(book="b3", market="moneyline", selection="Duke", american=-220),
        ],
    )
    best = game.best_moneyline("home")
    assert best is not None
    assert best.american == -200
