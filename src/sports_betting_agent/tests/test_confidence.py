"""Tests for confidence scoring."""

from __future__ import annotations

from sports_betting_agent.confidence import weighted_consensus
from sports_betting_agent.models_schema import GameOdds, OddsLine


def _game():
    g = GameOdds(sport="x", league="x", home_team="A", away_team="B")
    g.lines.extend(
        [
            OddsLine(book="pinnacle", market="moneyline", selection="A", american=-200),
            OddsLine(book="bovada", market="moneyline", selection="A", american=-190),
            OddsLine(book="draftkings", market="moneyline", selection="A", american=-195),
            OddsLine(book="pinnacle", market="moneyline", selection="B", american=+170),
            OddsLine(book="bovada", market="moneyline", selection="B", american=+160),
            OddsLine(book="draftkings", market="moneyline", selection="B", american=+170),
        ]
    )
    return g


def test_weighted_consensus_two_sides():
    reports = weighted_consensus(_game())
    assert "home" in reports and "away" in reports
    home = reports["home"]
    away = reports["away"]
    assert home.book_count == 3
    assert away.book_count == 3
    assert home.confidence + away.confidence > 0.5
    # Home favorite should have the higher confidence.
    assert home.confidence > away.confidence


def test_weighted_consensus_handles_empty_side():
    g = GameOdds(sport="x", league="x", home_team="A", away_team="B")
    g.lines.append(OddsLine(book="pinnacle", market="moneyline", selection="A", american=-200))
    reports = weighted_consensus(g)
    assert reports["home"].book_count == 1
    assert reports["away"].book_count == 0
