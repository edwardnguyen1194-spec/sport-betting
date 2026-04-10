"""Tests for the heavy favorite strategy."""

from __future__ import annotations

from datetime import datetime, timezone

from sports_betting_agent.config import Settings
from sports_betting_agent.models_schema import GameOdds, OddsLine
from sports_betting_agent.strategies.heavy_favorite import HeavyFavoriteStrategy


def _game_with_lines(home_prices, away_prices):
    game = GameOdds(
        sport="baseball_ncaa",
        league="ncaa",
        home_team="Duke",
        away_team="Virginia",
        commence_time=datetime(2026, 4, 10, 23, 0, tzinfo=timezone.utc),
    )
    for book, price in home_prices.items():
        game.lines.append(OddsLine(book=book, market="moneyline", selection="Duke", american=price))
    for book, price in away_prices.items():
        game.lines.append(OddsLine(book=book, market="moneyline", selection="Virginia", american=price))
    return game


def test_heavy_favorite_hits_chalk_band():
    # Duke is -180 across three books; consensus prob ~0.64 -- inside target band.
    game = _game_with_lines(
        home_prices={"bovada": -180, "pinnacle": -185, "draftkings": -175},
        away_prices={"bovada": +160, "pinnacle": +160, "draftkings": +155},
    )
    strategy = HeavyFavoriteStrategy(Settings(heavy_fav_min_books=2))
    recs = strategy.generate([game])
    assert len(recs) == 1
    rec = recs[0]
    assert rec.selection == "Duke"
    assert rec.american == -175  # best available (highest decimal) for a favorite
    assert 0.55 <= rec.confidence <= 0.85
    assert rec.stake_fraction > 0


def test_heavy_favorite_skips_too_heavy():
    # Duke is -600 -- outside the configured max_american window.
    game = _game_with_lines(
        home_prices={"bovada": -600, "pinnacle": -610, "draftkings": -590},
        away_prices={"bovada": +450, "pinnacle": +450, "draftkings": +440},
    )
    strategy = HeavyFavoriteStrategy(Settings(heavy_fav_min_books=2))
    assert strategy.generate([game]) == []


def test_heavy_favorite_skips_thin_coverage():
    # Only one book covers the favorite -> skip.
    game = _game_with_lines(
        home_prices={"bovada": -180},
        away_prices={"bovada": +160},
    )
    strategy = HeavyFavoriteStrategy(Settings(heavy_fav_min_books=2))
    assert strategy.generate([game]) == []
