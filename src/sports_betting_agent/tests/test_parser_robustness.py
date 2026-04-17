"""Regression tests for the defensive parsing paths.

These tests exist specifically because I cannot hit the live odds
endpoints from the build sandbox -- every assertion here corresponds
to a failure mode I spotted by code review.
"""

from __future__ import annotations

from sports_betting_agent.fetchers.actionnetwork import ActionNetworkFetcher
from sports_betting_agent.fetchers.bovada import BovadaFetcher
from sports_betting_agent.fetchers.espn import ESPNFetcher


# ---------------------------------------------------------------------------
# Bovada: malformed events + props groups must be skipped silently.
# ---------------------------------------------------------------------------


def test_bovada_skips_malformed_market_silently():
    payload = [
        {
            "events": [
                {
                    "id": "1",
                    "description": "A @ B",
                    "startTime": 1744326000000,
                    "competitors": [
                        {"id": "a", "name": "A", "home": False},
                        {"id": "b", "name": "B", "home": True},
                    ],
                    "displayGroups": [
                        {
                            "description": "Game Lines",
                            "markets": [
                                # This market is missing 'outcomes' entirely.
                                {"description": "Moneyline", "period": {"main": True}},
                                {
                                    "description": "Moneyline",
                                    "period": {"main": True},
                                    "outcomes": [
                                        {
                                            "description": "A",
                                            "price": {"american": "+150", "decimal": "2.50"},
                                        },
                                        {
                                            "description": "B",
                                            "price": {"american": "-180", "decimal": "1.56"},
                                        },
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ]
        }
    ]
    games = list(BovadaFetcher()._parse(payload, "baseball_ncaa", "ncaa"))
    assert len(games) == 1
    ml = [l for l in games[0].lines if l.market == "moneyline"]
    assert len(ml) == 2


def test_bovada_skips_props_group_entirely():
    payload = [
        {
            "events": [
                {
                    "id": "1",
                    "competitors": [
                        {"id": "a", "name": "A", "home": False},
                        {"id": "b", "name": "B", "home": True},
                    ],
                    "displayGroups": [
                        {
                            "description": "Player Props",
                            "markets": [
                                {
                                    "description": "Total Hits",
                                    "period": {"main": True},
                                    "outcomes": [
                                        {"description": "Over", "price": {"american": "-110", "handicap": "1.5"}},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    ]
    games = list(BovadaFetcher()._parse(payload, "baseball_ncaa", "ncaa"))
    # Team names are present so the event is emitted -- but with zero lines.
    assert len(games) == 1
    assert games[0].lines == []


def test_bovada_skips_soccer_3way_moneyline():
    payload = [
        {
            "events": [
                {
                    "id": "1",
                    "competitors": [
                        {"id": "a", "name": "LA Galaxy", "home": False},
                        {"id": "b", "name": "LAFC", "home": True},
                    ],
                    "displayGroups": [
                        {
                            "description": "Game Lines",
                            "markets": [
                                {
                                    "description": "Moneyline",
                                    "period": {"main": True},
                                    "outcomes": [
                                        {"description": "LA Galaxy", "price": {"american": "+240"}},
                                        {"description": "Draw", "price": {"american": "+290"}},
                                        {"description": "LAFC", "price": {"american": "+110"}},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    ]
    games = list(BovadaFetcher()._parse(payload, "soccer_mls", "mls"))
    assert len(games) == 1
    # 3-way is intentionally skipped.
    assert [l for l in games[0].lines if l.market == "moneyline"] == []


# ---------------------------------------------------------------------------
# ESPN: null consensus entry + pick'em spread.
# ---------------------------------------------------------------------------


def _espn_event(odds_entries):
    return {
        "events": [
            {
                "id": "1",
                "name": "A at B",
                "date": "2026-04-10T23:00Z",
                "competitions": [
                    {
                        "date": "2026-04-10T23:00Z",
                        "competitors": [
                            {
                                "homeAway": "home",
                                "team": {"displayName": "Team B"},
                            },
                            {
                                "homeAway": "away",
                                "team": {"displayName": "Team A"},
                            },
                        ],
                        "odds": odds_entries,
                    }
                ],
            }
        ]
    }


def test_espn_skips_null_consensus_entry():
    payload = _espn_event(
        [
            {
                "provider": {"name": "consensus"},
                "overUnder": None,
                "spread": None,
                "homeTeamOdds": {"moneyLine": None},
                "awayTeamOdds": {"moneyLine": None},
            },
            {
                "provider": {"name": "ESPN BET"},
                "overUnder": 8.5,
                "spread": -1.5,
                "homeTeamOdds": {"moneyLine": -180, "favorite": True},
                "awayTeamOdds": {"moneyLine": 160},
            },
        ]
    )
    games = list(ESPNFetcher()._parse(payload, "baseball_mlb", "mlb"))
    assert len(games) == 1
    books = {l.book for l in games[0].lines}
    assert "consensus" not in books
    assert "espn_bet" in books


def test_espn_skips_synthetic_spread_and_total():
    """ESPN's scoreboard exposes handicap/total numbers but not juice.

    We used to stamp american=-110 as a sentinel, which silently
    contaminated real book prices via provider-name collision (e.g.
    ESPN's "DraftKings" provider overwriting AN's real DK juice).
    Fix: skip spread/total emissions entirely. Only real moneyline
    prices flow through.
    """
    payload = _espn_event(
        [
            {
                "provider": {"name": "ESPN BET"},
                "overUnder": 8.5,
                "spread": 0,
                "homeTeamOdds": {"moneyLine": -105},
                "awayTeamOdds": {"moneyLine": -105},
            }
        ]
    )
    games = list(ESPNFetcher()._parse(payload, "baseball_mlb", "mlb"))
    assert len(games) == 1
    # Real moneyline prices survive (ESPN publishes these).
    mls = [l for l in games[0].lines if l.market == "moneyline"]
    assert len(mls) == 2
    assert all(l.american == -105 for l in mls)
    # Spread + total intentionally dropped — we never had juice for them.
    assert not any(l.market == "spread" for l in games[0].lines)
    assert not any(l.market == "total" for l in games[0].lines)


# ---------------------------------------------------------------------------
# Action Network: both list-shaped and dict-shaped odds blocks.
# ---------------------------------------------------------------------------


def _an_event(odds):
    return {
        "games": [
            {
                "id": 999,
                "home_team_id": 1,
                "away_team_id": 2,
                "start_time": "2026-04-10T23:00:00Z",
                "teams": [
                    {"id": 1, "full_name": "Home Team"},
                    {"id": 2, "full_name": "Away Team"},
                ],
                "odds": odds,
            }
        ]
    }


def test_action_network_list_shape():
    payload = _an_event(
        [
            {
                "book_id": 15,
                "ml_home": -180,
                "ml_away": 160,
                "spread_home": -1.5,
                "spread_home_line": -110,
                "spread_away": 1.5,
                "spread_away_line": -110,
                "total": 8.5,
                "over": -105,
                "under": -115,
            }
        ]
    )
    games = ActionNetworkFetcher()._parse(payload, "baseball_mlb", "mlb")
    assert len(games) == 1
    game = games[0]
    ml = [l for l in game.lines if l.market == "moneyline"]
    spreads = [l for l in game.lines if l.market == "spread"]
    totals = [l for l in game.lines if l.market == "total"]
    assert len(ml) == 2
    assert len(spreads) == 2
    assert len(totals) == 2
    assert all(l.book == "pinnacle" for l in ml)  # book_id 15


def test_action_network_dict_keyed_by_book_id():
    payload = _an_event(
        {
            "15": {
                "ml_home": -200,
                "ml_away": 170,
                "spread_home": -1.5,
                "spread_home_line": -110,
            },
            "75": {
                "ml_home": -195,
                "ml_away": 165,
            },
        }
    )
    games = ActionNetworkFetcher()._parse(payload, "baseball_mlb", "mlb")
    assert len(games) == 1
    ml_books = {l.book for l in games[0].lines if l.market == "moneyline"}
    assert {"pinnacle", "fanduel"} <= ml_books


def test_action_network_alternate_key_spellings():
    payload = _an_event(
        [
            {
                "book_id": 68,  # draftkings
                "home_ml": -150,
                "away_ml": 130,
                "over_under": 9.0,
                "over_price": -108,
                "under_price": -112,
            }
        ]
    )
    games = ActionNetworkFetcher()._parse(payload, "baseball_mlb", "mlb")
    assert len(games) == 1
    ml = [l for l in games[0].lines if l.market == "moneyline"]
    totals = [l for l in games[0].lines if l.market == "total"]
    assert len(ml) == 2
    assert len(totals) == 2
    assert totals[0].line == 9.0
