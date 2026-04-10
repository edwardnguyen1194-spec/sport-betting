"""Unit tests for the Bovada parser (no network)."""

from __future__ import annotations

from sports_betting_agent.fetchers.bovada import BovadaFetcher


SAMPLE_PAYLOAD = [
    {
        "path": [],
        "events": [
            {
                "id": "123",
                "description": "Duke Blue Devils @ Virginia Cavaliers",
                "startTime": 1744326000000,  # 2025-04-10 23:00 UTC-ish
                "competitors": [
                    {"id": "duke", "name": "Duke Blue Devils", "home": False},
                    {"id": "uva", "name": "Virginia Cavaliers", "home": True},
                ],
                "displayGroups": [
                    {
                        "description": "Game Lines",
                        "markets": [
                            {
                                "description": "Moneyline",
                                "key": "2W-12",
                                "period": {"description": "Match", "main": True},
                                "outcomes": [
                                    {
                                        "description": "Duke Blue Devils",
                                        "price": {"american": "+150", "decimal": "2.50"},
                                    },
                                    {
                                        "description": "Virginia Cavaliers",
                                        "price": {"american": "-180", "decimal": "1.56"},
                                    },
                                ],
                            },
                            {
                                "description": "Point Spread",
                                "period": {"description": "Match", "main": True},
                                "outcomes": [
                                    {
                                        "description": "Duke Blue Devils",
                                        "price": {
                                            "american": "-110",
                                            "decimal": "1.91",
                                            "handicap": "1.5",
                                        },
                                    },
                                    {
                                        "description": "Virginia Cavaliers",
                                        "price": {
                                            "american": "-110",
                                            "decimal": "1.91",
                                            "handicap": "-1.5",
                                        },
                                    },
                                ],
                            },
                            {
                                "description": "Total",
                                "period": {"description": "Match", "main": True},
                                "outcomes": [
                                    {
                                        "description": "Over",
                                        "price": {
                                            "american": "-110",
                                            "decimal": "1.91",
                                            "handicap": "10.5",
                                        },
                                    },
                                    {
                                        "description": "Under",
                                        "price": {
                                            "american": "-110",
                                            "decimal": "1.91",
                                            "handicap": "10.5",
                                        },
                                    },
                                ],
                            },
                        ],
                    }
                ],
            }
        ],
    }
]


def test_bovada_parse_basic():
    fetcher = BovadaFetcher()
    games = list(fetcher._parse(SAMPLE_PAYLOAD, "baseball_ncaa", "college-baseball"))
    assert len(games) == 1
    game = games[0]
    assert game.home_team == "Virginia Cavaliers"
    assert game.away_team == "Duke Blue Devils"
    assert game.source == "bovada"

    ml = [l for l in game.lines if l.market == "moneyline"]
    assert len(ml) == 2
    assert any(l.selection == "Duke Blue Devils" and l.american == 150 for l in ml)
    assert any(l.selection == "Virginia Cavaliers" and l.american == -180 for l in ml)

    spreads = [l for l in game.lines if l.market == "spread"]
    assert len(spreads) == 2
    assert {l.line for l in spreads} == {1.5, -1.5}

    totals = [l for l in game.lines if l.market == "total"]
    assert len(totals) == 2
    assert all(l.line == 10.5 for l in totals)


def test_bovada_parse_skips_malformed_events():
    fetcher = BovadaFetcher()
    bad = [{"events": [{"id": "no-teams"}]}]
    assert list(fetcher._parse(bad, "baseball_ncaa", "college-baseball")) == []
