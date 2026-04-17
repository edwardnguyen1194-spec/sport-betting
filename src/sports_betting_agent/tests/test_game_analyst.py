"""Tests for the GameAnalyst sub-agent and its ensemble wiring.

We stub out the Anthropic client so the tests run offline and
deterministically.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from sports_betting_agent.agents.base import AgentDecision, AgentLog
from sports_betting_agent.agents.game_analyst import (
    GameAnalyst,
    _book_lines_summary,
    _is_mlb,
    _recent_form_for,
    analyze_game,
)
from sports_betting_agent.models_schema import GameOdds, OddsLine
from sports_betting_agent.strategies.base import BetRecommendation, Strategy
from sports_betting_agent.strategies.ensemble import EnsembleStrategy


# ---------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------


class _FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeUsage:
    def __init__(self):
        self.input_tokens = 42
        self.output_tokens = 7


class _FakeResponse:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]
        self.usage = _FakeUsage()


class _FakeClient:
    """Stand-in for anthropic.Anthropic with a ``messages.create`` API."""

    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload
        self.messages = self

    def create(self, **_kw):
        return _FakeResponse(json.dumps(self._payload))


class _FakeScoring:
    """Minimal stand-in for TeamScoringTracker.team()."""

    def __init__(self, data: Dict[str, Dict[str, Dict[str, List[float]]]]):
        self._data = data

    def team(self, sport: str, team: str):
        return self._data.get(sport, {}).get(team)


class _FakeElo:
    def __init__(self, ratings: Dict[str, Dict[str, float]]):
        self._ratings = ratings

    def get_rating(self, sport: str, team: str):
        return self._ratings.get(sport, {}).get(team)


class _FakeNews:
    def __init__(self, headlines: List[Dict[str, str]]):
        self.headlines = headlines


def _make_agent(payload: Dict[str, Any], tmp_path) -> GameAnalyst:
    # Force the ``client`` property to return our fake by patching
    # Anthropic to a sentinel (so the "is None" early-return is
    # skipped) and pre-assigning ``_client``.
    import sports_betting_agent.agents.base as _base
    _base.Anthropic = object  # any non-None value
    log = AgentLog(str(tmp_path))
    agent = GameAnalyst(agent_log=log, api_key="fake-key", enabled=True)
    agent._client = _FakeClient(payload)
    return agent


def _mlb_game() -> GameOdds:
    g = GameOdds(
        sport="baseball_mlb",
        league="mlb",
        home_team="Boston Red Sox",
        away_team="New York Yankees",
        commence_time=datetime(2026, 4, 17, 23, 0, tzinfo=timezone.utc),
    )
    g.lines.extend(
        [
            OddsLine(book="bovada", market="total", selection="Over", american=-110, line=9.5),
            OddsLine(book="bovada", market="total", selection="Under", american=-110, line=9.5),
            OddsLine(book="draftkings", market="spread", selection="Boston Red Sox", american=-110, line=-1.5),
        ]
    )
    return g


# ---------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------


def test_is_mlb_detection():
    assert _is_mlb("baseball_mlb")
    assert _is_mlb("BASEBALL_MLB")
    assert not _is_mlb("basketball_nba")
    assert not _is_mlb("")


def test_book_lines_summary_picks_medians():
    game = _mlb_game()
    summary = _book_lines_summary(game)
    assert summary["posted_total_median"] == 9.5
    assert summary["home_spread_median"] == -1.5


def test_recent_form_reports_w_l_last_5():
    scoring = _FakeScoring(
        {
            "baseball_mlb": {
                "Boston Red Sox": {
                    # 6 games total; last 5 = [(5,3),(2,8),(4,1),(3,3),(7,6)]
                    # Wins = 3, Losses = 1, Pushes = 1
                    "home_scored": [1, 5, 2, 4, 3, 7],
                    "home_allowed": [2, 3, 8, 1, 3, 6],
                    "away_scored": [],
                    "away_allowed": [],
                }
            }
        }
    )
    form = _recent_form_for(scoring, "baseball_mlb", "Boston Red Sox")
    assert form["games_tracked"] == 6
    assert form["last_5_wl"].startswith("3-1")
    assert form["last_5_avg_scored"] > 0


def test_recent_form_handles_missing_team():
    scoring = _FakeScoring({"baseball_mlb": {}})
    assert _recent_form_for(scoring, "baseball_mlb", "Nope") == {}
    assert _recent_form_for(None, "baseball_mlb", "Nope") == {}


def test_game_analyst_parse_clamps_delta(tmp_path):
    # Agent says the angle is worth +0.50 but we must clamp to +0.05.
    agent = _make_agent(
        {
            "approved": True,
            "confidence_delta": 0.50,
            "reasoning": "Wind blowing out 18mph",
            "metadata": {
                "angles": ["wind blowing out", "0-5 road trip"],
                "best_bet_hint": "Under 8.5",
            },
        },
        tmp_path,
    )
    decision = agent.analyze(
        {
            "home_team": "Boston Red Sox",
            "away_team": "New York Yankees",
            "sport": "baseball_mlb",
        }
    )
    assert decision.approved is True
    assert decision.confidence_delta == 0.05
    assert "wind" in decision.reasoning.lower()
    assert decision.metadata["angles"][0] == "wind blowing out"
    assert decision.metadata["best_bet_hint"] == "Under 8.5"


def test_game_analyst_parse_clamps_negative(tmp_path):
    agent = _make_agent(
        {"approved": True, "confidence_delta": -0.99, "reasoning": "", "metadata": {}},
        tmp_path,
    )
    decision = agent.analyze({"home_team": "A", "away_team": "B", "sport": "x"})
    assert decision.confidence_delta == -0.05


def test_analyze_game_gathers_mlb_context(tmp_path):
    agent = _make_agent(
        {
            "approved": True,
            "confidence_delta": 0.02,
            "reasoning": "park + ump both favor unders",
            "metadata": {"angles": ["park 0.95", "ump -0.3"], "best_bet_hint": "Under 9.5"},
        },
        tmp_path,
    )
    scoring = _FakeScoring({"baseball_mlb": {}})
    elo = _FakeElo({"baseball_mlb": {"Boston Red Sox": 1520, "New York Yankees": 1565}})
    news = _FakeNews(
        [
            {"title": "Boston Red Sox ace placed on IL", "description": ""},
            {"title": "NFL draft results", "description": ""},
        ]
    )
    decision = analyze_game(
        agent, _mlb_game(), news_reader=news, scoring=scoring, elo=elo
    )
    assert decision.approved is True
    assert -0.05 <= decision.confidence_delta <= 0.05
    assert "park" in decision.reasoning.lower() or "ump" in decision.reasoning.lower()


def test_analyze_game_handles_none_inputs(tmp_path):
    agent = _make_agent(
        {"approved": True, "confidence_delta": 0.0, "reasoning": "", "metadata": {}},
        tmp_path,
    )
    decision = analyze_game(agent, _mlb_game())
    # Should still work with no news/scoring/elo injected.
    assert decision.error is None or decision.error == ""


def test_analyze_game_null_game_is_noop(tmp_path):
    agent = _make_agent({}, tmp_path)
    decision = analyze_game(agent, None)
    assert decision.reasoning == "no_input"


# ---------------------------------------------------------------------
# Ensemble wiring
# ---------------------------------------------------------------------


class _FakeStrategy(Strategy):
    name = "fake"

    def __init__(self, recs: List[BetRecommendation]):
        self._recs = recs

    def generate(self, games):
        return list(self._recs)


def _make_rec(game: GameOdds, market: str, selection: str, conf: float) -> BetRecommendation:
    # Pick matching line for book/american/decimal.
    match = None
    for ln in game.lines:
        if ln.market == market and ln.selection.lower() == selection.lower():
            match = ln
            break
    assert match is not None
    return BetRecommendation(
        game_key=game.game_key,
        sport=game.sport,
        league=game.league,
        home_team=game.home_team,
        away_team=game.away_team,
        market=market,
        selection=selection,
        american=match.american,
        decimal=match.decimal,
        line=match.line,
        book=match.book,
        strategy="fake",
        confidence=conf,
        edge=0.03,
    )


def test_ensemble_applies_game_analyst_delta(tmp_path, monkeypatch):
    game = _mlb_game()
    rec = _make_rec(game, "total", "Under", conf=0.60)

    # Build an analyst that always returns +0.04.
    agent = _make_agent(
        {
            "approved": True,
            "confidence_delta": 0.04,
            "reasoning": "wind blowing in",
            "metadata": {"angles": ["wind blowing in"], "best_bet_hint": "Under 9.5"},
        },
        tmp_path,
    )

    ensemble = EnsembleStrategy(
        strategies=[_FakeStrategy([rec])],
        game_analyst=agent,
    )

    # Force top-N to 5 (default) via env var to exercise the code path.
    monkeypatch.setenv("SBA_GAME_ANALYST_TOP_N", "3")

    out = ensemble.generate([game])
    # One rec, it should have received the analyst nudge.
    assert len(out) == 1
    r = out[0]
    assert r.confidence > 0.60  # nudged up
    assert r.confidence <= 0.72  # still under ensemble cap
    assert "game_analyst" in r.meta
    assert r.meta["game_analyst"]["angles"] == ["wind blowing in"]
    assert r.meta["game_analyst"]["best_bet_hint"] == "Under 9.5"
    assert "[game_analyst]" in r.reasoning


def test_ensemble_without_game_analyst_is_unchanged(tmp_path):
    game = _mlb_game()
    rec = _make_rec(game, "total", "Under", conf=0.60)
    ensemble = EnsembleStrategy(strategies=[_FakeStrategy([rec])])
    out = ensemble.generate([game])
    assert len(out) == 1
    assert "game_analyst" not in out[0].meta


def test_ensemble_top_n_env_zero_disables(tmp_path, monkeypatch):
    game = _mlb_game()
    rec = _make_rec(game, "total", "Under", conf=0.60)
    agent = _make_agent(
        {"approved": True, "confidence_delta": 0.04, "reasoning": "", "metadata": {}},
        tmp_path,
    )
    ensemble = EnsembleStrategy(
        strategies=[_FakeStrategy([rec])], game_analyst=agent
    )
    monkeypatch.setenv("SBA_GAME_ANALYST_TOP_N", "0")
    out = ensemble.generate([game])
    assert "game_analyst" not in out[0].meta
