"""Tests for the OpportunityScout sub-agent.

We exercise the pieces that don't require a live Anthropic API call:
the spread/total filter, prompt construction, and response parsing.
The base class handles the HTTP path itself.
"""

from __future__ import annotations

import json
import os
import tempfile

from sports_betting_agent.agents.base import AgentLog
from sports_betting_agent.agents.opportunity_scout import (
    OpportunityScout,
    _filter_spread_total,
    _rec_to_summary,
    scout_top_picks,
)
from sports_betting_agent.strategies.base import BetRecommendation


def _rec(market: str, selection: str, conf: float = 0.6, edge: float = 0.05) -> BetRecommendation:
    return BetRecommendation(
        game_key="x",
        sport="basketball_nba",
        league="NBA",
        home_team="Lakers",
        away_team="Celtics",
        market=market,
        selection=selection,
        american=-110,
        decimal=1.91,
        line=-3.5,
        book="pinnacle",
        strategy="spread_value",
        confidence=conf,
        edge=edge,
        reasoning="test rec",
    )


def _agent() -> OpportunityScout:
    tmp = tempfile.mkdtemp(prefix="scout-test-")
    log = AgentLog(tmp)
    return OpportunityScout(agent_log=log, api_key=None)


def test_filter_drops_moneyline():
    recs = [
        _rec("moneyline", "Lakers"),
        _rec("spread", "Lakers -3.5"),
        _rec("total", "Over 220"),
        _rec("player_prop", "LeBron 25+"),
    ]
    kept = _filter_spread_total(recs)
    markets = {r.market for r in kept}
    assert markets == {"spread", "total"}


def test_filter_handles_dicts():
    recs = [
        {"market": "moneyline", "selection": "Lakers"},
        {"market": "spread", "selection": "Lakers -3.5"},
    ]
    kept = _filter_spread_total(recs)
    assert len(kept) == 1
    assert kept[0]["market"] == "spread"


def test_rec_to_summary_extracts_core_fields():
    r = _rec("spread", "Lakers -3.5", conf=0.62, edge=0.04)
    s = _rec_to_summary(r)
    assert s["market"] == "spread"
    assert s["selection"] == "Lakers -3.5"
    assert s["confidence"] == 0.62
    assert s["edge"] == 0.04
    assert "Celtics @ Lakers" in s["game"]


def test_build_user_message_excludes_moneyline_and_ranks():
    agent = _agent()
    recs = [
        _rec("moneyline", "Lakers", conf=0.9, edge=0.2),  # must be dropped
        _rec("spread", "Lakers -3.5", conf=0.55, edge=0.02),
        _rec("spread", "Celtics +3.5", conf=0.7, edge=0.05),  # highest score
        _rec("total", "Over 220", conf=0.6, edge=0.03),
    ]
    msg = agent.build_user_message({
        "recs": recs,
        "headlines": [{"title": "Star player questionable", "sport": "basketball_nba"}],
        "max_picks": 3,
    })
    # Moneyline candidate must not leak into the prompt at all.
    assert "moneyline" not in msg.lower()
    # Candidates in the payload should be ranked, top first.
    payload_start = msg.index("{")
    payload = json.loads(msg[payload_start:])
    cands = payload["candidates"]
    assert len(cands) == 3
    assert cands[0]["selection"] == "Celtics +3.5"
    assert payload["summary_stats"]["candidate_count"] == 3
    assert payload["recent_headlines"]  # headline title passed through


def test_parse_response_extracts_picks_and_markdown():
    agent = _agent()
    raw = json.dumps({
        "top_picks": [
            {"rec_summary": "Lakers -3.5 @ pinnacle (-110)",
             "why": "Sharp steam on Lakers spread; Celtics traveling b2b."},
            {"rec_summary": "Over 220.5 @ draftkings (-105)",
             "why": "Pace-up matchup, elite offenses, no injury flags."},
        ],
        "markdown": "**Uncle — top plays**\n\n**Pick 1:** Lakers -3.5\nSteam.",
    })
    decision = agent.parse_response(raw, {})
    assert decision.approved is True
    assert decision.confidence_delta == 0.0
    assert decision.stake_multiplier == 1.0
    assert "Uncle" in decision.reasoning
    assert len(decision.metadata["top_picks"]) == 2
    assert decision.metadata["top_picks"][0]["rec_summary"].startswith("Lakers")


def test_parse_response_handles_fenced_json():
    agent = _agent()
    raw = (
        "```json\n"
        + json.dumps({
            "top_picks": [{"rec_summary": "Celtics +3.5", "why": "value"}],
            "markdown": "**Pick 1:** Celtics +3.5",
        })
        + "\n```"
    )
    decision = agent.parse_response(raw, {})
    assert decision.error is None
    assert "Celtics" in decision.reasoning
    assert decision.metadata["top_picks"][0]["rec_summary"] == "Celtics +3.5"


def test_parse_response_synthesizes_markdown_if_missing():
    agent = _agent()
    raw = json.dumps({
        "top_picks": [
            {"rec_summary": "Lakers -3.5", "why": "Sharp steam."},
        ],
        # no "markdown" key
    })
    decision = agent.parse_response(raw, {})
    assert "Pick 1" in decision.reasoning
    assert "Lakers -3.5" in decision.reasoning


def test_parse_response_returns_error_on_bad_json():
    agent = _agent()
    decision = agent.parse_response("this is not json", {})
    assert decision.error and decision.error.startswith("parse_error")
    assert decision.approved is True  # fail safe — stays informational


def test_analyze_no_client_returns_error():
    """With no API key, analyze() must fail safe — never raise."""
    agent = _agent()
    decision = agent.analyze({"recs": [_rec("spread", "Lakers -3.5")],
                              "headlines": []})
    assert decision.error == "no_client"
    assert decision.approved is True


def test_scout_top_picks_helper_delegates_to_agent():
    agent = _agent()
    decision = scout_top_picks(
        agent,
        all_recs=[_rec("spread", "Lakers -3.5")],
        news_headlines=[],
        max_picks=3,
    )
    # Without an API key we expect the no_client safety path.
    assert decision.error == "no_client"
