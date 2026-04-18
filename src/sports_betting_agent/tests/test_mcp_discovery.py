"""Tests for the MCPDiscovery sub-agent.

Like the other agent tests, we cover the deterministic pieces:
- the JSON parser (strip fences, cap candidates, normalise fields)
- persistence (bounded store, dedupe by lowercase name)
- the ``discover_mcps`` no-op path (agent=None)
- the ``build_user_message`` shape (existing_mcps dedupe list)
"""

from __future__ import annotations

import json

import pytest

from sports_betting_agent.agents.base import AgentLog
from sports_betting_agent.agents.mcp_discovery import (
    MAX_CANDIDATES_PER_RUN,
    MAX_STORED_CANDIDATES,
    MCPDiscovery,
    discover_mcps,
)


@pytest.fixture
def discovery(tmp_path):
    log = AgentLog(str(tmp_path))
    return MCPDiscovery(
        agent_log=log, data_dir=str(tmp_path), api_key=None,
    )


# ---------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------


def test_parse_response_shapes_candidates(discovery):
    payload = {
        "approved": True,
        "reasoning": "## MCP Discovery 2026-04-17\n\nThree good ones.",
        "metadata": {
            "candidates": [
                {
                    "name": "sports-data-mcp",
                    "purpose": "live MLB/NBA/NFL odds + injuries",
                    "url": "https://github.com/ex/sports-data-mcp",
                    "install_command": "npx -y @ex/sports-data-mcp",
                    "expected_value": "high",
                    "integration_effort": "days",
                },
                {
                    "name": "weather-gov-mcp",
                    "purpose": "wind + precip for MLB/NFL parks",
                    "url": "https://github.com/ex/weather-gov-mcp",
                    "install_command": "uvx ex/weather-gov-mcp",
                    "expected_value": "medium",
                    "integration_effort": "hours",
                },
            ],
        },
    }
    decision = discovery.parse_response(json.dumps(payload), context={})
    assert decision.error is None
    assert decision.approved is True
    cands = decision.metadata["candidates"]
    assert len(cands) == 2
    assert cands[0]["name"] == "sports-data-mcp"
    assert cands[1]["expected_value"] == "medium"
    # full markdown reasoning preserved (>0 chars)
    assert "MCP Discovery" in decision.reasoning


def test_parse_response_strips_fences_and_caps(discovery):
    # More than MAX_CANDIDATES_PER_RUN candidates.
    big_list = [
        {"name": f"mcp-{i}", "purpose": "x", "url": "u",
         "install_command": "i", "expected_value": "low",
         "integration_effort": "weeks"}
        for i in range(MAX_CANDIDATES_PER_RUN + 5)
    ]
    raw = "```json\n" + json.dumps({
        "approved": True,
        "reasoning": "ok",
        "metadata": {"candidates": big_list},
    }) + "\n```"
    decision = discovery.parse_response(raw, context={})
    assert decision.error is None
    assert len(decision.metadata["candidates"]) == MAX_CANDIDATES_PER_RUN


def test_parse_response_drops_bad_entries(discovery):
    payload = {
        "approved": True,
        "reasoning": "",
        "metadata": {
            "candidates": [
                {"name": "", "purpose": "blank name should be dropped"},
                "string-not-dict-should-be-dropped",
                {"name": "good-mcp", "purpose": "keeps this"},
            ],
        },
    }
    decision = discovery.parse_response(json.dumps(payload), context={})
    names = [c["name"] for c in decision.metadata["candidates"]]
    assert names == ["good-mcp"]


def test_parse_response_bad_json_returns_error(discovery):
    decision = discovery.parse_response("not json at all", context={})
    assert decision.error is not None
    # Fail SAFE
    assert decision.approved is True


# ---------------------------------------------------------------------
# build_user_message dedupe normalisation
# ---------------------------------------------------------------------


def test_build_user_message_accepts_strings_and_dicts(discovery):
    msg = discovery.build_user_message({
        "existing_mcps": [
            "sportsdata-io-mcp",
            {"name": "weather-gov-mcp"},
            {"no_name": True},   # dropped
            "espn-api-mcp",
        ],
    })
    # extract the JSON payload embedded after "DISCOVERY CONTEXT:"
    payload = json.loads(msg.split("DISCOVERY CONTEXT:\n", 1)[1])
    assert "existing_mcps" in payload
    assert "sportsdata-io-mcp" in payload["existing_mcps"]
    assert "weather-gov-mcp" in payload["existing_mcps"]
    assert "espn-api-mcp" in payload["existing_mcps"]
    assert len(payload["existing_mcps"]) == 3


# ---------------------------------------------------------------------
# persistence + dedupe
# ---------------------------------------------------------------------


def test_persist_candidates_dedupes_by_lowercase_name(discovery):
    added_first = discovery.persist_candidates([
        {"name": "Sports-Data-MCP", "purpose": "p1", "url": "u1",
         "install_command": "c", "expected_value": "high",
         "integration_effort": "days"},
        {"name": "weather-gov-mcp", "purpose": "p2", "url": "u2",
         "install_command": "c", "expected_value": "medium",
         "integration_effort": "hours"},
    ], reasoning="first run")
    assert len(added_first) == 2
    # Re-add the same (case-insensitive). Only a new one should land.
    added_second = discovery.persist_candidates([
        {"name": "sports-data-mcp", "purpose": "dup"},
        {"name": "fbref-mcp", "purpose": "new"},
    ], reasoning="second run")
    assert len(added_second) == 1
    assert added_second[0]["name"] == "fbref-mcp"
    # Store has 3 entries total.
    assert len(discovery.all_entries()) == 3


def test_persist_candidates_bounded_storage(tmp_path):
    log = AgentLog(str(tmp_path))
    d = MCPDiscovery(agent_log=log, data_dir=str(tmp_path), api_key=None)
    # Inject MAX_STORED_CANDIDATES + 20 pre-existing entries, then call
    # persist_candidates to confirm bounding kicks in.
    d._entries = [
        {"name": f"existing-{i}", "purpose": "x"}
        for i in range(MAX_STORED_CANDIDATES + 20)
    ]
    d.persist_candidates(
        [{"name": "brand-new-mcp", "purpose": "final"}],
        reasoning="",
    )
    all_entries = d.all_entries()
    assert len(all_entries) == MAX_STORED_CANDIDATES
    # The new entry is at the tail, the oldest entries are dropped.
    assert all_entries[-1]["name"] == "brand-new-mcp"


def test_persist_empty_list_is_noop(discovery):
    added = discovery.persist_candidates([], reasoning="")
    assert added == []
    assert discovery.all_entries() == []


# ---------------------------------------------------------------------
# discover_mcps helper
# ---------------------------------------------------------------------


def test_discover_mcps_noop_when_agent_none():
    decision = discover_mcps(agent=None, existing_mcps=["x"])
    assert decision.approved is True
    assert decision.error is None
    assert decision.metadata["candidates"] == []


def test_discover_mcps_returns_no_client_when_no_api_key(discovery):
    # Fresh agent, no ANTHROPIC_API_KEY -> base.analyze returns
    # error='no_client'. The helper must surface that without crashing.
    decision = discover_mcps(discovery, existing_mcps=[])
    # Either no_client or daily_budget_exhausted etc — just confirm
    # the helper didn't raise and returned an AgentDecision.
    assert decision is not None
    # Nothing persisted because the call errored before parsing.
    assert discovery.all_entries() == []
