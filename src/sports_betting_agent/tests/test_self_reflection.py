"""Tests for the SelfReflection sub-agent.

The real ``analyze()`` path needs an Anthropic client, so these tests
cover the deterministic pieces:
- the JSON parser (markdown preserved, patches capped at 5, invalid
  kinds dropped)
- the decisions_rollup aggregator
- the proposals persistence helpers (save + load + latest per agent)
- the ``run_self_reflection`` convenience wrapper's no-op paths
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from sports_betting_agent.agents.base import (
    AgentLog,
    AgentLogEntry,
    AgentDecision,
)
from sports_betting_agent.agents.self_reflection import (
    SelfReflection,
    VALID_TARGETS,
    VALID_PATCH_KINDS,
    _decisions_rollup,
    latest_proposals_by_agent,
    load_proposals,
    run_self_reflection,
    save_proposals,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reflector(tmp_path):
    log = AgentLog(str(tmp_path))
    return SelfReflection(agent_log=log, api_key=None)


@pytest.fixture
def seeded_log(tmp_path):
    log = AgentLog(str(tmp_path))
    for i in range(10):
        approved = (i % 4) != 0  # ~25% vetoed
        log.record(AgentLogEntry(
            ts=datetime(2026, 4, 10 + (i % 5), tzinfo=timezone.utc).isoformat(),
            agent="pick_reviewer",
            context_summary=f"Yankees -1.5 spread book{i}",
            decision={
                "approved": approved,
                "confidence_delta": 0.0 if approved else -0.05,
                "stake_multiplier": 1.0 if approved else 0.0,
                "reasoning": "approve" if approved else "veto",
                "error": None if i != 7 else "parse_error: bad json",
            },
            tokens_in=150,
            tokens_out=80 + i,
            latency_ms=400 + i * 10,
        ))
    # Add some noise from a different agent to make sure filtering works.
    log.record(AgentLogEntry(
        ts=datetime(2026, 4, 12, tzinfo=timezone.utc).isoformat(),
        agent="news_triage",
        context_summary="Red Sox vs Yanks",
        decision={"approved": True, "reasoning": "green"},
        tokens_in=50, tokens_out=20, latency_ms=120,
    ))
    return log


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


def test_parse_response_preserves_markdown_and_caps_patches(reflector):
    payload = {
        "approved": True,
        "reasoning": (
            "### SelfReflection audit for pick_reviewer 2026-04-20\n\n"
            "**Sample:** 50 decisions reviewed.\n\n"
            "**Patterns noticed:**\n"
            "- Approved 8 road -4.5 favorites that lost.\n"
            "- Vetoed 2 correct bets citing stale headlines.\n\n"
            "**Proposed patches:** 2\n"
        ),
        "metadata": {
            "target_agent": "pick_reviewer",
            "proposed_patches": [
                {
                    "kind": "system_prompt",
                    "old": "",
                    "new": "Be extra cautious on injury news after 8pm ET.",
                    "rationale": "5 of 10 road favorites after 8pm ET lost.",
                },
                {
                    "kind": "max_tokens",
                    "old": "512",
                    "new": "768",
                    "rationale": "Outputs truncated in 4/50 entries.",
                },
                {
                    "kind": "temperature",
                    "old": "0.7",
                    "new": "0.4",
                    "rationale": "Repetitive reasoning text.",
                },
                {
                    "kind": "schema",
                    "old": "reasoning,metadata",
                    "new": "stale_line_flag",
                    "rationale": "Need to track when the line looks stale.",
                },
                {
                    "kind": "system_prompt",
                    "old": "something",
                    "new": "something else",
                    "rationale": "fifth patch",
                },
                # The sixth must be dropped by the hard cap.
                {"kind": "system_prompt", "old": "x", "new": "y", "rationale": "6th"},
                # Invalid kind — must be filtered out entirely.
                {"kind": "delete_prompt", "old": "x", "new": "y", "rationale": "bad"},
            ],
        },
    }
    decision = reflector.parse_response(
        json.dumps(payload), context={"target_agent": "pick_reviewer"}
    )
    assert decision.error is None
    assert decision.approved is True
    assert "SelfReflection audit for pick_reviewer" in decision.reasoning
    # Markdown preserved beyond the base 500-char truncate.
    assert "Proposed patches" in decision.reasoning
    # Hard cap at 5.
    assert len(decision.metadata["proposed_patches"]) == 5
    # Bad-kind entry was dropped BEFORE the cap, so the cap catches
    # the duplicate system_prompt at position 6 instead.
    kinds = [p["kind"] for p in decision.metadata["proposed_patches"]]
    assert "delete_prompt" not in kinds
    for kind in kinds:
        assert kind in VALID_PATCH_KINDS
    assert decision.metadata["target_agent"] == "pick_reviewer"


def test_parse_response_strips_code_fences(reflector):
    raw = "```json\n" + json.dumps({
        "approved": True,
        "reasoning": "hi",
        "metadata": {
            "target_agent": "news_triage",
            "proposed_patches": [],
        },
    }) + "\n```"
    decision = reflector.parse_response(raw, context={})
    assert decision.error is None
    assert decision.reasoning == "hi"
    assert decision.metadata["target_agent"] == "news_triage"
    assert decision.metadata["proposed_patches"] == []


def test_parse_response_bad_json_returns_error_but_stays_approved(reflector):
    decision = reflector.parse_response("not json", context={})
    assert decision.error is not None
    assert decision.approved is True  # fail SAFE


def test_parse_response_filters_unknown_patch_kind(reflector):
    payload = {
        "approved": True,
        "reasoning": "short",
        "metadata": {
            "target_agent": "game_analyst",
            "proposed_patches": [
                {"kind": "nuke_agent", "old": "a", "new": "b", "rationale": "r"},
                {"kind": "system_prompt", "old": "x", "new": "y", "rationale": "r"},
            ],
        },
    }
    decision = reflector.parse_response(json.dumps(payload), context={})
    patches = decision.metadata["proposed_patches"]
    assert len(patches) == 1
    assert patches[0]["kind"] == "system_prompt"


# ---------------------------------------------------------------------------
# _decisions_rollup
# ---------------------------------------------------------------------------


def test_decisions_rollup_empty_input():
    out = _decisions_rollup([])
    assert out["count"] == 0
    assert out["approved"] == 0
    assert out["vetoed"] == 0


def test_decisions_rollup_counts_and_averages(seeded_log):
    entries = seeded_log.recent(n=100, agent="pick_reviewer")
    out = _decisions_rollup(entries)
    assert out["count"] == 10
    # In the fixture, i % 4 == 0 means veto (i=0,4,8 = 3 vetoes).
    assert out["vetoed"] == 3
    assert out["approved"] == 7
    assert out["approval_rate"] == 70.0
    # One entry has an error in the fixture (i == 7).
    assert out["errors"].get("parse_error") == 1
    assert out["avg_tokens_in"] == 150
    # avg_tokens_out averages 80..89 = 84 (integer div).
    assert 80 <= out["avg_tokens_out"] <= 89
    assert out["max_tokens_out"] == 89
    assert out["avg_latency_ms"] > 0


# ---------------------------------------------------------------------------
# save_proposals / load_proposals / latest_proposals_by_agent
# ---------------------------------------------------------------------------


def test_save_proposals_creates_and_appends(tmp_path):
    data_dir = str(tmp_path)
    decision = AgentDecision(
        approved=True,
        reasoning="### SelfReflection audit for pick_reviewer 2026-04-20\n...",
        metadata={
            "target_agent": "pick_reviewer",
            "proposed_patches": [
                {
                    "kind": "system_prompt",
                    "old": "",
                    "new": "Be more cautious after 8pm ET.",
                    "rationale": "Losses clustered after 8pm.",
                },
            ],
        },
    )
    path = save_proposals(data_dir, "pick_reviewer", decision, audit_date="2026-04-19")
    assert path.endswith("agent_prompt_proposals.json")
    assert os.path.exists(path)

    # Second save appends (newest last).
    decision2 = AgentDecision(
        approved=True,
        reasoning="second pass",
        metadata={"target_agent": "pick_reviewer", "proposed_patches": []},
    )
    save_proposals(data_dir, "pick_reviewer", decision2, audit_date="2026-04-26")

    loaded = load_proposals(data_dir)
    assert "pick_reviewer" in loaded
    assert len(loaded["pick_reviewer"]) == 2
    assert loaded["pick_reviewer"][-1]["audit_date"] == "2026-04-26"
    assert loaded["pick_reviewer"][0]["audit_date"] == "2026-04-19"


def test_save_proposals_segregates_by_agent(tmp_path):
    data_dir = str(tmp_path)
    d1 = AgentDecision(
        reasoning="pick_reviewer reflection",
        metadata={"target_agent": "pick_reviewer", "proposed_patches": []},
    )
    d2 = AgentDecision(
        reasoning="news_triage reflection",
        metadata={"target_agent": "news_triage", "proposed_patches": []},
    )
    save_proposals(data_dir, "pick_reviewer", d1, audit_date="2026-04-19")
    save_proposals(data_dir, "news_triage", d2, audit_date="2026-04-19")
    loaded = load_proposals(data_dir)
    assert set(loaded.keys()) == {"pick_reviewer", "news_triage"}


def test_load_proposals_missing_file_returns_empty_dict(tmp_path):
    assert load_proposals(str(tmp_path)) == {}


def test_latest_proposals_by_agent_keys_cover_all_targets(tmp_path):
    data_dir = str(tmp_path)
    d = AgentDecision(
        reasoning="only pick_reviewer so far",
        metadata={"target_agent": "pick_reviewer", "proposed_patches": []},
    )
    save_proposals(data_dir, "pick_reviewer", d, audit_date="2026-04-19")
    out = latest_proposals_by_agent(data_dir)
    assert set(out.keys()) == set(VALID_TARGETS)
    assert out["pick_reviewer"] is not None
    assert out["pick_reviewer"]["audit_date"] == "2026-04-19"
    # All other targets are None until their first reflection fires.
    for name in VALID_TARGETS:
        if name == "pick_reviewer":
            continue
        assert out[name] is None


# ---------------------------------------------------------------------------
# run_self_reflection convenience wrapper
# ---------------------------------------------------------------------------


def test_run_self_reflection_none_agent_returns_noop():
    class _EmptyLog:
        def recent(self, n, agent=None):
            return []

    decision = run_self_reflection(
        agent=None, target_name="pick_reviewer", agent_log=_EmptyLog(), n=50
    )
    assert decision.error is None
    assert decision.approved is True
    assert decision.metadata["proposed_patches"] == []
    assert decision.metadata["target_agent"] == "pick_reviewer"


def test_run_self_reflection_unknown_target_returns_error(reflector, seeded_log):
    decision = run_self_reflection(
        agent=reflector,
        target_name="not_an_agent",
        agent_log=seeded_log,
        n=50,
    )
    assert decision.error is not None
    assert "unknown_target" in decision.error


def test_run_self_reflection_without_client_returns_no_client(reflector, seeded_log):
    """BaseAgent without a client fails SAFE with error='no_client'.
    The wrapper must surface this rather than raising.
    """
    decision = run_self_reflection(
        agent=reflector,
        target_name="pick_reviewer",
        agent_log=seeded_log,
        n=50,
    )
    assert decision.error == "no_client"
    assert decision.approved is True


def test_valid_targets_cover_the_six_subagents():
    assert set(VALID_TARGETS) == {
        "pick_reviewer",
        "news_triage",
        "post_mortem",
        "game_analyst",
        "opportunity_scout",
        "strategy_auditor",
    }
