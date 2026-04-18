"""Tests for the HooksDiscovery sub-agent.

We cover the deterministic pieces — normalisation, default-catalog
fallback, persistence, the settings.local.json template shape, and the
safety path when no API key is available.
"""

from __future__ import annotations

import json
import os

import pytest

from sports_betting_agent.agents.base import AgentDecision, AgentLog
from sports_betting_agent.agents.hooks_discovery import (
    DEFAULT_HOOKS,
    HooksDiscovery,
    _HOOK_CANDIDATES_MAX,
    _normalise_hook,
    _settings_local_shape,
    discover_hooks,
    recent_hook_candidates,
    write_hooks_template,
)


@pytest.fixture
def agent(tmp_path) -> HooksDiscovery:
    log = AgentLog(str(tmp_path))
    return HooksDiscovery(agent_log=log, api_key=None)


# ---------------------------------------------------------------------------
# _normalise_hook
# ---------------------------------------------------------------------------


def test_normalise_hook_accepts_valid_entry():
    norm = _normalise_hook({
        "name": "ledger_guard",
        "trigger": "preEdit",
        "script": "exit 2",
        "rationale": "Ledger is book of record.",
        "safety_level": "destructive",
    })
    assert norm is not None
    assert norm["name"] == "ledger_guard"
    assert norm["trigger"] == "preEdit"
    assert norm["safety_level"] == "destructive"


def test_normalise_hook_rejects_missing_fields():
    assert _normalise_hook({"name": "x"}) is None
    assert _normalise_hook({"trigger": "preEdit"}) is None
    assert _normalise_hook(None) is None
    assert _normalise_hook("not a dict") is None


def test_normalise_hook_rejects_unknown_trigger():
    assert _normalise_hook({
        "name": "x", "trigger": "onSave", "script": "echo",
    }) is None


def test_normalise_hook_defaults_invalid_safety():
    norm = _normalise_hook({
        "name": "x", "trigger": "postEdit", "script": "echo",
        "safety_level": "nuclear",
    })
    assert norm is not None
    assert norm["safety_level"] == "moderate"


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


def test_parse_response_extracts_valid_hooks(agent):
    payload = json.dumps({
        "approved": True,
        "reasoning": "## Proposals\n- ledger guard\n- pytest gate",
        "metadata": {
            "hooks": [
                {"name": "ledger_guard", "trigger": "preEdit",
                 "script": "exit 2", "rationale": "protect ledger",
                 "safety_level": "destructive"},
                {"name": "pytest_gate", "trigger": "postEdit",
                 "script": "pytest -q", "rationale": "regression",
                 "safety_level": "moderate"},
                # Malformed entry must be dropped silently.
                {"name": "bogus", "trigger": "onSave", "script": "x"},
            ],
        },
    })
    decision = agent.parse_response(payload, {})
    assert decision.approved is True
    assert decision.error is None
    hooks = decision.metadata["hooks"]
    assert len(hooks) == 2
    assert hooks[0]["name"] == "ledger_guard"
    assert hooks[1]["trigger"] == "postEdit"


def test_parse_response_handles_fenced_json(agent):
    raw = (
        "```json\n"
        + json.dumps({
            "approved": True,
            "reasoning": "ok",
            "metadata": {"hooks": [{
                "name": "a", "trigger": "prePush", "script": "x",
                "rationale": "r", "safety_level": "safe",
            }]},
        })
        + "\n```"
    )
    decision = agent.parse_response(raw, {})
    assert decision.error is None
    assert decision.metadata["hooks"][0]["name"] == "a"


def test_parse_response_returns_error_on_bad_json(agent):
    decision = agent.parse_response("not json at all", {})
    assert decision.error is not None
    assert decision.error.startswith("parse_error")
    assert decision.approved is True  # fail safe


def test_parse_response_caps_hook_count(agent):
    many = [{
        "name": f"h{i}", "trigger": "postEdit", "script": "x",
        "rationale": "r", "safety_level": "safe",
    } for i in range(40)]
    payload = json.dumps({
        "approved": True, "reasoning": "",
        "metadata": {"hooks": many},
    })
    decision = agent.parse_response(payload, {})
    # Parser's hard cap is 16 (to keep the log sane).
    assert len(decision.metadata["hooks"]) <= 16


# ---------------------------------------------------------------------------
# discover_hooks (the public helper)
# ---------------------------------------------------------------------------


def test_discover_hooks_none_agent_returns_default_catalog(tmp_path):
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / "data"
    decision = discover_hooks(
        None,
        existing_hooks=[],
        data_dir=str(data_dir),
        repo_root=str(repo_root),
    )
    assert decision.approved is True
    hooks = decision.metadata["hooks"]
    assert len(hooks) == len(DEFAULT_HOOKS)
    triggers = {h["trigger"] for h in hooks}
    # Must touch every major focus area.
    assert {"preEdit", "postEdit", "preBash", "prePush"} <= triggers


def test_discover_hooks_persists_and_writes_template(tmp_path):
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / "data"
    discover_hooks(
        None,
        existing_hooks=[],
        data_dir=str(data_dir),
        repo_root=str(repo_root),
    )

    # Rolling audit log exists and has one entry.
    entries = recent_hook_candidates(str(data_dir))
    assert len(entries) == 1
    assert "decision" in entries[0]
    assert entries[0]["decision"]["metadata"]["hooks"]

    # Settings.local.json template exists in the expected place.
    template = repo_root / ".claude" / "hooks_proposed.json"
    assert template.exists()
    body = json.loads(template.read_text())
    assert "hooks" in body
    # Every bucket must be a list of normalised entries.
    for trig, items in body["hooks"].items():
        assert trig in ("preEdit", "postEdit", "preBash", "prePush")
        for item in items:
            assert {"name", "script", "rationale", "safety_level"} <= item.keys()


def test_discover_hooks_no_client_falls_back(tmp_path, agent):
    """With no API key analyze() errors — helper must still return the
    deterministic catalog so the dashboard never sees an empty list."""
    decision = discover_hooks(
        agent,
        existing_hooks=[],
        data_dir=str(tmp_path / "data"),
        repo_root=str(tmp_path / "repo"),
    )
    assert decision.approved is True
    hooks = decision.metadata["hooks"]
    assert len(hooks) > 0
    # Default catalog kicks in on the no_client path.
    assert any(h["name"] == "ledger_write_guard" for h in hooks)


def test_recent_hook_candidates_bounded(tmp_path):
    data_dir = tmp_path / "data"
    # Fire the helper many times to push past the bound.
    for _ in range(_HOOK_CANDIDATES_MAX + 20):
        discover_hooks(None, existing_hooks=[], data_dir=str(data_dir),
                       repo_root=str(tmp_path / "repo"))
    entries = recent_hook_candidates(str(data_dir), n=1000)
    assert len(entries) == _HOOK_CANDIDATES_MAX


# ---------------------------------------------------------------------------
# _settings_local_shape
# ---------------------------------------------------------------------------


def test_settings_local_shape_buckets_by_trigger():
    hooks = [
        {"name": "a", "trigger": "preEdit", "script": "x",
         "rationale": "r", "safety_level": "safe"},
        {"name": "b", "trigger": "preEdit", "script": "y",
         "rationale": "r2", "safety_level": "destructive"},
        {"name": "c", "trigger": "prePush", "script": "z",
         "rationale": "r3", "safety_level": "moderate"},
    ]
    shape = _settings_local_shape(hooks)
    assert set(shape["hooks"].keys()) == {"preEdit", "prePush"}
    assert len(shape["hooks"]["preEdit"]) == 2
    assert shape["hooks"]["preEdit"][0]["name"] == "a"
    # Bogus triggers are never emitted.
    assert "onSave" not in shape["hooks"]


def test_write_hooks_template_creates_directory(tmp_path):
    repo_root = str(tmp_path / "brand-new-repo")
    path = write_hooks_template(repo_root, DEFAULT_HOOKS)
    assert path and os.path.exists(path)
    assert path.endswith(os.path.join(".claude", "hooks_proposed.json"))


# ---------------------------------------------------------------------------
# analyze() fail-safe path
# ---------------------------------------------------------------------------


def test_analyze_no_client_returns_error(agent):
    decision = agent.analyze({"existing_hooks": []})
    assert decision.error == "no_client"
    assert decision.approved is True  # never blocks callers
