"""Tests for the StrategyAuditor sub-agent.

The real ``analyze()`` path needs an Anthropic client, so these tests
cover the deterministic pieces:
- the JSON parser (markdown preserved, proposed_changes capped at 3)
- the closed-bets 7-day filter
- the per-strategy ROI aggregator
- the markdown persistence helpers
- the ``run_weekly_audit`` convenience wrapper's no-op path
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

import pytest

from sports_betting_agent.agents.base import AgentLog
from sports_betting_agent.agents.strategy_auditor import (
    StrategyAuditor,
    _collect_config_snapshot,
    _last_7_days_bets,
    _roi_by_strategy,
    list_recent_weekly_audits,
    run_weekly_audit,
    save_weekly_audit_markdown,
)


class _DummyClv:
    def stats_by_strategy(self):
        return {
            "spread_value": {
                "bets": 20, "bets_with_clv": 18, "average_clv_pct": 0.8,
                "positive_clv_pct": 66.7, "win_rate": 58.0, "record": "11-8",
            },
            "public_fade": {
                "bets": 12, "bets_with_clv": 11, "average_clv_pct": -1.1,
                "positive_clv_pct": 27.3, "win_rate": 41.7, "record": "5-7",
            },
        }


@pytest.fixture
def auditor(tmp_path):
    log = AgentLog(str(tmp_path))
    # No API key -> client is None, but that's fine; we test parse/helpers.
    return StrategyAuditor(agent_log=log, api_key=None)


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


def test_parse_response_preserves_markdown_and_caps_proposals(auditor):
    payload = {
        "approved": True,
        "reasoning": (
            "## Weekly Strategy Audit 2026-04-20\n\n"
            "### Top performer: spread_value (ROI +8.2%, CLV +0.8%)\n"
            "Consistently beating the close.\n\n"
            "### Hidden drag: public_fade (ROI -6.2%, CLV -1.1%)\n"
            "CLV confirms the ROI dip is a real problem."
        ),
        "metadata": {
            "proposed_changes": [
                {"file": "src/sports_betting_agent/config.py",
                 "line": "contrarian_public_threshold",
                 "diff": "raise from 0.65 to 0.72"},
                {"file": "src/sports_betting_agent/config.py",
                 "line": "spread_value_min_edge",
                 "diff": "nudge from 0.025 to 0.027"},
                {"file": "a", "line": "b", "diff": "c"},
                # Fourth entry must be dropped by the parser's hard cap.
                {"file": "should", "line": "not", "diff": "appear"},
            ]
        }
    }
    decision = auditor.parse_response(json.dumps(payload), context={})
    assert decision.approved is True
    assert decision.error is None
    # Markdown kept intact (NOT truncated to 500 chars by the
    # BaseAgent default parser). The test fixture's markdown is only
    # ~220 chars but the point is that StrategyAuditor overrode the
    # 500-char cap — verify by sending a payload that DOES exceed
    # 500 chars and confirming full preservation.
    long_markdown = (
        "## Weekly Strategy Audit 2026-04-20\n\n"
        "### Top performer: spread_value (ROI +8.2%, CLV +0.8%)\n"
        + ("spread_value has been consistently beating the close on MLB run-lines "
           "and NHL puck-lines when Pinnacle is >=2 cents tighter than Bovada. "
           "That vig-removal edge holds up over the 7-day sample we're reviewing. "
           "No tuning needed this week.\n\n") * 2
        + "### Hidden drag: public_fade (ROI -6.2%, CLV -1.1%)\n"
          "CLV confirms ROI dip is real, not variance. Recommend a threshold bump."
    )
    payload2 = dict(payload)
    payload2["reasoning"] = long_markdown
    long_decision = auditor.parse_response(json.dumps(payload2), context={})
    assert "## Weekly Strategy Audit" in long_decision.reasoning
    assert "Hidden drag: public_fade" in long_decision.reasoning
    # The bug we are guarding against: base class truncates to 500.
    # With the override, the full long_markdown (>500 chars) survives.
    assert len(long_decision.reasoning) > 500
    assert len(long_decision.reasoning) == len(long_markdown)
    # Hard cap at 3 proposals.
    assert len(decision.metadata["proposed_changes"]) == 3
    assert decision.metadata["proposed_changes"][0]["line"] == "contrarian_public_threshold"


def test_parse_response_strips_code_fences(auditor):
    raw = "```json\n" + json.dumps({"approved": True, "reasoning": "hi", "metadata": {}}) + "\n```"
    decision = auditor.parse_response(raw, context={})
    assert decision.error is None
    assert decision.reasoning == "hi"
    assert decision.metadata["proposed_changes"] == []


def test_parse_response_bad_json_returns_error(auditor):
    decision = auditor.parse_response("not actually json", context={})
    assert decision.error is not None
    assert decision.approved is True  # fail SAFE


# ---------------------------------------------------------------------------
# date + aggregation helpers
# ---------------------------------------------------------------------------


def _bet(strategy, status, stake, decimal, days_ago):
    when = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return SimpleNamespace(
        strategy=strategy, status=status, stake=stake, decimal=decimal,
        placed_at=when.isoformat(), settled_at=when.isoformat(),
        sport="baseball_mlb", market="spread", edge=0.04, confidence=0.62,
    )


def test_last_7_days_filter_drops_old_bets():
    bets = [
        _bet("spread_value", "won", 100, 1.91, days_ago=2),
        _bet("spread_value", "lost", 100, 1.91, days_ago=3),
        _bet("public_fade", "lost", 100, 1.91, days_ago=30),   # too old
    ]
    recent = _last_7_days_bets(bets)
    assert len(recent) == 2
    assert all(b["strategy"] != "public_fade" for b in recent)


def test_roi_by_strategy_computes_roi_and_record():
    bets = [
        _bet("spread_value", "won", 100, 2.00, days_ago=1),   # +100 profit
        _bet("spread_value", "lost", 100, 2.00, days_ago=2),  # -100
        _bet("spread_value", "won", 100, 2.00, days_ago=3),   # +100
        _bet("public_fade", "lost", 50, 1.91, days_ago=1),    # -50
    ]
    plain = _last_7_days_bets(bets)
    roi = _roi_by_strategy(plain)
    assert roi["spread_value"]["record"] == "2-1"
    assert roi["spread_value"]["roi_pct"] == pytest.approx(33.33, rel=1e-2)
    assert roi["public_fade"]["record"] == "0-1"
    assert roi["public_fade"]["roi_pct"] == pytest.approx(-100.0, rel=1e-2)


def test_roi_by_strategy_splits_ensemble_credit():
    bet = _bet("spread_value+public_fade", "won", 100, 2.00, days_ago=1)
    roi = _roi_by_strategy(_last_7_days_bets([bet]))
    assert "spread_value" in roi
    assert "public_fade" in roi
    assert roi["spread_value"]["bets"] == 1
    assert roi["public_fade"]["bets"] == 1


# ---------------------------------------------------------------------------
# config snapshot
# ---------------------------------------------------------------------------


def test_collect_config_snapshot_pulls_known_knobs():
    settings = SimpleNamespace(
        spread_value_min_edge=0.025,
        total_value_min_edge=0.03,
        contrarian_public_threshold=0.65,
        # An extra attr that we shouldn't expose.
        anthropic_api_key="SECRET",
    )
    snap = _collect_config_snapshot(settings)
    assert snap["spread_value_min_edge"] == 0.025
    assert snap["contrarian_public_threshold"] == 0.65
    assert "anthropic_api_key" not in snap


# ---------------------------------------------------------------------------
# markdown persistence
# ---------------------------------------------------------------------------


def test_save_and_list_weekly_audits(tmp_path):
    data_dir = str(tmp_path)
    path = save_weekly_audit_markdown(data_dir, "2026-04-20", "# audit body")
    assert path.endswith("2026-04-20.md")
    assert os.path.exists(path)

    path2 = save_weekly_audit_markdown(data_dir, "2026-04-13", "# older")
    assert os.path.exists(path2)

    audits = list_recent_weekly_audits(data_dir, limit=5)
    assert len(audits) == 2
    # Newest-first.
    assert audits[0]["date"] == "2026-04-20"
    assert audits[1]["date"] == "2026-04-13"
    assert "# audit body" in audits[0]["markdown"]


def test_list_recent_weekly_audits_empty_dir(tmp_path):
    assert list_recent_weekly_audits(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# run_weekly_audit convenience wrapper
# ---------------------------------------------------------------------------


def test_run_weekly_audit_returns_noop_when_agent_none():
    decision = run_weekly_audit(agent=None, paper=SimpleNamespace(closed_bets=[]),
                                clv=_DummyClv(), config_snapshot={})
    assert decision.approved is True
    assert decision.metadata["proposed_changes"] == []
    assert decision.error is None


def test_run_weekly_audit_shape_without_client(auditor):
    """With no Anthropic client, analyze() returns an `error='no_client'`
    AgentDecision — run_weekly_audit just returns that without blowing up."""
    paper = SimpleNamespace(closed_bets=[
        _bet("spread_value", "won", 100, 1.91, days_ago=1),
    ])
    decision = run_weekly_audit(
        agent=auditor, paper=paper, clv=_DummyClv(),
        config_snapshot=SimpleNamespace(spread_value_min_edge=0.025),
    )
    # BaseAgent fails SAFE when no client is configured.
    assert decision.error == "no_client"
    assert decision.approved is True
