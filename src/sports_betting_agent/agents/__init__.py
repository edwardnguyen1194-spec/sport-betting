"""Claude sub-agents that add contextual intelligence on top of the
hardcoded strategies. See base.py for the design contract."""

from .base import BaseAgent, AgentDecision, AgentLog, AgentLogEntry
from .game_analyst import GameAnalyst, analyze_game
from .opportunity_scout import OpportunityScout, scout_top_picks
from .pick_reviewer import PickReviewer, review_rec
from .post_mortem import PostMortem, analyze_loss
from .strategy_auditor import (
    StrategyAuditor,
    run_weekly_audit,
    save_weekly_audit_markdown,
    list_recent_weekly_audits,
)

__all__ = [
    "BaseAgent",
    "AgentDecision",
    "AgentLog",
    "AgentLogEntry",
    "GameAnalyst",
    "analyze_game",
    "OpportunityScout",
    "scout_top_picks",
    "PickReviewer",
    "review_rec",
    "PostMortem",
    "analyze_loss",
    "StrategyAuditor",
    "run_weekly_audit",
    "save_weekly_audit_markdown",
    "list_recent_weekly_audits",
]
