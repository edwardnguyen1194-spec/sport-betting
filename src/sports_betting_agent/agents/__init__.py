"""Claude sub-agents that add contextual intelligence on top of the
hardcoded strategies. See base.py for the design contract."""

from .base import BaseAgent, AgentDecision, AgentLog, AgentLogEntry
from .game_analyst import GameAnalyst, analyze_game
from .news_triage import NewsTriage, triage_open_bets, persist_triage_audit
from .opportunity_scout import OpportunityScout, scout_top_picks
from .pick_reviewer import PickReviewer, review_rec
from .post_mortem import PostMortem, analyze_loss
from .strategy_auditor import (
    StrategyAuditor,
    run_weekly_audit,
    save_weekly_audit_markdown,
    list_recent_weekly_audits,
)
from .skills_learner import SkillsLearner, LearningEntry, RESEARCH_TOPICS

__all__ = [
    "BaseAgent",
    "AgentDecision",
    "AgentLog",
    "AgentLogEntry",
    "GameAnalyst",
    "analyze_game",
    "NewsTriage",
    "triage_open_bets",
    "persist_triage_audit",
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
    "SkillsLearner",
    "LearningEntry",
    "RESEARCH_TOPICS",
]
