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
from .hooks_discovery import (
    HooksDiscovery,
    discover_hooks,
    recent_hook_candidates,
    write_hooks_template,
    DEFAULT_HOOKS,
)
from .mcp_discovery import MCPDiscovery, discover_mcps
from .self_reflection import (
    SelfReflection,
    VALID_TARGETS as SELF_REFLECTION_TARGETS,
    run_self_reflection,
    save_proposals as save_self_reflection_proposals,
    load_proposals as load_self_reflection_proposals,
    latest_proposals_by_agent as latest_self_reflection_proposals,
)

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
    "HooksDiscovery",
    "discover_hooks",
    "recent_hook_candidates",
    "write_hooks_template",
    "DEFAULT_HOOKS",
    "MCPDiscovery",
    "discover_mcps",
    "SelfReflection",
    "SELF_REFLECTION_TARGETS",
    "run_self_reflection",
    "save_self_reflection_proposals",
    "load_self_reflection_proposals",
    "latest_self_reflection_proposals",
]
