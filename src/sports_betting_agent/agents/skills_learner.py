"""SkillsLearner — daily self-improvement sub-agent.

Every morning, reads the previous day's closed bets + CLV, scans
the web for new sports-betting research / tools / MCPs / hooks,
and saves structured "learnings" the agent can apply over time.

Output categories
-----------------

1. **new_techniques** — Sports-betting concepts we haven't implemented
   yet (e.g. "NBA rest-days pace adjustment", "Kelly uncertainty
   shrinkage", "MLB umpire zone K% coefficient").
2. **new_tools** — Free Python packages / APIs / datasets that could
   enrich our data pipeline (nba_api, FBref Stathead, umpscorecards).
3. **new_mcps** — Model Context Protocol servers that would give
   sub-agents real-time context (sports-data MCP, weather MCP,
   news MCP).
4. **new_hooks** — Claude Code hooks that would catch regressions
   (pre-commit market-guard, post-edit pytest, pre-push deploy-check).
5. **prompt_improvements** — Better system/user prompts for the
   6 existing sub-agents based on observed failure modes in the
   agent_log.
6. **code_changes** — Concrete diffs the learner is confident enough
   to propose as shippable improvements. Safer items get auto-shipped
   by the StrategyAuditor on Monday review.

Storage
-------

All learnings persisted to ``/data/sba/learnings.json`` (bounded at
365 entries = 1 year of daily learnings). Dashboard surfaces via
``/api/learnings``.

Safety
------

- Learner NEVER auto-modifies strategy code. Proposals flow to the
  dashboard for Uncle's approval or to StrategyAuditor for weekly
  batch review.
- Daily token budget: 20k tokens max per learning cycle.
- Idempotent: one learning per UTC day. Duplicate calls skip.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentDecision, AgentLog


logger = logging.getLogger(__name__)


# Rotating research topics so daily learnings don't repeat too tightly.
# Cycle once per week (7 buckets). Friday / weekend lean into
# high-frequency topics (RLM, steam) since weekend slates are biggest.
RESEARCH_TOPICS = [
    "closing line value sports betting 2026 CLV",
    "Dixon-Coles soccer totals calibration season-end",
    "Kelly fractional bankroll sports correlation 2026",
    "NBA pace rest days totals betting edge",
    "MLB umpire park factor strike zone K%",
    "NFL key number half-point buying 3 7 history",
    "reverse line movement steam detection 2026 research",
]

# Suggested MCP servers to evaluate. Kept here so the learner can
# track which we've already evaluated / integrated / rejected.
MCP_CANDIDATES = [
    ("sportsdata-io-mcp", "Real-time sports scores + odds API wrapper"),
    ("weather-gov-mcp", "Replace our ad-hoc weather.gov fetcher"),
    ("espn-api-mcp", "Structured ESPN headlines + injury scraper"),
    ("fbref-mcp", "Soccer xG + pace data feed"),
]


@dataclass
class LearningEntry:
    date: str
    topic: str
    new_techniques: List[Dict[str, Any]] = field(default_factory=list)
    new_tools: List[Dict[str, Any]] = field(default_factory=list)
    new_mcps: List[Dict[str, Any]] = field(default_factory=list)
    new_hooks: List[Dict[str, Any]] = field(default_factory=list)
    prompt_improvements: List[Dict[str, Any]] = field(default_factory=list)
    code_changes: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""


class SkillsLearner(BaseAgent):
    """Daily web-scanning learner. Inherits BaseAgent for HTTP + budget."""

    name = "skills_learner"
    max_tokens = 4096   # larger than other agents — this is a deep synthesis

    def __init__(self, agent_log: AgentLog, data_dir: str,
                 daily_token_budget: int = 100_000) -> None:
        super().__init__(agent_log=agent_log, daily_token_budget=daily_token_budget)
        self.data_dir = data_dir
        self.storage_path = os.path.join(data_dir, "learnings.json")
        self._entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path) as fh:
                self._entries = json.load(fh).get("entries", [])
        except Exception:
            self._entries = []

    def _save(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        # Bounded at 365 = 1 year of daily learnings.
        self._entries = self._entries[-365:]
        tmp = self.storage_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"entries": self._entries}, fh, indent=2)
        os.replace(tmp, self.storage_path)

    def _already_learned_today(self) -> bool:
        today = datetime.now(timezone.utc).date().isoformat()
        return any(e.get("date", "").startswith(today) for e in self._entries)

    def system_prompt(self) -> str:
        return (
            "You are the SkillsLearner for Uncle Phung's sports-betting AI. "
            "Every day you propose concrete improvements in SIX categories: "
            "new_techniques (betting concepts we should implement), new_tools "
            "(Python packages/APIs/datasets), new_mcps (Model Context Protocol "
            "servers), new_hooks (Claude Code hooks), prompt_improvements "
            "(for existing sub-agents), and code_changes (shippable diffs). "
            "RULES: (1) spreads + over/under only — no moneyline / props. "
            "(2) Every proposal must cite a source URL and a concrete rationale. "
            "(3) Code changes include file path + function + pseudocode. "
            "(4) Be conservative: propose 2-4 items per category, highest-ROI "
            "only. (5) Output strict JSON only — the parser will reject free text."
        )

    def build_user_message(self, context: Dict[str, Any]) -> str:
        topic = context.get("topic", RESEARCH_TOPICS[0])
        current_strategies = context.get("current_strategies", [])
        recent_closed_stats = context.get("recent_closed_stats", {})
        return json.dumps({
            "today_topic": topic,
            "current_strategies": current_strategies,
            "recent_closed_stats": recent_closed_stats,
            "mcp_candidates_seen": [c[0] for c in MCP_CANDIDATES],
            "instructions": (
                "Return JSON with exactly these keys: "
                "new_techniques, new_tools, new_mcps, new_hooks, "
                "prompt_improvements, code_changes, summary. "
                "Each list item is a dict with {name, rationale, citation, "
                "proposed_action}. summary is 3-5 sentences. Keep each list "
                "to 2-4 items."
            ),
        })

    def parse_response(self, text: str, context: Dict[str, Any]) -> AgentDecision:
        # Re-use base parser for the JSON header, but stash the full
        # structured output in metadata since the SkillsLearner
        # output doesn't map to the confidence_delta/stake_multiplier
        # schema the base uses.
        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```", 2)[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError) as exc:
            return AgentDecision(
                approved=True, reasoning="",
                error=f"skills_parse_error: {exc}",
            )
        return AgentDecision(
            approved=True,
            confidence_delta=0.0,
            stake_multiplier=1.0,
            reasoning=str(data.get("summary", ""))[:2000],
            metadata={
                "new_techniques": data.get("new_techniques", [])[:5],
                "new_tools": data.get("new_tools", [])[:5],
                "new_mcps": data.get("new_mcps", [])[:5],
                "new_hooks": data.get("new_hooks", [])[:5],
                "prompt_improvements": data.get("prompt_improvements", [])[:5],
                "code_changes": data.get("code_changes", [])[:5],
            },
        )

    def learn_today(
        self,
        current_strategies: List[str],
        recent_closed_stats: Dict[str, Any],
        force: bool = False,
        mcp_discovery: Optional[Any] = None,
    ) -> Optional[LearningEntry]:
        """Main entry point. Fires once per UTC day unless ``force=True``.

        If ``mcp_discovery`` is provided, we also run the MCPDiscovery
        sub-agent after the main synthesis succeeds and merge its
        candidate list into ``entry.new_mcps``. Keeps the two agents
        loosely coupled — caller decides when (and whether) to run
        the optional MCP scan.
        """
        if not force and self._already_learned_today():
            return None
        today = datetime.now(timezone.utc)
        # Rotate topic by UTC day-of-year modulo bucket count.
        topic_idx = today.timetuple().tm_yday % len(RESEARCH_TOPICS)
        topic = RESEARCH_TOPICS[topic_idx]

        decision = self.analyze({
            "topic": topic,
            "current_strategies": current_strategies,
            "recent_closed_stats": recent_closed_stats,
        })
        if decision.error:
            logger.warning("SkillsLearner error: %s", decision.error)
            return None
        meta = decision.metadata or {}
        new_mcps = list(meta.get("new_mcps", []) or [])

        # OPTIONAL sub-call: weekly MCP registry scan. Best-effort —
        # any failure here is swallowed so the main learning entry
        # still persists. Dedupe against MCP_CANDIDATES + whatever
        # Claude already surfaced in new_mcps above.
        if mcp_discovery is not None:
            try:
                from .mcp_discovery import discover_mcps
                seen = [c[0] for c in MCP_CANDIDATES]
                seen += [
                    (m.get("name") if isinstance(m, dict) else str(m))
                    for m in new_mcps
                ]
                mcp_decision = discover_mcps(mcp_discovery, existing_mcps=seen)
                extra = (mcp_decision.metadata or {}).get("candidates") or []
                if extra:
                    new_mcps.extend(extra)
            except Exception as exc:  # pragma: no cover — best-effort
                logger.warning("mcp_discovery sub-call failed: %s", exc)

        entry = LearningEntry(
            date=today.isoformat(),
            topic=topic,
            new_techniques=meta.get("new_techniques", []),
            new_tools=meta.get("new_tools", []),
            new_mcps=new_mcps,
            new_hooks=meta.get("new_hooks", []),
            prompt_improvements=meta.get("prompt_improvements", []),
            code_changes=meta.get("code_changes", []),
            summary=decision.reasoning,
        )
        self._entries.append(asdict(entry))
        self._save()
        logger.info(
            "skills_learner: %s recorded for %s (%d techniques, %d tools, %d mcps, %d hooks, %d prompts, %d code)",
            entry.date, topic,
            len(entry.new_techniques), len(entry.new_tools),
            len(entry.new_mcps), len(entry.new_hooks),
            len(entry.prompt_improvements), len(entry.code_changes),
        )
        return entry

    def recent(self, n: int = 30) -> List[Dict[str, Any]]:
        return self._entries[-n:]

    def latest(self) -> Optional[Dict[str, Any]]:
        return self._entries[-1] if self._entries else None
