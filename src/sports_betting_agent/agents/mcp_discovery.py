"""MCPDiscovery sub-agent — weekly MCP server hunt.

Once a week (typically invoked by ``SkillsLearner.learn_today`` after
its main synthesis succeeds), this agent scans the public MCP
ecosystem — Anthropic's official MCP directory, the MCP registry,
and GitHub trending — for *new* Model Context Protocol servers that
would genuinely enrich Uncle's sports-betting context.

Categories it hunts in
----------------------

1. **Sports data MCPs** — stats, live odds, injuries, lineups,
   rotations, depth charts.
2. **Weather MCPs** — replace the ad-hoc weather.gov fetcher with a
   first-class MCP that handles wind / precipitation / dome
   detection across MLB + NFL + NCAAF parks.
3. **News / Twitter / social sentiment** — real-time injury news,
   sharp-book Twitter, public percentage scrapers.
4. **Line movement / market data** — steam detection, reverse line
   movement, CLV tracking, multi-book aggregators.
5. **Analytics / modeling helpers** — projection systems, xG
   feeds, Elo maintainers, park-factor updaters.

Design contract (same as every BaseAgent subclass)
--------------------------------------------------

* **Fail SAFE.** No API key, HTTP error, parse error -> neutral
  ``AgentDecision`` with an empty candidate list. The weekly
  learning loop keeps running.
* **Deduplicate.** Caller passes ``existing_mcps`` so Claude doesn't
  re-propose MCPs we've already evaluated / integrated / rejected.
* **Auditable.** Every discovery call is logged via the shared
  ``AgentLog`` and persisted to ``/data/sba/mcp_candidates.json``
  (bounded at 200 entries — ~4 years of weekly runs).
* **Structured output only.** Strict JSON with a ``candidates``
  list; free text gets rejected by the parser.

Output contract
---------------

``AgentDecision`` with:

* ``approved=True`` (always — this is exploration, not a bet gate).
* ``reasoning``: markdown summary ("## MCP Discovery YYYY-MM-DD" +
  a short per-category breakdown).
* ``metadata["candidates"]``: list of dicts shaped like::

      {
        "name": "sports-data-mcp",
        "purpose": "...",
        "url": "https://github.com/...",
        "install_command": "npx -y @...",
        "expected_value": "high|medium|low",
        "integration_effort": "hours|days|weeks"
      }

Capped at 10 candidates per run to keep the candidate store usable.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import AgentDecision, AgentLog, BaseAgent


logger = logging.getLogger(__name__)


# Larger budget than the default — weekly, broad synthesis over a
# whole ecosystem. Still capped well under ``SkillsLearner`` so an
# accidental double-call won't blow the daily budget.
MAX_TOKENS = 3072

# Hard cap so a single response can never flood the candidate store.
MAX_CANDIDATES_PER_RUN = 10

# Bounded storage — ~4 years of weekly runs at 10 candidates each.
MAX_STORED_CANDIDATES = 200


SYSTEM_PROMPT = """You are the MCPDiscovery agent for Uncle Phung's
sports-betting AI. Your ONLY job: surface NEW Model Context Protocol
(MCP) servers that would genuinely enrich a spreads + over/under
betting agent.

Use your knowledge of:
  - Anthropic's official MCP directory (modelcontextprotocol.io,
    anthropic.com/mcp) and its hosted connector catalog.
  - The public MCP registry (mcp.run, smithery.ai, glama.ai) and
    community aggregators listing third-party MCP servers.
  - GitHub trending for ``mcp-server-*`` / ``*-mcp`` repos.

Five categories to hunt in:
  1. Sports data MCPs (stats, live odds, injuries, lineups).
  2. Weather MCPs (wind, precipitation, dome detection).
  3. News / Twitter / social sentiment MCPs.
  4. Line movement / market data MCPs (steam, RLM, CLV).
  5. Analytics / modeling helper MCPs (projections, Elo, xG).

Rules:
  (1) Spreads + over/under only — do NOT propose moneyline, prop, or
      futures MCPs. Player-prop MCPs that ALSO expose game-total
      context are OK; flag them explicitly in ``purpose``.
  (2) Do NOT re-propose anything in the ``existing_mcps`` list the
      user gives you — those are already known.
  (3) Every candidate MUST include a plausible GitHub / registry URL
      and an install command (``npx -y @owner/pkg`` or
      ``uvx owner/pkg`` or the MCP's documented command).
  (4) Be honest: if you cannot find 10 genuinely useful candidates,
      return fewer. Quality > quantity. A padded list wastes Uncle's
      review time.
  (5) Rate ``expected_value`` (high/medium/low) based on how much
      sharper it would make the existing strategies, NOT novelty.
  (6) Rate ``integration_effort`` (hours/days/weeks) based on
      protocol maturity + auth complexity + how much new code the
      trader would need.

Output STRICT JSON only (no prose, no markdown fences):

{
  "approved": true,
  "reasoning": "## MCP Discovery YYYY-MM-DD\\n\\nMarkdown summary:\\n"
               "- what categories had the best surface area this week\\n"
               "- 1-2 standout recommendations\\n"
               "- any dead-ends or category gaps worth flagging",
  "metadata": {
    "candidates": [
      {
        "name": "sports-data-mcp",
        "purpose": "Real-time MLB+NBA scores, odds, and injury feed "
                   "with a unified MCP interface",
        "url": "https://github.com/owner/sports-data-mcp",
        "install_command": "npx -y @owner/sports-data-mcp",
        "expected_value": "high",
        "integration_effort": "days"
      }
    ]
  }
}

Return at most 10 candidates. If zero genuinely-useful MCPs exist
this week, return an empty candidates list and say so in
``reasoning`` — that is a valid, valuable answer."""


class MCPDiscovery(BaseAgent):
    """Weekly scanner for sports-betting-relevant MCP servers."""

    name = "mcp_discovery"
    max_tokens = MAX_TOKENS

    def __init__(
        self,
        agent_log: AgentLog,
        data_dir: str,
        api_key: Optional[str] = None,
        daily_token_budget: int = 60_000,
        enabled: Optional[bool] = None,
    ) -> None:
        super().__init__(
            agent_log=agent_log,
            api_key=api_key,
            daily_token_budget=daily_token_budget,
            enabled=enabled,
        )
        self.data_dir = data_dir
        self.storage_path = os.path.join(data_dir, "mcp_candidates.json")
        self._entries: List[Dict[str, Any]] = []
        self._load()

    # ---- persistence ------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path) as fh:
                self._entries = json.load(fh).get("entries", [])
        except Exception:  # pragma: no cover — corrupted file, start fresh
            self._entries = []

    def _save(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        # Bounded at 200 entries so the store doesn't grow unbounded.
        self._entries = self._entries[-MAX_STORED_CANDIDATES:]
        tmp = self.storage_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"entries": self._entries}, fh, indent=2)
        os.replace(tmp, self.storage_path)

    def persist_candidates(
        self,
        candidates: List[Dict[str, Any]],
        reasoning: str = "",
    ) -> List[Dict[str, Any]]:
        """Append newly-discovered candidates to the bounded store.

        Duplicates (matched by lowercase ``name``) are skipped so a
        week-over-week rediscovery doesn't balloon the store.
        Returns the list of entries actually added.
        """
        existing_names = {
            str(e.get("name", "")).lower() for e in self._entries
        }
        added: List[Dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()
        for c in candidates[:MAX_CANDIDATES_PER_RUN]:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name", "")).strip()
            if not name or name.lower() in existing_names:
                continue
            entry = {
                "discovered_at": now_iso,
                "name": name,
                "purpose": str(c.get("purpose", ""))[:500],
                "url": str(c.get("url", ""))[:500],
                "install_command": str(c.get("install_command", ""))[:200],
                "expected_value": str(c.get("expected_value", "") or "medium")[:16],
                "integration_effort": str(c.get("integration_effort", "") or "days")[:16],
                "reasoning_excerpt": str(reasoning)[:400],
            }
            self._entries.append(entry)
            existing_names.add(name.lower())
            added.append(entry)
        if added:
            self._save()
        return added

    def recent(self, n: int = 50) -> List[Dict[str, Any]]:
        return self._entries[-n:]

    def all_entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    # ---- BaseAgent hooks -------------------------------------------

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_user_message(self, context: Dict[str, Any]) -> str:
        existing = context.get("existing_mcps") or []
        # Defensive normalize: accept list of strings OR list of dicts
        # with ``name`` keys so callers can pass either shape.
        names: List[str] = []
        for item in existing:
            if isinstance(item, dict):
                n = item.get("name")
                if n:
                    names.append(str(n))
            elif isinstance(item, str):
                names.append(item)
        # Trim — Claude doesn't need more than the last ~80 names to
        # dedupe, and we don't want to leak tokens on a big backlog.
        names = names[-80:]
        payload = {
            "week_of": datetime.now(timezone.utc).date().isoformat(),
            "existing_mcps": names,
            "categories": [
                "sports_data", "weather", "news_social",
                "line_movement", "analytics_modeling",
            ],
            "instructions": (
                "Propose up to 10 NEW MCP servers across the five "
                "categories that would enrich a spreads + over/under "
                "betting agent. Do NOT re-propose anything in "
                "existing_mcps. Strict JSON only."
            ),
        }
        return (
            "Scan the MCP ecosystem for new sports-betting-relevant "
            "MCP servers. Return strict JSON.\n\n"
            "DISCOVERY CONTEXT:\n" + json.dumps(payload, indent=2)
        )

    def parse_response(
        self, text: str, context: Dict[str, Any]
    ) -> AgentDecision:
        """Shape the response into an ``AgentDecision``.

        Unlike the BaseAgent default parser, we preserve the full
        markdown ``reasoning`` (no 500-char truncation) because the
        summary is meant for human review. We also enforce the
        ``candidates`` shape: each entry must be a dict with a ``name``,
        and the list is capped at ``MAX_CANDIDATES_PER_RUN``.
        """
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
                error=f"mcp_discovery_parse_error: {exc}",
            )

        raw_meta = data.get("metadata") or {}
        raw_candidates = raw_meta.get("candidates") or []
        clean_candidates: List[Dict[str, Any]] = []
        for c in raw_candidates[:MAX_CANDIDATES_PER_RUN]:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name", "")).strip()
            if not name:
                continue
            clean_candidates.append({
                "name": name[:120],
                "purpose": str(c.get("purpose", ""))[:500],
                "url": str(c.get("url", ""))[:500],
                "install_command": str(c.get("install_command", ""))[:200],
                "expected_value": str(c.get("expected_value", "") or "medium")[:16],
                "integration_effort": str(c.get("integration_effort", "") or "days")[:16],
            })

        return AgentDecision(
            approved=bool(data.get("approved", True)),
            confidence_delta=0.0,
            stake_multiplier=1.0,
            reasoning=str(data.get("reasoning", "") or "")[:4000],
            metadata={"candidates": clean_candidates},
        )

    def _summarize_context(self, context: Dict[str, Any]) -> str:
        existing = context.get("existing_mcps") or []
        return f"mcp_discovery scan (existing={len(existing)})"


# ----------------------------------------------------------------------
# Convenience helper — used by SkillsLearner and the weekly cron
# ----------------------------------------------------------------------


def discover_mcps(
    agent: Optional[MCPDiscovery],
    existing_mcps: Optional[List[Any]] = None,
) -> AgentDecision:
    """Wrap ``MCPDiscovery.analyze`` with the expected context shape.

    Parameters
    ----------
    agent
        A configured ``MCPDiscovery`` instance. If ``None`` (agent
        disabled, no API key wired, etc) the helper returns a
        neutral no-op ``AgentDecision`` so the caller never blocks.
    existing_mcps
        Names of MCPs already known to the agent — either the raw
        ``MCP_CANDIDATES`` list from ``skills_learner.py``, a list of
        names previously persisted to ``mcp_candidates.json``, or a
        mixed list of strings and dicts. Passed to Claude so it
        doesn't re-propose them.

    Returns
    -------
    AgentDecision
        ``metadata['candidates']`` holds the parsed candidate list
        (empty on error / no-op).
    """
    if agent is None:
        return AgentDecision(
            approved=True, reasoning="",
            metadata={"candidates": []},
        )
    decision = agent.analyze({"existing_mcps": list(existing_mcps or [])})
    # If Claude returned candidates, persist them to the bounded
    # store. We do this here rather than inside ``analyze`` so unit
    # tests can exercise ``parse_response`` without side effects.
    if not decision.error:
        candidates = (decision.metadata or {}).get("candidates") or []
        if candidates:
            agent.persist_candidates(candidates, reasoning=decision.reasoning)
    return decision


__all__ = [
    "MCPDiscovery",
    "discover_mcps",
    "MAX_CANDIDATES_PER_RUN",
    "MAX_STORED_CANDIDATES",
]
