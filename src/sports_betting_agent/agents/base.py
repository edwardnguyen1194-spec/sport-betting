"""Shared framework for Claude sub-agents.

Every sub-agent (PickReviewer, NewsTriage, PostMortem, etc.) extends
``BaseAgent`` and implements a single ``analyze()`` method. The base
class owns the Anthropic client, enforces a structured-JSON output
contract, applies a token budget, logs every decision to disk, and
fails SAFE (returns no-op on any error so the hardcoded strategies
remain the source of truth).

Design principles
-----------------

1. **Strategies are the floor, agents are the polish.** Agents can
   adjust confidence within ±10%, veto a bet, or add reasoning.
   They CANNOT invent picks or raise stakes. The hardcoded math is
   always in charge.

2. **Structured output only.** Every agent returns a Pydantic-style
   dict with known fields. No free-form text reaches the paper
   trader. Hallucinations surface as schema-violation errors.

3. **Fail open.** Network error, token budget exceeded, schema
   mismatch → return a neutral no-op result. The trading loop
   never blocks on the agent.

4. **Auditable.** Every agent call is logged to
   ``/data/sba/agent_log.json`` with timestamp, agent name, input
   summary, decision, token usage. Dashboard surfaces via
   ``/api/agent-log``.

5. **Budgeted.** A global per-day token ceiling; when exhausted,
   agents silently skip until tomorrow.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None  # agents silently disabled if SDK missing


logger = logging.getLogger(__name__)


DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_DAILY_TOKEN_BUDGET = 500_000


@dataclass
class AgentDecision:
    """Structured result every agent returns.

    Fields
    ------
    approved
        True = proceed, False = veto. Default True so a bug or
        skipped call never blocks a bet.
    confidence_delta
        Signed multiplier applied to the rec's confidence.
        Clamped by the caller to [-0.10, +0.10]. 0 = no change.
    stake_multiplier
        0.0-1.0. 1.0 = full stake, 0.5 = half, 0.0 = veto (same as
        ``approved=False``). Useful when the agent wants to dip a
        toe rather than pass entirely.
    reasoning
        Short human-readable justification. Goes into rec.reasoning
        and the dashboard.
    metadata
        Arbitrary structured diagnostics (news headlines, weather,
        etc). Surfaces in /api/agent-log for debugging.
    error
        Populated when the agent errored. Caller should treat as
        no-op.
    """

    approved: bool = True
    confidence_delta: float = 0.0
    stake_multiplier: float = 1.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class AgentLogEntry:
    ts: str
    agent: str
    context_summary: str
    decision: Dict[str, Any]
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0


class AgentLog:
    """Rolling on-disk log of every agent decision (bounded)."""

    MAX_ENTRIES = 5000

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "agent_log.json")
        self.entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as fh:
                self.entries = json.load(fh).get("entries", [])
        except Exception:
            self.entries = []

    def record(self, entry: AgentLogEntry) -> None:
        self.entries.append(asdict(entry))
        self.entries = self.entries[-self.MAX_ENTRIES:]
        os.makedirs(self.data_dir, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"entries": self.entries}, fh, indent=2)
        os.replace(tmp, self.path)

    def recent(self, n: int = 50, agent: Optional[str] = None) -> List[Dict]:
        xs = [e for e in self.entries if agent is None or e.get("agent") == agent]
        return xs[-n:]

    def today_token_usage(self) -> Dict[str, int]:
        """Sum token usage across today's entries. Used for budget."""
        today = datetime.now(timezone.utc).date().isoformat()
        total_in = total_out = 0
        for e in self.entries:
            if not e.get("ts", "").startswith(today):
                continue
            total_in += e.get("tokens_in", 0)
            total_out += e.get("tokens_out", 0)
        return {"tokens_in": total_in, "tokens_out": total_out,
                "total": total_in + total_out}


class BaseAgent(ABC):
    """All sub-agents inherit from this class.

    Subclasses only need to implement ``system_prompt()`` and
    ``build_user_message(context)`` plus an optional
    ``parse_response(text, context)`` that turns Claude's text into
    a strict ``AgentDecision``. The base class handles HTTP, token
    budget, structured-JSON parsing, error isolation, and logging.
    """

    name: str = "base_agent"
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __init__(
        self,
        agent_log: AgentLog,
        api_key: Optional[str] = None,
        daily_token_budget: int = DEFAULT_DAILY_TOKEN_BUDGET,
        enabled: Optional[bool] = None,
    ) -> None:
        self.agent_log = agent_log
        self.daily_token_budget = daily_token_budget
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Optional[Anthropic] = None
        # Per-agent env-gate: SBA_AGENT_<NAME>_ENABLED=0 disables it.
        env_key = f"SBA_AGENT_{self.name.upper()}_ENABLED"
        env_val = os.environ.get(env_key)
        if enabled is not None:
            self.enabled = enabled
        elif env_val is not None:
            self.enabled = env_val.lower() not in ("0", "false", "no", "off")
        else:
            self.enabled = True

    @property
    def client(self) -> Optional[Anthropic]:
        if Anthropic is None or not self._api_key:
            return None
        if self._client is None:
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    # -- subclass hooks ---------------------------------------------

    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @abstractmethod
    def build_user_message(self, context: Dict[str, Any]) -> str:
        ...

    def parse_response(
        self, text: str, context: Dict[str, Any]
    ) -> AgentDecision:
        """Default parser expects a JSON object with the decision
        fields. Subclasses can override for more elaborate shapes."""
        try:
            # Strip common fenced-code wrappers Claude sometimes emits.
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```", 2)[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError) as exc:
            return AgentDecision(
                approved=True, reasoning="", error=f"parse_error: {exc}"
            )
        return AgentDecision(
            approved=bool(data.get("approved", True)),
            confidence_delta=self._clamp(
                float(data.get("confidence_delta", 0.0)), -0.10, 0.10
            ),
            stake_multiplier=self._clamp(
                float(data.get("stake_multiplier", 1.0)), 0.0, 1.0
            ),
            reasoning=str(data.get("reasoning", "") or "")[:500],
            metadata=data.get("metadata", {}) or {},
        )

    # -- public API -------------------------------------------------

    def analyze(self, context: Dict[str, Any]) -> AgentDecision:
        if not self.enabled:
            return AgentDecision(reasoning="disabled")
        if self.client is None:
            return AgentDecision(error="no_client")

        # Daily budget enforcement.
        usage = self.agent_log.today_token_usage()
        if usage["total"] >= self.daily_token_budget:
            return AgentDecision(error="daily_budget_exhausted")

        system = self.system_prompt()
        user_msg = self.build_user_message(context)
        start = time.time()
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception as exc:
            logger.warning("agent %s HTTP error: %s", self.name, exc)
            return AgentDecision(error=f"http_error: {exc}")

        latency_ms = int((time.time() - start) * 1000)
        text = ""
        if resp.content:
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    text += block.text
        tokens_in = getattr(resp.usage, "input_tokens", 0)
        tokens_out = getattr(resp.usage, "output_tokens", 0)

        decision = self.parse_response(text, context)
        self.agent_log.record(AgentLogEntry(
            ts=datetime.now(timezone.utc).isoformat(),
            agent=self.name,
            context_summary=self._summarize_context(context),
            decision=asdict(decision),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        ))
        return decision

    # -- helpers ----------------------------------------------------

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """Short string for the log. Subclasses can override."""
        rec = context.get("rec")
        if rec is not None:
            return f"{getattr(rec, 'selection', '?')} {getattr(rec, 'market', '?')} @ {getattr(rec, 'book', '?')}"
        game = context.get("game")
        if game is not None:
            return f"{getattr(game, 'away_team', '?')} @ {getattr(game, 'home_team', '?')}"
        return str(context)[:120]
