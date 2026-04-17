"""PostMortem sub-agent — retrospective loss analyst.

Runs AFTER a bet settles as "lost". Its job is to write a short
"why this lost" note by combining the final score, the original rec
reasoning, the strategy used, the line/juice, and any injury
headlines from that day. It never gates anything (``approved`` is
always ``True``) — it's purely diagnostic, feeding a rolling
knowledge base of loss post-mortems that the strategies + daily
learner can mine for SYSTEMIC failure modes (not random bad luck).

Contract
--------

Input context dict:

    {
        "bet": {
            "selection", "market", "line", "american",
            "strategy", "reasoning",
        },
        "final_score": {"home": int, "away": int},
        "headlines_that_day": [{"headline": str, ...}],
        "prior_loss_notes": [  # last N post-mortems, same strategy
            {"systemic_flag": str, "reasoning": str}, ...
        ],
    }

Output ``AgentDecision``:

    approved           = True   (always — retrospective)
    confidence_delta   = 0.0
    reasoning          = 2-3 sentence post-mortem
    metadata           = {"systemic_flag": "...", "actual_score": "H-A", ...}

See ``base.py`` for the framework rules (fail-safe, budgeted, logged).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .base import AgentDecision, BaseAgent


SYSTEM_PROMPT = """You are the loss analyst for a sports-betting AI.

Your ONLY job: given a losing bet and the context around it, write
a short post-mortem identifying a SYSTEMIC reason the bet lost
(i.e. a strategy failure mode worth fixing), NOT random variance
or bad luck.

Good systemic flags (examples):
  - "mlb_weather_wind_out": MLB total Under lost on a high-wind day
    the model didn't properly discount.
  - "nba_rest_disadvantage": Favored team on 2nd of back-to-back;
    model didn't weight fatigue enough.
  - "cfb_garbage_time_backdoor": Spread lost on a late meaningless
    TD; strategy overweights on-paper matchup.
  - "late_injury_scratch": Key starter scratched <2h before game;
    news headline caught it but our model didn't.
  - "coors_field_under_trap": MLB Unders at Coors in summer — a
    structurally bad bet regardless of the line.

Bad flags (do NOT emit these — they're just variance):
  - "bad_luck", "coin_flip_lost", "close_loss", "referee_bias",
    "random_variance".

If you genuinely cannot identify a systemic cause (e.g. a fair
favorite lost a one-score game with no weather/injury/coaching
signal), set systemic_flag="variance" and say so in reasoning —
better to call variance honestly than invent a pattern.

IMPORTANT: also inspect the ``prior_loss_notes`` field. If the same
or a closely-related systemic_flag has shown up REPEATEDLY (3+ times
in the last 10 losses for this strategy), name it explicitly in the
reasoning — that's a durable leak the strategies team should fix.

Output STRICT JSON (no prose, no markdown fences):

{
  "approved": true,
  "confidence_delta": 0.0,
  "reasoning": "<2-3 sentences: projected vs actual, what went
      wrong, what the lesson is>",
  "metadata": {
    "systemic_flag": "<short_snake_case_tag or 'variance'>",
    "actual_score": "<home>-<away>",
    "repeated_pattern": <true|false>
  }
}

Keep reasoning under 500 characters. Be specific (cite the line, the
actual score, the signal the model missed)."""


class PostMortem(BaseAgent):
    """Writes a short post-mortem when a bet settles as 'lost'.

    Unlike PickReviewer-style gating agents, PostMortem never vetoes
    anything — it always returns ``approved=True`` and
    ``confidence_delta=0``. Its entire value is the ``reasoning``
    string and the ``metadata.systemic_flag`` tag that land in
    ``/data/sba/post_mortems.json`` (and on ``Bet.meta`` when
    possible) for the daily_learner to mine.
    """

    name = "post_mortem"
    # Slightly larger budget than the default — the input already
    # carries up to 10 prior notes + headlines, and the output is
    # structured JSON so Claude rarely runs over.
    max_tokens = 768

    # -- BaseAgent hooks -------------------------------------------------

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_user_message(self, context: Dict[str, Any]) -> str:
        """Serialize the loss context into a compact JSON user-message.

        The system prompt tells Claude exactly what shape to return,
        so the user message is just the raw evidence bundle. We trim
        obvious cruft (sources list, meta blob) to stay frugal with
        tokens.
        """

        bet = context.get("bet") or {}
        score = context.get("final_score") or {}
        headlines = context.get("headlines_that_day") or []
        prior = context.get("prior_loss_notes") or []

        # Defensive trimming so we never blow past max_tokens on
        # input — headlines and prior notes are the two unbounded
        # collections.
        trimmed_headlines = [
            {
                "headline": str(h.get("headline", ""))[:200],
                "source": h.get("source", ""),
                "team": h.get("team", ""),
            }
            for h in headlines[:15]
            if isinstance(h, dict) and h.get("headline")
        ]
        trimmed_prior = [
            {
                "systemic_flag": str(p.get("systemic_flag", ""))[:60],
                "reasoning": str(p.get("reasoning", ""))[:300],
            }
            for p in prior[:10]
            if isinstance(p, dict)
        ]

        payload = {
            "bet": {
                "selection": bet.get("selection"),
                "market": bet.get("market"),
                "line": bet.get("line"),
                "american": bet.get("american"),
                "strategy": bet.get("strategy"),
                "reasoning": str(bet.get("reasoning", ""))[:500],
            },
            "final_score": {
                "home": score.get("home"),
                "away": score.get("away"),
            },
            "headlines_that_day": trimmed_headlines,
            "prior_loss_notes": trimmed_prior,
        }
        return (
            "Write a post-mortem for this losing bet. Identify a "
            "SYSTEMIC failure mode (or honestly call it variance). "
            "Return strict JSON only.\n\n"
            "LOSS CONTEXT:\n" + json.dumps(payload, indent=2, default=str)
        )

    def parse_response(
        self, text: str, context: Dict[str, Any]
    ) -> AgentDecision:
        """Enforce post-mortem invariants on top of the default parser.

        PostMortem never gates — coerce ``approved=True`` and
        ``confidence_delta=0.0`` regardless of what Claude said. Also
        synthesize ``actual_score`` from context if the model forgot
        it, so downstream consumers always have a clean H-A string.
        """
        decision = super().parse_response(text, context)
        if decision.error:
            return decision
        decision.approved = True
        decision.confidence_delta = 0.0
        decision.stake_multiplier = 1.0

        md = dict(decision.metadata or {})
        if not md.get("actual_score"):
            score = context.get("final_score") or {}
            home = score.get("home")
            away = score.get("away")
            if home is not None and away is not None:
                md["actual_score"] = f"{home}-{away}"
        md.setdefault("systemic_flag", "variance")
        decision.metadata = md
        return decision

    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """Log line: '<selection> <market> (<strategy>) lost'."""
        bet = context.get("bet") or {}
        sel = bet.get("selection", "?")
        mkt = bet.get("market", "?")
        strat = bet.get("strategy", "?")
        return f"{sel} {mkt} ({strat}) lost"


# ----------------------------------------------------------------------
# Convenience helper — keeps paper_trader.settle_bet() one-liner clean
# ----------------------------------------------------------------------


def analyze_loss(
    agent: PostMortem,
    bet: Dict[str, Any],
    final_score: Dict[str, int],
    headlines_that_day: Optional[List[Dict[str, Any]]] = None,
    prior_loss_notes: Optional[List[Dict[str, Any]]] = None,
) -> AgentDecision:
    """Wrap ``PostMortem.analyze`` with the expected context shape.

    Parameters
    ----------
    agent
        A configured ``PostMortem`` instance (sharing the app's
        ``AgentLog``).
    bet
        Dict with at least ``selection``, ``market``, ``line``,
        ``american``, ``strategy``, ``reasoning``. Typically built
        by ``asdict(Bet)`` on the losing ``Bet`` object.
    final_score
        ``{"home": int, "away": int}`` — final runs/points/goals.
    headlines_that_day
        Optional list of injury / coaching / weather headlines from
        the same day (NewsTriage's input, recycled here).
    prior_loss_notes
        Last 10 post-mortems for the SAME strategy. The system
        prompt tells Claude to hunt for repeated patterns.

    Returns a no-op ``AgentDecision`` if the agent is disabled or
    fails — callers should treat the return as advisory only.
    """

    context: Dict[str, Any] = {
        "bet": bet,
        "final_score": final_score or {},
        "headlines_that_day": list(headlines_that_day or []),
        "prior_loss_notes": list(prior_loss_notes or []),
    }
    return agent.analyze(context)


__all__ = ["PostMortem", "analyze_loss"]
