"""Claude AI chat integration.

Wraps the Anthropic SDK with a small, opinionated client: the
dashboard sends a user message plus a short summary of current
recommendations + bankroll, and gets back a Vietnamese-language
answer. If the Anthropic SDK isn't installed or the API key is
missing, the chat endpoint returns a deterministic fallback so the
UI never breaks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import Settings, get_settings
from .strategies.base import BetRecommendation


logger = logging.getLogger(__name__)


SYSTEM_PROMPT_VI = """Bạn là trợ lý AI cá cược thể thao "Sport Betting AI".
Nhiệm vụ của bạn là phân tích kèo (odds), lý giải các bet đề xuất, và
trả lời người dùng bằng tiếng Việt một cách rõ ràng, súc tích, luôn
nhắc nhở quản lý vốn (bankroll management) và cá cược có trách nhiệm.
Dữ liệu dưới đây là dữ liệu THỰC từ hệ thống (Bovada, ESPN, SBR,
ScoresAndOdds, VegasInsider, Covers, Action Network). KHÔNG bịa số.
Khi người dùng hỏi về một trận đấu cụ thể, hãy trích dẫn chiến lược
(heavy favorite / value bets / LSTM player props) và tỷ lệ tự tin."""

SYSTEM_PROMPT_EN = """You are "Sport Betting AI", an assistant that explains
betting recommendations. All odds data comes from real sources
(Bovada, ESPN, SBR, ScoresAndOdds, VegasInsider, Covers, Action
Network). Never fabricate numbers. Always remind the user about
bankroll management and responsible gambling."""


@dataclass
class ChatContext:
    bankroll: float
    stats: Dict
    recommendations: List[BetRecommendation]

    def summarize(self, max_recs: int = 10) -> str:
        lines = [
            f"Bankroll: ${self.bankroll:,.2f}",
            f"Win rate: {self.stats.get('win_rate', 0) * 100:.1f}% over "
            f"{self.stats.get('closed_bets', 0)} settled bets",
            f"Open bets: {self.stats.get('open_bets', 0)}",
            "",
            "Top recommendations:",
        ]
        for rec in self.recommendations[:max_recs]:
            lines.append(
                f"- [{rec.strategy}] {rec.selection} @ {rec.american:+.0f} "
                f"({rec.book}) conf={rec.confidence:.2f} edge={rec.edge:+.3f}"
            )
            if rec.reasoning:
                lines.append(f"  reason: {rec.reasoning.splitlines()[0]}")
        return "\n".join(lines)


class ClaudeChat:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._client = None
        self._model = "claude-opus-4-7"
        self._system = SYSTEM_PROMPT_VI if self.settings.dashboard_language == "vi" else SYSTEM_PROMPT_EN

        api_key = self.settings.anthropic_api_key
        if not api_key:
            logger.info("ClaudeChat: ANTHROPIC_API_KEY not set, using fallback responses")
            return
        try:
            import anthropic  # type: ignore

            self._client = anthropic.Anthropic(api_key=api_key)
            logger.info("ClaudeChat: Anthropic SDK initialised")
        except Exception as exc:
            logger.warning("ClaudeChat: could not init Anthropic SDK: %s", exc)

    def reply(self, user_message: str, context: ChatContext) -> Dict[str, str]:
        """Return a dict ``{role, content}``."""

        if self._client is None:
            return {"role": "assistant", "content": self._fallback(user_message, context)}

        try:
            message = self._client.messages.create(  # type: ignore[attr-defined]
                model=self._model,
                max_tokens=1024,
                system=self._system,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{context.summarize()}\n\n"
                            f"Question: {user_message}"
                        ),
                    }
                ],
            )
            # SDK returns a list of content blocks.
            text = "".join(getattr(b, "text", "") for b in message.content)
            if not text.strip():
                text = self._fallback(user_message, context)
            return {"role": "assistant", "content": text}
        except Exception as exc:  # pragma: no cover
            logger.warning("ClaudeChat: API call failed: %s", exc)
            return {"role": "assistant", "content": self._fallback(user_message, context)}

    def _fallback(self, user_message: str, context: ChatContext) -> str:
        lang_vi = self.settings.dashboard_language == "vi"
        intro = (
            "Mình chưa kết nối được tới Claude API nên trả lời offline. "
            if lang_vi
            else "Offline mode (no Anthropic API key). "
        )
        summary = context.summarize(max_recs=5)
        hint = (
            "\n\nBạn có thể hỏi mình về một trận cụ thể, ví dụ: "
            "\"giải thích bet heavy favorite ở trận NCAA baseball tối nay\"."
            if lang_vi
            else "\n\nAsk me about a specific game to see strategy reasoning."
        )
        return intro + "\n\n" + summary + hint
