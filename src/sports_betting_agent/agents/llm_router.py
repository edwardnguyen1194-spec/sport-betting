"""100% free LLM router for sub-agents.

Every sub-agent used to call Anthropic Claude directly. Uncle said
"free 100%" so we route to free providers only, with cascade fallback
on rate-limit or error:

    1. Gemini 2.0 Flash  — Google AI Studio free tier
                           (15 req/min, 1M req/day)
    2. OpenRouter        — rotating free models:
                           google/gemini-2.0-flash-exp:free
                           meta-llama/llama-3.3-70b-instruct:free
                           qwen/qwen-2.5-72b-instruct:free
    3. Groq              — llama-3.3-70b-versatile
                           (generous free tier, fastest inference)

No paid providers. Router returns the first provider that gives us a
usable response. All three stale at once → router returns ``None`` and
the agent falls back to its no-op path (same as a schema mismatch).

Design notes
------------
* The router speaks a tiny unified interface — ``complete(system_prompt,
  user_message, max_tokens) -> CompletionResult`` — and lets the
  caller decide what to do with the text. Agents already have their
  own JSON parsers.
* Each provider is wrapped in a thin adapter that translates the
  unified call into that vendor's native API. We DON'T use LangChain
  or LiteLLM — those drag in 30 dependencies we don't need.
* Rate-limit detection is status-code based (429) with a 30-second
  per-provider cool-down. A cool provider is skipped until the
  window expires. After 30s we try again.
* Token counting: Gemini and OpenRouter return real usage. Groq's
  OpenAI-compatible endpoint also does. All of them surface to
  ``CompletionResult.tokens_in/out`` so the dashboard cost chart
  works uniformly (even though cost is $0 everywhere).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib import request as _urlreq
from urllib import error as _urlerr


logger = logging.getLogger(__name__)


# Per-provider cool-down window in seconds. Bumps when we get a 429
# or a repeated 5xx. Expires naturally, no manual reset required.
COOL_DOWN_SECONDS = 30


@dataclass
class CompletionResult:
    """What the router returns to callers.

    Fields mirror the Anthropic response shape agents already know:
    ``text`` is the assistant's reply, ``tokens_in/out`` count usage,
    ``provider`` and ``model`` tell the log which backend answered.
    ``error`` is populated only if ALL providers failed in the same
    call — in that case ``text`` is empty and the caller should
    treat it like a no-op.
    """

    text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    error: Optional[str] = None


class _ProviderBase:
    """Tiny interface every provider adapter implements.

    We don't import the vendor SDKs because:
      a) adds 10+ deps we don't want,
      b) their auth flows all look the same (Bearer token),
      c) urllib + JSON is fine for the 1-2 KB prompts agents send.
    """

    name: str = "unknown"
    model: str = ""

    def __init__(self) -> None:
        self._cooldown_until: float = 0.0

    def ready(self) -> bool:  # pragma: no cover
        """Does this provider have an API key configured?"""
        raise NotImplementedError

    def is_cool(self) -> bool:
        """Are we currently in a cool-down window after a 429?"""
        return time.time() < self._cooldown_until

    def mark_cooldown(self, seconds: int = COOL_DOWN_SECONDS) -> None:
        self._cooldown_until = time.time() + seconds

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
    ) -> CompletionResult:  # pragma: no cover
        raise NotImplementedError


class GeminiProvider(_ProviderBase):
    """Google Gemini 2.0 Flash via Google AI Studio free tier.

    Auth: ``?key=`` query-string param (not a Bearer header).
    Endpoint: generativelanguage.googleapis.com
    Free tier: 15 req/min, 1M req/day — plenty for our 10 sub-agents.
    """

    name = "gemini"
    model = "gemini-2.0-flash"
    ENDPOINT = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "{model}:generateContent?key={key}"
    )

    def ready(self) -> bool:
        return bool(os.environ.get("GOOGLE_AI_API_KEY"))

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
    ) -> CompletionResult:
        key = os.environ["GOOGLE_AI_API_KEY"]
        url = self.ENDPOINT.format(model=self.model, key=key)
        body = {
            # Gemini takes system_instruction as a separate field; it
            # handles it better than smashing it into the user turn.
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {"role": "user", "parts": [{"text": user_message}]}
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": max_tokens,
                # response_mime_type nudges Gemini toward valid JSON
                # without us having to scrape markdown code fences
                # — big win for agents that already expect JSON.
                "responseMimeType": "application/json",
            },
        }
        start = time.time()
        req = _urlreq.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with _urlreq.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
        except _urlerr.HTTPError as exc:
            if exc.code == 429:
                self.mark_cooldown()
                return CompletionResult(
                    provider=self.name, model=self.model,
                    error=f"rate_limit:{exc.code}",
                )
            return CompletionResult(
                provider=self.name, model=self.model,
                error=f"http_{exc.code}",
            )
        except Exception as exc:
            return CompletionResult(
                provider=self.name, model=self.model, error=str(exc),
            )

        latency_ms = int((time.time() - start) * 1000)
        try:
            data = json.loads(raw)
        except Exception as exc:
            return CompletionResult(
                provider=self.name, model=self.model,
                error=f"bad_json:{exc}", latency_ms=latency_ms,
            )

        text = ""
        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                text += part.get("text", "")

        usage = data.get("usageMetadata", {}) or {}
        return CompletionResult(
            text=text,
            tokens_in=int(usage.get("promptTokenCount", 0)),
            tokens_out=int(usage.get("candidatesTokenCount", 0)),
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
        )


class OpenRouterProvider(_ProviderBase):
    """OpenRouter free models (OpenAI-compatible API).

    Rotates through a list of free models so a single rate-limit
    doesn't kill the fallback tier. Auth via Bearer header.
    """

    name = "openrouter"
    # Listed best-first. Each call picks the next model in the list
    # (round-robin) so we spread load across free quotas.
    # Verified live 2026-04-18 — old flash-exp / llama-3.3-instruct /
    # qwen-2.5-72b were all deprecated. These are the current crop.
    MODELS = [
        "openai/gpt-oss-120b:free",           # OpenAI OSS 120B, json-native
        "qwen/qwen3-next-80b-a3b-instruct:free",  # Qwen 3 Next 80B
        "z-ai/glm-4.5-air:free",              # GLM 4.5 air, 131k ctx
        "google/gemma-4-31b-it:free",         # Google Gemma 4 31B
    ]
    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self) -> None:
        super().__init__()
        self._cursor = 0

    def ready(self) -> bool:
        return bool(os.environ.get("OPENROUTER_API_KEY"))

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
    ) -> CompletionResult:
        key = os.environ["OPENROUTER_API_KEY"]
        # Round-robin through free models
        model = self.MODELS[self._cursor % len(self.MODELS)]
        self._cursor += 1
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "response_format": {"type": "json_object"},
        }
        start = time.time()
        req = _urlreq.Request(
            self.ENDPOINT,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                # OpenRouter asks for these for analytics — harmless.
                "HTTP-Referer": "https://sports-betting-ai-agent.fly.dev",
                "X-Title": "Sports Betting AI Agent",
            },
            method="POST",
        )
        try:
            with _urlreq.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
        except _urlerr.HTTPError as exc:
            if exc.code == 429:
                self.mark_cooldown()
                return CompletionResult(
                    provider=self.name, model=model,
                    error=f"rate_limit:{exc.code}",
                )
            return CompletionResult(
                provider=self.name, model=model,
                error=f"http_{exc.code}",
            )
        except Exception as exc:
            return CompletionResult(
                provider=self.name, model=model, error=str(exc),
            )

        latency_ms = int((time.time() - start) * 1000)
        try:
            data = json.loads(raw)
        except Exception as exc:
            return CompletionResult(
                provider=self.name, model=model,
                error=f"bad_json:{exc}", latency_ms=latency_ms,
            )

        choices = data.get("choices") or []
        text = (
            choices[0].get("message", {}).get("content", "")
            if choices else ""
        )
        usage = data.get("usage", {}) or {}
        return CompletionResult(
            text=text,
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            provider=self.name,
            model=model,
            latency_ms=latency_ms,
        )


class GroqProvider(_ProviderBase):
    """Groq llama-3.3-70b-versatile (OpenAI-compatible, blazing fast).

    Groq's free tier gives ~6k tokens/min + 30 req/min. Their hardware
    is FPGA-like so p50 latency is ~200ms for short prompts — fastest
    of our three providers by a wide margin.
    """

    name = "groq"
    model = "llama-3.3-70b-versatile"
    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def ready(self) -> bool:
        return bool(os.environ.get("GROQ_API_KEY"))

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
    ) -> CompletionResult:
        key = os.environ["GROQ_API_KEY"]
        # Groq rejects response_format=json_object if the word "json"
        # doesn't appear in any message. Suffix the system prompt with
        # the magic word so they stop screaming 400 at us.
        sys_msg = system_prompt
        if "json" not in (sys_msg + user_message).lower():
            sys_msg = sys_msg + " Reply in JSON."
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_message},
            ],
            "response_format": {"type": "json_object"},
        }
        start = time.time()
        req = _urlreq.Request(
            self.ENDPOINT,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with _urlreq.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
        except _urlerr.HTTPError as exc:
            if exc.code == 429:
                self.mark_cooldown()
                return CompletionResult(
                    provider=self.name, model=self.model,
                    error=f"rate_limit:{exc.code}",
                )
            return CompletionResult(
                provider=self.name, model=self.model,
                error=f"http_{exc.code}",
            )
        except Exception as exc:
            return CompletionResult(
                provider=self.name, model=self.model, error=str(exc),
            )

        latency_ms = int((time.time() - start) * 1000)
        try:
            data = json.loads(raw)
        except Exception as exc:
            return CompletionResult(
                provider=self.name, model=self.model,
                error=f"bad_json:{exc}", latency_ms=latency_ms,
            )

        choices = data.get("choices") or []
        text = (
            choices[0].get("message", {}).get("content", "")
            if choices else ""
        )
        usage = data.get("usage", {}) or {}
        return CompletionResult(
            text=text,
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
        )


class LLMRouter:
    """Try each free provider in order; return the first success.

    Usage
    -----
    >>> router = LLMRouter()
    >>> result = router.complete(
    ...     system_prompt="You are an expert at X.",
    ...     user_message="Analyze Y. Return JSON.",
    ...     max_tokens=512,
    ... )
    >>> if result.error is None:
    ...     parsed = json.loads(result.text)

    The router never raises — a total failure surfaces as
    ``CompletionResult(error="all_providers_failed")`` with empty
    text. Callers should treat that as a no-op.
    """

    def __init__(self, providers: Optional[List[_ProviderBase]] = None) -> None:
        if providers is None:
            # Default order is Gemini → OpenRouter → Groq. Gemini
            # first because its 1M req/day is the most generous and
            # its quality is closest to Claude.
            providers = [GeminiProvider(), OpenRouterProvider(), GroqProvider()]
        self.providers = providers

    def available(self) -> List[str]:
        """List of providers that have their API key set."""
        return [p.name for p in self.providers if p.ready()]

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        errors: List[str] = []
        tried_any = False
        for provider in self.providers:
            if not provider.ready():
                errors.append(f"{provider.name}:no_key")
                continue
            if provider.is_cool():
                errors.append(f"{provider.name}:cooldown")
                continue
            tried_any = True
            result = provider.complete(
                system_prompt=system_prompt,
                user_message=user_message,
                max_tokens=max_tokens,
            )
            if result.error is None and result.text:
                return result
            errors.append(f"{provider.name}:{result.error or 'empty'}")
            logger.warning(
                "llm_router: %s failed (%s), trying next",
                provider.name,
                result.error or "empty",
            )

        return CompletionResult(
            error="all_providers_failed" if tried_any else "no_providers_configured",
            text="",
            provider="none",
            model="",
        )


# Module-level singleton so every agent shares the same cool-down
# state. Instantiated lazily so unit tests can patch providers.
_router: Optional[LLMRouter] = None


def get_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router
