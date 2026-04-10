"""Runtime configuration.

All settings are environment-driven so the package can be deployed to
fly.io, Docker, or a local machine without touching code. The fly.io
deployment at https://sports-betting-ai-agent.fly.dev only needs the
API keys set as secrets -- every free source works without any key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    return [part.strip() for part in raw.split(",") if part.strip()]


DEFAULT_FREE_SOURCES = [
    "bovada",
    "espn",
    "scoresandodds",
    "vegasinsider",
    "covers",
    "actionnetwork",
    "sbr",
]


@dataclass(frozen=True)
class Settings:
    """Immutable settings snapshot."""

    # --- API keys (all optional; free sources work without them) ---
    the_odds_api_key: str = field(default_factory=lambda: os.getenv("THE_ODDS_API_KEY", ""))
    sharp_api_key: str = field(default_factory=lambda: os.getenv("SHARP_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # --- Source selection ---
    enabled_sources: List[str] = field(
        default_factory=lambda: _env_list("SBA_ENABLED_SOURCES", DEFAULT_FREE_SOURCES)
    )

    # --- HTTP behaviour ---
    http_timeout: float = field(default_factory=lambda: _env_float("SBA_HTTP_TIMEOUT", 12.0))
    http_user_agent: str = field(
        default_factory=lambda: os.getenv(
            "SBA_USER_AGENT",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        )
    )

    # --- Paper-trading engine ---
    bankroll_start: float = field(default_factory=lambda: _env_float("SBA_BANKROLL", 10_000.0))
    min_bet: float = field(default_factory=lambda: _env_float("SBA_MIN_BET", 50.0))
    max_bet_pct: float = field(default_factory=lambda: _env_float("SBA_MAX_BET_PCT", 0.05))
    kelly_fraction: float = field(default_factory=lambda: _env_float("SBA_KELLY_FRACTION", 0.25))

    # --- Strategy thresholds ---
    heavy_fav_min_american: int = field(
        default_factory=lambda: _env_int("SBA_HEAVY_FAV_MIN_AMERICAN", -400)
    )
    heavy_fav_max_american: int = field(
        default_factory=lambda: _env_int("SBA_HEAVY_FAV_MAX_AMERICAN", -150)
    )
    heavy_fav_min_books: int = field(default_factory=lambda: _env_int("SBA_HEAVY_FAV_MIN_BOOKS", 2))
    min_confidence: float = field(
        default_factory=lambda: _env_float("SBA_MIN_CONFIDENCE", 0.65)
    )

    # --- Auto trading loop ---
    auto_trade_enabled: bool = field(
        default_factory=lambda: _env_bool("SBA_AUTO_TRADE", True)
    )
    auto_trade_interval_seconds: int = field(
        default_factory=lambda: _env_int("SBA_AUTO_TRADE_INTERVAL", 300)
    )

    # --- Dashboard ---
    dashboard_host: str = field(default_factory=lambda: os.getenv("SBA_HOST", "0.0.0.0"))
    dashboard_port: int = field(default_factory=lambda: _env_int("PORT", 8080))
    dashboard_language: str = field(default_factory=lambda: os.getenv("SBA_LANG", "vi"))

    # --- Storage ---
    data_dir: str = field(
        default_factory=lambda: os.getenv("SBA_DATA_DIR", "./data/sba")
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance.

    Cached so every module sees a consistent snapshot. Tests can clear
    the cache via ``get_settings.cache_clear()``.
    """

    return Settings()
