"""Base class and shared HTTP helpers for odds fetchers."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional

import requests

from ..config import Settings, get_settings
from ..models_schema import GameOdds


logger = logging.getLogger(__name__)


class FetcherError(RuntimeError):
    """Raised when a fetcher cannot return usable data."""


class BaseFetcher:
    """Common HTTP behaviour for all odds fetchers.

    Subclasses implement :meth:`fetch` and reuse the ``_get`` helper,
    which adds a realistic User-Agent, handles retries, and returns
    JSON (or raw text).
    """

    source_name: str = "base"
    #: Map of normalized sport keys -> source-specific path/identifier.
    supported_sports: Dict[str, str] = {}

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.settings.http_user_agent,
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    # ------------------------------------------------------------------
    # Networking
    # ------------------------------------------------------------------

    def _get(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: int = 1,
        backoff: float = 0.5,
        expect_json: bool = True,
    ) -> Any:
        """HTTP GET with retry + exponential backoff."""

        err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                resp = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.settings.http_timeout,
                )
                resp.raise_for_status()
                if expect_json:
                    return resp.json()
                return resp.text
            except Exception as exc:  # pragma: no cover - network paths
                err = exc
                logger.debug(
                    "%s GET %s failed (attempt %d/%d): %s",
                    self.source_name,
                    url,
                    attempt,
                    retries,
                    exc,
                )
                time.sleep(backoff ** attempt)
        raise FetcherError(f"{self.source_name}: GET {url} failed after {retries} retries: {err}")

    # ------------------------------------------------------------------
    # Contract
    # ------------------------------------------------------------------

    def fetch(self, sport_key: str) -> List[GameOdds]:  # pragma: no cover - interface
        """Return a list of normalized :class:`GameOdds` for ``sport_key``.

        ``sport_key`` is the normalized aggregator key, e.g.
        ``"baseball_ncaa"`` or ``"basketball_nba"``.
        """

        raise NotImplementedError

    def fetch_many(self, sport_keys: Iterable[str]) -> List[GameOdds]:
        """Fetch multiple sports and combine results, swallowing errors."""

        results: List[GameOdds] = []
        for key in sport_keys:
            try:
                results.extend(self.fetch(key))
            except FetcherError as exc:
                logger.warning("%s fetch failed for %s: %s", self.source_name, key, exc)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("%s unexpected error for %s: %s", self.source_name, key, exc)
        return results
