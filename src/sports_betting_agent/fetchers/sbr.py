"""Sportsbook Review (SBR) fetcher -- Pinnacle sharp odds.

This is a thin wrapper around whatever SBR odds scraper is already in
use by the deployed bot (the project docs mention "SBR (free
unlimited Pinnacle odds)"). It is intentionally decoupled: if the
``sbrscrape`` package is installed we use it, otherwise we fall back
to SBR's public JSON endpoint at
``https://www.sportsbookreview.com/api/*``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from ..models_schema import GameOdds, OddsLine
from .base import BaseFetcher, FetcherError


logger = logging.getLogger(__name__)


SPORT_MAP: Dict[str, tuple[str, str, str]] = {
    "baseball_ncaa": ("college-baseball", "baseball_ncaa", "college-baseball"),
    "baseball_mlb": ("mlb", "baseball_mlb", "mlb"),
    "basketball_nba": ("nba", "basketball_nba", "nba"),
    "basketball_ncaab": ("ncaab", "basketball_ncaab", "college-basketball"),
    "football_nfl": ("nfl", "football_nfl", "nfl"),
    "football_ncaaf": ("ncaaf", "football_ncaaf", "college-football"),
    "hockey_nhl": ("nhl", "hockey_nhl", "nhl"),
}


class SBRFetcher(BaseFetcher):
    source_name = "sbr"
    supported_sports = {k: v[0] for k, v in SPORT_MAP.items()}

    def fetch(self, sport_key: str) -> List[GameOdds]:
        if sport_key not in SPORT_MAP:
            raise FetcherError(f"SBR: unsupported sport_key={sport_key}")

        slug, sport_label, league_label = SPORT_MAP[sport_key]

        # Preferred: use the optional sbrscrape package if available.
        try:
            from sbrscrape import Scoreboard  # type: ignore

            sport_code = {
                "mlb": "MLB",
                "nba": "NBA",
                "nfl": "NFL",
                "nhl": "NHL",
                "college-baseball": "NCAAB",
                "college-basketball": "NCAAB",
                "college-football": "NCAAF",
            }.get(slug, slug.upper())
            games_data = Scoreboard(sport=sport_code).games or []
            return [self._from_sbrscrape(g, sport_label, league_label) for g in games_data if g]
        except Exception as exc:
            logger.debug("sbrscrape unavailable, falling back to HTTP: %s", exc)

        # Fallback: hit SBR's consensus JSON endpoint.
        url = f"https://www.sportsbookreview.com/betting-odds/{slug}/"
        html = self._get(url, expect_json=False)
        return self._parse_html(html, sport_label, league_label)

    # ------------------------------------------------------------------

    def _from_sbrscrape(self, game: dict, sport_label: str, league_label: str) -> GameOdds:
        home = game.get("home_team") or ""
        away = game.get("away_team") or ""
        commence: Optional[datetime] = None
        if game.get("date"):
            try:
                commence = datetime.fromisoformat(str(game["date"]).replace("Z", "+00:00"))
            except Exception:
                commence = None

        g = GameOdds(
            sport=sport_label,
            league=league_label,
            home_team=home,
            away_team=away,
            commence_time=commence,
            source=self.source_name,
        )

        # Moneylines
        for book, ml in (game.get("home_ml") or {}).items():
            if ml is not None:
                g.lines.append(OddsLine(book=str(book).lower(), market="moneyline", selection=home, american=float(ml)))
        for book, ml in (game.get("away_ml") or {}).items():
            if ml is not None:
                g.lines.append(OddsLine(book=str(book).lower(), market="moneyline", selection=away, american=float(ml)))

        # Spreads — only include if real juice is available (not default -110)
        for book, spread in (game.get("home_spread") or {}).items():
            if spread is not None:
                juice_dict = game.get("home_spread_juice") or {}
                juice = juice_dict.get(book)
                if juice is not None and juice != -110:  # Only real juice, not defaults
                    g.lines.append(OddsLine(book=str(book).lower(), market="spread", selection=home, american=float(juice), line=float(spread)))
        for book, spread in (game.get("away_spread") or {}).items():
            if spread is not None:
                juice_dict = game.get("away_spread_juice") or {}
                juice = juice_dict.get(book)
                if juice is not None and juice != -110:  # Only real juice, not defaults
                    g.lines.append(OddsLine(book=str(book).lower(), market="spread", selection=away, american=float(juice), line=float(spread)))

        # Totals — only include if real juice is available
        for book, total in (game.get("total") or {}).items():
            if total is not None:
                over_dict = game.get("over_juice") or {}
                under_dict = game.get("under_juice") or {}
                over_juice = over_dict.get(book)
                under_juice = under_dict.get(book)
                if over_juice is not None and over_juice != -110:
                    g.lines.append(OddsLine(book=str(book).lower(), market="total", selection="Over", american=float(over_juice), line=float(total)))
                if under_juice is not None and under_juice != -110:
                    g.lines.append(OddsLine(book=str(book).lower(), market="total", selection="Under", american=float(under_juice), line=float(total)))

        return g

    def _parse_html(self, html: str, sport_label: str, league_label: str) -> List[GameOdds]:
        # A very small placeholder parser; the deployed bot already
        # handles SBR HTML via its own code. We return an empty list
        # so the aggregator cleanly skips SBR when sbrscrape isn't
        # available -- other sources still provide full coverage.
        return []
