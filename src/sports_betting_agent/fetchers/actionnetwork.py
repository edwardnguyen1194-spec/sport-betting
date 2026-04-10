"""Action Network odds fetcher.

Action Network's public website is backed by a JSON API hosted at
``https://api.actionnetwork.com/web/v1/scoreboard/<league>`` which
returns full line data for every game on the board, including odds
from every book Action tracks. The endpoint requires no API key -- a
browser-style ``User-Agent`` is sufficient.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models_schema import GameOdds, OddsLine
from .base import BaseFetcher, FetcherError


AN_BASE = "https://api.actionnetwork.com/web/v1/scoreboard"

SPORT_PATHS: Dict[str, tuple[str, str, str]] = {
    "baseball_ncaa": ("ncaab", "baseball_ncaa", "college-baseball"),
    "baseball_mlb": ("mlb", "baseball_mlb", "mlb"),
    "basketball_nba": ("nba", "basketball_nba", "nba"),
    "basketball_ncaab": ("ncaab", "basketball_ncaab", "college-basketball"),
    "football_nfl": ("nfl", "football_nfl", "nfl"),
    "football_ncaaf": ("ncaaf", "football_ncaaf", "college-football"),
    "hockey_nhl": ("nhl", "hockey_nhl", "nhl"),
}

# Action Network internal book-id -> human-readable label (partial map).
BOOK_LABELS = {
    15: "pinnacle",
    75: "fanduel",
    68: "draftkings",
    69: "betmgm",
    30: "caesars",
    264: "bet365",
    123: "pointsbet",
    79: "barstool",
    972: "bovada",
    1005: "hardrock",
    59: "espn_bet",
}


class ActionNetworkFetcher(BaseFetcher):
    source_name = "actionnetwork"
    supported_sports = {k: v[0] for k, v in SPORT_PATHS.items()}

    def fetch(self, sport_key: str) -> List[GameOdds]:
        if sport_key not in SPORT_PATHS:
            raise FetcherError(f"ActionNetwork: unsupported sport_key={sport_key}")

        slug, sport_label, league_label = SPORT_PATHS[sport_key]
        url = f"{AN_BASE}/{slug}"
        payload = self._get(
            url,
            params={"bookIds": ",".join(str(b) for b in BOOK_LABELS), "periods": "event"},
            headers={"Origin": "https://www.actionnetwork.com", "Referer": "https://www.actionnetwork.com/"},
        )
        return self._parse(payload, sport_label, league_label)

    # ------------------------------------------------------------------

    def _parse(
        self,
        payload: Dict[str, Any],
        sport_label: str,
        league_label: str,
    ) -> List[GameOdds]:
        games: List[GameOdds] = []
        for event in payload.get("games", []) or []:
            game = self._parse_game(event, sport_label, league_label)
            if game is not None:
                games.append(game)
        return games

    def _parse_game(
        self,
        ev: Dict[str, Any],
        sport_label: str,
        league_label: str,
    ) -> Optional[GameOdds]:
        teams = ev.get("teams") or []
        if len(teams) < 2:
            return None
        home = next((t for t in teams if t.get("id") == ev.get("home_team_id")), teams[0])
        away = next((t for t in teams if t.get("id") == ev.get("away_team_id")), teams[1])
        home_team = home.get("full_name") or home.get("display_name") or ""
        away_team = away.get("full_name") or away.get("display_name") or ""
        if not home_team or not away_team:
            return None

        commence = None
        start = ev.get("start_time")
        if start:
            try:
                commence = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
            except ValueError:
                commence = None

        game = GameOdds(
            sport=sport_label,
            league=league_label,
            home_team=home_team,
            away_team=away_team,
            commence_time=commence,
            source=self.source_name,
            event_id=str(ev.get("id") or ""),
        )

        odds_block = ev.get("odds") or []
        if isinstance(odds_block, dict):
            odds_block = odds_block.get("event") or []

        for entry in odds_block:
            if not isinstance(entry, dict):
                continue
            book_id = entry.get("book_id")
            book = BOOK_LABELS.get(book_id, f"book_{book_id}")
            # Moneyline
            if entry.get("ml_home") is not None:
                game.lines.append(
                    OddsLine(book=book, market="moneyline", selection=home_team, american=_n(entry.get("ml_home")))
                )
            if entry.get("ml_away") is not None:
                game.lines.append(
                    OddsLine(book=book, market="moneyline", selection=away_team, american=_n(entry.get("ml_away")))
                )
            # Spread
            if entry.get("spread_home") is not None:
                game.lines.append(
                    OddsLine(
                        book=book,
                        market="spread",
                        selection=home_team,
                        american=_n(entry.get("spread_home_line")) or -110.0,
                        line=_n(entry.get("spread_home")),
                    )
                )
            if entry.get("spread_away") is not None:
                game.lines.append(
                    OddsLine(
                        book=book,
                        market="spread",
                        selection=away_team,
                        american=_n(entry.get("spread_away_line")) or -110.0,
                        line=_n(entry.get("spread_away")),
                    )
                )
            # Total
            if entry.get("total") is not None:
                total = _n(entry.get("total"))
                game.lines.append(
                    OddsLine(
                        book=book,
                        market="total",
                        selection="Over",
                        american=_n(entry.get("over")) or -110.0,
                        line=total,
                    )
                )
                game.lines.append(
                    OddsLine(
                        book=book,
                        market="total",
                        selection="Under",
                        american=_n(entry.get("under")) or -110.0,
                        line=total,
                    )
                )

        return game


def _n(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
