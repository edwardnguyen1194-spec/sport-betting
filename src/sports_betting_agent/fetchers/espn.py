"""ESPN hidden-API odds fetcher.

ESPN's public website is backed by the undocumented but completely
free ``site.api.espn.com`` endpoints. The scoreboard response for
most sports embeds an ``odds`` array with moneyline, spread, and
total information from a consensus provider (often ESPN BET /
Caesars / DraftKings depending on sport).

Docs: https://gist.github.com/akeaswaran/b48b02f1c94f873c6655e7129910fc3b
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from ..models_schema import GameOdds, OddsLine
from .base import BaseFetcher, FetcherError


ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"


SPORT_PATHS: Dict[str, tuple[str, str, str]] = {
    "baseball_ncaa": ("baseball/college-baseball", "baseball_ncaa", "college-baseball"),
    "baseball_mlb": ("baseball/mlb", "baseball_mlb", "mlb"),
    "basketball_nba": ("basketball/nba", "basketball_nba", "nba"),
    "basketball_ncaab": (
        "basketball/mens-college-basketball",
        "basketball_ncaab",
        "mens-college-basketball",
    ),
    "football_nfl": ("football/nfl", "football_nfl", "nfl"),
    "football_ncaaf": (
        "football/college-football",
        "football_ncaaf",
        "college-football",
    ),
    "hockey_nhl": ("hockey/nhl", "hockey_nhl", "nhl"),
    "soccer_mls": ("soccer/usa.1", "soccer_mls", "mls"),
}


class ESPNFetcher(BaseFetcher):
    source_name = "espn"
    supported_sports = {k: v[0] for k, v in SPORT_PATHS.items()}

    def fetch(self, sport_key: str) -> List[GameOdds]:
        if sport_key not in SPORT_PATHS:
            raise FetcherError(f"ESPN: unsupported sport_key={sport_key}")

        path, sport_label, league_label = SPORT_PATHS[sport_key]
        url = f"{ESPN_BASE}/{path}/scoreboard"
        payload = self._get(url, params={"limit": 200})
        return list(self._parse(payload, sport_label, league_label))

    def _parse(
        self,
        payload: Dict[str, Any],
        sport_label: str,
        league_label: str,
    ) -> Iterable[GameOdds]:
        for event in payload.get("events", []) or []:
            game = self._parse_event(event, sport_label, league_label)
            if game is not None:
                yield game

    def _parse_event(
        self,
        event: Dict[str, Any],
        sport_label: str,
        league_label: str,
    ) -> Optional[GameOdds]:
        competitions = event.get("competitions") or []
        if not competitions:
            return None
        comp = competitions[0]
        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if home is None or away is None:
            return None

        home_team = (home.get("team") or {}).get("displayName") or ""
        away_team = (away.get("team") or {}).get("displayName") or ""
        if not home_team or not away_team:
            return None

        commence = None
        date_str = event.get("date") or comp.get("date")
        if date_str:
            try:
                commence = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except ValueError:
                commence = None

        game = GameOdds(
            sport=sport_label,
            league=league_label,
            home_team=home_team,
            away_team=away_team,
            commence_time=commence,
            source=self.source_name,
            event_id=str(event.get("id") or ""),
            meta={"name": event.get("name", "")},
        )

        for odds in comp.get("odds", []) or []:
            provider = (odds.get("provider") or {}).get("name", "espn")
            book = provider.lower().replace(" ", "_")

            # Spread / total come as numbers at the top level.
            over_under = _parse_float(odds.get("overUnder"))
            spread = _parse_float(odds.get("spread"))

            home_odds = odds.get("homeTeamOdds") or {}
            away_odds = odds.get("awayTeamOdds") or {}

            home_ml = _parse_float(home_odds.get("moneyLine"))
            away_ml = _parse_float(away_odds.get("moneyLine"))
            if home_ml is not None:
                game.lines.append(
                    OddsLine(
                        book=book,
                        market="moneyline",
                        selection=home_team,
                        american=home_ml,
                    )
                )
            if away_ml is not None:
                game.lines.append(
                    OddsLine(
                        book=book,
                        market="moneyline",
                        selection=away_team,
                        american=away_ml,
                    )
                )

            if spread is not None:
                # ESPN's spread is from the favorite's perspective; we
                # still record it as a line on the favored side.
                favored = home_team if (home_odds.get("favorite") or spread < 0) else away_team
                under = away_team if favored == home_team else home_team
                game.lines.append(
                    OddsLine(
                        book=book,
                        market="spread",
                        selection=favored,
                        american=-110,
                        line=-abs(spread),
                    )
                )
                game.lines.append(
                    OddsLine(
                        book=book,
                        market="spread",
                        selection=under,
                        american=-110,
                        line=abs(spread),
                    )
                )

            if over_under is not None:
                game.lines.append(
                    OddsLine(
                        book=book,
                        market="total",
                        selection="Over",
                        american=-110,
                        line=over_under,
                    )
                )
                game.lines.append(
                    OddsLine(
                        book=book,
                        market="total",
                        selection="Under",
                        american=-110,
                        line=over_under,
                    )
                )

        return game


def _parse_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
