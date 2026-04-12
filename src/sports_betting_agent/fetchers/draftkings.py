"""DraftKings Sportsbook odds fetcher.

DraftKings exposes an internal JSON API at sportsbook-nash.draftkings.com
that returns full odds for all markets. No API key needed.

Group IDs:
  MLB=84240, NBA=42648, NFL=88808, NHL=42133
  NCAAB=92483, NCAAF=87637
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models_schema import GameOdds, OddsLine
from .base import BaseFetcher, FetcherError


DK_BASE = "https://sportsbook-nash.draftkings.com/sites/US-SB/api/v5/eventgroups"

SPORT_GROUPS: Dict[str, tuple[str, str, str]] = {
    "baseball_mlb": ("84240", "baseball_mlb", "mlb"),
    "basketball_nba": ("42648", "basketball_nba", "nba"),
    "football_nfl": ("88808", "football_nfl", "nfl"),
    "hockey_nhl": ("42133", "hockey_nhl", "nhl"),
    "basketball_ncaab": ("92483", "basketball_ncaab", "college-basketball"),
    "football_ncaaf": ("87637", "football_ncaaf", "college-football"),
}


class DraftKingsFetcher(BaseFetcher):
    source_name = "draftkings"
    supported_sports = {k: v[0] for k, v in SPORT_GROUPS.items()}

    def fetch(self, sport_key: str) -> List[GameOdds]:
        if sport_key not in SPORT_GROUPS:
            return []

        group_id, sport_label, league_label = SPORT_GROUPS[sport_key]
        url = f"{DK_BASE}/{group_id}"
        try:
            data = self._get(url, params={"format": "json"})
        except Exception:
            return []
        return self._parse(data, sport_label, league_label)

    def _parse(self, data: Dict[str, Any], sport: str, league: str) -> List[GameOdds]:
        games: List[GameOdds] = []

        events = data.get("eventGroup", {}).get("events", [])
        if not events:
            # Try alternate structure
            for cat in data.get("eventGroup", {}).get("offerCategories", []):
                for sub in cat.get("offerSubcategoryDescriptors", []):
                    for offer in sub.get("offerSubcategory", {}).get("offers", []):
                        pass  # DK structure varies
            return games

        for event in events:
            try:
                game = self._parse_event(event, sport, league)
                if game and game.lines:
                    games.append(game)
            except Exception:
                continue

        return games

    def _parse_event(self, event: Dict, sport: str, league: str) -> Optional[GameOdds]:
        name = event.get("name", "")
        if " @ " not in name and " vs " not in name.lower():
            return None

        # Parse team names
        sep = " @ " if " @ " in name else " vs "
        parts = name.split(sep, 1)
        if len(parts) != 2:
            return None

        away_team = parts[0].strip()
        home_team = parts[1].strip()

        # Parse start time
        commence = None
        start = event.get("startDate")
        if start:
            try:
                commence = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        game = GameOdds(
            sport=sport,
            league=league,
            home_team=home_team,
            away_team=away_team,
            commence_time=commence,
            source=self.source_name,
            event_id=str(event.get("eventId", "")),
        )

        # Parse offers (odds)
        for category in event.get("displayGroups", []):
            for market in category.get("markets", []):
                market_type = market.get("description", "").lower()
                for outcome in market.get("outcomes", []):
                    self._add_line(game, market_type, outcome, home_team, away_team)

        return game

    def _add_line(self, game: GameOdds, market_type: str, outcome: Dict,
                  home_team: str, away_team: str):
        american = _to_float(outcome.get("oddsAmerican"))
        if american is None:
            return

        label = outcome.get("label", "")
        line = _to_float(outcome.get("line"))

        if "moneyline" in market_type or "money line" in market_type:
            game.lines.append(OddsLine(
                book="draftkings", market="moneyline",
                selection=label, american=american,
            ))
        elif "spread" in market_type or "run line" in market_type or "puck line" in market_type:
            game.lines.append(OddsLine(
                book="draftkings", market="spread",
                selection=label, american=american, line=line,
            ))
        elif "total" in market_type or "over/under" in market_type:
            sel = "Over" if "over" in label.lower() else "Under"
            game.lines.append(OddsLine(
                book="draftkings", market="total",
                selection=sel, american=american, line=line,
            ))


def _to_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(str(val).replace("+", ""))
    except (ValueError, TypeError):
        return None
