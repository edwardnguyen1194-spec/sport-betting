"""Covers.com odds fetcher.

Covers.com publishes free consensus odds and betting trends. Their
pages follow a ``/sport/<league>/odds`` pattern, e.g.::

    https://www.covers.com/sport/baseball/college-baseball/odds
    https://www.covers.com/sport/baseball/mlb/odds
    https://www.covers.com/sport/basketball/nba/odds

Like the other HTML sources, this fetcher is permissive: it extracts
any table row that has two team names plus a numeric line, rather
than binding too tightly to their (changing) CSS classes.
"""

from __future__ import annotations

from typing import Dict, List

from bs4 import BeautifulSoup

from ..models_schema import GameOdds, OddsLine
from .base import BaseFetcher, FetcherError


COVERS_BASE = "https://www.covers.com"

SPORT_PATHS: Dict[str, tuple[str, str, str]] = {
    "baseball_ncaa": (
        "sport/baseball/college-baseball/odds",
        "baseball_ncaa",
        "college-baseball",
    ),
    "baseball_mlb": ("sport/baseball/mlb/odds", "baseball_mlb", "mlb"),
    "basketball_nba": ("sport/basketball/nba/odds", "basketball_nba", "nba"),
    "basketball_ncaab": (
        "sport/basketball/college-basketball/odds",
        "basketball_ncaab",
        "college-basketball",
    ),
    "football_nfl": ("sport/football/nfl/odds", "football_nfl", "nfl"),
    "football_ncaaf": (
        "sport/football/ncaaf/odds",
        "football_ncaaf",
        "college-football",
    ),
    "hockey_nhl": ("sport/hockey/nhl/odds", "hockey_nhl", "nhl"),
}


class CoversFetcher(BaseFetcher):
    source_name = "covers"
    supported_sports = {k: v[0] for k, v in SPORT_PATHS.items()}

    def fetch(self, sport_key: str) -> List[GameOdds]:
        if sport_key not in SPORT_PATHS:
            raise FetcherError(f"Covers: unsupported sport_key={sport_key}")
        path, sport_label, league_label = SPORT_PATHS[sport_key]
        url = f"{COVERS_BASE}/{path}"
        html = self._get(url, expect_json=False)
        return self._parse(html, sport_label, league_label)

    def _parse(self, html: str, sport_label: str, league_label: str) -> List[GameOdds]:
        soup = BeautifulSoup(html, "html.parser")
        games: List[GameOdds] = []
        for box in soup.select("div[data-game-id]") or soup.select("tr[data-game-id]"):
            home_el = box.select_one("[data-home-team], [data-home]")
            away_el = box.select_one("[data-away-team], [data-away]")
            if home_el is None or away_el is None:
                continue
            home_team = _clean(home_el.get_text(" ", strip=True))
            away_team = _clean(away_el.get_text(" ", strip=True))
            if not home_team or not away_team:
                continue

            game = GameOdds(
                sport=sport_label,
                league=league_label,
                home_team=home_team,
                away_team=away_team,
                source=self.source_name,
                event_id=box.get("data-game-id"),
            )

            home_ml = _num(box.get("data-home-ml") or box.get("data-ml-home"))
            away_ml = _num(box.get("data-away-ml") or box.get("data-ml-away"))
            if home_ml is not None:
                game.lines.append(
                    OddsLine(book="covers_consensus", market="moneyline", selection=home_team, american=home_ml)
                )
            if away_ml is not None:
                game.lines.append(
                    OddsLine(book="covers_consensus", market="moneyline", selection=away_team, american=away_ml)
                )

            spread = _num(box.get("data-spread"))
            total = _num(box.get("data-total"))
            if spread is not None:
                favored = home_team if spread < 0 else away_team
                underdog = away_team if favored == home_team else home_team
                game.lines.append(
                    OddsLine(book="covers_consensus", market="spread", selection=favored, american=-110, line=-abs(spread))
                )
                game.lines.append(
                    OddsLine(book="covers_consensus", market="spread", selection=underdog, american=-110, line=abs(spread))
                )
            if total is not None:
                game.lines.append(
                    OddsLine(book="covers_consensus", market="total", selection="Over", american=-110, line=total)
                )
                game.lines.append(
                    OddsLine(book="covers_consensus", market="total", selection="Under", american=-110, line=total)
                )

            if game.lines:
                games.append(game)
        return games


def _clean(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _num(value):
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace("+", ""))
    except ValueError:
        return None
