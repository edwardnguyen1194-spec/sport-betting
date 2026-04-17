"""ScoresAndOdds.com HTML scraper.

ScoresAndOdds publishes a free, consensus-style odds grid for every
major US sport. There is no public JSON endpoint, so this fetcher
downloads the HTML page and parses it with BeautifulSoup.

Pages use predictable slugs such as::

    https://www.scoresandodds.com/mlb
    https://www.scoresandodds.com/nba
    https://www.scoresandodds.com/nfl
    https://www.scoresandodds.com/nhl
    https://www.scoresandodds.com/ncaaf
    https://www.scoresandodds.com/ncaab

Each game is a ``div.event-card`` containing ``tr.event-card-row``
rows with ``data-side="away"``/``data-side="home"``. Team names are
in ``a[aria-label]``. Odds are in ``td[data-field]`` cells with
``span.data-value`` for the number.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

from ..models_schema import GameOdds, OddsLine
from .base import BaseFetcher, FetcherError


SO_BASE = "https://www.scoresandodds.com"

SPORT_PATHS: Dict[str, tuple[str, str, str]] = {
    "baseball_ncaa": ("ncaab", "baseball_ncaa", "college-baseball"),
    "baseball_mlb": ("mlb", "baseball_mlb", "mlb"),
    "basketball_nba": ("nba", "basketball_nba", "nba"),
    "basketball_ncaab": ("ncaab", "basketball_ncaab", "college-basketball"),
    "football_nfl": ("nfl", "football_nfl", "nfl"),
    "football_ncaaf": ("ncaaf", "football_ncaaf", "college-football"),
    "hockey_nhl": ("nhl", "hockey_nhl", "nhl"),
}


class ScoresAndOddsFetcher(BaseFetcher):
    source_name = "scoresandodds"
    supported_sports = {k: v[0] for k, v in SPORT_PATHS.items()}

    def fetch(self, sport_key: str) -> List[GameOdds]:
        if sport_key not in SPORT_PATHS:
            raise FetcherError(f"ScoresAndOdds: unsupported sport_key={sport_key}")

        slug, sport_label, league_label = SPORT_PATHS[sport_key]
        url = f"{SO_BASE}/{slug}"
        html = self._get(url, expect_json=False)
        return self._parse(html, sport_label, league_label)

    def _parse(self, html: str, sport_label: str, league_label: str) -> List[GameOdds]:
        soup = BeautifulSoup(html, "html.parser")
        games: List[GameOdds] = []

        for card in soup.select("div.event-card"):
            rows = card.select("tr.event-card-row")
            away_row = None
            home_row = None
            for row in rows:
                side = row.get("data-side", "")
                if side == "away":
                    away_row = row
                elif side == "home":
                    home_row = row

            if away_row is None or home_row is None:
                continue

            away_team = _get_team_name(away_row)
            home_team = _get_team_name(home_row)
            if not away_team or not home_team:
                continue

            game = GameOdds(
                sport=sport_label,
                league=league_label,
                home_team=home_team,
                away_team=away_team,
                source=self.source_name,
                meta={},
            )

            _add_row_lines(game, away_row, away_team)
            _add_row_lines(game, home_row, home_team)

            if game.lines:
                games.append(game)

        return games


def _get_team_name(row) -> str:
    el = row.select_one("a[aria-label]")
    if el:
        return el.get("aria-label", "").strip()
    el = row.select_one("span.team-name a")
    if el:
        return el.get_text(strip=True)
    return ""


def _add_row_lines(game: GameOdds, row, team_name: str) -> None:
    side = row.get("data-side", "")

    for td in row.select("td[data-field]"):
        field = td.get("data-field", "")
        val_el = td.select_one("span.data-value")
        odds_el = td.select_one("small.data-odds")

        raw_val = val_el.get_text(strip=True) if val_el else ""
        raw_odds = odds_el.get_text(strip=True) if odds_el else ""

        if "moneyline" in field:
            ml = _parse_american(raw_val)
            if ml is not None:
                game.lines.append(
                    OddsLine(
                        book="scoresandodds_consensus",
                        market="moneyline",
                        selection=team_name,
                        american=ml,
                    )
                )

        elif "spread" in field:
            spread = _parse_spread_val(raw_val, raw_odds)
            if spread is not None:
                line_val, juice = spread
                game.lines.append(
                    OddsLine(
                        book="scoresandodds_consensus",
                        market="spread",
                        selection=team_name,
                        american=juice,
                        line=line_val,
                    )
                )

        elif "total" in field:
            total = _parse_total_val(raw_val, raw_odds)
            if total is not None:
                line_val, juice, ou_side = total
                if ou_side is None:
                    ou_side = "Over" if side == "away" else "Under"
                game.lines.append(
                    OddsLine(
                        book="scoresandodds_consensus",
                        market="total",
                        selection=ou_side,
                        american=juice,
                        line=line_val,
                    )
                )


def _parse_american(value: str) -> Optional[float]:
    if not value:
        return None
    cleaned = value.replace("EVEN", "+100").replace("even", "+100").replace("PK", "+100").strip()
    cleaned = re.sub(r"[^0-9.+\-]", "", cleaned)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_spread_val(value: str, odds: str) -> Optional[tuple[float, float]]:
    if not value:
        return None
    cleaned = re.sub(r"[^0-9.+\-]", "", value)
    try:
        line = float(cleaned) if cleaned else None
    except ValueError:
        return None
    if line is None:
        return None
    if value.startswith("+"):
        line = abs(line)
    elif value.startswith("-"):
        line = -abs(line)
    # Real juice only. Previously defaulted to -110 when scraped odds
    # were missing or unparseable, contaminating the pipeline with fake
    # "10-cent juice" that silently flowed through to strategies and
    # the dashboard. Return None so downstream knows juice is unknown.
    juice = _parse_american(odds) if odds else None
    return line, juice


def _parse_total_val(value: str, odds: str) -> Optional[tuple[float, float, Optional[str]]]:
    if not value:
        return None
    upper = value.upper().strip()
    ou_side = None
    if upper.startswith("O"):
        ou_side = "Over"
        upper = upper[1:].strip()
    elif upper.startswith("U"):
        ou_side = "Under"
        upper = upper[1:].strip()
    cleaned = re.sub(r"[^0-9.]", "", upper)
    try:
        line = float(cleaned) if cleaned else None
    except ValueError:
        return None
    if line is None:
        return None
    # Real juice only. Previously defaulted to -110 when scraped odds
    # were missing or unparseable, contaminating the pipeline with fake
    # "10-cent juice" that silently flowed through to strategies and
    # the dashboard. Return None so downstream knows juice is unknown.
    juice = _parse_american(odds) if odds else None
    return line, juice, ou_side
