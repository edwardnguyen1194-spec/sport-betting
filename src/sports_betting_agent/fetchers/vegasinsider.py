"""VegasInsider.com odds scraper.

VegasInsider publishes consensus odds tables at URLs like::

    https://www.vegasinsider.com/college-baseball/odds/las-vegas/
    https://www.vegasinsider.com/mlb/odds/las-vegas/
    https://www.vegasinsider.com/nba/odds/las-vegas/

Each row contains the away / home teams and a sequence of book cells
with moneyline, spread, and total values. We take the first available
book as the consensus price, and additionally emit a per-book line
whenever a known sportsbook column is present.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from bs4 import BeautifulSoup

from ..models_schema import GameOdds, OddsLine
from .base import BaseFetcher, FetcherError


VI_BASE = "https://www.vegasinsider.com"

SPORT_PATHS: Dict[str, tuple[str, str, str]] = {
    "baseball_ncaa": (
        "college-baseball/odds/las-vegas/",
        "baseball_ncaa",
        "college-baseball",
    ),
    "baseball_mlb": ("mlb/odds/las-vegas/", "baseball_mlb", "mlb"),
    "basketball_nba": ("nba/odds/las-vegas/", "basketball_nba", "nba"),
    "basketball_ncaab": (
        "college-basketball/odds/las-vegas/",
        "basketball_ncaab",
        "college-basketball",
    ),
    "football_nfl": ("nfl/odds/las-vegas/", "football_nfl", "nfl"),
    "football_ncaaf": (
        "college-football/odds/las-vegas/",
        "football_ncaaf",
        "college-football",
    ),
    "hockey_nhl": ("nhl/odds/las-vegas/", "hockey_nhl", "nhl"),
}


class VegasInsiderFetcher(BaseFetcher):
    source_name = "vegasinsider"
    supported_sports = {k: v[0] for k, v in SPORT_PATHS.items()}

    def fetch(self, sport_key: str) -> List[GameOdds]:
        if sport_key not in SPORT_PATHS:
            raise FetcherError(f"VegasInsider: unsupported sport_key={sport_key}")
        path, sport_label, league_label = SPORT_PATHS[sport_key]
        url = f"{VI_BASE}/{path}"
        html = self._get(url, expect_json=False)
        return self._parse(html, sport_label, league_label)

    def _parse(self, html: str, sport_label: str, league_label: str) -> List[GameOdds]:
        soup = BeautifulSoup(html, "html.parser")
        games: List[GameOdds] = []

        for table in soup.select("table.main-odds-table") or soup.select("table"):
            for row in table.select("tr.game-row") or table.select("tbody tr"):
                cells = row.find_all("td")
                if len(cells) < 4:
                    continue
                teams_block = cells[0].get_text("\n", strip=True).split("\n")
                if len(teams_block) < 2:
                    continue
                away_team = _clean(teams_block[0])
                home_team = _clean(teams_block[1])
                if not home_team or not away_team:
                    continue

                game = GameOdds(
                    sport=sport_label,
                    league=league_label,
                    home_team=home_team,
                    away_team=away_team,
                    source=self.source_name,
                )

                for cell in cells[1:]:
                    book = _clean(cell.get("data-book") or cell.get("class", [""])[0] or "vegasinsider")
                    text = cell.get_text(" ", strip=True)
                    parsed = _parse_generic_cell(text)
                    if not parsed:
                        continue
                    ml_home, ml_away, spread_fav, total_line = parsed
                    if ml_home is not None:
                        game.lines.append(
                            OddsLine(book=book, market="moneyline", selection=home_team, american=ml_home)
                        )
                    if ml_away is not None:
                        game.lines.append(
                            OddsLine(book=book, market="moneyline", selection=away_team, american=ml_away)
                        )
                    if spread_fav is not None:
                        side_line, juice = spread_fav
                        favored = home_team if side_line < 0 else away_team
                        underdog = away_team if favored == home_team else home_team
                        game.lines.append(
                            OddsLine(book=book, market="spread", selection=favored, american=juice, line=-abs(side_line))
                        )
                        game.lines.append(
                            OddsLine(book=book, market="spread", selection=underdog, american=juice, line=abs(side_line))
                        )
                    if total_line is not None:
                        line_val, juice = total_line
                        game.lines.append(
                            OddsLine(book=book, market="total", selection="Over", american=juice, line=line_val)
                        )
                        game.lines.append(
                            OddsLine(book=book, market="total", selection="Under", american=juice, line=line_val)
                        )

                if game.lines:
                    games.append(game)

        return games


def _clean(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _parse_generic_cell(text: str):
    """Very forgiving parser for VI consensus cells.

    VI's markup changes periodically; rather than over-fitting we
    extract any signed number that looks like a moneyline (>=100 or
    <=-100) or spread (abs < 100) from the text.
    """

    if not text:
        return None
    tokens = text.replace(",", " ").replace("(", " ").replace(")", " ").split()
    nums: list[float] = []
    for tok in tokens:
        try:
            nums.append(float(tok.replace("+", "").replace("EVEN", "100")))
        except ValueError:
            continue
    if not nums:
        return None

    moneylines = [n for n in nums if abs(n) >= 100]
    spreads = [n for n in nums if abs(n) < 100]

    ml_home = moneylines[0] if moneylines else None
    ml_away = moneylines[1] if len(moneylines) > 1 else None
    spread_fav = None
    total_line = None
    if spreads:
        # VegasInsider's scrape exposes only the handicap / total number;
        # per-book juice isn't on the page. We used to stamp a fake
        # american=-110 on both which contaminated the pipeline. Now:
        # return juice=None. Downstream skips lines with no real juice,
        # so VI contributes the reference handicap without a phantom
        # bet price.
        spread_fav = (spreads[0], None)
        if len(spreads) > 1:
            total_line = (spreads[1], None)
    return ml_home, ml_away, spread_fav, total_line
