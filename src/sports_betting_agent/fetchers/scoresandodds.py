"""ScoresAndOdds.com HTML scraper.

ScoresAndOdds publishes a free, consensus-style odds grid for every
major US sport. There is no public JSON endpoint, so this fetcher
downloads the HTML page and parses it with BeautifulSoup.

Pages use predictable slugs such as::

    https://www.scoresandodds.com/ncaabb       -> NCAA baseball
    https://www.scoresandodds.com/mlb
    https://www.scoresandodds.com/nba
    https://www.scoresandodds.com/ncaab
    https://www.scoresandodds.com/nfl
    https://www.scoresandodds.com/ncaaf
    https://www.scoresandodds.com/nhl

Each game row exposes ``data-team``, ``data-value-ml`` (moneyline),
``data-value-spread``, ``data-value-total`` attributes that we read
directly -- far more stable than scraping the rendered text.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from bs4 import BeautifulSoup

from ..models_schema import GameOdds, OddsLine
from .base import BaseFetcher, FetcherError


SO_BASE = "https://www.scoresandodds.com"

SPORT_PATHS: Dict[str, tuple[str, str, str]] = {
    "baseball_ncaa": ("ncaabb", "baseball_ncaa", "college-baseball"),
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

        for box in soup.select("div.event-card") or soup.select("article.event-card"):
            teams = box.select("tr[data-team]")
            if len(teams) < 2:
                continue
            away_row, home_row = teams[0], teams[1]
            away_team = _clean(away_row.get("data-team", ""))
            home_team = _clean(home_row.get("data-team", ""))
            if not home_team or not away_team:
                continue

            game = GameOdds(
                sport=sport_label,
                league=league_label,
                home_team=home_team,
                away_team=away_team,
                source=self.source_name,
                meta={},
            )

            self._add_row_lines(game, home_row, home_team, home=True)
            self._add_row_lines(game, away_row, away_team, home=False)

            if game.lines:
                games.append(game)

        return games

    # ------------------------------------------------------------------

    def _add_row_lines(
        self,
        game: GameOdds,
        row,
        team_name: str,
        *,
        home: bool,
    ) -> None:
        ml = _parse_american(row.get("data-value-ml"))
        spread = _parse_spread(row.get("data-value-spread"))
        total = _parse_total(row.get("data-value-total"))

        if ml is not None:
            game.lines.append(
                OddsLine(
                    book="scoresandodds_consensus",
                    market="moneyline",
                    selection=team_name,
                    american=ml,
                )
            )

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

        if total is not None:
            line_val, juice, side = total
            if side is None:
                side = "Over" if home else "Under"
            game.lines.append(
                OddsLine(
                    book="scoresandodds_consensus",
                    market="total",
                    selection=side,
                    american=juice,
                    line=line_val,
                )
            )


def _clean(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _parse_american(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    cleaned = value.replace("EVEN", "100").replace("PK", "100").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_spread(value: Optional[str]) -> Optional[tuple[float, float]]:
    if not value:
        return None
    # Format examples: "-1.5 -110", "+1.5 -110", "PK -110"
    parts = value.replace("PK", "0").split()
    try:
        line = float(parts[0])
        juice = float(parts[1]) if len(parts) > 1 else -110.0
    except (ValueError, IndexError):
        return None
    return line, juice


def _parse_total(value: Optional[str]) -> Optional[tuple[float, float, Optional[str]]]:
    if not value:
        return None
    parts = value.upper().split()
    try:
        side = None
        if parts[0] in {"O", "OVER"}:
            side = "Over"
            parts = parts[1:]
        elif parts[0] in {"U", "UNDER"}:
            side = "Under"
            parts = parts[1:]
        line = float(parts[0])
        juice = float(parts[1]) if len(parts) > 1 else -110.0
    except (ValueError, IndexError):
        return None
    return line, juice, side
