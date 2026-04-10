"""Bovada free odds fetcher.

Bovada exposes a public JSON endpoint used by their own web client:

    https://www.bovada.lv/services/sports/event/v2/events/A/description/{path}

The ``{path}`` segment is the same slug you see in the URL of the
sport's page on bovada.lv. No API key is required, no rate limit is
documented. Bovada publishes a rich moneyline / spread / total for
every game; including the 37 NCAA baseball matchups requested in this
task.

Response shape (what we care about)::

    [
      {
        "events": [
          {
            "id": "...",
            "description": "Team A @ Team B",
            "startTime": 1712764800000,  # ms since epoch, UTC
            "competitors": [
              {"id": "...", "name": "Team A", "home": false},
              {"id": "...", "name": "Team B", "home": true}
            ],
            "displayGroups": [
              {
                "description": "Game Lines",
                "markets": [
                  {
                    "description": "Moneyline",
                    "key": "2W-12",
                    "period": {"description": "Match", "main": true},
                    "outcomes": [
                      {"description": "Team A",
                       "price": {"american": "+150", "decimal": "2.50"}}
                    ]
                  },
                  {"description": "Point Spread", ... },
                  {"description": "Total",       ... },
                ]
              }
            ]
          }
        ]
      }
    ]
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from ..models_schema import GameOdds, OddsLine
from .base import BaseFetcher, FetcherError


BOVADA_BASE = "https://www.bovada.lv/services/sports/event/v2/events/A/description"


# Normalized sport_key -> (Bovada path slug, sport label, league label)
SPORT_PATHS: Dict[str, tuple[str, str, str]] = {
    "baseball_ncaa": ("baseball/college-baseball", "baseball_ncaa", "college-baseball"),
    "baseball_mlb": ("baseball/mlb", "baseball_mlb", "mlb"),
    "basketball_nba": ("basketball/nba", "basketball_nba", "nba"),
    "basketball_ncaab": (
        "basketball/college-basketball",
        "basketball_ncaab",
        "college-basketball",
    ),
    "football_nfl": ("football/nfl", "football_nfl", "nfl"),
    "football_ncaaf": (
        "football/college-football",
        "football_ncaaf",
        "college-football",
    ),
    "hockey_nhl": ("hockey/nhl", "hockey_nhl", "nhl"),
    "soccer_mls": ("soccer/north-america/united-states/mls", "soccer_mls", "mls"),
    "tennis_atp": ("tennis/atp", "tennis_atp", "atp"),
    "mma_ufc": ("ufc-mma/ufc", "mma_ufc", "ufc"),
}


class BovadaFetcher(BaseFetcher):
    """Free, unlimited odds from Bovada."""

    source_name = "bovada"
    supported_sports = {k: v[0] for k, v in SPORT_PATHS.items()}

    # ------------------------------------------------------------------

    def fetch(self, sport_key: str) -> List[GameOdds]:
        if sport_key not in SPORT_PATHS:
            raise FetcherError(f"Bovada: unsupported sport_key={sport_key}")

        path, sport_label, league_label = SPORT_PATHS[sport_key]
        url = f"{BOVADA_BASE}/{path}"
        payload = self._get(
            url,
            params={
                "marketFilterId": "def",
                "preMatchOnly": "true",
                "eventsLimit": 500,
                "lang": "en",
            },
            headers={"Referer": f"https://www.bovada.lv/sports/{path}"},
        )
        return list(self._parse(payload, sport_label, league_label))

    # ------------------------------------------------------------------

    def _parse(
        self,
        payload: Any,
        sport_label: str,
        league_label: str,
    ) -> Iterable[GameOdds]:
        if not isinstance(payload, list):
            return

        for block in payload:
            if not isinstance(block, dict):
                continue
            for event in block.get("events", []) or []:
                try:
                    game = self._parse_event(event, sport_label, league_label)
                except Exception:
                    # Defensive: skip malformed events rather than
                    # letting one bad game crash the whole fetch.
                    continue
                if game is not None:
                    yield game

    def _parse_event(
        self,
        event: Dict[str, Any],
        sport_label: str,
        league_label: str,
    ) -> Optional[GameOdds]:
        competitors = event.get("competitors") or []
        home = next((c for c in competitors if c.get("home")), None)
        away = next((c for c in competitors if not c.get("home")), None)
        if home is None or away is None:
            return None

        home_name = str(home.get("name") or "").strip()
        away_name = str(away.get("name") or "").strip()
        if not home_name or not away_name:
            return None

        start_ms = event.get("startTime")
        commence = None
        if isinstance(start_ms, (int, float)):
            commence = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc)

        game = GameOdds(
            sport=sport_label,
            league=league_label,
            home_team=home_name,
            away_team=away_name,
            commence_time=commence,
            source=self.source_name,
            event_id=str(event.get("id") or ""),
            meta={"description": event.get("description", ""), "live": event.get("live", False)},
        )

        for group in event.get("displayGroups", []) or []:
            group_desc = str(group.get("description") or "").lower()
            # Only parse the "Game Lines" group by default. Everything
            # else (props, futures, alternate lines, team totals,
            # 1st-half/quarter lines) lives in other groups and would
            # pollute the moneyline/spread/total normalisation. A
            # dedicated props fetcher can opt in later.
            if group_desc and "game lines" not in group_desc and group_desc != "lines":
                continue
            for market in group.get("markets", []) or []:
                try:
                    period = market.get("period") or {}
                    # ``main`` defaults to True because some markets
                    # omit the period entirely and those are always the
                    # full-game lines.
                    if period and period.get("main") is False:
                        continue
                    market_desc = str(market.get("description") or "").lower()
                    # Soccer's 3-way market is labelled "Moneyline"
                    # too but has three outcomes including a draw.
                    # Skip it here -- the 2-way merge math in the
                    # strategy layer can't use it, and a future 3-way
                    # strategy can parse it from raw data.
                    outcomes = market.get("outcomes", []) or []
                    if "moneyline" in market_desc and len(outcomes) > 2:
                        continue
                    for outcome in outcomes:
                        line = self._outcome_to_line(
                            outcome,
                            market_desc,
                            home_name=home_name,
                            away_name=away_name,
                        )
                        if line is not None:
                            game.lines.append(line)
                except Exception:
                    # Never let one malformed market kill the event.
                    continue

        return game

    def _outcome_to_line(
        self,
        outcome: Dict[str, Any],
        market_desc: str,
        *,
        home_name: str,
        away_name: str,
    ) -> Optional[OddsLine]:
        price = outcome.get("price") or {}
        american_raw = price.get("american")
        decimal_raw = price.get("decimal")
        handicap_raw = price.get("handicap")

        american = _parse_american(american_raw)
        decimal = _parse_float(decimal_raw)
        handicap = _parse_float(handicap_raw)
        if american is None and decimal is None:
            return None

        description = str(outcome.get("description") or "").strip()

        if "moneyline" in market_desc or market_desc == "money line":
            market = "moneyline"
            selection = description
        elif "spread" in market_desc or "runline" in market_desc or "puckline" in market_desc:
            market = "spread"
            selection = description
        elif "total" in market_desc:
            market = "total"
            selection = description  # "Over"/"Under"
        else:
            market = market_desc.replace(" ", "_") or "other"
            selection = description

        return OddsLine(
            book="bovada",
            market=market,
            selection=selection,
            american=american,
            decimal=decimal,
            line=handicap,
        )


# ---------------------------------------------------------------------------
# Number parsing helpers
# ---------------------------------------------------------------------------


def _parse_american(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    text = str(value).strip().replace("EVEN", "100").replace("PK", "100")
    try:
        return float(text)
    except ValueError:
        return None


def _parse_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None
