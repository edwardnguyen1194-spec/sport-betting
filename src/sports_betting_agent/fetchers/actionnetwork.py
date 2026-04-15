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
    # AN's NCAA baseball endpoint was mistakenly mapped to "ncaab"
    # (college basketball); AN itself has no college-baseball scoreboard
    # feed, so we leave it out rather than pollute the aggregator.
    "baseball_mlb": ("mlb", "baseball_mlb", "mlb"),
    "basketball_nba": ("nba", "basketball_nba", "nba"),
    "basketball_ncaab": ("ncaab", "basketball_ncaab", "college-basketball"),
    "basketball_wnba": ("wnba", "basketball_wnba", "wnba"),
    "football_nfl": ("nfl", "football_nfl", "nfl"),
    "football_ncaaf": ("ncaaf", "football_ncaaf", "college-football"),
    "hockey_nhl": ("nhl", "hockey_nhl", "nhl"),
    "soccer_mls": ("mls", "soccer_mls", "mls"),
    "soccer_epl": ("epl", "soccer_epl", "epl"),
    "soccer_ucl": ("ucl", "soccer_ucl", "ucl"),
}

# Action Network internal book-id -> human-readable label.
# Only IDs listed here are accepted — anything else is skipped so the
# dashboard never shows "book_21" style placeholders and outlier filters
# don't get confused by lines tagged to unmapped sources.
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
    21: "circa",          # Sharp book, reputable
    71: "betrivers",      # Formerly SugarHouse
    76: "foxbet",
}


class ActionNetworkFetcher(BaseFetcher):
    source_name = "actionnetwork"
    supported_sports = {k: v[0] for k, v in SPORT_PATHS.items()}

    def fetch(self, sport_key: str) -> List[GameOdds]:
        if sport_key not in SPORT_PATHS:
            raise FetcherError(f"ActionNetwork: unsupported sport_key={sport_key}")

        slug, sport_label, league_label = SPORT_PATHS[sport_key]
        url = f"{AN_BASE}/{slug}"
        from datetime import datetime, timezone, timedelta
        # Fetch today + next 2 days so the aggregator has upcoming games
        # to show when today's slate is already in progress. Without this,
        # late-evening Pacific cycles would only see AN's current-day
        # (running/finished) games and drop them all via the commence
        # filter — we'd lose every non-Bovada spread/total line.
        now = datetime.now(timezone.utc)
        games: List[GameOdds] = []
        seen_keys: set = set()
        for day_offset in (0, 1, 2):
            date_str = (now + timedelta(days=day_offset)).strftime("%Y%m%d")
            try:
                payload = self._get(
                    url,
                    params={
                        "bookIds": ",".join(str(b) for b in BOOK_LABELS),
                        "periods": "event",
                        "date": date_str,
                    },
                    headers={
                        "Origin": "https://www.actionnetwork.com",
                        "Referer": "https://www.actionnetwork.com/",
                    },
                )
            except Exception:
                continue
            for g in self._parse(payload, sport_label, league_label):
                if g.game_key in seen_keys:
                    continue
                seen_keys.add(g.game_key)
                games.append(g)
        return games

    # ------------------------------------------------------------------

    def _parse(
        self,
        payload: Dict[str, Any],
        sport_label: str,
        league_label: str,
    ) -> List[GameOdds]:
        games: List[GameOdds] = []
        # AN has also used ``events`` as the top-level key in some
        # API versions -- accept either.
        raw_games = payload.get("games") or payload.get("events") or []
        for event in raw_games:
            try:
                game = self._parse_game(event, sport_label, league_label)
            except Exception:
                continue
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

        # The "odds" block has drifted across Action Network API
        # versions:
        # * Older: a flat list of entries, each with a book_id.
        # * Newer: a dict { "event": [...] } or a dict keyed by
        #   book_id -> entry.
        odds_block = ev.get("odds") or []
        if isinstance(odds_block, dict):
            nested = odds_block.get("event")
            if isinstance(nested, list):
                odds_block = nested
            else:
                # dict keyed by book_id
                flat = []
                for key, value in odds_block.items():
                    if isinstance(value, dict):
                        value = dict(value)
                        value.setdefault("book_id", key)
                        flat.append(value)
                    elif isinstance(value, list):
                        for v in value:
                            if isinstance(v, dict):
                                v = dict(v)
                                v.setdefault("book_id", key)
                                flat.append(v)
                odds_block = flat

        # Only keep the FIRST odds entry per book (main market)
        # Action Network returns alternate lines, half-time, quarters etc.
        seen_books: set = set()
        for entry in odds_block:
            if not isinstance(entry, dict):
                continue
            book_id = entry.get("book_id")
            if book_id in seen_books:
                continue  # Skip duplicate entries for same book
            seen_books.add(book_id)
            try:
                self._apply_entry(entry, game, home_team, away_team)
            except Exception:
                continue

        # Extract public betting percentages if available
        self._extract_public_pcts(ev, game)

        return game

    @staticmethod
    def _extract_public_pcts(ev: Dict[str, Any], game: GameOdds) -> None:
        """Try to extract public betting % from various AN API key paths."""
        # AN has used different structures across versions
        for key in ("betting", "public_betting", "betting_splits", "ticket_counts"):
            data = ev.get(key)
            if isinstance(data, dict):
                for src_key, meta_key in (
                    ("ml_home_pct", "public_ml_home_pct"),
                    ("ml_away_pct", "public_ml_away_pct"),
                    ("home_ml_pct", "public_ml_home_pct"),
                    ("away_ml_pct", "public_ml_away_pct"),
                    ("spread_home_pct", "public_spread_home_pct"),
                    ("spread_away_pct", "public_spread_away_pct"),
                    ("total_over_pct", "public_total_over_pct"),
                    ("total_under_pct", "public_total_under_pct"),
                ):
                    val = _n(data.get(src_key))
                    if val is not None:
                        game.meta[meta_key] = val / 100.0 if val > 1.0 else val
                return

        # Also check top-level event keys
        for src_key, meta_key in (
            ("home_ticket_pct", "public_ml_home_pct"),
            ("away_ticket_pct", "public_ml_away_pct"),
            ("home_money_pct", "public_ml_home_pct"),
            ("away_money_pct", "public_ml_away_pct"),
        ):
            val = _n(ev.get(src_key))
            if val is not None:
                game.meta[meta_key] = val / 100.0 if val > 1.0 else val

    @staticmethod
    def _apply_entry(
        entry: Dict[str, Any],
        game: GameOdds,
        home_team: str,
        away_team: str,
    ) -> None:
        book_id = entry.get("book_id")
        try:
            book_id_int = int(book_id) if book_id is not None else None
        except (TypeError, ValueError):
            book_id_int = None
        book = BOOK_LABELS.get(book_id_int)
        if book is None:
            # Unknown book id — skip rather than leak "book_21" style
            # placeholders into the dashboard. If a real book shows up
            # we extend BOOK_LABELS above; everything else is noise.
            return

        # Moneyline -- accept multiple historical key spellings.
        ml_home = _first_float(entry, ("ml_home", "home_ml", "moneyline_home", "ml"))
        ml_away = _first_float(entry, ("ml_away", "away_ml", "moneyline_away"))
        if ml_home is not None:
            game.lines.append(
                OddsLine(book=book, market="moneyline", selection=home_team, american=ml_home)
            )
        if ml_away is not None:
            game.lines.append(
                OddsLine(book=book, market="moneyline", selection=away_team, american=ml_away)
            )

        # Spread (handicap + juice separately).
        spread_home = _first_float(entry, ("spread_home", "home_spread", "spread"))
        spread_home_juice = _first_float(
            entry,
            ("spread_home_line", "home_spread_line", "spread_home_price", "spread_line"),
        )
        spread_away = _first_float(entry, ("spread_away", "away_spread"))
        spread_away_juice = _first_float(
            entry,
            ("spread_away_line", "away_spread_line", "spread_away_price"),
        )
        if spread_home is not None and spread_home_juice is not None:
            game.lines.append(
                OddsLine(
                    book=book,
                    market="spread",
                    selection=home_team,
                    american=spread_home_juice,
                    line=spread_home,
                )
            )
        if spread_away is not None and spread_away_juice is not None:
            game.lines.append(
                OddsLine(
                    book=book,
                    market="spread",
                    selection=away_team,
                    american=spread_away_juice,
                    line=spread_away,
                )
            )

        # Total — only add if real juice is available
        total = _first_float(entry, ("total", "over_under", "ou"))
        if total is not None:
            over_juice = _first_float(entry, ("over", "over_line", "over_price"))
            under_juice = _first_float(entry, ("under", "under_line", "under_price"))
            if over_juice is not None:
                game.lines.append(
                    OddsLine(book=book, market="total", selection="Over", american=over_juice, line=total)
                )
            if under_juice is not None:
                game.lines.append(
                    OddsLine(book=book, market="total", selection="Under", american=under_juice, line=total)
                )


def _n(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(entry: Dict[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    """Return the first non-None numeric value from ``keys`` in ``entry``."""

    for key in keys:
        if key not in entry:
            continue
        parsed = _n(entry.get(key))
        if parsed is not None:
            return parsed
    return None
