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
# v2 scoreboard returns public-betting bet_info (tickets + money %) for
# every outcome. We call it as a *supplementary* request — v1 still
# drives the main line/price data because our parser is battle-tested
# against its structure and occasional API drift.
AN_V2_BASE = "https://api.actionnetwork.com/web/v2/scoreboard"

SPORT_PATHS: Dict[str, tuple[str, str, str]] = {
    # Expanded 2026-04-17 to give Uncle coverage across all major
    # world sport leagues. Each tuple is (an_slug, sport_label, league_label).
    # AN's scoreboard endpoint exposes these leagues via /web/v1/scoreboard/<slug>.
    # Baseball
    "baseball_mlb": ("mlb", "baseball_mlb", "mlb"),
    # NOTE: AN has no college-baseball feed — left out rather than
    # pollute the aggregator.

    # Basketball
    "basketball_nba": ("nba", "basketball_nba", "nba"),
    "basketball_ncaab": ("ncaab", "basketball_ncaab", "college-basketball"),
    "basketball_wnba": ("wnba", "basketball_wnba", "wnba"),
    "basketball_euroleague": ("euroleague", "basketball_euroleague", "euroleague"),

    # American football
    "football_nfl": ("nfl", "football_nfl", "nfl"),
    "football_ncaaf": ("ncaaf", "football_ncaaf", "college-football"),
    "football_cfl": ("cfl", "football_cfl", "cfl"),

    # Hockey
    "hockey_nhl": ("nhl", "hockey_nhl", "nhl"),
    "hockey_khl": ("khl", "hockey_khl", "khl"),

    # Soccer — the big 5 European leagues + MLS + UEFA competitions.
    # NOTE: AN uses the NO-HYPHEN form for most slugs ("laliga",
    # "seriea", "ligue1"). Verified via empirical probe on 2026-04-17.
    "soccer_mls": ("mls", "soccer_mls", "mls"),
    "soccer_epl": ("epl", "soccer_epl", "epl"),
    "soccer_ucl": ("ucl", "soccer_ucl", "ucl"),
    "soccer_uel": ("uel", "soccer_uel", "uel"),
    "soccer_esp": ("laliga", "soccer_esp", "laliga"),
    "soccer_ita": ("seriea", "soccer_ita", "seriea"),
    "soccer_ger": ("bundesliga", "soccer_ger", "bundesliga"),
    "soccer_fra": ("ligue1", "soccer_fra", "ligue1"),

    # Combat + tennis (spread/total coverage varies; some have only ML)
    "mma_ufc": ("ufc", "mma_ufc", "ufc"),
    "tennis_atp": ("atp", "tennis_atp", "atp"),
    "tennis_wta": ("wta", "tennis_wta", "wta"),
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
        # One supplementary v2 call per day — gives us every outcome's
        # tickets-% / money-% so RLM and PublicFade actually have data.
        # Keyed on event_id so the per-game parse can look it up cheaply.
        self._public_cache: Dict[str, Dict[str, float]] = {}
        for day_offset in (0, 1, 2):
            date_str = (now + timedelta(days=day_offset)).strftime("%Y%m%d")
            try:
                self._load_public_pcts(slug, date_str)
            except Exception:
                # Public-pct fetch is best-effort — never let a v2 miss
                # block the main v1 line fetch.
                pass
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

    def _load_public_pcts(self, slug: str, date_str: str) -> None:
        """Query the v2 scoreboard endpoint once per day and aggregate
        public-betting percentages (tickets + money) per game.

        Writes into ``self._public_cache`` keyed by ``str(event_id)``.
        Fails silently — the v1 parse still runs even when this misses.
        """
        url = f"{AN_V2_BASE}/{slug}"
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
        for g in (payload.get("games") or []):
            event_id = str(g.get("id") or "")
            if not event_id:
                continue
            sh_t, sa_t, to_t, tu_t = [], [], [], []   # ticket % per side
            sh_m, sa_m, to_m, tu_m = [], [], [], []   # money % per side
            for _book_id, book in (g.get("markets") or {}).items():
                if not isinstance(book, dict):
                    continue
                event = book.get("event", {}) or {}
                for entry in event.get("spread", []) or []:
                    bi = entry.get("bet_info") or {}
                    t_pct = (bi.get("tickets") or {}).get("percent")
                    m_pct = (bi.get("money") or {}).get("percent")
                    side = entry.get("side")
                    if t_pct and side == "home": sh_t.append(t_pct)
                    if t_pct and side == "away": sa_t.append(t_pct)
                    if m_pct and side == "home": sh_m.append(m_pct)
                    if m_pct and side == "away": sa_m.append(m_pct)
                for entry in event.get("total", []) or []:
                    bi = entry.get("bet_info") or {}
                    t_pct = (bi.get("tickets") or {}).get("percent")
                    m_pct = (bi.get("money") or {}).get("percent")
                    side = entry.get("side")
                    if t_pct and side == "over": to_t.append(t_pct)
                    if t_pct and side == "under": tu_t.append(t_pct)
                    if m_pct and side == "over": to_m.append(m_pct)
                    if m_pct and side == "under": tu_m.append(m_pct)

            def avg(xs):
                return sum(xs) / len(xs) / 100.0 if xs else None

            meta = {}
            if sh_t: meta["public_spread_home_pct"] = avg(sh_t)
            if sa_t: meta["public_spread_away_pct"] = avg(sa_t)
            if to_t: meta["public_total_over_pct"] = avg(to_t)
            if tu_t: meta["public_total_under_pct"] = avg(tu_t)
            # Money-% variants let RLM look for the signature sharp
            # signal: low tickets % AND high money % on the same side.
            if sh_m: meta["public_spread_home_money_pct"] = avg(sh_m)
            if sa_m: meta["public_spread_away_money_pct"] = avg(sa_m)
            if to_m: meta["public_total_over_money_pct"] = avg(to_m)
            if tu_m: meta["public_total_under_money_pct"] = avg(tu_m)

            if meta:
                self._public_cache[event_id] = meta

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

        # Keep ONLY the main-line entry per book. Previous "first entry"
        # approach leaked alternate lines (puck line +2.5, 1st period,
        # half spreads) into the main market — some books list alts
        # first, causing the dashboard to show a book with Utah -1.5
        # when the true main line is Utah +1.5. Filter strictly to
        # ``type == "game"`` (or missing type, which AN uses as default
        # for the game-level line) before deduping.
        seen_books: set = set()
        for entry in odds_block:
            if not isinstance(entry, dict):
                continue
            # Skip non-game markets: 1H, 2H, 1Q, 2Q, 3Q, 4Q, 1P, 2P, 3P, etc.
            entry_type = str(entry.get("type", "game")).lower()
            if entry_type not in ("game", "", "full_game", "fulltime", "fullgame"):
                continue
            # Skip alt spreads / alt totals — AN tags these in meta.
            meta = entry.get("meta") or {}
            if isinstance(meta, dict):
                if meta.get("is_alternate") or meta.get("alt"):
                    continue
            book_id = entry.get("book_id")
            if book_id in seen_books:
                continue
            seen_books.add(book_id)
            try:
                self._apply_entry(entry, game, home_team, away_team)
            except Exception:
                continue

        # Extract public betting percentages if available
        self._extract_public_pcts(ev, game)

        return game

    def _extract_public_pcts(self, ev: Dict[str, Any], game: GameOdds) -> None:
        """Merge public-betting percentages into ``game.meta``.

        Preferred source: the v2 scoreboard cache populated in ``fetch()``.
        It has tickets-% and money-% per side for every outcome — strong
        enough for RLM's sharp-money divergence signal.

        Legacy fallback: scan the v1 event payload for the old key paths
        (``betting``, ``public_betting``, …). AN hasn't populated these
        in years but leaving the fallback in costs nothing and covers us
        if the v2 endpoint flakes.
        """
        # -- v2 cache path (preferred) ---------------------------------
        event_id = str(ev.get("id") or "")
        cache = getattr(self, "_public_cache", None)
        if cache and event_id in cache:
            for meta_key, val in cache[event_id].items():
                if val is None:
                    continue
                game.meta[meta_key] = val
            return

        # -- legacy v1 fallback ----------------------------------------
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
