"""Odds aggregator.

Runs every enabled fetcher in parallel and merges their output into
a single, de-duplicated list of :class:`GameOdds`, with sources
tracked in each game's ``meta``.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Type

from ..config import Settings, get_settings
from ..models_schema import GameOdds, merge_games
from .actionnetwork import ActionNetworkFetcher
from .base import BaseFetcher, FetcherError
from .bovada import BovadaFetcher
from .covers import CoversFetcher
from .draftkings import DraftKingsFetcher
from .espn import ESPNFetcher
from .sbr import SBRFetcher
from .scoresandodds import ScoresAndOddsFetcher
from .vegasinsider import VegasInsiderFetcher


logger = logging.getLogger(__name__)


_FETCHER_REGISTRY: Dict[str, Type[BaseFetcher]] = {
    "bovada": BovadaFetcher,
    "espn": ESPNFetcher,
    "draftkings": DraftKingsFetcher,
    "scoresandodds": ScoresAndOddsFetcher,
    "vegasinsider": VegasInsiderFetcher,
    "covers": CoversFetcher,
    "actionnetwork": ActionNetworkFetcher,
    "sbr": SBRFetcher,
}


class OddsAggregator:
    """Pull + merge odds from every enabled source."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        fetchers: Optional[Dict[str, BaseFetcher]] = None,
    ) -> None:
        self.settings = settings or get_settings()
        if fetchers is not None:
            self.fetchers = fetchers
        else:
            self.fetchers = {
                name: cls(self.settings)
                for name, cls in _FETCHER_REGISTRY.items()
                if name in self.settings.enabled_sources
            }

    # ------------------------------------------------------------------

    def list_sources(self) -> List[str]:
        return sorted(self.fetchers.keys())

    def fetch_sport(self, sport_key: str) -> List[GameOdds]:
        """Fetch a single sport from every enabled source in parallel."""

        results: List[GameOdds] = []
        if not self.fetchers:
            return results

        # Previously 5s which was too aggressive: ActionNetwork returns a
        # 500KB+ JSON payload that can't be fully received from Fly.io in
        # that window, so we silently lost its multi-book spreads and
        # totals on every cycle. Must exceed SBA_HTTP_TIMEOUT (default 25s)
        # or the HTTP call finishes after the aggregator already gave up.
        # Auto-trade runs every 300s so 30s is ~10% of a cycle — safe.
        AGG_TIMEOUT = 30
        with ThreadPoolExecutor(max_workers=max(4, len(self.fetchers))) as pool:
            future_map = {
                pool.submit(self._safe_fetch, fetcher, sport_key): name
                for name, fetcher in self.fetchers.items()
            }
            try:
                for fut in as_completed(future_map, timeout=AGG_TIMEOUT):
                    name = future_map[fut]
                    try:
                        results.extend(fut.result())
                    except Exception as exc:  # pragma: no cover
                        logger.warning("aggregator: %s raised %s", name, exc)
            except TimeoutError:
                logger.warning("aggregator: some sources timed out for %s", sport_key)

        merged = merge_games(results)
        # Stamp every line's last_update to the fetch time if the
        # fetcher didn't set one. The paper trader rejects bets with
        # stale-line timestamps (>10 min). Without this backfill, most
        # lines would have last_update=None and the freshness guard
        # would never bite. We use the current fetch moment as "now" —
        # the actual upstream timestamp is often not exposed by the
        # scraping source, so fetch-time is the best proxy.
        _fetch_now = datetime.now(timezone.utc)
        for game in merged:
            for line in game.lines:
                if line.last_update is None:
                    line.last_update = _fetch_now
        # Remove outlier lines (bad data from alternate markets)
        for game in merged:
            self._remove_outlier_lines(game)
        # Cross-game variance filter: a book whose american price is
        # the SAME value across 10+ unrelated games in the same sport +
        # market is serving fake juice, not a real scraped price. The
        # canonical failure mode this catches: a fetcher that stamps
        # -110 on every outcome when the source doesn't publish juice
        # (ESPN used to do this and contaminated real DraftKings prices
        # via book-name collision). Defense in depth on top of the
        # per-fetcher fixes — if a future fetcher regresses, this kicks
        # the bad data out before strategies see it.
        self._drop_stuck_price_books(merged)
        # Strict filtering:
        # 1. Drop games with no commence_time (can't verify they're upcoming)
        # 2. Drop games that already started or start within 5 min (lines are stale)
        # 3. Drop games more than 24 HOURS away (Uncle's rule: "only
        #    today or 24 hours games only"). Previous 3-day window was
        #    producing bets on tomorrow's and Sunday's games which
        #    Uncle does NOT want. Overridable via SBA_MAX_HOURS_AHEAD.
        now = datetime.now(timezone.utc)
        from datetime import timedelta
        cutoff_soon = now + timedelta(minutes=5)
        max_hours = float(os.environ.get("SBA_MAX_HOURS_AHEAD", "24"))
        cutoff_far = now + timedelta(hours=max_hours)
        upcoming = [
            g for g in merged
            if g.commence_time is not None
            and g.commence_time > cutoff_soon
            and g.commence_time < cutoff_far
        ]
        dropped_no_time = sum(1 for g in merged if g.commence_time is None)
        dropped_past = sum(1 for g in merged if g.commence_time is not None and (g.commence_time <= cutoff_soon or g.commence_time >= cutoff_far))
        logger.info(
            "aggregator.fetch_sport(%s): %d raw -> %d merged -> %d upcoming "
            "(dropped %d no-time, %d past/soon) across %d sources",
            sport_key,
            len(results),
            len(merged),
            len(upcoming),
            dropped_no_time,
            dropped_past,
            len(self.fetchers),
        )
        return upcoming

    def fetch_sports(self, sport_keys: Iterable[str]) -> List[GameOdds]:
        """Fetch and merge multiple sports."""

        all_games: List[GameOdds] = []
        for sport_key in sport_keys:
            all_games.extend(self.fetch_sport(sport_key))
        return all_games

    # ------------------------------------------------------------------

    @staticmethod
    def _remove_outlier_lines(game: GameOdds) -> None:
        """Remove moneyline outliers that are likely alternate/quarter lines.

        If a line's implied probability differs from the median by more
        than 20 percentage points, it's probably from the wrong market.
        Example: Caesars -120 when everyone else is -380.
        """
        from ..models_schema import american_to_implied
        # Group ML lines by team
        teams: Dict[str, list] = {}
        for line in game.lines:
            if line.market != "moneyline" or line.american is None:
                continue
            key = line.selection.lower()
            teams.setdefault(key, []).append(line)

        bad_lines = set()
        for team, lines in teams.items():
            if len(lines) < 3:
                continue
            implieds = [american_to_implied(l.american) for l in lines]
            implieds.sort()
            median = implieds[len(implieds) // 2]
            for line in lines:
                imp = american_to_implied(line.american)
                if abs(imp - median) > 0.20:  # 20% off median = outlier
                    bad_lines.add(id(line))

        if bad_lines:
            game.lines = [l for l in game.lines if id(l) not in bad_lines]

        # Spread-sign sanity check: if most books have team X as the
        # favorite (negative handicap), drop any book that shows X as
        # the dog. Alt-line leaks from ActionNetwork caused this —
        # e.g. FanDuel listing Utah +1.5 while everyone else had Utah
        # -1.5. The odds are real, but the labeled side is wrong.
        signs_by_team: Dict[str, List[tuple]] = {}
        for line in game.lines:
            if line.market != "spread" or line.line is None:
                continue
            key = line.selection.lower()
            signs_by_team.setdefault(key, []).append(line)

        drop_ids = set()
        for team, lines in signs_by_team.items():
            if len(lines) < 3:
                continue
            # Count favorite vs dog votes.
            fav_votes = sum(1 for l in lines if l.line < 0)
            dog_votes = sum(1 for l in lines if l.line > 0)
            if fav_votes == 0 or dog_votes == 0:
                continue   # All books agree — nothing to prune.
            # Minority side is suspect.
            if fav_votes > dog_votes:
                drop_ids.update(id(l) for l in lines if l.line > 0)
            elif dog_votes > fav_votes:
                drop_ids.update(id(l) for l in lines if l.line < 0)
        if drop_ids:
            game.lines = [l for l in game.lines if id(l) not in drop_ids]

        # Spread-magnitude alt-line filter — sport-aware tolerance.
        # NBA spreads legitimately vary by 1+ point across books on
        # the same main line (Caesars -7.5, Pinnacle -8.5 is normal).
        # NFL key numbers cluster around 3 and 7. NHL/MLB run lines
        # are tight (always +/-1.5). Per-sport thresholds:
        SPREAD_TOL = {
            "basketball_nba":   2.5,
            "basketball_ncaab": 3.5,
            "basketball_wnba":  2.5,
            "football_nfl":     3.5,
            "football_ncaaf":   4.0,
            "hockey_nhl":       0.5,
            "baseball_mlb":     0.5,
            "baseball_ncaa":    1.0,
            # Soccer Asian-handicaps span -1.5 to +0 on same favorite.
            "soccer_mls":       2.0,
            "soccer_epl":       2.0,
            "soccer_ucl":       2.0,
        }
        spread_tol = SPREAD_TOL.get(game.sport, 1.0)
        handicaps_by_team: Dict[str, List] = {}
        for line in game.lines:
            if line.market != "spread" or line.line is None:
                continue
            handicaps_by_team.setdefault(line.selection.lower(), []).append(line)
        mag_drops = set()
        for team, lines in handicaps_by_team.items():
            if len(lines) < 3:
                continue
            sorted_h = sorted(abs(l.line) for l in lines)
            median_abs = sorted_h[len(sorted_h) // 2]
            for l in lines:
                if abs(abs(l.line) - median_abs) > spread_tol:
                    mag_drops.add(id(l))
        if mag_drops:
            game.lines = [l for l in game.lines if id(l) not in mag_drops]

        # Total-number alt-line filter — same per-sport tolerance.
        TOTAL_TOL = {
            "basketball_nba":   8.0,
            "basketball_ncaab": 8.0,
            "basketball_wnba":  6.0,
            "football_nfl":     3.0,
            "football_ncaaf":   4.0,
            "hockey_nhl":       0.5,
            "baseball_mlb":     1.0,
            "baseball_ncaa":    1.5,
            "soccer_mls":       1.0,
            "soccer_epl":       1.0,
            "soccer_ucl":       1.0,
        }
        total_tol = TOTAL_TOL.get(game.sport, 2.0)
        totals_by_side: Dict[str, List] = {}
        for line in game.lines:
            if line.market != "total" or line.line is None:
                continue
            totals_by_side.setdefault(line.selection.lower(), []).append(line)
        tot_drops = set()
        for side, lines in totals_by_side.items():
            if len(lines) < 3:
                continue
            sorted_t = sorted(l.line for l in lines)
            median_t = sorted_t[len(sorted_t) // 2]
            for l in lines:
                if abs(l.line - median_t) > total_tol:
                    tot_drops.add(id(l))
        if tot_drops:
            game.lines = [l for l in game.lines if id(l) not in tot_drops]

        # Cross-fetcher dedup: if both Bovada (direct fetcher) and
        # ActionNetwork mapped a Bovada line for the same selection
        # at the same handicap, keep the one with the more recent
        # last_update or just the first encountered. Same book name
        # appearing twice with different numbers means an alt leak —
        # keep the entry whose handicap matches the consensus median.
        from collections import defaultdict as _dd
        dup_keys = _dd(list)
        for l in game.lines:
            key = (l.book.lower(), l.market, l.selection.lower(), l.line)
            dup_keys[key].append(l)
        # Drop later duplicates with the same key — prefer the entry
        # with real juice over one that only has a handicap. (Without
        # this, ESPN/Covers/VI lines with american=None can win the
        # dedup race if they arrive first and shadow ActionNetwork's
        # real-juice line with the same handicap.)
        keep = set()
        # Direct-source preference: when both the direct fetcher (e.g.
        # BovadaFetcher) AND ActionNetwork's secondary mapping return
        # a line for the same book/market/selection/handicap, the
        # direct fetcher is almost always fresher. ActionNetwork's
        # book prices can lag several minutes — which is why Uncle
        # saw Bovada MIN -0.5 at -108 while the real market was -135.
        # Prefer direct > mapped > anything else.
        DIRECT_SOURCE_PREF = {
            "bovada": "bovada",
            "draftkings": "draftkings",
        }
        for key, lines in dup_keys.items():
            book_lower = key[0]
            with_juice = [l for l in lines if l.american is not None]
            if not with_juice:
                keep.add(id(lines[0]))
                continue
            # If this is a known direct-source book, prefer the line
            # whose ``source`` attribute matches the book name itself.
            preferred_src = DIRECT_SOURCE_PREF.get(book_lower)
            best = None
            if preferred_src:
                direct = [
                    l for l in with_juice
                    if getattr(l, "source", "").lower() == preferred_src
                ]
                if direct:
                    best = direct[0]
            if best is None:
                best = with_juice[0]
            keep.add(id(best))
        # Also collapse (book, market, selection) across handicaps to
        # one entry per book — pick the line whose |handicap| matches
        # the per-team median.
        bms = _dd(list)
        for l in game.lines:
            if l.market != "spread" or l.line is None:
                continue
            bms[(l.book.lower(), l.selection.lower())].append(l)
        cross_drops = set()
        for (book, sel), lines in bms.items():
            if len(lines) < 2:
                continue
            # Same book reported multiple handicaps — pick closest to
            # team median, drop the rest.
            team_lines = handicaps_by_team.get(sel, [])
            if not team_lines:
                continue
            sorted_h = sorted(abs(t.line) for t in team_lines if id(t) not in mag_drops)
            if not sorted_h:
                continue
            median_abs = sorted_h[len(sorted_h) // 2]
            best = min(lines, key=lambda x: abs(abs(x.line) - median_abs))
            for l in lines:
                if l is not best:
                    cross_drops.add(id(l))
        if cross_drops:
            game.lines = [l for l in game.lines if id(l) not in cross_drops]

    @staticmethod
    def _drop_stuck_price_books(games: List[GameOdds]) -> None:
        """Drop (book, sport, market) triples whose american price is
        suspiciously uniform across many games.

        The bug this guards against: a fetcher that stamps a fake
        default price (e.g. -110) on every outcome because the upstream
        source doesn't publish juice. Real sportsbooks produce varied
        juice (-105, -115, -120, +105 etc.); a stuck feed produces the
        same number across 20+ unrelated games. That's the tell.

        Thresholds chosen generously so a small slate can't trigger it:
        need ≥10 distinct games, AND ≥80% of those games quote the
        same exact price. Only scoped to ``spread`` and ``total`` —
        moneyline prices naturally cluster near even on pick'ems and
        would false-positive.
        """
        from collections import Counter, defaultdict as _dd

        MIN_GAMES = 10
        DOM_RATIO = 0.80
        MARKETS = ("spread", "total")

        # (book, sport, market) -> {game_id: price}
        buckets: Dict[tuple[str, str, str], Dict[int, float]] = _dd(dict)
        for g in games:
            for l in g.lines:
                if l.market not in MARKETS or l.american is None:
                    continue
                # Dedup by game — both sides of one game at the same
                # price (pick'em) shouldn't count twice.
                buckets[(l.book.lower(), g.sport, l.market)][id(g)] = l.american

        bad_keys: set = set()
        for key, per_game in buckets.items():
            if len(per_game) < MIN_GAMES:
                continue
            top_price, top_ct = Counter(per_game.values()).most_common(1)[0]
            if top_ct / len(per_game) >= DOM_RATIO:
                logger.warning(
                    "stuck-price: dropping %s %s %s — %d/%d games at %s",
                    key[0], key[1], key[2], top_ct, len(per_game), top_price,
                )
                bad_keys.add(key)

        if not bad_keys:
            return
        for g in games:
            if not g.lines:
                continue
            g.lines = [
                l for l in g.lines
                if (l.book.lower(), g.sport, l.market) not in bad_keys
            ]

    @staticmethod
    def _safe_fetch(fetcher: BaseFetcher, sport_key: str) -> List[GameOdds]:
        try:
            return fetcher.fetch(sport_key)
        except FetcherError as exc:
            logger.info("%s: skip %s (%s)", fetcher.source_name, sport_key, exc)
            return []
        except Exception as exc:  # pragma: no cover
            logger.exception("%s crashed on %s: %s", fetcher.source_name, sport_key, exc)
            return []
