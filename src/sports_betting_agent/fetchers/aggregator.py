"""Odds aggregator.

Runs every enabled fetcher in parallel and merges their output into
a single, de-duplicated list of :class:`GameOdds`, with sources
tracked in each game's ``meta``.
"""

from __future__ import annotations

import logging
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
        # Remove outlier lines (bad data from alternate markets)
        for game in merged:
            self._remove_outlier_lines(game)
        # Strict filtering:
        # 1. Drop games with no commence_time (can't verify they're upcoming)
        # 2. Drop games that already started or start within 5 min (lines are stale)
        # 3. Drop games more than 3 days away (odds not reliable yet)
        now = datetime.now(timezone.utc)
        from datetime import timedelta
        cutoff_soon = now + timedelta(minutes=5)
        cutoff_far = now + timedelta(days=3)
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
        # Drop later duplicates with the same key.
        keep = set()
        for key, lines in dup_keys.items():
            keep.add(id(lines[0]))
            for extra in lines[1:]:
                pass   # not added to keep set => removed below
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
    def _safe_fetch(fetcher: BaseFetcher, sport_key: str) -> List[GameOdds]:
        try:
            return fetcher.fetch(sport_key)
        except FetcherError as exc:
            logger.info("%s: skip %s (%s)", fetcher.source_name, sport_key, exc)
            return []
        except Exception as exc:  # pragma: no cover
            logger.exception("%s crashed on %s: %s", fetcher.source_name, sport_key, exc)
            return []
