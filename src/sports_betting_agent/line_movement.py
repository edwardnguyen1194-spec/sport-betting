"""Line-movement tracking and steam detection.

Stores per-game odds snapshots and exposes helpers that identify
steam moves — synchronized line movements across multiple sharp
books, widely regarded as the clearest signal of professional money.

World-class bettors (Pinnacle's own trading desk included) track
these movements because they compress all available information —
injuries, lineup changes, weather, sharp action — into a single
number that shifts in near-real-time.

References
----------
* Pinnacle's CLV / closing-line research.
* ThunderBet / PicktheOdds "steam move" methodology: 3+ sharp books
  moving in the same direction within a tight window.
* SharpEdge public-vs-sharp-tracker patterns.

Steam heuristic used here
-------------------------
For a given (game, market, selection):
  1. Compare the last-known snapshot to the newest.
  2. Require at least STEAM_MIN_BOOKS sharp books (Pinnacle, Circa,
     Bovada, Bookmaker, BetOnline) to have moved in the SAME
     direction.
  3. The summed American-odds delta must exceed STEAM_MIN_CENTS on
     moneyline/total juice, or the spread must have moved at least
     STEAM_MIN_SPREAD points across the sharp consensus.

The detector is deliberately conservative — a handful of false
positives costs nothing, but false negatives mean we miss the move.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from .models_schema import GameOdds


logger = logging.getLogger(__name__)


SHARP_BOOKS = {"pinnacle", "circa", "bovada", "bookmaker", "betonline"}
STEAM_MIN_BOOKS = 2                 # at least this many sharps moving together
STEAM_MIN_CENTS = 15.0              # summed juice change threshold
STEAM_MIN_SPREAD = 0.5              # half-point swing on spreads/totals
SNAPSHOT_KEEP = 48                  # retain up to this many snapshots per key


@dataclass
class Snapshot:
    ts: str                          # ISO 8601 UTC
    book: str
    american: float
    line: Optional[float] = None    # handicap for spread, total number for totals


@dataclass
class SteamMove:
    game_key: str
    market: str                      # "spread" | "total" | "moneyline"
    selection: str
    direction: str                   # "shorten" (odds tighter / line shorter) | "lengthen"
    sharp_books_moving: List[str]
    juice_delta: float               # summed American-odds change across sharps
    line_delta: float                # average spread/total movement across sharps
    from_ts: str
    to_ts: str


class LineMovementStore:
    """Persist odds snapshots and detect steam moves.

    Keyed by (game_key, market, selection). Each key holds a rolling
    list of snapshots; we trim to SNAPSHOT_KEEP to keep the JSON file
    small enough to reload quickly on each request.
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "line_snapshots.json")
        # key -> list[Snapshot]
        self._store: Dict[str, List[Snapshot]] = {}
        self._load()

    # -- persistence -------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._store = {
                k: [Snapshot(**s) for s in v] for k, v in raw.items()
            }
        except Exception as exc:
            logger.warning("LineMovementStore load failed: %s", exc)
            self._store = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {k: [asdict(s) for s in v] for k, v in self._store.items()}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, self.path)

    # -- ingestion ---------------------------------------------------

    @staticmethod
    def _key(game_key: str, market: str, selection: str) -> str:
        return f"{game_key}|{market}|{selection.lower().strip()}"

    def ingest(self, games: Iterable[GameOdds]) -> None:
        """Append the current snapshot for every line in every game."""
        now = datetime.now(timezone.utc).isoformat()
        touched = False
        for game in games:
            for line in game.lines:
                if line.american is None:
                    continue
                key = self._key(game.game_key, line.market, line.selection)
                snap = Snapshot(
                    ts=now,
                    book=line.book.lower(),
                    american=float(line.american),
                    line=line.line,
                )
                history = self._store.setdefault(key, [])
                # Only record when something actually changed — keeps the
                # store lean and makes deltas meaningful.
                prior = next(
                    (s for s in reversed(history) if s.book == snap.book),
                    None,
                )
                if prior and prior.american == snap.american and prior.line == snap.line:
                    continue
                history.append(snap)
                if len(history) > SNAPSHOT_KEEP:
                    del history[:-SNAPSHOT_KEEP]
                touched = True
        if touched:
            self._save()

    # -- steam detection --------------------------------------------

    def detect_steam(self) -> List[SteamMove]:
        """Scan all tracked keys for recent steam moves."""
        moves: List[SteamMove] = []
        for key, history in self._store.items():
            if len(history) < 2:
                continue
            game_key, market, selection = key.split("|", 2)

            # Group snapshots by book; we need the earliest + latest per book.
            per_book: Dict[str, Tuple[Snapshot, Snapshot]] = {}
            for snap in history:
                if snap.book not in SHARP_BOOKS:
                    continue
                first, _ = per_book.get(snap.book, (snap, snap))
                per_book[snap.book] = (first, snap)

            if len(per_book) < STEAM_MIN_BOOKS:
                continue

            # Compute per-book deltas.
            deltas_juice: List[Tuple[str, float]] = []
            deltas_line: List[Tuple[str, float]] = []
            for book, (first, last) in per_book.items():
                deltas_juice.append((book, last.american - first.american))
                if first.line is not None and last.line is not None:
                    deltas_line.append((book, last.line - first.line))

            # Require agreement on direction.
            juice_signs = {1 if d > 0 else -1 if d < 0 else 0 for _, d in deltas_juice}
            juice_signs.discard(0)
            if len(juice_signs) != 1:
                # Books disagree on direction — not steam.
                continue

            summed_juice = sum(d for _, d in deltas_juice)
            line_avg = (
                sum(d for _, d in deltas_line) / len(deltas_line)
                if deltas_line
                else 0.0
            )

            qualifies = abs(summed_juice) >= STEAM_MIN_CENTS or abs(line_avg) >= STEAM_MIN_SPREAD
            if not qualifies:
                continue

            moves.append(
                SteamMove(
                    game_key=game_key,
                    market=market,
                    selection=selection,
                    direction="shorten" if summed_juice < 0 else "lengthen",
                    sharp_books_moving=sorted(per_book.keys()),
                    juice_delta=round(summed_juice, 1),
                    line_delta=round(line_avg, 2),
                    from_ts=min(f.ts for f, _ in per_book.values()),
                    to_ts=max(l.ts for _, l in per_book.values()),
                )
            )
        return moves

    # -- helpers for dashboard --------------------------------------

    def stats(self) -> Dict:
        total_keys = len(self._store)
        total_snapshots = sum(len(v) for v in self._store.values())
        return {
            "tracked_lines": total_keys,
            "total_snapshots": total_snapshots,
        }
