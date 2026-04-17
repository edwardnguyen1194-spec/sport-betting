"""Closing Line Value (CLV) tracker.

CLV is the #1 predictor of long-term betting profitability, as
established by Stanford Wong in "Sharp Sports Betting" and confirmed
by Marco Blume (former Pinnacle Trading Director).

A bettor who consistently gets CLV-positive bets will be profitable
long-term regardless of short-term variance. This module tracks CLV
for every bet placed and reports whether the AI agent is genuinely
sharp.

CLV = (your_decimal_odds / closing_decimal_odds) - 1
Positive CLV = you got a better price than the market close = +EV
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .models_schema import american_to_decimal


logger = logging.getLogger(__name__)


@dataclass
class CLVRecord:
    bet_id: str
    game_key: str
    selection: str
    market: str
    bet_american: float
    bet_decimal: float
    # ``line`` is the handicap/total number. Critical for totals:
    # "Over" is not unique without the line (Over 8.5 vs Over 9.0 are
    # different bets). Added 2026-04-17 after Agent-2 audit caught the
    # silent bug where every total was matching the first "Over"/"Under"
    # entry seen in the feed, regardless of line number.
    line: Optional[float] = None
    # ``strategy`` lets us break down CLV per strategy for the
    # auto-tuner and the dashboard scorecard.
    strategy: Optional[str] = None
    closing_american: Optional[float] = None
    closing_decimal: Optional[float] = None
    clv: Optional[float] = None
    result: Optional[str] = None  # "won", "lost", "push"
    timestamp: str = ""


class CLVTracker:
    """Track Closing Line Value for every bet the agent places."""

    def __init__(self, data_dir: str = "/data/sba"):
        self.data_dir = data_dir
        self.records: List[CLVRecord] = []
        self._load()

    def _path(self) -> str:
        return os.path.join(self.data_dir, "clv_records.json")

    def _load(self):
        path = self._path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                self.records = [CLVRecord(**r) for r in data]
            except Exception:
                self.records = []

    def _save(self):
        path = self._path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)

    def record_bet(self, bet_id: str, game_key: str, selection: str,
                   market: str, american: float,
                   line: Optional[float] = None,
                   strategy: Optional[str] = None):
        """Record a new bet at the time it's placed.

        ``line`` and ``strategy`` are optional for backwards
        compatibility with old callers; new call sites should pass both.
        """
        rec = CLVRecord(
            bet_id=bet_id,
            game_key=game_key,
            selection=selection,
            market=market,
            bet_american=american,
            bet_decimal=american_to_decimal(american),
            line=line,
            strategy=strategy,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.records.append(rec)
        self._save()

    def record_closing_line(self, game_key: str, selection: str,
                            market: str, closing_american: float,
                            line: Optional[float] = None):
        """Record the closing line for a game (called at game start).

        Matches on (game_key, market, selection, line) so totals at
        different lines don't cross-pollute. For spread markets line is
        the handicap; for totals it's the over/under number; None means
        match any line (legacy fallback).
        """
        closing_dec = american_to_decimal(closing_american)
        for rec in self.records:
            if rec.closing_american is not None:
                continue
            if rec.game_key != game_key:
                continue
            if rec.market != market:
                continue
            if rec.selection.lower() != selection.lower():
                continue
            # Line match: if both sides have a line, it must match.
            # Either-None fallback preserves backwards compat on legacy
            # records placed before ``line`` was a field.
            if rec.line is not None and line is not None:
                if abs(rec.line - line) > 0.01:
                    continue
            rec.closing_american = closing_american
            rec.closing_decimal = closing_dec
            rec.clv = (rec.bet_decimal / closing_dec) - 1
        self._save()

    def record_result(self, bet_id: str, result: str):
        """Record win/loss/push for a bet."""
        for rec in self.records:
            if rec.bet_id == bet_id:
                rec.result = result
                break
        self._save()

    def average_clv(self, min_bets: int = 10) -> Optional[float]:
        """Average CLV across all bets with closing line data."""
        clv_bets = [r for r in self.records if r.clv is not None]
        if len(clv_bets) < min_bets:
            return None
        return sum(r.clv for r in clv_bets) / len(clv_bets)

    def is_sharp(self) -> Optional[bool]:
        """A bettor with avg CLV > 0 over 100+ bets is likely sharp."""
        avg = self.average_clv(min_bets=100)
        if avg is None:
            return None
        return avg > 0

    def stats(self) -> Dict:
        """Summary stats for the dashboard."""
        clv_bets = [r for r in self.records if r.clv is not None]
        total = len(self.records)
        with_clv = len(clv_bets)
        avg_clv = sum(r.clv for r in clv_bets) / with_clv if clv_bets else 0
        positive_clv = sum(1 for r in clv_bets if r.clv > 0)

        won = sum(1 for r in self.records if r.result == "won")
        lost = sum(1 for r in self.records if r.result == "lost")

        return {
            "total_bets_tracked": total,
            "bets_with_clv": with_clv,
            "average_clv": round(avg_clv * 100, 2),  # as percentage
            "positive_clv_pct": round(positive_clv / with_clv * 100, 1) if with_clv else 0,
            "is_sharp": self.is_sharp(),
            "win_rate": round(won / (won + lost) * 100, 1) if (won + lost) > 0 else 0,
        }

    def stats_by_strategy(self) -> Dict[str, Dict]:
        """Per-strategy CLV scoreboard. Lets the daily auto-tuner move
        each strategy's threshold independently — a bad strategy no
        longer drags the global edge knob down."""
        bucket: Dict[str, List[CLVRecord]] = {}
        for r in self.records:
            strat = r.strategy or "unknown"
            # Ensemble concatenates names with '+' when multiple
            # strategies agree. Split so each gets credit.
            for name in strat.split("+"):
                bucket.setdefault(name.strip() or "unknown", []).append(r)
        out: Dict[str, Dict] = {}
        for name, recs in bucket.items():
            clv_bets = [r for r in recs if r.clv is not None]
            won = sum(1 for r in recs if r.result == "won")
            lost = sum(1 for r in recs if r.result == "lost")
            avg = sum(r.clv for r in clv_bets) / len(clv_bets) if clv_bets else 0
            pos = sum(1 for r in clv_bets if r.clv > 0)
            out[name] = {
                "bets": len(recs),
                "bets_with_clv": len(clv_bets),
                "average_clv_pct": round(avg * 100, 2),
                "positive_clv_pct": (
                    round(pos / len(clv_bets) * 100, 1) if clv_bets else 0
                ),
                "win_rate": (
                    round(won / (won + lost) * 100, 1) if (won + lost) else 0
                ),
                "record": f"{won}-{lost}",
            }
        return out
