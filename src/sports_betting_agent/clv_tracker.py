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
                   market: str, american: float):
        """Record a new bet at the time it's placed."""
        rec = CLVRecord(
            bet_id=bet_id,
            game_key=game_key,
            selection=selection,
            market=market,
            bet_american=american,
            bet_decimal=american_to_decimal(american),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.records.append(rec)
        self._save()

    def record_closing_line(self, game_key: str, selection: str,
                            market: str, closing_american: float):
        """Record the closing line for a game (called at game start)."""
        closing_dec = american_to_decimal(closing_american)
        for rec in self.records:
            if (rec.game_key == game_key and
                rec.selection.lower() == selection.lower() and
                rec.market == market and
                rec.closing_american is None):
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
