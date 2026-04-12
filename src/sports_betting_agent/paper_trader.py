"""24/7 paper-trading engine.

Consumes :class:`BetRecommendation` objects from the ensemble and
simulates bankroll evolution. Results are persisted as JSON so the
Flask dashboard and the Claude chat layer can reason about historical
performance across restarts.

Design notes
============

* **Thread-safe.** The :meth:`PaperTrader.run_forever` loop is
  designed to be started in a background thread from the Flask app.
* **$50 minimum bet, fractional-Kelly sizing** (matches the deployed
  bot's existing rules).
* **Settlement is explicit.** The trader doesn't try to scrape
  results; it just exposes :meth:`settle_bet` so the dashboard (or a
  separate results scraper) can mark a bet won/lost. Open bets stay
  tracked in ``open_bets`` until they're settled.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional

from .config import Settings, get_settings
from .strategies.base import BetRecommendation


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ledger dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Bet:
    id: str
    placed_at: str
    game_key: str
    sport: str
    league: str
    home_team: str
    away_team: str
    market: str
    selection: str
    american: float
    decimal: float
    line: Optional[float]
    book: str
    strategy: str
    confidence: float
    edge: float
    stake: float
    status: str = "open"        # open / won / lost / push / void
    result: Optional[float] = None  # profit/loss
    settled_at: Optional[str] = None
    reasoning: str = ""

    @classmethod
    def from_recommendation(cls, rec: BetRecommendation, stake: float, bet_id: str) -> "Bet":
        return cls(
            id=bet_id,
            placed_at=datetime.now(timezone.utc).isoformat(),
            game_key=rec.game_key,
            sport=rec.sport,
            league=rec.league,
            home_team=rec.home_team,
            away_team=rec.away_team,
            market=rec.market,
            selection=rec.selection,
            american=rec.american,
            decimal=rec.decimal,
            line=rec.line,
            book=rec.book,
            strategy=rec.strategy,
            confidence=rec.confidence,
            edge=rec.edge,
            stake=stake,
            reasoning=rec.reasoning,
        )


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class PaperTrader:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        state_path: Optional[str] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.state_path = state_path or os.path.join(self.settings.data_dir, "ledger.json")
        self._lock = threading.RLock()
        self._stop = threading.Event()

        self.bankroll: float = self.settings.bankroll_start
        self.open_bets: Dict[str, Bet] = {}
        self.closed_bets: List[Bet] = []
        self._next_id: int = 1

        self._load_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.warning("PaperTrader: could not read %s: %s", self.state_path, exc)
            return
        self.bankroll = float(data.get("bankroll", self.settings.bankroll_start))
        self._next_id = int(data.get("next_id", 1))
        self.open_bets = {b["id"]: Bet(**b) for b in data.get("open_bets", [])}
        self.closed_bets = [Bet(**b) for b in data.get("closed_bets", [])]

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        payload = {
            "bankroll": self.bankroll,
            "next_id": self._next_id,
            "open_bets": [asdict(b) for b in self.open_bets.values()],
            "closed_bets": [asdict(b) for b in self.closed_bets[-5_000:]],
        }
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.state_path)

    # ------------------------------------------------------------------
    # Bet lifecycle
    # ------------------------------------------------------------------

    def place(self, rec: BetRecommendation) -> Optional[Bet]:
        with self._lock:
            stake = self._size_stake(rec)
            if stake < self.settings.min_bet:
                logger.debug("skipping rec %s: stake %.2f below min", rec.selection, stake)
                return None
            if stake > self.bankroll:
                logger.debug("skipping rec %s: insufficient bankroll", rec.selection)
                return None
            # Don't double-book the same market — check BOTH open AND closed bets.
            dup_key = (rec.game_key, rec.market, rec.selection.lower())
            for existing in self.open_bets.values():
                if (existing.game_key, existing.market, existing.selection.lower()) == dup_key:
                    return None
            for existing in self.closed_bets:
                if (existing.game_key, existing.market, existing.selection.lower()) == dup_key:
                    return None
            bet_id = f"pt-{self._next_id:06d}"
            self._next_id += 1
            bet = Bet.from_recommendation(rec, stake, bet_id)
            self.open_bets[bet_id] = bet
            self.bankroll -= stake
            self._save_state()
            logger.info(
                "[PAPER] placed %s on %s at %+.0f stake=$%.2f (conf=%.2f edge=%+.3f)",
                bet.strategy,
                bet.selection,
                bet.american,
                bet.stake,
                bet.confidence,
                bet.edge,
            )
            return bet

    def place_many(self, recs: Iterable[BetRecommendation], max_bets: int = 5) -> List[Bet]:
        """Place bets from recommendations, capped at max_bets per cycle.
        Takes the top picks sorted by confidence * edge."""
        sorted_recs = sorted(recs, key=lambda r: r.confidence * max(r.edge, 0), reverse=True)
        placed = []
        for r in sorted_recs:
            if len(placed) >= max_bets:
                break
            bet = self.place(r)
            if bet is not None:
                placed.append(bet)
        return placed

    def settle_bet(self, bet_id: str, outcome: str) -> Optional[Bet]:
        """Mark a bet won/lost/push/void and update bankroll."""

        with self._lock:
            bet = self.open_bets.pop(bet_id, None)
            if bet is None:
                return None
            outcome = outcome.lower()
            if outcome == "won":
                profit = bet.stake * (bet.decimal - 1.0)
                self.bankroll += bet.stake + profit
                bet.result = profit
            elif outcome == "lost":
                bet.result = -bet.stake
            elif outcome in {"push", "void"}:
                self.bankroll += bet.stake
                bet.result = 0.0
            else:
                # Unknown -> treat as push.
                self.bankroll += bet.stake
                bet.result = 0.0
                outcome = "push"
            bet.status = outcome
            bet.settled_at = datetime.now(timezone.utc).isoformat()
            self.closed_bets.append(bet)
            self._save_state()
            return bet

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def _size_stake(self, rec: BetRecommendation) -> float:
        frac = rec.stake_fraction
        if frac <= 0:
            frac = 0.01
        stake = max(self.settings.min_bet, self.bankroll * frac)
        stake = min(stake, self.bankroll * self.settings.max_bet_pct)
        return round(stake, 2)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        with self._lock:
            closed = self.closed_bets
            wins = sum(1 for b in closed if b.status == "won")
            losses = sum(1 for b in closed if b.status == "lost")
            pushes = sum(1 for b in closed if b.status in {"push", "void"})
            graded = wins + losses
            win_rate = wins / graded if graded else 0.0
            pnl = sum(b.result or 0.0 for b in closed)
            roi = pnl / sum(b.stake for b in closed) if closed else 0.0
            return {
                "bankroll": round(self.bankroll, 2),
                "open_bets": len(self.open_bets),
                "closed_bets": len(closed),
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "win_rate": round(win_rate, 4),
                "pnl": round(pnl, 2),
                "roi": round(roi, 4),
            }

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def run_forever(
        self,
        recommender: Callable[[], List[BetRecommendation]],
        interval: Optional[int] = None,
    ) -> None:
        """Pull recommendations on an interval and place qualifying bets.

        ``recommender`` is a zero-arg callable the caller provides
        (typically a closure over the aggregator + ensemble strategy).
        """

        interval = interval or self.settings.auto_trade_interval_seconds
        logger.info("PaperTrader: auto-trade loop started (interval=%ds)", interval)
        while not self._stop.is_set():
            try:
                recs = recommender()
                placed = self.place_many(recs)
                logger.info("PaperTrader: cycle placed %d bets (bankroll=$%.2f)", len(placed), self.bankroll)
            except Exception as exc:  # pragma: no cover
                logger.exception("PaperTrader: cycle failed: %s", exc)
            # Sleep in small chunks so stop() is responsive.
            for _ in range(interval):
                if self._stop.is_set():
                    break
                time.sleep(1)

    def start_background(self, recommender: Callable[[], List[BetRecommendation]]) -> threading.Thread:
        thread = threading.Thread(
            target=self.run_forever,
            args=(recommender,),
            name="paper-trader",
            daemon=True,
        )
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()
