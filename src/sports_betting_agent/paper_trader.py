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
from .clv_tracker import CLVTracker
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

        # Closing Line Value tracker — the #1 predictor of long-run profit
        # for any spread/total bettor. Records every placement and lets us
        # compute whether we are genuinely beating the closing market.
        self.clv = CLVTracker(self.settings.data_dir)

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
            # Cap total exposure at 75% of starting bankroll. Previously
            # 50% which fit exactly 10 min-bets — Uncle's dashboard hit
            # the ceiling with 9 spreads, leaving zero room for totals
            # (each min $50). 75% gives the strategies breathing room
            # for real market diversity while still keeping bankroll
            # safe (~1/4 untouched).
            total_exposed = sum(b.stake for b in self.open_bets.values())
            if total_exposed + stake > self.settings.bankroll_start * 0.75:
                logger.debug("skipping rec %s: total exposure %.2f would exceed 75%% cap", rec.selection, total_exposed + stake)
                return None
            # Don't double-book the same market — check BOTH open AND closed bets.
            # Exact-selection dup (e.g. two Over 9.0 picks for same game).
            dup_key = (rec.game_key, rec.market, rec.selection.lower())
            for existing in self.open_bets.values():
                if (existing.game_key, existing.market, existing.selection.lower()) == dup_key:
                    return None
            for existing in self.closed_bets:
                if (existing.game_key, existing.market, existing.selection.lower()) == dup_key:
                    return None
            # Cross-side dup: per game, the spread market has ONLY TWO
            # sides — one at -1.5 and one at +1.5. Having placed either
            # side means we should NOT also place the other side.
            # Previously the agent booked Anaheim +1.5 AND Nashville
            # +1.5 on the same game because upstream alt-line filters
            # let both sides appear with the same sign.
            if rec.market in ("spread", "total"):
                for existing in list(self.open_bets.values()) + self.closed_bets:
                    if existing.game_key == rec.game_key and existing.market == rec.market:
                        logger.info(
                            "skipping %s %s: game already has a %s bet on %s",
                            rec.selection, rec.market, rec.market, existing.selection,
                        )
                        return None
            bet_id = f"pt-{self._next_id:06d}"
            self._next_id += 1
            bet = Bet.from_recommendation(rec, stake, bet_id)
            self.open_bets[bet_id] = bet
            self.bankroll -= stake
            # Record for CLV tracking — we'll capture the closing line later
            # (record_closing_lines) and compute CLV = bet_decimal/close_decimal - 1.
            try:
                self.clv.record_bet(
                    bet_id=bet_id,
                    game_key=bet.game_key,
                    selection=bet.selection,
                    market=bet.market,
                    american=bet.american,
                )
            except Exception as exc:
                logger.warning("CLV record_bet failed for %s: %s", bet_id, exc)
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

        Enforces a balanced market mix so the dashboard never shows
        all-spreads or all-totals. Quotas per cycle of 5 bets:

          * Up to 2 spread picks (top-scored)
          * Up to 2 total picks (top-scored)
          * 1 wildcard slot filled by the next best across any market

        Each quota is skipped when no eligible recs exist. Scoring is
        still confidence * max(edge, 0) so we always prefer the agent's
        highest-conviction plays within each market.
        """
        recs = list(recs)
        sorted_recs = sorted(recs, key=lambda r: r.confidence * max(r.edge, 0), reverse=True)

        placed: List[Bet] = []
        picked_ids: set = set()

        def _place_first_n(candidates, n_target):
            placed_here = 0
            for r in candidates:
                if len(placed) >= max_bets or placed_here >= n_target:
                    break
                if id(r) in picked_ids:
                    continue
                bet = self.place(r)
                if bet is not None:
                    placed.append(bet)
                    picked_ids.add(id(r))
                    placed_here += 1

        spreads = [r for r in sorted_recs if r.market == "spread"]
        totals = [r for r in sorted_recs if r.market == "total"]

        # Up to 2 spreads AND up to 2 totals before anything else —
        # guarantees Uncle's dashboard shows both markets as long as
        # recs exist for each.
        _place_first_n(spreads, 2)
        _place_first_n(totals, 2)

        # Remaining slot(s): best-scored across any market we haven't
        # already placed.
        _place_first_n(sorted_recs, max_bets - len(placed))

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
            try:
                self.clv.record_result(bet_id, outcome)
            except Exception as exc:
                logger.warning("CLV record_result failed for %s: %s", bet_id, exc)
            self._save_state()
            return bet

    def record_closing_lines(self, games) -> int:
        """Call this as games are about to start (or just kicked off) so
        each open bet gets its market-close reference line recorded. The
        resulting CLV is the sharpest measurement of whether the picks
        are genuinely beating the market, independent of short-run luck.
        Returns the number of bets that received a closing line update.
        """
        updated = 0
        with self._lock:
            open_bet_keys = {
                (b.game_key, b.market, b.selection.lower().strip()): b
                for b in self.open_bets.values()
            }
            if not open_bet_keys:
                return 0
            for game in games:
                for line in game.lines:
                    if line.american is None:
                        continue
                    key = (game.game_key, line.market, line.selection.lower().strip())
                    bet = open_bet_keys.get(key)
                    if bet is None:
                        continue
                    try:
                        self.clv.record_closing_line(
                            game_key=game.game_key,
                            selection=line.selection,
                            market=line.market,
                            closing_american=float(line.american),
                        )
                        updated += 1
                    except Exception as exc:
                        logger.warning("CLV record_closing_line failed: %s", exc)
        return updated

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
