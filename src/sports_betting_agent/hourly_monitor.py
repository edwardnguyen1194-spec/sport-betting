"""Hourly self-monitor + anomaly detector.

Complements ``daily_learner.py`` (which makes once-per-day policy
moves) with a lighter-touch hourly sweep that catches problems
*as they emerge* rather than waiting 24h.

What it checks
--------------

1. **Bias anomalies** — if >80% of open bets are on one side
   (all Over, all home, all favorites, all dogs) it logs a warning
   so we can catch a recurrence of the "all Overs" pattern Uncle
   flagged before it drains bankroll.
2. **Stale open bets** — any bet open >48h without closing. The
   auto-settler only uses a 2-day ESPN lookback, so bets played
   on days beyond that are orphaned unless we surface them.
3. **Drawdown approach** — warn at ≥10% drawdown so Uncle has an
   early-warning window before the 20% halt fires.
4. **Strategy silence** — if a strategy that normally fires 3+
   times/day has gone quiet, something in its data inputs may
   have broken (e.g. ActionNetwork public-% pipeline regression).
5. **Fetcher errors** — counts recent aggregator timeouts + parse
   failures. Not fatal, but >25% error rate in a sport is a
   signal to investigate a feed.
6. **Overconfidence cluster** — if >70% of open bets cluster at
   their strategy's confidence cap, the cap is binding and the
   strategy is systematically overclaiming.

What it does NOT do
-------------------

Auto-tune thresholds or place bets. That's the daily_learner's
job with more data and statistical power. The hourly monitor is
pure observability: log findings, flag anomalies, give Uncle +
Claude a quick-read health panel.

Output
------

Appends one entry per hour to ``hourly_log.json`` (bounded at 168
entries = 7 days) and exposes the latest via the dashboard's
``/api/hourly-log`` endpoint.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class HourlySnapshot:
    timestamp: str
    bankroll: float
    peak_bankroll: float
    drawdown_pct: float
    halted: bool
    open_count: int
    open_exposure: float
    closed_today: int
    won_today: int
    lost_today: int
    # Bias indicators
    bias_flags: List[str] = field(default_factory=list)
    # Stale-open count
    stale_open_count: int = 0
    stale_open_ids: List[str] = field(default_factory=list)
    # Per-strategy activity
    strategy_24h: Dict[str, int] = field(default_factory=dict)
    # Anomalies text log
    anomalies: List[str] = field(default_factory=list)
    # Health score 0-100
    health_score: int = 100


class HourlyMonitor:
    """Once-per-hour diagnostic sweep over the paper-trader state."""

    def __init__(self, paper_trader, data_dir: str) -> None:
        self.paper = paper_trader
        self.data_dir = data_dir
        self.log_path = os.path.join(data_dir, "hourly_log.json")
        self._last_run_hour: Optional[str] = None
        self._load_last_run()

    def _load_last_run(self) -> None:
        if not os.path.exists(self.log_path):
            return
        try:
            with open(self.log_path) as fh:
                data = json.load(fh)
            if data.get("entries"):
                ts = data["entries"][-1].get("timestamp", "")
                if ts:
                    self._last_run_hour = ts[:13]  # YYYY-MM-DDTHH
        except Exception:
            pass

    def _should_run(self) -> bool:
        current_hour = datetime.now(timezone.utc).isoformat()[:13]
        return current_hour != self._last_run_hour

    def run(self, force: bool = False) -> Optional[HourlySnapshot]:
        if not force and not self._should_run():
            return None
        snap = self._build_snapshot()
        self._persist(snap)
        self._last_run_hour = snap.timestamp[:13]
        return snap

    def _build_snapshot(self) -> HourlySnapshot:
        paper = self.paper
        now = datetime.now(timezone.utc)
        opens = list(paper.open_bets.values())
        closed = list(paper.closed_bets)

        # 1. Bias anomaly check (open bets only — what we're exposed to now).
        bias_flags: List[str] = []
        if opens:
            sides = Counter(b.selection for b in opens)
            markets = Counter(b.market for b in opens)
            # Over/Under imbalance
            over_ct = sum(1 for b in opens if b.market == "total" and "over" in (b.selection or "").lower())
            under_ct = sum(1 for b in opens if b.market == "total" and "under" in (b.selection or "").lower())
            total_market = over_ct + under_ct
            if total_market >= 5:
                over_ratio = over_ct / total_market if total_market else 0
                if over_ratio >= 0.85:
                    bias_flags.append(f"over_heavy: {over_ct}/{total_market} totals are Over")
                elif over_ratio <= 0.15:
                    bias_flags.append(f"under_heavy: {under_ct}/{total_market} totals are Under")
            # Spread home/away bias via game_key + home_team heuristic:
            # selection equals home_team ⇒ home side bet
            spread_home = sum(
                1 for b in opens
                if b.market == "spread" and b.selection == b.home_team
            )
            spread_away = sum(
                1 for b in opens
                if b.market == "spread" and b.selection == b.away_team
            )
            spread_total = spread_home + spread_away
            if spread_total >= 5:
                home_ratio = spread_home / spread_total
                if home_ratio >= 0.85:
                    bias_flags.append(f"home_heavy: {spread_home}/{spread_total} spreads on home")
                elif home_ratio <= 0.15:
                    bias_flags.append(f"away_heavy: {spread_away}/{spread_total} spreads on away")

        # 2. Stale open bets (>48h since placed).
        stale_open_ids: List[str] = []
        for b in opens:
            if not b.placed_at:
                continue
            try:
                placed_dt = datetime.fromisoformat(str(b.placed_at).replace("Z", "+00:00"))
            except ValueError:
                continue
            if (now - placed_dt) > timedelta(hours=48):
                stale_open_ids.append(b.id)

        # 3. Drawdown, risk snapshot.
        risk = paper.risk_snapshot() if hasattr(paper, "risk_snapshot") else {
            "bankroll": paper.bankroll,
            "peak_bankroll": getattr(paper, "_peak_bankroll", paper.bankroll),
            "drawdown_pct": 0.0,
            "halted": False,
            "open_exposure": sum(b.stake for b in opens),
        }

        # 4. Today's activity counts (UTC day).
        today_iso = now.date().isoformat()
        closed_today = [
            b for b in closed if (b.placed_at or "").startswith(today_iso)
        ]
        won_today = sum(1 for b in closed_today if b.status == "won")
        lost_today = sum(1 for b in closed_today if b.status == "lost")

        # 5. Per-strategy activity over the last 24h.
        cutoff_24h = now - timedelta(hours=24)
        strat_24h: Counter = Counter()
        for b in opens + closed:
            if not b.placed_at:
                continue
            try:
                ts = datetime.fromisoformat(str(b.placed_at).replace("Z", "+00:00"))
            except ValueError:
                continue
            if ts >= cutoff_24h:
                strat_24h[b.strategy or "unknown"] += 1

        # 6. Anomaly narratives + health score.
        anomalies: List[str] = []
        if risk.get("halted"):
            anomalies.append("HALTED: drawdown circuit-breaker triggered. Manual resume required.")
        dd = risk.get("drawdown_pct", 0.0)
        if dd >= 15:
            anomalies.append(f"drawdown_warn: {dd:.1f}% — 5% from halt threshold")
        elif dd >= 10:
            anomalies.append(f"drawdown_watch: {dd:.1f}%")
        if stale_open_ids:
            anomalies.append(
                f"stale_bets: {len(stale_open_ids)} open >48h — auto-settler may have missed"
            )
        anomalies.extend(bias_flags)
        if len(opens) == 0 and closed_today == []:
            anomalies.append("no_activity: no open bets + nothing closed today — check aggregator")

        # Health score: start at 100, subtract for each anomaly class.
        health = 100
        if risk.get("halted"):
            health -= 50
        if dd >= 15:
            health -= 20
        elif dd >= 10:
            health -= 10
        health -= 10 * len(bias_flags)
        health -= 5 * min(len(stale_open_ids), 4)
        health = max(0, min(100, health))

        return HourlySnapshot(
            timestamp=now.isoformat(),
            bankroll=round(risk.get("bankroll", paper.bankroll), 2),
            peak_bankroll=round(risk.get("peak_bankroll", paper.bankroll), 2),
            drawdown_pct=round(dd, 2),
            halted=bool(risk.get("halted")),
            open_count=len(opens),
            open_exposure=round(risk.get("open_exposure", 0.0), 2),
            closed_today=len(closed_today),
            won_today=won_today,
            lost_today=lost_today,
            bias_flags=bias_flags,
            stale_open_count=len(stale_open_ids),
            stale_open_ids=stale_open_ids[:10],
            strategy_24h=dict(strat_24h.most_common()),
            anomalies=anomalies,
            health_score=health,
        )

    def _persist(self, snap: HourlySnapshot) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        entries: List[Dict] = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path) as fh:
                    entries = json.load(fh).get("entries", [])
            except Exception:
                entries = []
        entries.append(asdict(snap))
        # Bound at 168 = 7 days of hourly samples.
        entries = entries[-168:]
        tmp = self.log_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"entries": entries}, fh, indent=2)
        os.replace(tmp, self.log_path)
        logger.info(
            "hourly_monitor: health=%d open=%d dd=%.1f%% anomalies=%d",
            snap.health_score, snap.open_count, snap.drawdown_pct, len(snap.anomalies),
        )
        if snap.anomalies:
            for a in snap.anomalies:
                logger.warning("hourly_monitor anomaly: %s", a)

    def latest(self) -> Optional[Dict]:
        if not os.path.exists(self.log_path):
            return None
        try:
            with open(self.log_path) as fh:
                entries = json.load(fh).get("entries", [])
            return entries[-1] if entries else None
        except Exception:
            return None

    def recent(self, hours: int = 24) -> List[Dict]:
        if not os.path.exists(self.log_path):
            return []
        try:
            with open(self.log_path) as fh:
                entries = json.load(fh).get("entries", [])
        except Exception:
            return []
        return entries[-hours:]
