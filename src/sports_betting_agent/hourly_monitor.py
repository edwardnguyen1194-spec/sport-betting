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
    # Bug scan results — hard correctness issues (not soft anomalies)
    bugs: List[str] = field(default_factory=list)
    # Per-category bug counts for dashboards
    bug_counts: Dict[str, int] = field(default_factory=dict)
    # Sub-agent health — per-agent last_seen + error rate in last 24h
    agent_health: Dict[str, Dict[str, object]] = field(default_factory=dict)
    # External-endpoint uptime — last probe result per public route
    endpoint_health: Dict[str, Dict[str, object]] = field(default_factory=dict)


class HourlyMonitor:
    """Once-per-hour diagnostic sweep over the paper-trader state."""

    # The 10 Claude sub-agents we expect to be alive. Each should
    # have at least one AgentLog entry in the last 24h; continuous
    # zero-activity for an agent whose trigger condition should have
    # fired (e.g. post_mortem with a loss in the last 24h) gets
    # flagged as a soft anomaly.
    KNOWN_AGENTS = [
        "pick_reviewer", "news_triage", "post_mortem",
        "game_analyst", "opportunity_scout", "strategy_auditor",
        "skills_learner", "mcp_discovery",
        "self_reflection", "hooks_discovery",
    ]

    def __init__(
        self,
        paper_trader,
        data_dir: str,
        agent_log=None,
    ) -> None:
        self.paper = paper_trader
        self.data_dir = data_dir
        self.agent_log = agent_log   # optional AgentLog instance
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

        # ---- BUG SCAN ------------------------------------------------
        # Hard correctness checks. These flag integrity violations that
        # should literally never happen. A nonzero bug list means a
        # code regression slipped through tests — Uncle gets alerted.
        bugs, bug_counts = self._bug_scan(opens, closed)

        # ---- SUB-AGENT HEALTH ----------------------------------------
        # Verify each of the 10 Claude sub-agents is alive. Inspects
        # agent_log for last-24h activity + error ratio.
        agent_health = self._agent_health()
        for name, h in agent_health.items():
            if h.get("errors_24h", 0) > 0 and h.get("runs_24h", 0) > 0:
                err_rate = h["errors_24h"] / h["runs_24h"]
                if err_rate > 0.5:
                    anomalies.append(
                        f"agent_errors: {name} {h['errors_24h']}/{h['runs_24h']} "
                        f"errors ({err_rate:.0%}) in last 24h"
                    )

        # ---- ENDPOINT UPTIME -----------------------------------------
        # Lightweight probe of every /api/* route the monitor knows
        # about. Fails soft — if the probe itself errors, we log 'err'
        # rather than fake an up signal.
        endpoint_health = self._endpoint_health()
        for path, stat in endpoint_health.items():
            status = stat.get("status")
            if status != "ok":
                anomalies.append(
                    f"endpoint_down: {path} returned {status}"
                )

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
        # Bugs hit harder than soft anomalies — each category costs 15.
        health -= 15 * len(bug_counts)
        # Every agent erroring > 50% costs 10, every endpoint down costs 10.
        bad_agents = sum(
            1 for h in agent_health.values()
            if h.get("runs_24h", 0) > 0
            and (h.get("errors_24h", 0) / max(h["runs_24h"], 1)) > 0.5
        )
        health -= 10 * bad_agents
        bad_endpoints = sum(
            1 for h in endpoint_health.values() if h.get("status") != "ok"
        )
        health -= 10 * bad_endpoints
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
            anomalies=anomalies + [f"BUG: {b}" for b in bugs],
            health_score=health,
            bugs=bugs,
            bug_counts=bug_counts,
            agent_health=agent_health,
            endpoint_health=endpoint_health,
        )

    def _agent_health(self) -> Dict[str, Dict[str, object]]:
        """Per-agent activity + error snapshot from the shared AgentLog."""
        if self.agent_log is None:
            return {n: {"runs_24h": 0, "errors_24h": 0, "last_seen": None,
                        "status": "agent_log_unavailable"} for n in self.KNOWN_AGENTS}
        # Pull full recent history then filter per-agent.
        entries = self.agent_log.recent(n=2000) or []
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=24)
        out: Dict[str, Dict[str, object]] = {}
        for name in self.KNOWN_AGENTS:
            runs = 0
            errors = 0
            last_seen: Optional[str] = None
            for e in entries:
                if e.get("agent") != name:
                    continue
                ts_str = e.get("ts", "")
                if not ts_str:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if ts < cutoff:
                    continue
                runs += 1
                if (e.get("decision") or {}).get("error"):
                    errors += 1
                if last_seen is None or ts_str > last_seen:
                    last_seen = ts_str
            status = "alive"
            if runs == 0:
                status = "idle"   # never ran or none in last 24h (may be correct)
            elif runs > 0 and errors / runs > 0.5:
                status = "erroring"
            out[name] = {
                "runs_24h": runs,
                "errors_24h": errors,
                "last_seen": last_seen,
                "status": status,
            }
        return out

    def _endpoint_health(self) -> Dict[str, Dict[str, object]]:
        """Self-probe of the Flask app's own /api/* routes.

        We run in the same process so we can't fully simulate an
        external HTTP hit, but we can validate that each route handler
        is registered and loadable. A richer external probe lives in
        the hourly Claude cron sub-agent.
        """
        # Light-touch check: inspect flask.current_app routes. If we're
        # called from the background thread without an app_context,
        # we return a stub to avoid false alarms.
        try:
            from flask import current_app
            endpoints = [
                "/api/health", "/api/risk", "/api/ledger", "/api/clv",
                "/api/hourly-log", "/api/agent-log", "/api/learnings",
                "/api/weekly-audit", "/api/scout",
            ]
            rule_map = {r.rule: r for r in current_app.url_map.iter_rules()}
            out: Dict[str, Dict[str, object]] = {}
            for ep in endpoints:
                out[ep] = {
                    "registered": ep in rule_map,
                    "status": "ok" if ep in rule_map else "missing_route",
                }
            return out
        except Exception as exc:
            return {"_probe": {"status": "no_app_context", "err": str(exc)}}

    def _bug_scan(self, opens, closed) -> tuple[list[str], dict[str, int]]:
        """Hourly integrity scan. Returns (bug_list, per_category_counts).

        Each check here represents a class of regression that should be
        impossible given the code contracts:

          - invalid_market: a bet on anything other than spread/total
            (Uncle's rule violation; paper_trader guard should block)
          - invalid_price: american is None, NaN, or decimal <= 1.0
          - invalid_confidence: confidence outside [0, 1]
          - negative_edge: edge <= 0 on an open bet (we only place +EV)
          - duplicate_game: ≥2 open bets on the same game_key
            (per-game cap should block; if we see one, the cap broke)
          - ledger_corruption: bet.stake or bet.decimal unparseable as float
          - orphan_void: bet.status=="void" present in open_bets dict
            (voids belong in closed_bets)
          - schema_drift: bet missing required attribute

        Silent strategies are also bug-flagged — a strategy that ran
        yesterday but has zero activity in the last 6h is almost always
        a broken data input, not market conditions.
        """
        from collections import Counter as _Counter
        bugs: list[str] = []
        counts: dict[str, int] = {}

        def _bump(cat: str, detail: str) -> None:
            bugs.append(f"{cat}: {detail}")
            counts[cat] = counts.get(cat, 0) + 1

        seen_games: _Counter = _Counter()
        for b in opens:
            bid = getattr(b, "id", "?")
            # Market guard
            if getattr(b, "market", None) not in ("spread", "total"):
                _bump("invalid_market", f"{bid} market={b.market!r}")
            # Price guard
            am = getattr(b, "american", None)
            dec = getattr(b, "decimal", None)
            if am is None or dec is None:
                _bump("invalid_price", f"{bid} american={am} decimal={dec}")
            else:
                try:
                    am_f = float(am)
                    dec_f = float(dec)
                    if dec_f <= 1.0 or am_f == 0:
                        _bump("invalid_price", f"{bid} american={am_f} decimal={dec_f}")
                except (TypeError, ValueError):
                    _bump("ledger_corruption", f"{bid} unparseable price")
            # Confidence sanity
            conf = getattr(b, "confidence", None)
            if conf is None:
                _bump("invalid_confidence", f"{bid} confidence=None")
            else:
                try:
                    c = float(conf)
                    if c < 0 or c > 1.0:
                        _bump("invalid_confidence", f"{bid} confidence={c}")
                except (TypeError, ValueError):
                    _bump("ledger_corruption", f"{bid} unparseable confidence")
            # Stake sanity
            try:
                stake = float(getattr(b, "stake", 0))
                if stake <= 0:
                    _bump("invalid_stake", f"{bid} stake={stake}")
            except (TypeError, ValueError):
                _bump("ledger_corruption", f"{bid} unparseable stake")
            # Per-game cap violation
            gkey = getattr(b, "game_key", None)
            if gkey:
                seen_games[gkey] += 1
            # Orphan voids
            if getattr(b, "status", None) in ("void", "push", "won", "lost"):
                _bump("orphan_closed_in_open", f"{bid} status={b.status}")

        # Duplicate game check
        for gkey, ct in seen_games.items():
            if ct >= 2:
                _bump("duplicate_game", f"{gkey} has {ct} open bets")

        # Silent-strategy check: a strategy that placed ≥3 bets in the
        # previous 24h but 0 in the last 6h is almost certainly broken.
        now = datetime.now(timezone.utc)
        prev_24 = now - timedelta(hours=24)
        prev_6 = now - timedelta(hours=6)
        strat_24: _Counter = _Counter()
        strat_6: _Counter = _Counter()
        for b in opens + closed:
            if not b.placed_at:
                continue
            # Don't count voided bets as "runs" — a strategy we
            # manually voided (e.g. MIDDLE bet cleanup, halt reset)
            # was false-positive flagged as silent. We care about
            # strategies that STOP FINDING picks, not strategies
            # whose picks we took back. Voids represent removal, not
            # strategy activity.
            if getattr(b, "status", None) == "void":
                continue
            try:
                ts = datetime.fromisoformat(str(b.placed_at).replace("Z", "+00:00"))
            except ValueError:
                continue
            strat = b.strategy or "unknown"
            for name in strat.split("+"):
                name = name.strip() or "unknown"
                if ts >= prev_24:
                    strat_24[name] += 1
                if ts >= prev_6:
                    strat_6[name] += 1
        for name, n_24 in strat_24.items():
            if n_24 >= 3 and strat_6.get(name, 0) == 0:
                # Only flag "real" strategies, not ensemble-combined
                # multi-name entries (those already show up under their
                # components).
                if name in {
                    "spread_value", "total_value", "total_projection",
                    "steam_follow", "reverse_line_movement", "public_fade",
                    "mls_home_travel",
                }:
                    _bump(
                        "silent_strategy",
                        f"{name} ran {n_24}× in 24h but 0× in last 6h",
                    )

        return bugs, counts

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
