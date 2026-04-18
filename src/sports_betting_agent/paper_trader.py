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
from .agents.pick_reviewer import PickReviewer, review_rec
from .agents.post_mortem import PostMortem, analyze_loss


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
    # ISO timestamp of when the game is scheduled to start. Pulled from
    # rec.meta["commence_time"] (stamped by the ensemble). Lets Uncle
    # see on each card "when is this game actually played" — the
    # placed_at + settled_at tell the bet lifecycle, this tells the
    # match time.
    game_time: Optional[str] = None

    @classmethod
    def from_recommendation(cls, rec: BetRecommendation, stake: float, bet_id: str) -> "Bet":
        meta = rec.meta or {}
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
            game_time=meta.get("commence_time") or meta.get("game_commence_time"),
        )


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class PaperTrader:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        state_path: Optional[str] = None,
        pick_reviewer: Optional[PickReviewer] = None,
        post_mortem: Optional[PostMortem] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.state_path = state_path or os.path.join(self.settings.data_dir, "ledger.json")
        self._lock = threading.RLock()
        self._stop = threading.Event()
        # Optional Claude reviewer. Default None so tests never need
        # to mock Anthropic; production wiring injects it lazily.
        self.pick_reviewer = pick_reviewer
        # Optional PostMortem agent — runs retrospectively on lost
        # bets to write a systemic-failure note. Does not gate
        # anything; purely diagnostic.
        self.post_mortem: Optional[PostMortem] = post_mortem
        # Optional hook the app can register so post-mortems get
        # injury/weather headlines from that day. Signature:
        # ``fn(bet: Bet) -> list[dict]``. Default: no headlines.
        self._post_mortem_headlines_provider: Optional[
            Callable[[Bet], List[Dict]]
        ] = None
        self.post_mortems_path = os.path.join(
            self.settings.data_dir, "post_mortems.json"
        )

        self.bankroll: float = self.settings.bankroll_start
        self.open_bets: Dict[str, Bet] = {}
        self.closed_bets: List[Bet] = []
        self._next_id: int = 1
        # Risk-management persistent state (Agent-3 safeguards).
        self._peak_bankroll: float = self.settings.bankroll_start
        self._halted: bool = False

        # Closing Line Value tracker — the #1 predictor of long-run profit
        # for any spread/total bettor. Records every placement and lets us
        # compute whether we are genuinely beating the closing market.
        self.clv = CLVTracker(self.settings.data_dir)

        self._load_state()

    # ------------------------------------------------------------------
    # PostMortem wiring
    # ------------------------------------------------------------------

    def attach_post_mortem(
        self,
        agent: PostMortem,
        headlines_provider: Optional[Callable[[Bet], List[Dict]]] = None,
    ) -> None:
        """Register a :class:`PostMortem` agent for loss analysis.

        ``headlines_provider`` is an optional callable that gets the
        losing ``Bet`` and returns a list of headline dicts from that
        day. If omitted, the post-mortem runs without news context.
        """
        self.post_mortem = agent
        self._post_mortem_headlines_provider = headlines_provider

    def _recent_loss_notes(
        self, strategy: str, limit: int = 10
    ) -> List[Dict]:
        """Read the last ``limit`` post-mortems for ``strategy`` off disk.

        The on-disk file is the full history (bounded to MAX_ENTRIES
        in the writer); we scan it newest-first and keep entries
        matching the strategy. Returns at most ``limit`` dicts of
        shape ``{"systemic_flag": str, "reasoning": str}``.
        """
        try:
            if not os.path.exists(self.post_mortems_path):
                return []
            with open(self.post_mortems_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            entries = data.get("entries", []) if isinstance(data, dict) else []
        except Exception as exc:
            logger.warning("post_mortem read failed: %s", exc)
            return []
        out: List[Dict] = []
        for e in reversed(entries):
            if e.get("strategy") != strategy:
                continue
            out.append({
                "systemic_flag": (e.get("metadata") or {}).get("systemic_flag", ""),
                "reasoning": e.get("reasoning", ""),
            })
            if len(out) >= limit:
                break
        return out

    def _append_post_mortem(self, entry: Dict) -> None:
        """Append one post-mortem entry to ``/data/sba/post_mortems.json``."""
        os.makedirs(os.path.dirname(self.post_mortems_path) or ".", exist_ok=True)
        existing: List[Dict] = []
        try:
            if os.path.exists(self.post_mortems_path):
                with open(self.post_mortems_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, dict):
                        existing = list(data.get("entries", []))
        except Exception:
            existing = []
        existing.append(entry)
        # Bound to 5k entries — matches AgentLog.MAX_ENTRIES.
        existing = existing[-5000:]
        tmp = self.post_mortems_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"entries": existing}, fh, indent=2)
        os.replace(tmp, self.post_mortems_path)

    def _run_post_mortem(self, bet: Bet) -> None:
        """Background worker: run the agent, persist the note.

        Fails SAFE — any exception is logged and swallowed so a bad
        post-mortem can never corrupt settlement. Writes the note
        onto the bet's ``meta`` if available (via closed_bets lookup),
        and appends to ``post_mortems.json``.
        """
        if self.post_mortem is None:
            return
        try:
            headlines: List[Dict] = []
            if self._post_mortem_headlines_provider is not None:
                try:
                    headlines = list(
                        self._post_mortem_headlines_provider(bet) or []
                    )
                except Exception as exc:
                    logger.warning(
                        "post_mortem headlines_provider failed: %s", exc,
                    )
            prior = self._recent_loss_notes(bet.strategy, limit=10)
            # ESPN final score isn't wired through the trader yet —
            # the scraper that calls settle_bet("lost") is the only
            # place that actually knows the score. If the caller has
            # tucked it onto ``bet.meta`` we use it; otherwise 0-0
            # so Claude still writes a note (just without a score).
            meta = {}
            try:
                from dataclasses import asdict as _asdict
                meta = (_asdict(bet) or {}).get("meta", {}) or {}
            except Exception:
                meta = {}
            final_score = meta.get("final_score") or {"home": None, "away": None}
            bet_ctx = {
                "selection": bet.selection,
                "market": bet.market,
                "line": bet.line,
                "american": bet.american,
                "strategy": bet.strategy,
                "reasoning": bet.reasoning,
            }
            decision = analyze_loss(
                self.post_mortem,
                bet=bet_ctx,
                final_score=final_score,
                headlines_that_day=headlines,
                prior_loss_notes=prior,
            )
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "bet_id": bet.id,
                "game_key": bet.game_key,
                "strategy": bet.strategy,
                "selection": bet.selection,
                "market": bet.market,
                "line": bet.line,
                "american": bet.american,
                "reasoning": decision.reasoning,
                "metadata": decision.metadata,
                "error": decision.error,
            }
            self._append_post_mortem(entry)
            # Tuck the flag onto the bet's reasoning for downstream
            # visibility. We intentionally avoid mutating the bet
            # object's schema (Bet is a dataclass with fixed fields);
            # instead we append a compact suffix to ``reasoning``.
            flag = (decision.metadata or {}).get("systemic_flag", "")
            if flag and flag != "variance" and not decision.error:
                with self._lock:
                    for b in self.closed_bets:
                        if b.id == bet.id:
                            suffix = f" [post_mortem: {flag}]"
                            if suffix not in (b.reasoning or ""):
                                b.reasoning = (b.reasoning or "") + suffix
                            break
                    self._save_state()
        except Exception as exc:  # pragma: no cover — fail-safe net
            logger.warning(
                "post_mortem run failed for %s: %s", bet.id, exc,
            )

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
        # Persistent risk state. Backfilled for pre-upgrade ledgers.
        self._peak_bankroll = float(data.get("peak_bankroll", self.bankroll))
        self._halted = bool(data.get("halted", False))

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        payload = {
            "bankroll": self.bankroll,
            "next_id": self._next_id,
            "open_bets": [asdict(b) for b in self.open_bets.values()],
            "closed_bets": [asdict(b) for b in self.closed_bets[-5_000:]],
            "peak_bankroll": getattr(self, "_peak_bankroll", self.bankroll),
            "halted": getattr(self, "_halted", False),
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
            # ------------------------------------------------------------
            # Hard market guard: spread + over/under ONLY. Uncle's rule.
            # Any rec that reached us with a moneyline / prop / middle
            # market is silently dropped. Belt-and-suspenders for the
            # ensemble.py filter; if a future strategy emits an
            # unexpected market type, this prevents a phantom bet.
            # ------------------------------------------------------------
            if rec.market not in ("spread", "total"):
                logger.debug(
                    "paper_trader skip: market %r not in {spread,total}", rec.market,
                )
                return None

            # ------------------------------------------------------------
            # Line integrity: never place a bet on a price that might
            # be stale. Uncle's rule: "always has correct betting lines
            # and odds". If the rec carries a ``line_observed_at``
            # meta, require it to be within 10 min. No timestamp means
            # the rec just came out of the aggregator fetch that
            # triggered this cycle, so it's implicitly fresh.
            # ------------------------------------------------------------
            line_ts = (rec.meta or {}).get("line_observed_at")
            if line_ts:
                try:
                    from datetime import datetime as _dt, timezone as _tz
                    observed = _dt.fromisoformat(str(line_ts).replace("Z", "+00:00"))
                    age_s = (_dt.now(_tz.utc) - observed).total_seconds()
                    if age_s > 600:   # > 10 min = stale
                        logger.warning(
                            "paper_trader skip stale line: %s %s at %s age=%.0fs",
                            rec.selection, rec.market, rec.book, age_s,
                        )
                        return None
                except (ValueError, TypeError):
                    pass
            # Sanity: american must be an actual number, not None.
            if rec.american is None or rec.decimal is None or rec.decimal <= 1.0:
                logger.warning(
                    "paper_trader skip: invalid price american=%r decimal=%r",
                    rec.american, rec.decimal,
                )
                return None

            # ------------------------------------------------------------
            # OFF-MARKET PRICE GUARD (Uncle's bug report 2026-04-17).
            # If one book's price is WAY better than the market
            # consensus, it's either:
            #   (a) Truly the best price (rare — sharps would've hit it)
            #   (b) Stale/phantom price from a broken scraper
            #   (c) A book-specific promo we can't actually access
            # In every one of those scenarios, placing the bet at the
            # phantom price inflates our apparent edge by 3-10% and
            # produces fake paper P&L.
            #
            # Rule: if this rec's decimal price is >8% better than the
            # median decimal price across 3+ other books at the same
            # line, skip the bet. The ensemble should pick from the
            # median-fair book instead.
            #
            # Env override: SBA_OFF_MARKET_TOLERANCE=0.08
            try:
                rec_meta = rec.meta or {}
                all_lines = rec_meta.get("all_book_prices") or []
                if len(all_lines) >= 4 and rec.decimal:
                    import statistics as _st
                    other_decimals = [
                        float(l.get("decimal"))
                        for l in all_lines
                        if l.get("book") != rec.book
                        and l.get("decimal")
                        and abs(float(l.get("line", 0)) - float(rec.line or 0)) < 0.01
                    ]
                    if len(other_decimals) >= 3:
                        median_decimal = _st.median(other_decimals)
                        tolerance = float(_os.environ.get(
                            "SBA_OFF_MARKET_TOLERANCE", "0.08"
                        ))
                        uplift = (rec.decimal - median_decimal) / median_decimal
                        if uplift > tolerance:
                            logger.warning(
                                "off-market skip: %s %s at %s decimal=%.3f is "
                                "%.1f%% better than median %.3f across %d other "
                                "books — likely stale/phantom price",
                                rec.selection, rec.market, rec.book,
                                rec.decimal, uplift * 100,
                                median_decimal, len(other_decimals),
                            )
                            return None
            except Exception as exc:  # pragma: no cover
                logger.debug("off-market guard skipped: %s", exc)

            # ------------------------------------------------------------
            # Risk management gates (world-class safeguards per Agent-3
            # audit). Order matters — cheapest checks first.
            # ------------------------------------------------------------

            # 1. Drawdown circuit-breaker — loosened from 20% to 40%
            #    per Uncle's explicit "remove tam dung, keep going"
            #    instruction. Paper-trading natural variance easily
            #    produces 20% drawdowns during exploration; the halt
            #    was firing too aggressively and interrupting learning.
            #    40% = only fires on CATASTROPHIC runs (bankroll down
            #    to 60% of peak), the true blow-up region. Still
            #    configurable via env var SBA_MAX_DRAWDOWN_PCT.
            import os as _os
            halt_threshold = float(
                _os.environ.get("SBA_MAX_DRAWDOWN_PCT", "0.40")
            )
            peak = max(
                self.settings.bankroll_start,
                getattr(self, "_peak_bankroll", self.settings.bankroll_start),
            )
            self._peak_bankroll = max(peak, self.bankroll)
            drawdown = (self._peak_bankroll - self.bankroll) / self._peak_bankroll
            if drawdown >= halt_threshold and not getattr(self, "_halted", False):
                logger.critical(
                    "DRAWDOWN HALT: bankroll $%.0f is %.1f%% below peak $%.0f — "
                    "pausing all new bets. POST /api/reset-halt to resume.",
                    self.bankroll, drawdown * 100, self._peak_bankroll,
                )
                self._halted = True
                self._save_state()
                return None
            if getattr(self, "_halted", False):
                return None

            # 2. Tilt dampener: after 7-of-last-10 losses, half-size.
            #    At 8-of-10, skip. Recovers tilt-driven loss spirals.
            recent = [
                b for b in self.closed_bets[-10:] if b.status in ("won", "lost")
            ]
            losses_in_window = sum(1 for b in recent if b.status == "lost")
            tilt_halve = losses_in_window >= 7 and len(recent) >= 8
            if losses_in_window >= 8 and len(recent) >= 10:
                logger.warning(
                    "tilt skip: %d losses in last %d closed bets", losses_in_window, len(recent),
                )
                return None

            # 2b. CONCENTRATION GUARD (Uncle Phung's request 2026-04-17):
            #    public_fade + reverse_line_movement strategies are BY
            #    DESIGN contrarian to the public, and the public nearly
            #    always loads the Over. Result: 4-5 open Unders in a
            #    row, all highly correlated. A bad-weather night or a
            #    single sharp-miss pattern wipes the whole slate.
            #
            #    Rule:
            #      - Count open bets with the same total selection
            #        (Over vs Under). Spreads don't suffer this
            #        correlation so they're exempt.
            #      - 1st-3rd same-side Under/Over: allow normally.
            #      - 4th same-side:  require confidence >= 0.65.
            #      - 5th+ same-side: require edge >= 0.10 AND conf >= 0.65.
            #
            #    Env overrides so we can tune without redeploy:
            #      SBA_CONC_SOFT_LIMIT  (default 3 — when checks start)
            #      SBA_CONC_HARD_CONF   (default 0.65 — needed >= soft+1)
            #      SBA_CONC_HARD_EDGE   (default 0.10 — needed >= soft+2)
            if rec.market == "total" and rec.selection:
                sel_lower = rec.selection.lower().strip()
                if sel_lower in ("over", "under"):
                    soft_limit = int(_os.environ.get("SBA_CONC_SOFT_LIMIT", "3"))
                    hard_conf = float(_os.environ.get("SBA_CONC_HARD_CONF", "0.65"))
                    hard_edge = float(_os.environ.get("SBA_CONC_HARD_EDGE", "0.10"))
                    same_side = sum(
                        1 for b in self.open_bets.values()
                        if b.market == "total"
                        and (b.selection or "").lower().strip() == sel_lower
                    )
                    if same_side >= soft_limit:
                        # Hitting the concentration ceiling — require
                        # higher conviction to add more of the same side.
                        conf = rec.confidence or 0.0
                        edge = rec.edge or 0.0
                        if conf < hard_conf:
                            logger.warning(
                                "concentration skip: %s would be #%d open %s "
                                "but confidence %.3f < %.2f",
                                rec.selection, same_side + 1, sel_lower,
                                conf, hard_conf,
                            )
                            return None
                        if same_side >= soft_limit + 1 and edge < hard_edge:
                            logger.warning(
                                "concentration skip: %s would be #%d open %s "
                                "but edge %.3f < %.2f",
                                rec.selection, same_side + 1, sel_lower,
                                edge, hard_edge,
                            )
                            return None
                        logger.info(
                            "concentration warn: #%d %s passes tighter gates "
                            "(conf=%.3f edge=%.3f)",
                            same_side + 1, sel_lower, conf, edge,
                        )

            # 2bb. MARKET-MIX GUARD (Uncle's request 2026-04-17):
            #   "make sure agent and sub agents will bet on BOTH point
            #    spreads AND over/under please". Without a guard the
            #    ensemble emits mostly totals (4 strategies produce
            #    totals, only 2 produce spreads) so the open book
            #    naturally skews. Enforce a rough 60/40 ceiling —
            #    any single market type can be at most 70% of open
            #    bets beyond the first 3. After that, the minority
            #    market needs a slot or the bet is skipped.
            total_open = len(self.open_bets)
            if total_open >= 3:
                spread_count = sum(
                    1 for b in self.open_bets.values() if b.market == "spread"
                )
                total_count = sum(
                    1 for b in self.open_bets.values() if b.market == "total"
                )
                max_ratio = float(_os.environ.get("SBA_MAX_MARKET_RATIO", "0.70"))
                # If adding this bet would push its market past the
                # max ratio threshold, block it and wait for the
                # other market to catch up.
                hypothetical = total_open + 1
                if rec.market == "total":
                    new_total_pct = (total_count + 1) / hypothetical
                    if new_total_pct > max_ratio:
                        logger.warning(
                            "market-mix skip: %s would push totals to %.0f%% "
                            "of open (cap %.0f%%). Need a spread slot first.",
                            rec.selection, new_total_pct * 100, max_ratio * 100,
                        )
                        return None
                elif rec.market == "spread":
                    new_spread_pct = (spread_count + 1) / hypothetical
                    if new_spread_pct > max_ratio:
                        logger.warning(
                            "market-mix skip: %s would push spreads to %.0f%% "
                            "of open (cap %.0f%%). Need a total slot first.",
                            rec.selection, new_spread_pct * 100, max_ratio * 100,
                        )
                        return None

            # 2c. KEY-NUMBER GUARD (master-bettor rule).
            #    Historically, NFL games land on 3 ~15% of the time,
            #    NHL/MLB games land on 1-goal margin ~28% of the time,
            #    etc. Buying ONTO these key numbers (e.g. taking -3
            #    instead of -2.5) is systematically -EV. We downgrade
            #    stake when the rec is on the wrong side of a key
            #    number. Spreads only — totals have weaker cluster
            #    patterns per sport.
            try:
                from .agents.betting_expertise import KEY_NUMBERS_SPREAD
                if rec.market == "spread" and rec.line is not None:
                    sport_key = (rec.sport or "").lower()
                    key_list = KEY_NUMBERS_SPREAD.get(sport_key, [])
                    abs_line = abs(float(rec.line))
                    for k in key_list[:3]:  # top-3 key numbers per sport
                        key_val = float(k["number"])
                        # If our line is EXACTLY on a top-3 key number,
                        # note it in meta so PickReviewer sees it and
                        # can veto / shrink. We don't outright skip
                        # here — a strategy might legitimately have
                        # found that the current market is mispriced
                        # AT the key number itself.
                        if abs(abs_line - key_val) < 0.01:
                            (rec.meta or {}).setdefault(
                                "on_key_number", True
                            )
                            logger.info(
                                "key_number stamp: %s spread line=%.1f sits on key %.1f (%.1f%% hit rate)",
                                rec.selection, rec.line, key_val, k.get("hit_rate_pct", 0),
                            )
                            break
            except Exception as exc:  # pragma: no cover
                logger.debug("key_number guard skipped: %s", exc)

            stake = self._size_stake(rec)
            if tilt_halve:
                stake *= 0.5
            if stake < self.settings.min_bet:
                logger.debug("skipping rec %s: stake %.2f below min", rec.selection, stake)
                return None
            if stake > self.bankroll:
                logger.debug("skipping rec %s: insufficient bankroll", rec.selection)
                return None
            # 3. Dynamic exposure cap. Previously frozen at starting
            #    bankroll — a degraded $3k bankroll could have $7.5k open
            #    (>100% exposure). Now scales with current bankroll but
            #    with a floor so we don't collapse into a death spiral.
            live_cap_base = max(
                self.bankroll, self.settings.bankroll_start * 0.5
            )
            total_exposed = sum(b.stake for b in self.open_bets.values())
            if total_exposed + stake > live_cap_base * 0.75:
                logger.debug(
                    "skipping rec %s: exposure $%.0f would exceed 75%% of live cap $%.0f",
                    rec.selection, total_exposed + stake, live_cap_base,
                )
                return None
            # 4. Daily bet count cap. Prevents runaway "1,440 bets/day"
            #    scenario on a loose threshold day.
            from datetime import datetime as _dt, timezone as _tz
            today_iso = _dt.now(_tz.utc).date().isoformat()
            todays = sum(
                1
                for b in (list(self.open_bets.values()) + self.closed_bets)
                if b.placed_at and b.placed_at.startswith(today_iso)
            )
            DAILY_CAP = 25
            if todays >= DAILY_CAP:
                logger.info(
                    "daily cap reached (%d bets today) — skipping %s",
                    todays, rec.selection,
                )
                return None
            # 5. Per-game-per-market cap. Uncle wants BOTH spread + total
            #    firing per game (previous "1 total bet per game" cap
            #    was crowding out spread_value — ensemble sort put the
            #    higher-confidence total first, and the spread never
            #    got a look). Now we allow exactly 1 SPREAD and 1 TOTAL
            #    per game but no duplicates within a market, which
            #    matches Uncle's "spreads and over/under only" rule.
            #    Two bets are correlated through the outcome but not
            #    perfectly — spread covers vs total goes over are
            #    ~0.4-0.5 rho in practice, low enough that quarter-
            #    Kelly sizing on each absorbs the joint variance.
            per_game_market = sum(
                1 for b in self.open_bets.values()
                if b.game_key == rec.game_key and b.market == rec.market
            )
            if per_game_market >= 1:
                logger.debug(
                    "per-game-market cap: already have %d %s bet on %s",
                    per_game_market, rec.market, rec.game_key,
                )
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

            # ------------------------------------------------------------
            # Final gate: Claude PickReviewer sub-agent. Runs only when
            # one is injected (production wiring); tests leave it None
            # so no Anthropic call ever fires. The agent may veto, trim
            # the stake, or nudge confidence. BaseAgent fails SAFE —
            # any HTTP / parse / budget error returns a neutral approve
            # so the trading loop never blocks on the LLM.
            # ------------------------------------------------------------
            if self.pick_reviewer is not None:
                decision = review_rec(
                    self.pick_reviewer,
                    rec,
                    game_context={
                        "commence_time": (rec.meta or {}).get("commence_time"),
                        "recent_headlines": (rec.meta or {}).get("recent_headlines", []),
                        "weather": (rec.meta or {}).get("weather"),
                        "elo_diff": (rec.meta or {}).get("elo_diff"),
                    },
                )
                if not decision.approved or decision.stake_multiplier <= 0.0:
                    logger.info(
                        "pick_reviewer vetoed %s %s: %s",
                        rec.selection, rec.market, decision.reasoning or decision.error,
                    )
                    return None
                # Apply stake trim. Multiplier is already clamped to [0, 1].
                stake = round(stake * decision.stake_multiplier, 2)
                if stake < self.settings.min_bet:
                    logger.debug(
                        "pick_reviewer trimmed stake below min (%.2f) — skipping",
                        stake,
                    )
                    return None
                # Nudge confidence and fold reviewer's note into reasoning
                # so it surfaces on the dashboard. confidence_delta is
                # already clamped to [-0.10, 0.10] by BaseAgent; clamp
                # the final value to [0, 1] to keep downstream math sane.
                new_conf = max(0.0, min(1.0, rec.confidence + decision.confidence_delta))
                rec.confidence = new_conf
                if decision.reasoning:
                    suffix = f" | reviewer: {decision.reasoning}"
                    rec.reasoning = (rec.reasoning or "") + suffix

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
                    line=bet.line,
                    strategy=bet.strategy,
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

    def resume(self) -> Dict[str, float]:
        """Clear the drawdown halt and let bets flow again.

        Called via POST /api/reset-halt. Returns the current risk
        snapshot so Uncle can see how deep the drawdown was.
        """
        with self._lock:
            was_halted = getattr(self, "_halted", False)
            self._halted = False
            # Rebase the peak to current bankroll so the next halt is
            # measured from here forward, not from the pre-drawdown high.
            self._peak_bankroll = self.bankroll
            self._save_state()
            return {
                "was_halted": was_halted,
                "bankroll": self.bankroll,
                "peak_bankroll": self._peak_bankroll,
            }

    def risk_snapshot(self) -> Dict[str, float]:
        """Current risk posture for the dashboard + API."""
        peak = getattr(self, "_peak_bankroll", self.bankroll)
        dd = (peak - self.bankroll) / peak if peak > 0 else 0.0
        open_exposure = sum(b.stake for b in self.open_bets.values())
        return {
            "bankroll": self.bankroll,
            "peak_bankroll": peak,
            "drawdown_pct": round(dd * 100, 2),
            "halted": getattr(self, "_halted", False),
            "open_exposure": open_exposure,
            "exposure_pct_of_live": (
                round(open_exposure / max(self.bankroll, 1) * 100, 2)
            ),
        }

    def settle_bet(self, bet_id: str, outcome: str) -> Optional[Bet]:
        """Mark a bet won/lost/push/void and update bankroll.

        On a lost bet, kicks off the PostMortem agent asynchronously
        (if configured) so settlement is never blocked by an
        Anthropic call. The post-mortem note lands in
        ``/data/sba/post_mortems.json`` and (when non-trivial) gets
        appended to the bet's ``reasoning`` string.
        """

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

        # Post-mortem fires OUTSIDE the lock — it does its own disk
        # I/O + network call and we don't want to block concurrent
        # settlement. Spawn-and-forget; ``_run_post_mortem`` swallows
        # all exceptions internally.
        if outcome == "lost" and self.post_mortem is not None:
            t = threading.Thread(
                target=self._run_post_mortem,
                args=(bet,),
                name=f"post-mortem-{bet.id}",
                daemon=True,
            )
            t.start()
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
