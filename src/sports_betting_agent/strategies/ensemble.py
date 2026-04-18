"""Ensemble strategy runner.

Runs every strategy you hand it, merges duplicate recommendations on
the same game + selection, and blends their confidences. The output
is a single ranked list the paper trader or dashboard can consume.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional

from ..config import Settings, get_settings
from ..models_schema import GameOdds
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)


class EnsembleStrategy(Strategy):
    name = "ensemble"

    def __init__(
        self,
        strategies: List[Strategy],
        settings: Optional[Settings] = None,
        news_reader: Optional[Any] = None,
        game_analyst: Optional[Any] = None,
        scoring: Optional[Any] = None,
        elo: Optional[Any] = None,
    ) -> None:
        self.strategies = strategies
        self.settings = settings or get_settings()
        # Optional NewsReader — when present we downgrade confidence on
        # bets where cached headlines flag an injury for one of the
        # involved teams. Kept as a duck-typed attribute to avoid
        # circular imports and so tests can inject a fake.
        self.news_reader = news_reader
        # Optional GameAnalyst — if wired, we run it over the top-N
        # surviving recs after ensemble merge / news penalty to apply
        # a small context-based confidence polish.
        self.game_analyst = game_analyst
        self.scoring = scoring
        self.elo = elo

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        games = list(games)
        buckets: Dict[tuple, BetRecommendation] = {}

        # Hard rule per Uncle: ONLY spread and over/under (total).
        # Any strategy that sneaks in a moneyline / player-prop / middle
        # market gets filtered here before reaching the paper trader.
        ALLOWED_MARKETS = {"spread", "total"}

        for strat in self.strategies:
            try:
                for rec in strat.generate(games):
                    if rec.market not in ALLOWED_MARKETS:
                        continue
                    key = (rec.game_key, rec.market, rec.selection.lower())
                    if key not in buckets:
                        buckets[key] = rec
                        buckets[key].meta.setdefault("contributing_strategies", [])
                        buckets[key].meta["contributing_strategies"].append(strat.name)
                    else:
                        existing = buckets[key]
                        # Bayesian log-odds combination — independent
                        # strategies agreeing should raise confidence
                        # ABOVE either input, not stay between them.
                        # logit(post) = logit(prior) + logit(s1) + logit(s2) - 2*logit(prior)
                        # For a neutral prior of 0.5, the prior terms
                        # cancel and this reduces to log-odds addition.
                        import math as _m
                        def _logit(p):
                            p = min(max(p, 0.01), 0.99)
                            return _m.log(p / (1 - p))
                        def _sigmoid(x):
                            return 1.0 / (1.0 + _m.exp(-x))
                        combined_logit = _logit(existing.confidence) + _logit(rec.confidence) - _logit(0.5)
                        blended_conf = _sigmoid(combined_logit)
                        # Cap hard — two strategies agreeing doesn't
                        # grant certainty, only corroboration.
                        existing.confidence = round(min(0.72, blended_conf), 4)
                        # Edge is computed from blended confidence vs
                        # the best price we'll actually take. Re-derive
                        # instead of taking max which inflates by book
                        # count (Agent-1 finding).
                        best_decimal = max(existing.decimal or 0.0, rec.decimal or 0.0)
                        if best_decimal > 0:
                            implied_best = 1.0 / best_decimal
                            existing.edge = round(existing.confidence - implied_best, 4)
                        if (rec.decimal or 0.0) > (existing.decimal or 0.0):
                            existing.american = rec.american
                            existing.decimal = rec.decimal
                            existing.book = rec.book
                        strategies = existing.meta.setdefault("contributing_strategies", [])
                        if strat.name not in strategies:
                            strategies.append(strat.name)
                        existing.strategy = "+".join(strategies)
                        existing.reasoning += "\n" + rec.reasoning
            except Exception as exc:  # pragma: no cover
                logger.exception("ensemble: strategy %s failed: %s", strat.name, exc)

        recs = list(buckets.values())

        # Stamp line_observed_at + game commence_time on every rec.
        # - line_observed_at: paper-trader's freshness guard rejects
        #   stale quotes >10 min old.
        # - commence_time: surfaces on the dashboard bet cards so Uncle
        #   can see when the game actually kicks off.
        from datetime import datetime as _dt, timezone as _tz
        line_index: Dict[tuple, object] = {}
        game_commence: Dict[str, str] = {}
        for g in games:
            if g.commence_time is not None:
                game_commence[g.game_key] = (
                    g.commence_time.isoformat()
                    if hasattr(g.commence_time, "isoformat")
                    else str(g.commence_time)
                )
            for ln in g.lines:
                if ln.market not in ("spread", "total"):
                    continue
                key = (
                    g.game_key,
                    ln.market,
                    (ln.selection or "").lower().strip(),
                    ln.book.lower(),
                )
                line_index[key] = ln
        # Build a secondary index of peer prices per (game, market,
        # selection) across ALL books so paper_trader can compare this
        # rec's price vs the market consensus and reject phantom/stale
        # quotes (Uncle's bug report 2026-04-17: Bovada showed MIN
        # -0.5 at -108 while every other book had -130 to -145).
        peer_index: Dict[tuple, List[Dict[str, Any]]] = {}
        for g in games:
            for ln in g.lines:
                if ln.market not in ("spread", "total"):
                    continue
                if ln.american is None:
                    continue
                # Group by game+market+selection+LINE so we only
                # compare apples-to-apples (same handicap).
                peer_key = (
                    g.game_key,
                    ln.market,
                    (ln.selection or "").lower().strip(),
                    round(float(ln.line or 0), 2),
                )
                peer_index.setdefault(peer_key, []).append({
                    "book": ln.book,
                    "american": ln.american,
                    "decimal": (
                        1.0 + (ln.american / 100.0) if ln.american > 0
                        else 1.0 + (100.0 / abs(ln.american))
                    ),
                    "line": float(ln.line or 0),
                })

        now_iso = _dt.now(_tz.utc).isoformat()
        for rec in recs:
            k = (
                rec.game_key,
                rec.market,
                rec.selection.lower().strip(),
                rec.book.lower(),
            )
            ln = line_index.get(k)
            if ln is not None and getattr(ln, "last_update", None) is not None:
                rec.meta["line_observed_at"] = ln.last_update.isoformat()
                rec.meta["line_has_real_timestamp"] = True
            else:
                # Real timestamp missing — flag explicitly so the
                # paper-trader's freshness guard can treat as
                # suspect rather than trusting "now".
                rec.meta["line_observed_at"] = now_iso
                rec.meta["line_has_real_timestamp"] = False
            # Stamp the game start time if we know it.
            ct = game_commence.get(rec.game_key)
            if ct:
                rec.meta["commence_time"] = ct
            # Stamp peer prices — same game, market, selection, line
            # across ALL books. Paper trader uses this to detect when
            # this rec's price is an off-market outlier.
            peer_key = (
                rec.game_key,
                rec.market,
                rec.selection.lower().strip(),
                round(float(rec.line or 0), 2),
            )
            peers = peer_index.get(peer_key, [])
            rec.meta["all_book_prices"] = peers
            rec.meta["peer_book_count"] = len(peers)

        # Off-market filter at ENSEMBLE level (in addition to the
        # paper_trader guard). Reason: Uncle's dashboard surfaces
        # these recs BEFORE paper_trader sees them. We don't want
        # misleading phantom picks showing in "Gợi ý AI" even if
        # they'd get blocked at place time. Use a slightly looser
        # threshold here (12% vs 8% at place time) so borderline
        # cases still surface for review but clear outliers are gone.
        def _is_off_market(r):
            if not r.decimal or r.decimal <= 1.0:
                return False
            other_decs = [
                p.get("decimal") for p in r.meta.get("all_book_prices", [])
                if p.get("book") != r.book and p.get("decimal")
            ]
            if len(other_decs) < 3:
                return False
            other_decs_sorted = sorted(other_decs)
            median = other_decs_sorted[len(other_decs_sorted) // 2]
            if median <= 1.0:
                return False
            uplift = (r.decimal - median) / median
            return uplift > 0.12
        before_filter = len(recs)
        recs = [r for r in recs if not _is_off_market(r)]
        filtered_off = before_filter - len(recs)
        if filtered_off > 0:
            logger.warning(
                "ensemble: filtered %d off-market recs (>12%% above peer median)",
                filtered_off,
            )

        # News-aware injury/scratch filter. After strategy confidences
        # are combined we check cached ESPN headlines for injury terms
        # co-mentioned with either team. A match shaves confidence
        # (capped at 15% by NewsReader) and annotates the reasoning so
        # the dashboard/Claude chat can see *why* a bet was downgraded.
        if self.news_reader is not None and hasattr(self.news_reader, "penalty_for_game"):
            penalty_cache: Dict[tuple, tuple] = {}
            for rec in recs:
                key = (rec.home_team, rec.away_team)
                if key not in penalty_cache:
                    try:
                        penalty_cache[key] = self.news_reader.penalty_for_game(
                            rec.home_team, rec.away_team
                        )
                    except Exception as exc:  # pragma: no cover
                        logger.debug("ensemble: news penalty failed for %s: %s", key, exc)
                        penalty_cache[key] = (0.0, "")
                penalty_pct, reason = penalty_cache[key]
                if penalty_pct > 0:
                    before = rec.confidence
                    rec.confidence = round(rec.confidence * (1 - penalty_pct), 4)
                    # Re-derive edge from the penalized confidence so
                    # downstream ranking + Kelly sizing reflect the
                    # news-adjusted win probability.
                    if rec.decimal:
                        implied = 1.0 / rec.decimal
                        rec.edge = round(rec.confidence - implied, 4)
                    rec.reasoning = (rec.reasoning + "\n" + reason).strip()
                    rec.meta["news_penalty_pct"] = penalty_pct
                    rec.meta["news_penalty_reason"] = reason
                    logger.info(
                        "ensemble: news penalty %.0f%% on %s (%.3f -> %.3f): %s",
                        penalty_pct * 100,
                        rec.game_key,
                        before,
                        rec.confidence,
                        reason,
                    )

        # Optional GameAnalyst polish on the top-N recs.
        # Runs AFTER ensemble merge + news penalty so the agent sees
        # the final pre-filter confidence and can nudge up/down based
        # on per-game context (weather, park, umpire, form, Elo,
        # headlines). Capped at ±0.05 by the agent itself; we clamp
        # again here belt-and-suspenders style.
        if self.game_analyst is not None and recs:
            try:
                top_n_env = os.environ.get("SBA_GAME_ANALYST_TOP_N")
                top_n = int(top_n_env) if top_n_env else 5
            except (TypeError, ValueError):
                top_n = 5
            if top_n > 0:
                # Sort by current confidence * edge to pick the most
                # likely-to-be-placed picks; dedup per game so we
                # don't burn tokens on both sides of the same game.
                ranked = sorted(
                    recs,
                    key=lambda r: (r.confidence, r.edge or 0),
                    reverse=True,
                )
                game_index = {g.game_key: g for g in games}
                seen_games: set = set()
                targets = []
                for r in ranked:
                    if r.game_key in seen_games:
                        continue
                    seen_games.add(r.game_key)
                    targets.append(r)
                    if len(targets) >= top_n:
                        break

                try:
                    from ..agents.game_analyst import analyze_game as _analyze_game
                except Exception as exc:  # pragma: no cover
                    logger.debug(
                        "ensemble: game_analyst import failed: %s", exc
                    )
                    _analyze_game = None

                if _analyze_game is not None:
                    for rec in targets:
                        game = game_index.get(rec.game_key)
                        if game is None:
                            continue
                        try:
                            decision = _analyze_game(
                                self.game_analyst,
                                game,
                                settings=self.settings,
                                news_reader=self.news_reader,
                                scoring=self.scoring,
                                elo=self.elo,
                            )
                        except Exception as exc:  # pragma: no cover
                            logger.debug(
                                "ensemble: game_analyst failed on %s: %s",
                                rec.game_key,
                                exc,
                            )
                            continue
                        if decision is None or decision.error:
                            continue
                        delta = max(-0.05, min(0.05, decision.confidence_delta or 0.0))
                        if delta == 0.0 and not decision.metadata:
                            continue
                        before = rec.confidence
                        rec.confidence = round(
                            max(0.0, min(0.99, rec.confidence + delta)), 4
                        )
                        if rec.decimal:
                            implied = 1.0 / rec.decimal
                            rec.edge = round(rec.confidence - implied, 4)
                        if decision.reasoning:
                            rec.reasoning = (
                                rec.reasoning + "\n[game_analyst] " + decision.reasoning
                            ).strip()
                        rec.meta["game_analyst"] = {
                            "delta": delta,
                            "angles": decision.metadata.get("angles", []),
                            "best_bet_hint": decision.metadata.get("best_bet_hint", ""),
                        }
                        logger.info(
                            "ensemble: game_analyst %+.3f on %s (%.3f -> %.3f)",
                            delta,
                            rec.game_key,
                            before,
                            rec.confidence,
                        )

        # Gate on minimum confidence — MARKET-AWARE.
        # Spreads naturally fair at 51-54% (they're designed to be
        # ~coin-flip); a uniform 55% floor kills the entire spread
        # book. Use 0.50 for spreads so edge-based filtering (already
        # applied inside each strategy) is the real gate. Totals still
        # use the standard 0.55 floor since good total picks do reach
        # 55-60%. Edge must still be positive regardless.
        min_conf_total = self.settings.min_confidence    # 0.55 default
        min_conf_spread = max(0.50, min_conf_total - 0.05)
        def _passes(r):
            floor = min_conf_spread if r.market == "spread" else min_conf_total
            return r.confidence >= floor and (r.edge or 0) > 0
        filtered = [r for r in recs if _passes(r)]
        filtered.sort(key=lambda r: (r.confidence, r.edge), reverse=True)
        logger.info(
            "ensemble: %d recs from %d strategies "
            "(%d passed min_conf: spread>=%.2f, total>=%.2f)",
            len(recs),
            len(self.strategies),
            len(filtered),
            min_conf_spread,
            min_conf_total,
        )
        return filtered
