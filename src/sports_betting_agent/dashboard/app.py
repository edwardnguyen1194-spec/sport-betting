"""Flask app factory.

Wires together the aggregator, strategies, paper trader and Claude
chat into a single deployable web app. The routes are intentionally
small so the existing fly.io deployment at
``sports-betting-ai-agent.fly.dev`` can adopt this module by pointing
its ``FLASK_APP`` / ``gunicorn`` entrypoint at ``create_app``.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from functools import wraps
from flask import Flask, jsonify, render_template, request, Response

from ..auto_settler import AutoSettler
from ..brain import AgentBrain
from ..config import Settings, get_settings
from ..claude_chat import ChatContext, ClaudeChat
from ..daily_learner import DailyLearner
from ..hourly_monitor import HourlyMonitor
from ..fetchers.aggregator import OddsAggregator
from ..learning_log import LearningLog
from ..line_movement import LineMovementStore
from ..news_reader import NewsReader
from ..paper_trader import PaperTrader
from ..power_ratings import EloRatings
from ..strategies import (
    BetRecommendation,
    EloSpreadStrategy,
    EnsembleStrategy,
    SpreadValueStrategy,
    TotalValueStrategy,
    TotalProjectionStrategy,
    SteamFollowStrategy,
    PublicFadeStrategy,
    ReverseLineMovementStrategy,
    MLSHomeTravelStrategy,
)
from ..team_scoring import TeamScoringTracker


logger = logging.getLogger(__name__)


DEFAULT_SPORTS = [
    # Baseball
    "baseball_mlb",
    "baseball_ncaa",
    # Basketball
    "basketball_nba",
    "basketball_ncaab",
    "basketball_wnba",
    "basketball_euroleague",
    # American football
    "football_nfl",
    "football_ncaaf",
    "football_cfl",
    # Hockey
    "hockey_nhl",
    "hockey_khl",
    # Soccer — big 5 European + MLS + UEFA competitions
    "soccer_mls",
    "soccer_epl",
    "soccer_ucl",
    "soccer_uel",       # Europa League
    "soccer_esp",       # La Liga
    "soccer_ita",       # Serie A
    "soccer_ger",       # Bundesliga
    "soccer_fra",       # Ligue 1
    # Combat + individual
    "mma_ufc",
    "tennis_atp",
    "tennis_wta",
]


def create_app(
    settings: Optional[Settings] = None,
    *,
    start_background: Optional[bool] = None,
) -> Flask:
    settings = settings or get_settings()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, "templates"),
        static_folder=os.path.join(base_dir, "static"),
    )
    app.config["SETTINGS"] = settings

    SPORT_VI = {
        "baseball_ncaa": "Bóng chày NCAA",
        "baseball_mlb": "MLB",
        "basketball_nba": "NBA",
        "basketball_ncaab": "Bóng rổ NCAA",
        "football_nfl": "NFL",
        "football_ncaaf": "Bóng bầu dục NCAA",
        "hockey_nhl": "NHL",
    }
    MARKET_VI = {"moneyline": "Tỷ lệ thắng", "spread": "Kèo chấp", "total": "Tổng điểm"}

    @app.template_filter("sport_vi")
    def sport_vi_filter(s):
        return SPORT_VI.get(s, s)

    @app.template_filter("market_vi")
    def market_vi_filter(s):
        return MARKET_VI.get(s, s)

    _NEWS_VI = [
        ("Game Highlights", "Điểm nổi bật"),
        ("Highlight:", "Điểm nhấn:"),
        ("wins against", "thắng"),
        ("win against", "thắng"),
        ("beat", "đánh bại"),
        ("beats", "đánh bại"),
        ("tops", "vượt qua"),
        ("rally past", "ngược dòng thắng"),
        ("ends", "kết thúc"),
        ("end", "kết thúc"),
        ("hits", "đánh"),
        ("homer", "cú đánh xa"),
        ("home run", "cú đánh xa"),
        ("HRs", "cú đánh xa"),
        ("walk-off", "cú thắng phút cuối"),
        ("inning", "hiệp"),
        ("season", "mùa giải"),
        ("game skid", "chuỗi thua"),
        ("upset win", "chiến thắng bất ngờ"),
        ("snatches", "giật"),
        ("rides hot start", "khởi đầu nóng"),
        ("hold off", "cầm cự thắng"),
        ("completes hat trick", "hoàn thành hat-trick"),
        ("walk it off", "kết thúc trận"),
        ("stunning", "ngoạn mục"),
        ("tracker:", "theo dõi:"),
        ("rankings", "xếp hạng"),
        ("challenge system", "hệ thống thách thức"),
        ("Team, player", "Đội, cầu thủ"),
        ("vs.", "đấu"),
        ("1st", "đầu tiên"),
        ("No.", "Hạng"),
        ("helps", "giúp"),
        ("caps", "kết thúc với"),
        ("night with", "đêm với"),
        ("as", "khi"),
        ("of the", "của"),
        ("for the", "cho"),
        ("to", "để"),
        ("in the", "trong"),
        ("with", "với"),
    ]

    @app.template_filter("news_vi")
    def news_vi_filter(s):
        result = s
        for en, vi in _NEWS_VI:
            result = result.replace(en, vi)
        return result

    aggregator = OddsAggregator(settings)
    # Build the agent log + PickReviewer BEFORE the PaperTrader so we
    # can inject the reviewer into its constructor. If ANTHROPIC_API_KEY
    # is absent or SBA_AGENT_PICK_REVIEWER_ENABLED=0, BaseAgent silently
    # returns no-op decisions — the trader keeps running.
    from ..agents import AgentLog, PickReviewer, StrategyAuditor
    agent_log = AgentLog(settings.data_dir)
    pick_reviewer = PickReviewer(agent_log=agent_log)
    # Weekly deep-review agent. Fires Monday mornings from
    # daily_learner.run(); no-ops otherwise. Reuses the shared
    # ``agent_log`` so its decisions show up in /api/agent-log too.
    strategy_auditor = StrategyAuditor(agent_log=agent_log)
    paper = PaperTrader(settings, pick_reviewer=pick_reviewer)
    chat = ClaudeChat(settings)
    learn_log = LearningLog(settings.data_dir)
    news = NewsReader(settings.data_dir)
    brain = AgentBrain(settings.data_dir)
    # Team scoring tracker — fuels the model-based total projection strategy.
    # Wired into EloRatings so both update from the same ESPN feed in lockstep.
    scoring = TeamScoringTracker(settings.data_dir)
    elo = EloRatings(settings.data_dir, scoring_tracker=scoring)
    # Line-movement store feeds the steam-move detector — a core world-class
    # sharp signal (3+ sharp books moving the same direction = smart money).
    line_store = LineMovementStore(settings.data_dir)
    # Daily self-improvement — auto-reviews CLV + win-rate, nudges edge
    # thresholds, and pulls one fresh sharp-betting article per day.
    daily_learner = DailyLearner(
        settings, paper.clv, line_store,
        strategy_auditor=strategy_auditor,
    )
    # Hourly health monitor — logs per-hour snapshots and flags
    # anomalies (all-Over bias, stale open bets, drawdown approach,
    # quiet strategies). Read-only; it never auto-tunes.
    hourly_monitor = HourlyMonitor(paper, settings.data_dir)

    # OpportunityScout is an on-demand "top 3 plays right now" concierge
    # Uncle fires via a dashboard button. Read-only; does not affect the
    # paper trader. Reuses the shared ``agent_log`` created up above.
    from ..agents import OpportunityScout
    scout = OpportunityScout(agent_log=agent_log)

    # NewsTriage runs every trade cycle over OPEN bets. Read-only — it
    # only flags (red/yellow/green). A red alert emits a warning log
    # and the decision is persisted next to the CLV records for audit.
    from ..agents import NewsTriage, triage_open_bets, persist_triage_audit
    news_triage = NewsTriage(agent_log=agent_log)

    # Uncle wants SPREADS and OVER/UNDER only — no moneyline bets.
    # TotalProjectionStrategy is the model-based counterpart to the
    # market-based TotalValueStrategy and is the reason totals work
    # even when only one book quotes them.
    # World-class ensemble after the multi-agent audit (2026-04-17).
    # Disabled strategies:
    #   - MiddleDetectorStrategy (middle-prob math was 3-5× too loose:
    #     gap * 0.05 vs empirical 1.5%. Re-enable when we have a proper
    #     margin-of-victory distribution per sport.)
    #   - SituationalStrategy (hand-picked 9-factor additive scoring,
    #     favorites-only filter, cap 0.92 = hidden bankroll drag.)
    #   - NHLGoalieB2BStrategy (schedule-based B2B detection depended on
    #     yesterday's games appearing in today's feed — almost never
    #     fires, and when it does it's based on a data gap not a real
    #     starter signal. Re-enable when we wire up a real roster feed.)
    strategies = [
        # Core market-based: A-grade per audit, proper vig-removed arb.
        SpreadValueStrategy(settings),
        TotalValueStrategy(settings),
        # Model-based totals with Dixon-Coles Poisson for soccer +
        # market-regressed rolling scoring for other sports. B+ grade.
        TotalProjectionStrategy(settings, scoring=scoring),
        # A+ profit engine per audit — Pinnacle+sharp-book consensus.
        SteamFollowStrategy(settings, line_store=line_store),
        # A-grade: public + sharp movement joint signal.
        ReverseLineMovementStrategy(settings, line_store=line_store),
        # PublicFade: C grade but now has public-% data wired up. Keep
        # on with higher threshold; auto-tuner will adjust.
        PublicFadeStrategy(settings),
        # MLS travel fatigue — static TZ table, seasonal edge.
        MLSHomeTravelStrategy(settings),
        # Elo-derived ATS picks: home/away spread based on our own
        # power-rating model diverging from the posted main line.
        # Shares the same EloRatings instance that also powers
        # /api/elo and the scoring tracker.
        EloSpreadStrategy(settings, elo=elo),
    ]
    ensemble = EnsembleStrategy(strategies, settings, news_reader=news)

    app.extensions["sba"] = {
        "aggregator": aggregator,
        "ensemble": ensemble,
        "paper": paper,
        "chat": chat,
        "learn_log": learn_log,
    }

    def _current_recs(sports: Optional[List[str]] = None) -> List[BetRecommendation]:
        sports = sports or DEFAULT_SPORTS
        games = aggregator.fetch_sports(sports)
        # Feed every fetch into the line-movement store so steam/RLM can
        # be detected across cycles. Also refresh closing-line values for
        # every open bet — that's what powers the CLV scoreboard.
        try:
            line_store.ingest(games)
        except Exception as exc:
            logger.warning("line_store ingest failed: %s", exc)
        try:
            paper.record_closing_lines(games)
        except Exception as exc:
            logger.warning("record_closing_lines failed: %s", exc)
        # Filter to games that haven't started yet (plus a 10 min grace
        # window). Uncle flagged that we were placing bets on games that
        # were already in progress — a recipe for variance-free losses.
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        cutoff = _dt.now(_tz.utc) + _td(minutes=10)
        upcoming = [
            g for g in games
            if g.commence_time is None or g.commence_time >= cutoff
        ]
        dropped = len(games) - len(upcoming)
        if dropped:
            logger.info("Dropped %d in-progress/finished games from rec pool", dropped)
        return ensemble.generate(upcoming)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    # Simple password gate — Uncle's request. Password is checked via
    # HTTP Basic Auth on every page load. API endpoints (/api/*) are
    # left open so the auto-trader and monitoring scripts still work.
    DASHBOARD_PASSWORD = os.environ.get("SBA_PASSWORD", "phungga2026")

    def _check_auth(username, password):
        return password == DASHBOARD_PASSWORD

    def _auth_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            auth = request.authorization
            if not auth or not _check_auth(auth.username, auth.password):
                return Response(
                    "Đăng nhập để xem dashboard",
                    401,
                    {"WWW-Authenticate": 'Basic realm="AI Sports Betting"'},
                )
            return f(*args, **kwargs)
        return decorated

    @app.before_request
    def _protect_dashboard():
        # Only gate HTML pages, not /api/* endpoints
        if not request.path.startswith("/api/") and request.path != "/favicon.ico":
            auth = request.authorization
            if not auth or not _check_auth(auth.username, auth.password):
                return Response(
                    "Đăng nhập để xem dashboard",
                    401,
                    {"WWW-Authenticate": 'Basic realm="AI Sports Betting"'},
                )

    @app.after_request
    def _no_cache(response):
        """Prevent browser/proxy caching of the dashboard — Uncle kept
        seeing yesterday's ledger until a second hard reload."""
        # Only disable cache for HTML + JSON API responses; keep static
        # assets (CSS/JS/images) cacheable so the page still loads fast.
        ct = response.headers.get("Content-Type", "")
        if ct.startswith("text/html") or ct.startswith("application/json"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.route("/")
    def index():
        stats = paper.stats()
        # Update learning log on each page load
        learn_log.analyze_and_log(
            stats,
            list(paper.open_bets.values()),
            paper.closed_bets,
        )
        # Settlement and news done in background cycle, not on page load
        return render_template(
            "index.html",
            stats=paper.stats(),
            sources=aggregator.list_sources(),
            sports=DEFAULT_SPORTS,
            language=settings.dashboard_language,
            learning=learn_log.recent_entries(7),
            open_bets=list(paper.open_bets.values()),
            closed_bets=list(reversed(paper.closed_bets[-20:])),
            news_headlines=news.recent_headlines(n=12),
            brain_summary=brain.daily_summary(),
        )

    @app.route("/api/health")
    def health():
        return jsonify({"ok": True, "sources": aggregator.list_sources()})

    @app.route("/api/reset", methods=["POST"])
    def reset():
        """Reset paper trader to starting bankroll. Clears all bets."""
        with paper._lock:
            paper.bankroll = settings.bankroll_start
            paper.open_bets.clear()
            paper.closed_bets.clear()
            paper._next_id = 1
            paper._peak_bankroll = settings.bankroll_start
            paper._halted = False
            paper._save_state()
        return jsonify({"ok": True, "bankroll": paper.bankroll})

    @app.route("/api/risk")
    def api_risk():
        """Current risk snapshot: bankroll, drawdown %, halt status."""
        return jsonify(paper.risk_snapshot())

    @app.route("/api/reset-halt", methods=["POST"])
    def api_reset_halt():
        """Clear the drawdown halt so betting can resume."""
        result = paper.resume()
        return jsonify({"ok": True, **result})

    @app.route("/api/hourly-log")
    def api_hourly_log():
        """Latest + recent hourly-monitor snapshots.

        Returns the freshest entry plus the last 24 for sparkline +
        trend visualization. Query param ``hours`` overrides the window
        (max 168).
        """
        try:
            hours = min(168, max(1, int(request.args.get("hours", 24))))
        except (TypeError, ValueError):
            hours = 24
        return jsonify({
            "latest": hourly_monitor.latest(),
            "recent": hourly_monitor.recent(hours),
        })

    @app.route("/api/hourly-run", methods=["POST"])
    def api_hourly_run():
        """Force-run the hourly monitor right now (for debugging)."""
        from dataclasses import asdict as _asdict
        snap = hourly_monitor.run(force=True)
        return jsonify({"ok": True, "snapshot": _asdict(snap) if snap else None})

    @app.route("/api/agent-log")
    def api_agent_log():
        """Recent Claude sub-agent decisions.

        Query params:
          - n: number of entries (default 50, max 500)
          - agent: filter by agent name (pick_reviewer, news_triage, etc.)
        Also returns today's token usage so Uncle can see budget burn.
        """
        try:
            n = min(500, max(1, int(request.args.get("n", 50))))
        except (TypeError, ValueError):
            n = 50
        name = request.args.get("agent")
        return jsonify({
            "entries": agent_log.recent(n=n, agent=name),
            "today_tokens": agent_log.today_token_usage(),
        })

    @app.route("/api/purge-voids", methods=["POST"])
    def purge_voids():
        """Remove every status='void' entry from closed history.
        Leaves real win/loss results untouched so the agent's actual
        performance record stays intact. Uncle finds HÒA rows visually
        noisy on the dashboard."""
        with paper._lock:
            before = len(paper.closed_bets)
            paper.closed_bets = [b for b in paper.closed_bets if b.status != "void"]
            removed = before - len(paper.closed_bets)
            paper._save_state()
        return jsonify({"ok": True, "removed": removed, "remaining": len(paper.closed_bets)})

    _odds_cache: Dict = {}

    @app.route("/api/odds")
    def odds():
        import time as _time
        sport = request.args.get("sport", "baseball_ncaa")
        now = _time.time()
        # Cache odds for 60 seconds to avoid hammering sources on every page load
        cached = _odds_cache.get(sport)
        if cached and now - cached["ts"] < 60:
            return jsonify(cached["data"])
        games = aggregator.fetch_sport(sport)
        data = {"sport": sport, "count": len(games), "games": [g.to_dict() for g in games]}
        _odds_cache[sport] = {"ts": now, "data": data}
        return jsonify(data)

    @app.route("/api/recommendations")
    def recommendations():
        from ..models_schema import american_to_implied
        sports_param = request.args.get("sports")
        sports = [s.strip() for s in sports_param.split(",")] if sports_param else DEFAULT_SPORTS
        games = aggregator.fetch_sports(sports)

        # Uncle wants SPREADS and OVER/UNDER only — no moneyline bets.
        # Include the model-based total projection alongside the market
        # strategies so totals surface even when a single book quotes.
        all_recs = []
        for strat in [
            SpreadValueStrategy(settings),
            TotalValueStrategy(settings),
            TotalProjectionStrategy(settings, scoring=scoring),
            SteamFollowStrategy(settings, line_store=line_store),
            PublicFadeStrategy(settings),
        ]:
            all_recs.extend(strat.generate(games))

        all_recs.sort(key=lambda r: r.confidence, reverse=True)
        recs = all_recs[:15]
        return jsonify({
            "count": len(recs),
            "recommendations": [r.to_dict() for r in recs],
        })

    @app.route("/api/best-picks")
    def best_picks():
        """Top picks across ALL sports — the daily pick sheet."""
        all_recs = _current_recs(DEFAULT_SPORTS)
        scored = sorted(all_recs, key=lambda r: r.confidence * max(r.edge, 0), reverse=True)
        top = scored[:15]
        return jsonify({
            "count": len(top),
            "picks": [r.to_dict() for r in top],
        })

    @app.route("/api/ledger")
    def ledger():
        return jsonify(
            {
                "stats": paper.stats(),
                "open": [b.__dict__ for b in paper.open_bets.values()],
                "closed": [b.__dict__ for b in paper.closed_bets[-100:]],
            }
        )

    @app.route("/api/place", methods=["POST"])
    def api_place():
        """Manually place a bet from a rec dict (dashboard button)."""

        data = request.get_json(silent=True) or {}
        rec = BetRecommendation(
            game_key=data.get("game_key", ""),
            sport=data.get("sport", ""),
            league=data.get("league", ""),
            home_team=data.get("home_team", ""),
            away_team=data.get("away_team", ""),
            market=data.get("market", "moneyline"),
            selection=data.get("selection", ""),
            american=float(data.get("american", 0)),
            decimal=float(data.get("decimal", 1.0)),
            line=data.get("line"),
            book=data.get("book", ""),
            strategy=data.get("strategy", "manual"),
            confidence=float(data.get("confidence", 0.0)),
            edge=float(data.get("edge", 0.0)),
            stake_fraction=float(data.get("stake_fraction", 0.01)),
            reasoning=data.get("reasoning", ""),
        )
        bet = paper.place(rec)
        return jsonify({"ok": bet is not None, "bet": bet.__dict__ if bet else None})

    @app.route("/api/settle", methods=["POST"])
    def api_settle():
        data = request.get_json(silent=True) or {}
        bet_id = data.get("bet_id", "")
        outcome = data.get("outcome", "")
        bet = paper.settle_bet(bet_id, outcome)
        return jsonify({"ok": bet is not None, "bet": bet.__dict__ if bet else None})

    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        data = request.get_json(silent=True) or {}
        user_msg = data.get("message", "").strip()
        if not user_msg:
            return jsonify({"error": "empty message"}), 400
        recs = _current_recs()
        ctx = ChatContext(
            bankroll=paper.bankroll,
            stats=paper.stats(),
            recommendations=recs,
        )
        reply = chat.reply(user_msg, ctx)
        return jsonify(reply)

    @app.route("/api/scout", methods=["POST"])
    def api_scout():
        """On-demand OpportunityScout — Uncle's "top 3 plays right now"
        button. Pulls fresh recs + headlines, asks the scout agent for
        a markdown writeup, and returns the structured AgentDecision."""
        from dataclasses import asdict
        recs = _current_recs()
        headlines = news.recent_headlines(n=15)
        decision = scout.analyze({"recs": recs[:20], "headlines": headlines})
        return jsonify({"ok": True, "decision": asdict(decision)})

    # ------------------------------------------------------------------
    # Background auto-trading
    # ------------------------------------------------------------------

    settler = AutoSettler(paper)

    @app.route("/api/settle-auto", methods=["POST"])
    def api_settle_auto():
        results = settler.settle_completed_bets()
        return jsonify({"settled": len(results), "results": results})

    # Wrap the recommender to also settle bets, read news, and learn each cycle
    def _trade_and_settle() -> list:
        # 1. Settle completed bets
        settler.settle_completed_bets()
        # 1b. NewsTriage pass over the still-open bets. Runs AFTER the
        # settler so we only spend tokens on bets that survived this
        # cycle. Flags red/yellow/green per bet; a red alert logs a
        # warning but never vetoes — this agent only flags. Every
        # decision is persisted next to the CLV records for audit.
        try:
            triage_decisions = triage_open_bets(
                news_triage, paper.open_bets, news,
            )
            for bet_id, decision in triage_decisions.items():
                alert = (decision.metadata or {}).get("alert_level")
                if alert == "red":
                    logger.warning(
                        "news_triage RED alert on bet %s: %s",
                        bet_id, decision.reasoning,
                    )
            persist_triage_audit(paper.clv, triage_decisions)
        except Exception as exc:
            logger.warning("news_triage cycle failed: %s", exc)
        # 2. Read sports news
        news.fetch_all()
        # 3. Update Elo power ratings from ESPN results
        elo.update_from_espn()
        # 4. Brain learns from results
        brain.learn_from_results(paper.closed_bets)
        # 5. Brain learns from news
        brain.learn_from_news(news.recent_headlines(n=20))
        # 6. Update learning log
        learn_log.analyze_and_log(
            paper.stats(),
            list(paper.open_bets.values()),
            paper.closed_bets,
        )
        # 7. Every-morning self-improvement. Uncle asked for the agent
        # to learn new skills each morning while he sleeps. Window is
        # 12:00-17:00 UTC (5am-10am Pacific / 8am-1pm Eastern) — wide
        # enough that a single background cycle lands inside it every
        # day and the learner is idempotent across duplicate triggers.
        from datetime import datetime, timezone as _tz
        utc_hour = datetime.now(_tz.utc).hour
        if 12 <= utc_hour < 17:
            try:
                entry = daily_learner.run(paper.closed_bets, paper=paper)
                if entry is not None:
                    logger.info("daily_learner: ran for %s (skill=%s)", entry.date, entry.skill_title)
            except Exception as exc:
                logger.warning("daily_learner failed: %s", exc)
        # 8. Hourly self-monitor. Idempotent — fires once per UTC
        # hour regardless of how many trade cycles hit it. Logs a
        # snapshot + flags bias/stale/drawdown anomalies.
        try:
            snap = hourly_monitor.run()
            if snap is not None:
                logger.info(
                    "hourly_monitor snapshot: health=%d open=%d dd=%.1f%% anomalies=%d",
                    snap.health_score, snap.open_count,
                    snap.drawdown_pct, len(snap.anomalies),
                )
        except Exception as exc:
            logger.warning("hourly_monitor failed: %s", exc)
        # 9. Generate recommendations
        return _current_recs()

    @app.route("/api/elo")
    def elo_status():
        return jsonify(elo.summary())

    @app.route("/api/scoring")
    def scoring_status():
        """Rolling team scoring tracker — drives TotalProjectionStrategy.
        Shows how much history we have per sport."""
        return jsonify(scoring.summary())

    @app.route("/api/brain")
    def brain_status():
        return jsonify(brain.daily_summary())

    @app.route("/api/clv")
    def clv_status():
        """Closing Line Value scoreboard — the #1 world-class metric.
        Positive average CLV over 100+ bets = genuinely +EV picks.

        Also exposes per-strategy breakdown so the dashboard can render
        a W-L + CLV scoreboard without a second API round trip."""
        payload = dict(paper.clv.stats())
        try:
            payload["by_strategy"] = paper.clv.stats_by_strategy()
        except Exception as exc:
            logger.warning("stats_by_strategy failed: %s", exc)
            payload["by_strategy"] = {}
        # Per-strategy P&L from the closed ledger so Uncle sees ROI
        # alongside CLV. CLV records alone can't compute ROI because
        # they don't carry stake/payout.
        try:
            roi_by_strat: Dict[str, Dict[str, float]] = {}
            for b in paper.closed_bets:
                if not getattr(b, "strategy", None):
                    continue
                for name in b.strategy.split("+"):
                    key = name.strip() or "unknown"
                    bucket = roi_by_strat.setdefault(
                        key, {"stake": 0.0, "pnl": 0.0, "bets": 0}
                    )
                    bucket["stake"] += float(b.stake)
                    bucket["bets"] += 1
                    if b.status == "won":
                        bucket["pnl"] += float(b.stake) * (float(b.decimal) - 1.0)
                    elif b.status == "lost":
                        bucket["pnl"] -= float(b.stake)
                    # push/void: no P&L change
            for key, vals in roi_by_strat.items():
                vals["roi_pct"] = (
                    round(vals["pnl"] / vals["stake"] * 100, 2)
                    if vals["stake"] > 0 else 0.0
                )
            payload["roi_by_strategy"] = roi_by_strat
        except Exception as exc:
            logger.warning("roi_by_strategy failed: %s", exc)
            payload["roi_by_strategy"] = {}
        return jsonify(payload)

    @app.route("/api/learner/today")
    def learner_today():
        """Today's self-improvement entry (null if it hasn't run yet)."""
        entry = daily_learner.today()
        if entry is None:
            return jsonify({"status": "not_run_yet", "scheduled": "12:00-17:00 UTC daily (morning US)"})
        from dataclasses import asdict
        return jsonify(asdict(entry))

    @app.route("/api/learner/history")
    def learner_history():
        """Recent self-improvement history (last 30 days by default)."""
        from dataclasses import asdict
        n = int(request.args.get("n", 30))
        return jsonify({
            "count": len(daily_learner.entries),
            "recent": [asdict(e) for e in daily_learner.recent(n)],
        })

    @app.route("/api/learner/run", methods=["POST"])
    def learner_run():
        """Manual trigger — forces one learner cycle now (for testing)."""
        force = bool(request.json and request.json.get("force"))
        entry = daily_learner.run(paper.closed_bets, force=force, paper=paper)
        if entry is None:
            return jsonify({"ok": False, "reason": "already_ran_today"})
        from dataclasses import asdict
        return jsonify({"ok": True, "entry": asdict(entry)})

    @app.route("/api/weekly-audit")
    def weekly_audit():
        """StrategyAuditor output — the Monday-morning deep review.

        Returns the most recent audit plus up to N prior audits. Each
        audit is the markdown body saved to
        ``<data_dir>/weekly_audits/YYYY-MM-DD.md``. On cold systems
        (no audit has fired yet) ``latest`` is ``null`` and ``recent``
        is an empty list.
        """
        from ..agents import list_recent_weekly_audits
        try:
            n = max(1, min(52, int(request.args.get("n", 12))))
        except (TypeError, ValueError):
            n = 12
        audits = list_recent_weekly_audits(settings.data_dir, limit=n)
        return jsonify({
            "count": len(audits),
            "latest": audits[0] if audits else None,
            "recent": audits,
        })

    @app.route("/api/steam")
    def steam_status():
        """Current steam moves — 3+ sharp books moving the same direction.
        A strong indicator of professional (sharp) money."""
        moves = line_store.detect_steam()
        return jsonify({
            "count": len(moves),
            "store": line_store.stats(),
            "moves": [
                {
                    "game_key": m.game_key,
                    "market": m.market,
                    "selection": m.selection,
                    "direction": m.direction,
                    "sharp_books": m.sharp_books_moving,
                    "juice_delta": m.juice_delta,
                    "line_delta": m.line_delta,
                    "from": m.from_ts,
                    "to": m.to_ts,
                }
                for m in moves
            ],
        })

    should_start = settings.auto_trade_enabled if start_background is None else start_background
    if should_start:
        paper.start_background(_trade_and_settle)

    return app
