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
from typing import List, Optional

from functools import wraps
from flask import Flask, jsonify, render_template, request, Response

from ..auto_settler import AutoSettler
from ..brain import AgentBrain
from ..config import Settings, get_settings
from ..claude_chat import ChatContext, ClaudeChat
from ..daily_learner import DailyLearner
from ..fetchers.aggregator import OddsAggregator
from ..learning_log import LearningLog
from ..line_movement import LineMovementStore
from ..news_reader import NewsReader
from ..paper_trader import PaperTrader
from ..power_ratings import EloRatings
from ..strategies import (
    BetRecommendation,
    EnsembleStrategy,
    SpreadValueStrategy,
    TotalValueStrategy,
    TotalProjectionStrategy,
    SteamFollowStrategy,
    PublicFadeStrategy,
    ReverseLineMovementStrategy,
    NHLGoalieB2BStrategy,
    MLSHomeTravelStrategy,
    MiddleDetectorStrategy,
    SituationalStrategy,
    EloEdgeStrategy,
)
from ..team_scoring import TeamScoringTracker


logger = logging.getLogger(__name__)


DEFAULT_SPORTS = [
    "baseball_ncaa",
    "baseball_mlb",
    "basketball_nba",
    "basketball_ncaab",
    "basketball_wnba",
    "football_nfl",
    "football_ncaaf",
    "hockey_nhl",
    "soccer_mls",
    "soccer_epl",
    "soccer_ucl",
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
    paper = PaperTrader(settings)
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
    daily_learner = DailyLearner(settings, paper.clv, line_store)

    # Uncle wants SPREADS and OVER/UNDER only — no moneyline bets.
    # TotalProjectionStrategy is the model-based counterpart to the
    # market-based TotalValueStrategy and is the reason totals work
    # even when only one book quotes them.
    strategies = [
        SpreadValueStrategy(settings),
        TotalValueStrategy(settings),
        TotalProjectionStrategy(settings, scoring=scoring),
        # Follows Pinnacle + Bovada steam moves on spreads/totals — the
        # single biggest edge we can extract from the line-movement store.
        SteamFollowStrategy(settings, line_store=line_store),
        # Fade the public: research shows dogs covering ~63.8% when
        # public has <40% tickets. Uses ActionNetwork public-betting %.
        PublicFadeStrategy(settings),
        # Middle detector: when books disagree on spread by 1+ point
        # (Book A -3 / Book B +4), bet both sides — both win if result
        # lands in the middle (Stanford Wong, Sharp Sports Betting).
        MiddleDetectorStrategy(settings),
        # Reverse Line Movement: public ≥60% on side A but 2+ sharp
        # books move the line toward side B = sharp money. 56-58%
        # historical ATS win rate (Pinnacle research, SSRN).
        ReverseLineMovementStrategy(settings, line_store=line_store),
        # NHL puckline fade on back-to-back-scheduled teams — goalies
        # on short rest give up ~0.25 more goals per Schuckers 2020.
        NHLGoalieB2BStrategy(settings),
        # MLS home favorite when visitor crossed 2+ timezones — ~3%
        # historical ATS edge (largest soccer HFA effect in any league).
        MLSHomeTravelStrategy(settings),
    ]
    ensemble = EnsembleStrategy(strategies, settings)

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
            paper._save_state()
        return jsonify({"ok": True, "bankroll": paper.bankroll})

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
                entry = daily_learner.run(paper.closed_bets)
                if entry is not None:
                    logger.info("daily_learner: ran for %s (skill=%s)", entry.date, entry.skill_title)
            except Exception as exc:
                logger.warning("daily_learner failed: %s", exc)
        # 8. Generate recommendations
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
        Positive average CLV over 100+ bets = genuinely +EV picks."""
        return jsonify(paper.clv.stats())

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
        entry = daily_learner.run(paper.closed_bets, force=force)
        if entry is None:
            return jsonify({"ok": False, "reason": "already_ran_today"})
        from dataclasses import asdict
        return jsonify({"ok": True, "entry": asdict(entry)})

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
