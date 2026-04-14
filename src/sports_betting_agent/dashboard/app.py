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

from flask import Flask, jsonify, render_template, request

from ..auto_settler import AutoSettler
from ..brain import AgentBrain
from ..config import Settings, get_settings
from ..claude_chat import ChatContext, ClaudeChat
from ..fetchers.aggregator import OddsAggregator
from ..learning_log import LearningLog
from ..news_reader import NewsReader
from ..paper_trader import PaperTrader
from ..power_ratings import EloRatings
from ..strategies import (
    BetRecommendation,
    EnsembleStrategy,
    HeavyFavoriteStrategy,
    ValueBetStrategy,
    SpreadValueStrategy,
    TotalValueStrategy,
    ContrarianStrategy,
    MiddleDetectorStrategy,
    SituationalStrategy,
    EloEdgeStrategy,
    PythagoreanStrategy,
)


logger = logging.getLogger(__name__)


DEFAULT_SPORTS = [
    "baseball_ncaa",
    "baseball_mlb",
    "basketball_nba",
    "basketball_ncaab",
    "football_nfl",
    "football_ncaaf",
    "hockey_nhl",
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
    elo = EloRatings(settings.data_dir)

    # Uncle wants SPREADS and OVER/UNDER only — no moneyline bets
    strategies = [
        SpreadValueStrategy(settings),
        TotalValueStrategy(settings),
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
        return ensemble.generate(games)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

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

        # First try strategies
        all_recs = []
        for strat in [HeavyFavoriteStrategy(settings), ValueBetStrategy(settings),
                       SpreadValueStrategy(settings), TotalValueStrategy(settings),
                       ContrarianStrategy(settings), MiddleDetectorStrategy(settings),
                       SituationalStrategy(settings)]:
            all_recs.extend(strat.generate(games))

        # If strategies found nothing, build picks from raw odds (favorites)
        if not all_recs:
            for game in games:
                for side in ("home", "away"):
                    team = game.home_team if side == "home" else game.away_team
                    ml = [l for l in game.lines if l.market == "moneyline"
                          and l.selection.lower() == team.lower() and l.american is not None]
                    if not ml:
                        continue
                    best = max(ml, key=lambda l: l.decimal or 0)
                    if best.american is None or best.american > -120:
                        continue  # Only show favorites
                    implied = american_to_implied(best.american)
                    all_recs.append(BetRecommendation(
                        game_key=game.game_key, sport=game.sport, league=game.league,
                        home_team=game.home_team, away_team=game.away_team,
                        market="moneyline", selection=team,
                        american=best.american, decimal=best.decimal or 1.0,
                        book=best.book, strategy="ai_analysis",
                        confidence=round(implied, 4), edge=round(implied - 0.5, 4),
                        reasoning=f"AI phân tích: {team} là đội mạnh hơn ({implied:.0%} thắng) tại {best.book}",
                    ))

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
        # 7. Generate recommendations
        return _current_recs()

    @app.route("/api/elo")
    def elo_status():
        return jsonify(elo.summary())

    @app.route("/api/brain")
    def brain_status():
        return jsonify(brain.daily_summary())

    should_start = settings.auto_trade_enabled if start_background is None else start_background
    if should_start:
        paper.start_background(_trade_and_settle)

    return app
