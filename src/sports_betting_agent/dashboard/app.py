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

from ..config import Settings, get_settings
from ..claude_chat import ChatContext, ClaudeChat
from ..fetchers.aggregator import OddsAggregator
from ..paper_trader import PaperTrader
from ..strategies import (
    BetRecommendation,
    EnsembleStrategy,
    HeavyFavoriteStrategy,
    ValueBetStrategy,
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

    aggregator = OddsAggregator(settings)
    strategies = [
        HeavyFavoriteStrategy(settings),
        ValueBetStrategy(settings),
    ]
    ensemble = EnsembleStrategy(strategies, settings)
    paper = PaperTrader(settings)
    chat = ClaudeChat(settings)

    app.extensions["sba"] = {
        "aggregator": aggregator,
        "ensemble": ensemble,
        "paper": paper,
        "chat": chat,
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
        return render_template(
            "index.html",
            stats=stats,
            sources=aggregator.list_sources(),
            sports=DEFAULT_SPORTS,
            language=settings.dashboard_language,
        )

    @app.route("/api/health")
    def health():
        return jsonify({"ok": True, "sources": aggregator.list_sources()})

    @app.route("/api/odds")
    def odds():
        sport = request.args.get("sport", "baseball_ncaa")
        games = aggregator.fetch_sport(sport)
        return jsonify({"sport": sport, "count": len(games), "games": [g.to_dict() for g in games]})

    @app.route("/api/recommendations")
    def recommendations():
        sports_param = request.args.get("sports")
        sports = [s.strip() for s in sports_param.split(",")] if sports_param else DEFAULT_SPORTS
        recs = _current_recs(sports)
        return jsonify(
            {
                "count": len(recs),
                "recommendations": [r.to_dict() for r in recs],
            }
        )

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

    should_start = settings.auto_trade_enabled if start_background is None else start_background
    if should_start:
        paper.start_background(_current_recs)

    return app
