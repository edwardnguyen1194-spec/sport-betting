"""Command line interface.

Examples::

    python -m sports_betting_agent.cli odds --sport baseball_ncaa --source bovada
    python -m sports_betting_agent.cli recommend --sports baseball_ncaa,basketball_nba
    python -m sports_betting_agent.cli serve
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import List

from .config import get_settings
from .fetchers.aggregator import OddsAggregator
from .fetchers import (
    BovadaFetcher,
    ESPNFetcher,
    ScoresAndOddsFetcher,
    VegasInsiderFetcher,
    CoversFetcher,
    ActionNetworkFetcher,
    SBRFetcher,
)
from .line_movement import LineMovementStore
from .news_reader import NewsReader
from .team_scoring import TeamScoringTracker
from .strategies import (
    EnsembleStrategy,
    SpreadValueStrategy,
    TotalValueStrategy,
    TotalProjectionStrategy,
    SteamFollowStrategy,
    PublicFadeStrategy,
    ReverseLineMovementStrategy,
    MLSHomeTravelStrategy,
)


_SINGLE = {
    "bovada": BovadaFetcher,
    "espn": ESPNFetcher,
    "scoresandodds": ScoresAndOddsFetcher,
    "vegasinsider": VegasInsiderFetcher,
    "covers": CoversFetcher,
    "actionnetwork": ActionNetworkFetcher,
    "sbr": SBRFetcher,
}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sports_betting_agent")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_odds = sub.add_parser("odds", help="Print odds for a sport")
    p_odds.add_argument("--sport", default="baseball_ncaa")
    p_odds.add_argument(
        "--source",
        default="all",
        help=f"all or one of: {', '.join(_SINGLE)}",
    )

    p_rec = sub.add_parser("recommend", help="Print ensemble recommendations")
    p_rec.add_argument("--sports", default="baseball_ncaa,baseball_mlb,basketball_nba")

    sub.add_parser("serve", help="Run the Flask dashboard")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.cmd == "odds":
        return _cmd_odds(args)
    if args.cmd == "recommend":
        return _cmd_recommend(args)
    if args.cmd == "serve":
        return _cmd_serve()
    return 1


def _cmd_odds(args) -> int:
    settings = get_settings()
    if args.source == "all":
        agg = OddsAggregator(settings)
        games = agg.fetch_sport(args.sport)
    else:
        cls = _SINGLE.get(args.source)
        if cls is None:
            print(f"Unknown source: {args.source}")
            return 2
        games = cls(settings).fetch(args.sport)
    print(json.dumps([g.to_dict() for g in games], indent=2, default=str))
    return 0


def _cmd_recommend(args) -> int:
    """CLI mirror of the dashboard ensemble — useful for cron / debugging."""
    settings = get_settings()
    agg = OddsAggregator(settings)
    games = agg.fetch_sports([s.strip() for s in args.sports.split(",") if s.strip()])
    scoring = TeamScoringTracker(settings.data_dir)
    line_store = LineMovementStore(settings.data_dir)
    news = NewsReader(settings.data_dir)
    ensemble = EnsembleStrategy(
        [
            SpreadValueStrategy(settings),
            TotalValueStrategy(settings),
            TotalProjectionStrategy(settings, scoring=scoring),
            SteamFollowStrategy(settings, line_store=line_store),
            ReverseLineMovementStrategy(settings, line_store=line_store),
            PublicFadeStrategy(settings),
            MLSHomeTravelStrategy(settings),
        ],
        settings,
        news_reader=news,
    )
    recs = ensemble.generate(games)
    print(json.dumps([r.to_dict() for r in recs], indent=2, default=str))
    return 0


def _cmd_serve() -> int:
    from .dashboard import create_app

    settings = get_settings()
    app = create_app(settings)
    app.run(host=settings.dashboard_host, port=settings.dashboard_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
