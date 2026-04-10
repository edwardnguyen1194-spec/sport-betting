"""Live validation script for every fetcher.

Run this inside an environment that has outbound internet access
(your laptop, or `fly ssh console -a sports-betting-ai-agent` followed
by `python scripts/validate_live.py`). It will:

1. Hit every enabled source for every supported sport.
2. Report games parsed + moneyline lines extracted + sample teams.
3. Save the raw payload of the first successful fetch per source to
   ``validation_dumps/<source>_<sport>.json`` so you can eyeball the
   schema when a parser returns 0 games.
4. Print a red/green summary at the end.

This is the script I would have run myself if this sandbox had
network access. Exit code is 0 if every enabled source returns at
least one game for at least one sport; 1 otherwise.

Usage::

    PYTHONPATH=src python scripts/validate_live.py
    PYTHONPATH=src python scripts/validate_live.py --sources bovada,espn
    PYTHONPATH=src python scripts/validate_live.py --sport baseball_ncaa
    PYTHONPATH=src python scripts/validate_live.py --dump   # save raw payloads
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from typing import Any, Dict, List

from sports_betting_agent.config import get_settings
from sports_betting_agent.fetchers import (
    ActionNetworkFetcher,
    BovadaFetcher,
    CoversFetcher,
    ESPNFetcher,
    SBRFetcher,
    ScoresAndOddsFetcher,
    VegasInsiderFetcher,
)


ALL_FETCHERS = {
    "bovada": BovadaFetcher,
    "espn": ESPNFetcher,
    "actionnetwork": ActionNetworkFetcher,
    "scoresandodds": ScoresAndOddsFetcher,
    "vegasinsider": VegasInsiderFetcher,
    "covers": CoversFetcher,
    "sbr": SBRFetcher,
}

DEFAULT_SPORTS = [
    "baseball_ncaa",
    "baseball_mlb",
    "basketball_nba",
    "basketball_ncaab",
    "football_nfl",
    "football_ncaaf",
    "hockey_nhl",
]

# ANSI color codes. Fall back to plain if stdout is not a tty.
def _color(code: str) -> str:
    return code if sys.stdout.isatty() else ""

GREEN = _color("\033[32m")
RED = _color("\033[31m")
YELLOW = _color("\033[33m")
DIM = _color("\033[2m")
RESET = _color("\033[0m")


def _mark(ok: bool) -> str:
    return f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"


def run_one(name: str, fetcher, sport: str, *, dump: bool) -> Dict[str, Any]:
    """Fetch and summarize a single (source, sport) pair."""

    start = time.monotonic()
    try:
        games = fetcher.fetch(sport)
        elapsed = time.monotonic() - start
    except Exception as exc:
        return {
            "source": name,
            "sport": sport,
            "ok": False,
            "games": 0,
            "ml_lines": 0,
            "elapsed_s": round(time.monotonic() - start, 2),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    ml_lines = sum(1 for g in games for l in g.lines if l.market == "moneyline")
    total_lines = sum(len(g.lines) for g in games)
    sample = None
    if games:
        g = games[0]
        sample = {
            "away_team": g.away_team,
            "home_team": g.home_team,
            "commence_time": g.commence_time.isoformat() if g.commence_time else None,
            "lines_count": len(g.lines),
            "sample_lines": [
                {
                    "book": l.book,
                    "market": l.market,
                    "selection": l.selection,
                    "american": l.american,
                    "line": l.line,
                }
                for l in g.lines[:5]
            ],
        }

    if dump and games:
        path = os.path.join("validation_dumps", f"{name}_{sport}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([g.to_dict() for g in games], fh, indent=2, default=str)

    return {
        "source": name,
        "sport": sport,
        "ok": len(games) > 0,
        "games": len(games),
        "ml_lines": ml_lines,
        "total_lines": total_lines,
        "elapsed_s": round(elapsed, 2),
        "sample": sample,
        "error": None,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        default=",".join(ALL_FETCHERS.keys()),
        help="Comma-separated source names (default: all)",
    )
    parser.add_argument(
        "--sport",
        default=None,
        help="Restrict to a single sport key (default: all supported)",
    )
    parser.add_argument("--dump", action="store_true", help="Save successful payloads to validation_dumps/")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    settings = get_settings()
    wanted_sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    sports = [args.sport] if args.sport else DEFAULT_SPORTS

    results: List[Dict[str, Any]] = []
    for src_name in wanted_sources:
        fetcher_cls = ALL_FETCHERS.get(src_name)
        if fetcher_cls is None:
            print(f"{YELLOW}skip{RESET} {src_name}: unknown source")
            continue
        fetcher = fetcher_cls(settings)
        print(f"\n{DIM}== {src_name} =={RESET}")
        for sport in sports:
            if sport not in fetcher.supported_sports:
                continue
            result = run_one(src_name, fetcher, sport, dump=args.dump)
            results.append(result)
            if result["ok"]:
                sample = result["sample"] or {}
                print(
                    f"  {_mark(True)}  {sport:18s}  "
                    f"games={result['games']:3d}  "
                    f"ML={result['ml_lines']:3d}  "
                    f"lines={result['total_lines']:4d}  "
                    f"{result['elapsed_s']:5.2f}s  "
                    f"eg. {sample.get('away_team','?')} @ {sample.get('home_team','?')}"
                )
            else:
                err = result["error"] or "no games returned"
                print(
                    f"  {_mark(False)}  {sport:18s}  "
                    f"{result['elapsed_s']:5.2f}s  "
                    f"{RED}{err[:120]}{RESET}"
                )

    # Summary
    print("\n" + "=" * 72)
    by_source: Dict[str, Dict[str, int]] = {}
    for r in results:
        d = by_source.setdefault(r["source"], {"pass": 0, "fail": 0, "games": 0})
        d["pass" if r["ok"] else "fail"] += 1
        d["games"] += r["games"]

    any_source_ok = False
    for src, d in sorted(by_source.items()):
        passed = d["pass"]
        failed = d["fail"]
        status = _mark(passed > 0)
        print(
            f"{status}  {src:15s}  sports_pass={passed}  sports_fail={failed}  "
            f"total_games={d['games']}"
        )
        if passed > 0:
            any_source_ok = True

    return 0 if any_source_ok else 1


if __name__ == "__main__":
    sys.exit(main())
