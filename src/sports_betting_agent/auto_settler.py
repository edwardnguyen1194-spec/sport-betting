"""Auto-settlement of paper bets using ESPN scores.

Checks ESPN's scoreboard API for final scores, then settles
open bets automatically. Runs as part of the auto-trade loop.

Settlement logic:
- Moneyline: did the selected team win?
- Spread: did the selected team cover the spread?
- Total: did the combined score go over/under the line?
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import requests

from .paper_trader import PaperTrader, Bet
from .models_schema import _norm


logger = logging.getLogger(__name__)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

SPORT_ESPN_MAP = {
    "baseball_ncaa": "baseball/college-baseball",
    "baseball_mlb": "baseball/mlb",
    "basketball_nba": "basketball/nba",
    "basketball_ncaab": "basketball/college-basketball",
    "football_nfl": "football/nfl",
    "football_ncaaf": "football/college-football",
    "hockey_nhl": "hockey/nhl",
}


class AutoSettler:
    """Automatically settle open bets by checking ESPN final scores."""

    def __init__(self, paper: PaperTrader):
        self.paper = paper
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/537.36"
        )

    def settle_completed_bets(self) -> List[Dict]:
        """Check all open bets and settle any whose games are final.

        Only settles bets on games that started 4+ hours ago to ensure
        the game is truly final (avoids premature settlement).
        """
        settled = []
        open_bets = list(self.paper.open_bets.values())

        if not open_bets:
            return settled

        # Only try to settle bets on games that started 4+ hours ago
        now = datetime.now(timezone.utc)
        eligible = []
        for bet in open_bets:
            placed = datetime.fromisoformat(bet.placed_at)
            hours_since = (now - placed).total_seconds() / 3600
            if hours_since >= 4:
                eligible.append(bet)

        if not eligible:
            return settled
        open_bets = eligible

        # Group bets by sport to minimize API calls
        by_sport: Dict[str, List[Bet]] = {}
        for bet in open_bets:
            by_sport.setdefault(bet.sport, []).append(bet)

        for sport, bets in by_sport.items():
            espn_path = SPORT_ESPN_MAP.get(sport)
            if not espn_path:
                continue

            try:
                scores = self._fetch_scores(espn_path)
            except Exception as exc:
                logger.warning("auto_settler: failed to fetch %s scores: %s", sport, exc)
                continue

            for bet in bets:
                result = self._check_bet(bet, scores)
                if result is not None:
                    outcome, detail = result
                    settled_bet = self.paper.settle_bet(bet.id, outcome)
                    if settled_bet:
                        settled.append({
                            "bet_id": bet.id,
                            "selection": bet.selection,
                            "outcome": outcome,
                            "detail": detail,
                            "profit": settled_bet.result,
                        })
                        logger.info(
                            "[SETTLE] %s %s on %s: %s (P&L=$%.2f)",
                            outcome.upper(),
                            bet.strategy,
                            bet.selection,
                            detail,
                            settled_bet.result or 0,
                        )

        return settled

    def _fetch_scores(self, espn_path: str) -> List[Dict]:
        """Fetch today's and yesterday's scores from ESPN."""
        games = []
        now = datetime.now(timezone.utc)

        for days_ago in (0, 1):
            date = now - timedelta(days=days_ago)
            date_str = date.strftime("%Y%m%d")
            url = f"{ESPN_BASE}/{espn_path}/scoreboard?dates={date_str}"
            try:
                resp = self._session.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                for event in data.get("events", []):
                    game = self._parse_event(event)
                    if game:
                        games.append(game)
            except Exception as exc:
                logger.debug("auto_settler: %s %s: %s", espn_path, date_str, exc)

        return games

    @staticmethod
    def _parse_event(event: Dict) -> Optional[Dict]:
        """Parse an ESPN event into a simple score dict."""
        competition = (event.get("competitions") or [{}])[0]
        status = competition.get("status", {}).get("type", {})

        # Only settle final games
        if not status.get("completed", False):
            return None

        competitors = competition.get("competitors", [])
        if len(competitors) < 2:
            return None

        teams = {}
        for comp in competitors:
            team_data = comp.get("team", {})
            name = team_data.get("displayName") or team_data.get("shortDisplayName") or ""
            # Null-score guard (Agent-3 finding): ESPN occasionally
            # returns null for ``score`` on games in weird states.
            # Coercing to 0 silently graded those as losses. Treat as
            # not-yet-final and retry on the next cycle.
            raw_score = comp.get("score")
            if raw_score is None or raw_score == "":
                return None
            try:
                score = int(raw_score)
            except (TypeError, ValueError):
                return None
            home_away = comp.get("homeAway", "")
            teams[home_away] = {"name": name, "score": score}

        home = teams.get("home", {})
        away = teams.get("away", {})

        if not home.get("name") or not away.get("name"):
            return None

        return {
            "home_team": home["name"],
            "away_team": away["name"],
            "home_score": home["score"],
            "away_score": away["score"],
            "final": True,
        }

    def _check_bet(self, bet: Bet, scores: List[Dict]) -> Optional[tuple]:
        """Check if a bet's game is final and determine outcome.

        Returns (outcome, detail) or None if game not found/not final.
        """
        # Find the matching game
        game = None
        for g in scores:
            if self._teams_match(bet.home_team, g["home_team"]) and \
               self._teams_match(bet.away_team, g["away_team"]):
                game = g
                break

        if game is None:
            return None

        home_score = game["home_score"]
        away_score = game["away_score"]
        detail = f"{game['away_team']} {away_score} - {game['home_team']} {home_score}"

        if bet.market == "moneyline":
            return self._settle_moneyline(bet, home_score, away_score, detail)
        elif bet.market == "spread":
            return self._settle_spread(bet, home_score, away_score, detail)
        elif bet.market == "total":
            return self._settle_total(bet, home_score, away_score, detail)

        return None

    def _settle_moneyline(self, bet: Bet, home_score: int, away_score: int, detail: str):
        """Settle a moneyline bet."""
        if home_score == away_score:
            return ("push", detail)

        home_won = home_score > away_score
        bet_on_home = self._teams_match(bet.selection, bet.home_team)
        bet_on_away = self._teams_match(bet.selection, bet.away_team)

        if bet_on_home and home_won:
            return ("won", detail)
        elif bet_on_away and not home_won:
            return ("won", detail)
        elif bet_on_home or bet_on_away:
            return ("lost", detail)

        return None

    def _settle_spread(self, bet: Bet, home_score: int, away_score: int, detail: str):
        """Settle a spread bet."""
        if bet.line is None:
            return None

        bet_on_home = self._teams_match(bet.selection, bet.home_team)
        margin = home_score - away_score  # positive = home won

        if bet_on_home:
            adjusted = margin + bet.line  # e.g., home -3: margin + (-3)
        else:
            adjusted = (away_score - home_score) + bet.line

        if adjusted > 0:
            return ("won", f"{detail} (cover by {abs(adjusted):.1f})")
        elif adjusted < 0:
            return ("lost", f"{detail} (miss by {abs(adjusted):.1f})")
        else:
            return ("push", detail)

    def _settle_total(self, bet: Bet, home_score: int, away_score: int, detail: str):
        """Settle a total (over/under) bet."""
        if bet.line is None:
            return None

        total = home_score + away_score

        if bet.selection.lower() == "over":
            if total > bet.line:
                return ("won", f"{detail} (total {total}, line {bet.line})")
            elif total < bet.line:
                return ("lost", f"{detail} (total {total}, line {bet.line})")
            else:
                return ("push", f"{detail} (total {total}, push)")
        else:  # under
            if total < bet.line:
                return ("won", f"{detail} (total {total}, line {bet.line})")
            elif total > bet.line:
                return ("lost", f"{detail} (total {total}, line {bet.line})")
            else:
                return ("push", f"{detail} (total {total}, push)")

    @staticmethod
    def _teams_match(name_a: str, name_b: str) -> bool:
        """Fuzzy team name matching."""
        return _norm(name_a) == _norm(name_b)
