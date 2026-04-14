"""Elo Power Ratings built from ESPN game results.

Uses the FiveThirtyEight Elo methodology to rate every team.
The agent uses these ratings to generate its OWN win probabilities
independent of the market — this is what separates good bettors
from great ones.

Ratings are built from ESPN scoreboard results and persist to disk.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

SPORT_ESPN = {
    "baseball_mlb": "baseball/mlb",
    "basketball_nba": "basketball/nba",
    "hockey_nhl": "hockey/nhl",
    "football_nfl": "football/nfl",
    "baseball_ncaa": "baseball/college-baseball",
    "basketball_ncaab": "basketball/college-basketball",
    "football_ncaaf": "football/college-football",
}

# Calibrated per sport
SPORT_CONFIG = {
    "baseball_mlb": {"k": 4, "hfa": 24, "default": 1500, "regression": 0.33},
    "basketball_nba": {"k": 20, "hfa": 100, "default": 1500, "regression": 0.25},
    "hockey_nhl": {"k": 6, "hfa": 33, "default": 1500, "regression": 0.33},
    "football_nfl": {"k": 20, "hfa": 48, "default": 1505, "regression": 0.33},
    "baseball_ncaa": {"k": 8, "hfa": 30, "default": 1500, "regression": 0.40},
    "basketball_ncaab": {"k": 25, "hfa": 65, "default": 1500, "regression": 0.40},
    "football_ncaaf": {"k": 25, "hfa": 55, "default": 1500, "regression": 0.40},
}


class EloRatings:
    """Elo rating system for all sports."""

    def __init__(self, data_dir: str = "/data/sba", scoring_tracker=None):
        self.data_dir = data_dir
        self.ratings: Dict[str, Dict[str, float]] = {}
        self.games_processed: int = 0
        self._processed_ids: set = set()  # Track processed game IDs to avoid duplicates
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "Mozilla/5.0"
        # Optional: a TeamScoringTracker that gets fed each completed game
        # alongside the Elo update. Kept optional so unit tests can skip it.
        self.scoring_tracker = scoring_tracker
        self._load()

    def _path(self) -> str:
        return os.path.join(self.data_dir, "elo_ratings.json")

    def _load(self):
        path = self._path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                self.ratings = data.get("ratings", {})
                self.games_processed = data.get("games_processed", 0)
                self._processed_ids = set(data.get("processed_ids", []))
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self._path()), exist_ok=True)
        with open(self._path(), "w") as f:
            # Keep last 5000 processed IDs to prevent unbounded growth
            recent_ids = list(self._processed_ids)[-5000:]
            json.dump({
                "ratings": self.ratings,
                "games_processed": self.games_processed,
                "processed_ids": recent_ids,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

    def update_from_espn(self):
        """Fetch recent results from ESPN and update ratings."""
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        new_games = 0

        for sport, espn_path in SPORT_ESPN.items():
            cfg = SPORT_CONFIG.get(sport, SPORT_CONFIG["baseball_mlb"])

            # Check last 14 days for maximum data collection
            for days_ago in range(14):
                date = now - timedelta(days=days_ago)
                date_str = date.strftime("%Y%m%d")
                url = f"{ESPN_BASE}/{espn_path}/scoreboard?dates={date_str}"

                try:
                    resp = self._session.get(url, timeout=4)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception:
                    continue

                for event in data.get("events", []):
                    event_id = str(event.get("id", ""))
                    elo_already_seen = event_id in self._processed_ids

                    comp = (event.get("competitions") or [{}])[0]
                    status = comp.get("status", {}).get("type", {})
                    if not status.get("completed", False):
                        continue

                    competitors = comp.get("competitors", [])
                    if len(competitors) < 2:
                        continue

                    home = away = None
                    for c in competitors:
                        team_name = c.get("team", {}).get("displayName", "")
                        score = int(c.get("score", 0))
                        ha = c.get("homeAway", "")
                        if ha == "home":
                            home = {"name": team_name, "score": score}
                        else:
                            away = {"name": team_name, "score": score}

                    if not home or not away or not home["name"] or not away["name"]:
                        continue

                    # Scoring tracker has its own dedup (processed_ids on its
                    # side). We always offer the game to it — that's how the
                    # tracker backfills history on first run, without re-Eloing
                    # games the rating system already processed.
                    if self.scoring_tracker is not None:
                        try:
                            self.scoring_tracker.record_game(
                                sport=sport,
                                event_id=event_id,
                                home=home["name"],
                                away=away["name"],
                                home_score=home["score"],
                                away_score=away["score"],
                            )
                        except Exception as exc:
                            logger.warning("scoring_tracker record_game failed: %s", exc)

                    if elo_already_seen:
                        continue

                    # Update Elo
                    self._update_game(
                        sport, home["name"], away["name"],
                        home["score"], away["score"], cfg
                    )
                    self._processed_ids.add(event_id)
                    new_games += 1

        if new_games > 0:
            self.games_processed += new_games
            self._save()
            logger.info("elo: updated %d games, total %d processed", new_games, self.games_processed)
        # Always save the scoring tracker — on first runs after deploy the
        # tracker is backfilling games Elo already processed, so there
        # are no "new" games for Elo but plenty for the tracker.
        if self.scoring_tracker is not None:
            try:
                self.scoring_tracker.save()
            except Exception as exc:
                logger.warning("scoring_tracker save failed: %s", exc)

    def _update_game(self, sport: str, home: str, away: str,
                     home_score: int, away_score: int, cfg: Dict):
        """Update Elo ratings for a single completed game."""
        if sport not in self.ratings:
            self.ratings[sport] = {}

        h_elo = self.ratings[sport].get(home, cfg["default"])
        a_elo = self.ratings[sport].get(away, cfg["default"])

        margin = home_score - away_score
        result = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)

        # Expected score with home field advantage
        expected = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo - cfg["hfa"]) / 400.0))

        # Margin of victory multiplier with autocorrelation correction
        elo_diff = h_elo + cfg["hfa"] - a_elo
        if abs(margin) > 0:
            mov_mult = math.log(abs(margin) + 1) * (2.2 / (2.2 + 0.001 * abs(elo_diff)))
        else:
            mov_mult = 1.0

        k = cfg["k"] * mov_mult
        self.ratings[sport][home] = h_elo + k * (result - expected)
        self.ratings[sport][away] = a_elo + k * ((1 - result) - (1 - expected))

    def predict_win_prob(self, sport: str, home: str, away: str) -> Optional[float]:
        """Predict home team win probability."""
        if sport not in self.ratings:
            return None
        cfg = SPORT_CONFIG.get(sport, SPORT_CONFIG["baseball_mlb"])
        h_elo = self.ratings[sport].get(home)
        a_elo = self.ratings[sport].get(away)
        if h_elo is None or a_elo is None:
            return None
        return 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo - cfg["hfa"]) / 400.0))

    def get_rating(self, sport: str, team: str) -> Optional[float]:
        """Get a team's current Elo rating."""
        return self.ratings.get(sport, {}).get(team)

    def top_teams(self, sport: str, n: int = 10) -> list:
        """Get top N teams by Elo rating."""
        teams = self.ratings.get(sport, {})
        sorted_teams = sorted(teams.items(), key=lambda x: x[1], reverse=True)
        return sorted_teams[:n]

    def summary(self) -> Dict:
        """Summary for dashboard."""
        return {
            "total_games": self.games_processed,
            "sports_tracked": list(self.ratings.keys()),
            "teams_rated": sum(len(v) for v in self.ratings.values()),
        }
