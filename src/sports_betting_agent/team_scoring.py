"""Rolling team scoring tracker.

Complements the Elo ratings system: same ESPN completed-game feed,
but instead of a single strength rating per team we keep a short
history of ``scored``/``allowed`` values per side (home vs away).
This is the raw material for a model-based total-points projection.

Why a separate file
-------------------
Elo tells us *which* team is stronger, not *how many* points the
game is likely to produce. Serious total bettors maintain their own
pace/scoring models and compare them to the posted total — which is
exactly what the new TotalProjectionStrategy does with this data.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


WINDOW = 20            # games retained per team/venue
MIN_SAMPLES = 3        # below this we refuse to project a total
# Was 5 — too high for NCAA/MLS teams with sparse ESPN history.
# The confidence cap (62%) in TotalProjectionStrategy already
# guards against overconfidence on thin data.


class TeamScoringTracker:
    """Keep rolling scored/allowed averages per team, split by venue.

    ``record_game`` is idempotent — a given ESPN event id is only
    applied once. Persisted to ``team_scoring.json`` in the data dir.
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "team_scoring.json")
        # {sport: {team: {"home_scored": [...], "home_allowed": [...],
        #                  "away_scored": [...], "away_allowed": [...]}}}
        self.data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        self.processed_ids: set = set()
        self._load()

    # -- persistence -------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.data = raw.get("data", {})
            self.processed_ids = set(raw.get("processed_ids", []))
        except Exception as exc:
            logger.warning("TeamScoringTracker load failed: %s", exc)
            self.data = {}
            self.processed_ids = set()

    def _save(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        payload = {
            "data": self.data,
            # Bound the id set so the file doesn't grow forever.
            "processed_ids": list(self.processed_ids)[-20_000:],
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, self.path)

    # -- ingestion ---------------------------------------------------

    def record_game(
        self,
        sport: str,
        event_id: str,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
    ) -> bool:
        """Record one completed game. Returns True if applied, False if dup."""
        if event_id in self.processed_ids:
            return False
        sport_map = self.data.setdefault(sport, {})
        home_rec = sport_map.setdefault(
            home,
            {"home_scored": [], "home_allowed": [], "away_scored": [], "away_allowed": []},
        )
        away_rec = sport_map.setdefault(
            away,
            {"home_scored": [], "home_allowed": [], "away_scored": [], "away_allowed": []},
        )
        home_rec["home_scored"].append(float(home_score))
        home_rec["home_allowed"].append(float(away_score))
        away_rec["away_scored"].append(float(away_score))
        away_rec["away_allowed"].append(float(home_score))
        # Trim each list to WINDOW.
        for rec in (home_rec, away_rec):
            for key, lst in rec.items():
                if len(lst) > WINDOW:
                    del lst[:-WINDOW]
        self.processed_ids.add(event_id)
        return True

    def save(self) -> None:
        self._save()

    # -- queries -----------------------------------------------------

    def team(self, sport: str, team: str) -> Optional[Dict[str, List[float]]]:
        sport_data = self.data.get(sport, {})
        # Exact match first.
        exact = sport_data.get(team)
        if exact is not None:
            return exact
        # Fuzzy fallback: Bovada uses "Stanford" but ESPN stores
        # "Stanford Cardinal". Match when the query is a prefix of a
        # stored name, or the stored name starts with the query.
        needle = team.lower().strip()
        for stored_name, rec in sport_data.items():
            sn = stored_name.lower()
            if sn.startswith(needle) or needle.startswith(sn):
                return rec
        return None

    @staticmethod
    def _avg(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    def projected_total(
        self, sport: str, home: str, away: str
    ) -> Optional[Dict[str, float]]:
        """Project the expected total for ``home`` vs ``away``.

        Uses:
          * home-team's home scoring rate + away-team's away scoring rate
          * home-team's home runs allowed + away-team's away runs allowed

        Averaged the two viewpoints for a single number. Returns None
        when either side has fewer than MIN_SAMPLES relevant games.
        """
        h = self.team(sport, home)
        a = self.team(sport, away)
        if not h or not a:
            return None

        h_scored = h["home_scored"]
        h_allowed = h["home_allowed"]
        a_scored = a["away_scored"]
        a_allowed = a["away_allowed"]
        if (
            len(h_scored) < MIN_SAMPLES
            or len(h_allowed) < MIN_SAMPLES
            or len(a_scored) < MIN_SAMPLES
            or len(a_allowed) < MIN_SAMPLES
        ):
            return None

        # Viewpoint 1: home offense vs away defense, away offense vs home defense.
        proj_home = (self._avg(h_scored) + self._avg(a_allowed)) / 2.0
        proj_away = (self._avg(a_scored) + self._avg(h_allowed)) / 2.0
        projected = proj_home + proj_away

        # Samples used and per-team averages returned for transparency.
        return {
            "projected_total": round(projected, 2),
            "projected_home_score": round(proj_home, 2),
            "projected_away_score": round(proj_away, 2),
            "home_games_tracked": min(len(h_scored), len(h_allowed)),
            "away_games_tracked": min(len(a_scored), len(a_allowed)),
        }

    def summary(self) -> Dict:
        by_sport = {}
        for sport, teams in self.data.items():
            total_games = sum(
                len(rec["home_scored"]) + len(rec["away_scored"])
                for rec in teams.values()
            )
            by_sport[sport] = {
                "teams": len(teams),
                "game_records": total_games,
            }
        return {
            "sports": by_sport,
            "total_processed_ids": len(self.processed_ids),
        }
