"""24/7 Sports News Reader.

Scrapes ESPN headlines and injury reports every trade cycle.
The agent uses this data to:
1. Skip games with key player injuries
2. Boost confidence when news confirms an edge
3. Display breaking news on the dashboard

Sources: ESPN headlines API (free, no key needed)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

ESPN_NEWS = "https://site.api.espn.com/apis/site/v2/sports"
ESPN_HEADLINES = "https://site.api.espn.com/apis/site/v2/sports/{path}/news"

SPORT_ESPN = {
    "baseball_mlb": "baseball/mlb",
    "baseball_ncaa": "baseball/college-baseball",
    "basketball_nba": "basketball/nba",
    "basketball_ncaab": "basketball/college-basketball",
    "football_nfl": "football/nfl",
    "football_ncaaf": "football/college-football",
    "hockey_nhl": "hockey/nhl",
}


class NewsReader:
    """Read sports news and injury reports from ESPN."""

    def __init__(self, data_dir: str = "/data/sba"):
        self.data_dir = data_dir
        self.headlines: List[Dict] = []
        self.injuries: Dict[str, List[Dict]] = {}
        self.last_fetch: Optional[datetime] = None
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/537.36"
        )
        self._load()

    def _path(self) -> str:
        return os.path.join(self.data_dir, "news_cache.json")

    def _load(self):
        path = self._path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                self.headlines = data.get("headlines", [])
                self.injuries = data.get("injuries", {})
                ts = data.get("last_fetch")
                if ts:
                    self.last_fetch = datetime.fromisoformat(ts)
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self._path()), exist_ok=True)
        with open(self._path(), "w") as f:
            json.dump({
                "headlines": self.headlines[-50:],
                "injuries": self.injuries,
                "last_fetch": self.last_fetch.isoformat() if self.last_fetch else None,
            }, f, indent=2, ensure_ascii=False)

    def fetch_all(self):
        """Fetch news and injuries for all sports. Called each trade cycle."""
        now = datetime.now(timezone.utc)

        # Don't fetch more than once per 10 minutes
        if self.last_fetch and (now - self.last_fetch).total_seconds() < 600:
            return

        all_headlines = []

        for sport, path in SPORT_ESPN.items():
            try:
                headlines = self._fetch_headlines(path, sport)
                all_headlines.extend(headlines)
            except Exception as exc:
                logger.debug("news_reader: %s headlines failed: %s", sport, exc)

            try:
                injuries = self._fetch_injuries(path, sport)
                if injuries:
                    self.injuries[sport] = injuries
            except Exception as exc:
                logger.debug("news_reader: %s injuries failed: %s", sport, exc)

        # Keep newest 50 headlines, deduplicate by title
        seen = set()
        unique = []
        for h in all_headlines + self.headlines:
            if h["title"] not in seen:
                seen.add(h["title"])
                unique.append(h)
        self.headlines = unique[:50]
        self.last_fetch = now
        self._save()
        logger.info("news_reader: fetched %d headlines, %d sport injury reports",
                     len(all_headlines), len(self.injuries))

    def _fetch_headlines(self, espn_path: str, sport: str) -> List[Dict]:
        """Fetch latest headlines from ESPN."""
        url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_path}/news"
        try:
            resp = self._session.get(url, timeout=8, params={"limit": 10})
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        headlines = []
        for article in data.get("articles", []):
            headline = {
                "title": article.get("headline", ""),
                "description": article.get("description", ""),
                "sport": sport,
                "published": article.get("published", ""),
                "type": article.get("type", ""),
            }
            if headline["title"]:
                headlines.append(headline)
        return headlines

    def _fetch_injuries(self, espn_path: str, sport: str) -> List[Dict]:
        """Fetch injury reports from ESPN scoreboard."""
        # ESPN scoreboard sometimes includes injury data in competitions
        url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_path}/scoreboard"
        try:
            resp = self._session.get(url, timeout=8)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        injuries = []
        for event in data.get("events", []):
            for comp in event.get("competitions", []):
                for team in comp.get("competitors", []):
                    team_name = team.get("team", {}).get("displayName", "")
                    # Check for injury report in team data
                    for player in team.get("leaders", []):
                        # ESPN sometimes flags injured players
                        pass
                    # Check status notes
                    situation = comp.get("situation", {})
                    if situation:
                        note = situation.get("lastPlay", {}).get("text", "")
                        if "injur" in note.lower():
                            injuries.append({
                                "team": team_name,
                                "note": note,
                                "sport": sport,
                            })
        return injuries

    def recent_headlines(self, sport: Optional[str] = None, n: int = 10) -> List[Dict]:
        """Get recent headlines, optionally filtered by sport."""
        if sport:
            filtered = [h for h in self.headlines if h["sport"] == sport]
        else:
            filtered = self.headlines
        return filtered[:n]

    def get_injuries(self, sport: str) -> List[Dict]:
        """Get injury reports for a sport."""
        return self.injuries.get(sport, [])

    def has_injury_alert(self, team_name: str) -> bool:
        """Check if a team has any injury alerts."""
        team_lower = team_name.lower()
        for sport_injuries in self.injuries.values():
            for inj in sport_injuries:
                if team_lower in inj.get("team", "").lower():
                    return True
        return False

    def dashboard_data(self) -> Dict:
        """Data for the dashboard display."""
        return {
            "headlines": self.headlines[:15],
            "last_fetch": self.last_fetch.isoformat() if self.last_fetch else None,
            "injury_count": sum(len(v) for v in self.injuries.values()),
        }
