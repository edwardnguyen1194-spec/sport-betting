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

    # ------------------------------------------------------------------
    # Injury-aware confidence penalty
    # ------------------------------------------------------------------
    #
    # Scans cached headlines for injury keywords co-located with a team
    # name. If a headline mentions both the team and one of the injury
    # phrases below, we treat it as a real signal that our bet on this
    # team's game carries extra risk and shave confidence accordingly.
    #
    # Phrases are matched as whole words/phrases to avoid false positives
    # (e.g. "out" should not match "outstanding"). Multi-word phrases
    # are matched as substrings since they're already specific enough.
    INJURY_TERMS = (
        "ruled out",
        "day-to-day",
        "day to day",
        "placed on il",
        "questionable",
        "doubtful",
        "scratched",
        "inactive",
        "injured",
        "out",
        "il",
    )

    # Per-headline penalty when team + injury term both present.
    _PER_HEADLINE_PENALTY = 0.05
    # Hard cap so a news storm doesn't zero out a bet's confidence.
    _MAX_PENALTY = 0.15

    def penalty_for_game(self, home_team: str, away_team: str) -> tuple:
        """Return (penalty_pct, reason) for a game based on injury news.

        penalty_pct is in [0, 0.15]. Reason is a human-readable string
        (empty if no penalty). A headline must contain the team name AND
        an injury keyword in the same headline to count. Matches against
        both headline title and description when available.
        """
        import re

        teams = [(home_team or "").strip(), (away_team or "").strip()]
        teams = [t for t in teams if t]
        if not teams:
            return (0.0, "")

        matched_reasons: List[str] = []
        seen_titles: set = set()

        # Compile team-name patterns. We look for the last "word" of the
        # team name too (e.g. "LAFC" or "Lakers") because ESPN headlines
        # often drop the city. We use word-boundary matching so "LA" in
        # "LAFC" doesn't false-match a headline about "LA traffic".
        team_patterns = []
        for t in teams:
            parts = [p for p in t.split() if len(p) >= 3]
            variants = {t.lower()}
            if parts:
                variants.add(parts[-1].lower())  # e.g. "Lakers" from "LA Lakers"
            for v in variants:
                # Escape for regex, wrap in word boundaries.
                team_patterns.append((t, re.compile(r"\b" + re.escape(v) + r"\b", re.IGNORECASE)))

        for h in self.headlines:
            title = (h.get("title") or "").strip()
            desc = (h.get("description") or "").strip()
            if not title:
                continue
            if title in seen_titles:
                continue
            combined = f"{title} {desc}".lower()

            # Find a team match first — cheaper than scanning all injury
            # terms on every headline.
            matched_team = None
            for team_name, pat in team_patterns:
                if pat.search(title) or pat.search(desc):
                    matched_team = team_name
                    break
            if not matched_team:
                continue

            # Now check injury terms. Use word boundaries for short terms
            # like "out" / "il" so they don't false-match inside words.
            matched_term = None
            for term in self.INJURY_TERMS:
                if " " in term or "-" in term:
                    # Multi-word phrase — substring match is specific enough.
                    if term in combined:
                        matched_term = term
                        break
                else:
                    # Single short word — require word boundaries.
                    if re.search(r"\b" + re.escape(term) + r"\b", combined):
                        matched_term = term
                        break
            if not matched_term:
                continue

            seen_titles.add(title)
            matched_reasons.append(f"{matched_team}: '{title}' ({matched_term})")

        if not matched_reasons:
            return (0.0, "")

        raw_penalty = self._PER_HEADLINE_PENALTY * len(matched_reasons)
        penalty = min(self._MAX_PENALTY, raw_penalty)
        # Keep the reason short — we append it to rec.reasoning which
        # flows into the dashboard and chat summary.
        preview = "; ".join(matched_reasons[:2])
        extra = "" if len(matched_reasons) <= 2 else f" (+{len(matched_reasons) - 2} more)"
        reason = f"injury_penalty={penalty:.0%} from news: {preview}{extra}"
        return (penalty, reason)

    def dashboard_data(self) -> Dict:
        """Data for the dashboard display."""
        return {
            "headlines": self.headlines[:15],
            "last_fetch": self.last_fetch.isoformat() if self.last_fetch else None,
            "injury_count": sum(len(v) for v in self.injuries.values()),
        }
