"""Daily self-improvement loop.

World-class sharp bettors don't just place bets — they keep a log,
adjust thresholds based on CLV and win-rate, and read sharp content
constantly to stay ahead of market shifts. This module gives the
agent that same discipline:

1. Once per calendar day (UTC), the run() method kicks in.
2. It reads the paper trader's closed bets + CLV tracker + line-
   movement store and computes performance by strategy / sport /
   market / book.
3. It auto-tunes ``spread_value_min_edge`` and ``total_value_min_edge``
   — if recent CLV is clearly negative, the edge bar rises; if CLV
   is positive, the bar can safely drop a touch to catch more +EV
   bets.
4. It fetches a sharp-betting article of the day (Pinnacle Betting
   Resources RSS is the canonical free source) and logs the
   headline + first paragraph as a "skill learned" entry.
5. Everything is persisted to ``daily_learning.json`` so the
   dashboard can expose today's insight and the long-form history.

Failure modes
-------------
Every external fetch is wrapped in try/except; the learner never
takes the agent down, and a cycle that fails to fetch still records
the internal performance review. Scheduling is idempotent — calling
run() many times per day results in exactly one recorded pass.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterable, List, Optional
from urllib.request import Request, urlopen

from .clv_tracker import CLVTracker
from .config import Settings
from .line_movement import LineMovementStore


logger = logging.getLogger(__name__)


# A small rotating pool of high-quality free sources the agent can read
# to pick up new technique. The learner tries each in turn until one
# succeeds (network on Fly.io can be flaky to specific CDNs).
LEARNING_FEEDS: List[str] = [
    "https://www.pinnacle.com/en/betting-resources/rss",
    "https://www.boydsbets.com/feed/",
    "https://www.sportsbookreview.com/picks/feed/",
]

# Minimum recent samples before auto-tuning edge thresholds. Below
# this we only record the review; we don't touch the knobs.
MIN_SAMPLES_FOR_TUNE = 20

# Hard caps — never tune edge thresholds outside of [2%, 8%].
EDGE_FLOOR = 0.02
EDGE_CEIL = 0.08


@dataclass
class DailyEntry:
    """One day's worth of self-improvement output."""

    date: str                                 # YYYY-MM-DD (UTC)
    timestamp: str
    review: Dict                              # stats by strategy / sport / market / book
    avg_clv: Optional[float]
    win_rate: Optional[float]
    tuned: Dict                               # threshold changes applied this cycle
    skill_title: str = ""
    skill_summary: str = ""
    skill_source: str = ""
    notes: List[str] = field(default_factory=list)


class DailyLearner:
    """Runs once per UTC day. Self-tunes + logs new skills."""

    def __init__(
        self,
        settings: Settings,
        clv: CLVTracker,
        line_store: LineMovementStore,
    ) -> None:
        self.settings = settings
        self.clv = clv
        self.line_store = line_store
        self.data_dir = settings.data_dir
        self.entries: List[DailyEntry] = []
        self._load()

    # -- persistence -------------------------------------------------

    def _path(self) -> str:
        return os.path.join(self.data_dir, "daily_learning.json")

    def _load(self) -> None:
        path = self._path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.entries = [DailyEntry(**e) for e in raw]
        except Exception as exc:
            logger.warning("DailyLearner load failed: %s", exc)
            self.entries = []

    def _save(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        path = self._path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump([asdict(e) for e in self.entries], fh, indent=2)
        os.replace(tmp, path)

    # -- entry points ------------------------------------------------

    def already_ran_today(self) -> bool:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return any(e.date == today for e in self.entries)

    def run(self, closed_bets: Iterable, force: bool = False) -> Optional[DailyEntry]:
        """Execute one daily cycle. Idempotent unless ``force`` is set."""
        if not force and self.already_ran_today():
            return None

        now = datetime.now(timezone.utc)
        review = self._performance_review(list(closed_bets))
        clv_stats = self.clv.stats()
        tuned = self._auto_tune(review, clv_stats)
        skill_title, skill_summary, skill_source = self._fetch_daily_skill()

        notes = self._compose_notes(review, clv_stats, tuned)

        entry = DailyEntry(
            date=now.strftime("%Y-%m-%d"),
            timestamp=now.isoformat(),
            review=review,
            avg_clv=clv_stats.get("average_clv"),
            win_rate=clv_stats.get("win_rate"),
            tuned=tuned,
            skill_title=skill_title,
            skill_summary=skill_summary,
            skill_source=skill_source,
            notes=notes,
        )
        self.entries.append(entry)
        # Keep bounded so the JSON file stays a reasonable size.
        if len(self.entries) > 365:
            self.entries = self.entries[-365:]
        self._save()
        return entry

    # -- review ------------------------------------------------------

    def _performance_review(self, closed_bets: List) -> Dict:
        """Break performance down by strategy / sport / market / book."""
        by_strategy: Dict[str, Dict[str, int]] = defaultdict(lambda: {"won": 0, "lost": 0, "push": 0, "void": 0})
        by_sport: Dict[str, Dict[str, int]] = defaultdict(lambda: {"won": 0, "lost": 0})
        by_market: Dict[str, Dict[str, int]] = defaultdict(lambda: {"won": 0, "lost": 0})
        by_book: Dict[str, Dict[str, int]] = defaultdict(lambda: {"won": 0, "lost": 0})

        for bet in closed_bets:
            status = getattr(bet, "status", "open")
            strat = getattr(bet, "strategy", "?")
            sport = getattr(bet, "sport", "?")
            market = getattr(bet, "market", "?")
            book = getattr(bet, "book", "?")
            if status not in ("won", "lost", "push", "void"):
                continue
            by_strategy[strat][status] += 1
            if status in ("won", "lost"):
                by_sport[sport][status] += 1
                by_market[market][status] += 1
                by_book[book][status] += 1

        def _with_rate(d: Dict[str, Dict[str, int]]) -> Dict[str, Dict]:
            out: Dict[str, Dict] = {}
            for key, stats in d.items():
                won = stats.get("won", 0)
                lost = stats.get("lost", 0)
                total = won + lost
                out[key] = {
                    **stats,
                    "total_settled": total,
                    "win_rate": round(won / total * 100, 1) if total else None,
                }
            return out

        return {
            "by_strategy": _with_rate(by_strategy),
            "by_sport": _with_rate(by_sport),
            "by_market": _with_rate(by_market),
            "by_book": _with_rate(by_book),
        }

    # -- auto-tune ---------------------------------------------------

    def _auto_tune(self, review: Dict, clv_stats: Dict) -> Dict:
        """Nudge edge thresholds based on recent CLV + sample size.

        If average CLV is clearly negative we are overpaying vs. the
        market and should be stricter. If CLV is clearly positive we
        can afford to take slightly thinner edges. All changes are
        capped to a single 0.5pp nudge per day.
        """
        tuned: Dict[str, Dict] = {}
        avg_clv = clv_stats.get("average_clv")  # already in %
        samples = clv_stats.get("bets_with_clv", 0) or 0
        if avg_clv is None or samples < MIN_SAMPLES_FOR_TUNE:
            return {"skipped_reason": f"insufficient samples ({samples}/{MIN_SAMPLES_FOR_TUNE})"}

        # Convert CLV percent into a direction signal.
        direction = 0
        if avg_clv <= -1.0:
            direction = +1   # raise edge bar
        elif avg_clv >= +1.0:
            direction = -1   # lower edge bar

        for attr in ("spread_value_min_edge", "total_value_min_edge"):
            current = getattr(self.settings, attr, 0.025)
            if direction == 0:
                continue
            nudge = 0.005 * direction
            new_val = max(EDGE_FLOOR, min(EDGE_CEIL, round(current + nudge, 4)))
            if abs(new_val - current) < 1e-6:
                continue
            setattr(self.settings, attr, new_val)
            tuned[attr] = {"from": current, "to": new_val, "reason": f"avg_clv={avg_clv:.2f}%"}

        return tuned or {"note": "CLV within neutral band — thresholds unchanged"}

    # -- skill fetch -------------------------------------------------

    def _fetch_daily_skill(self) -> tuple[str, str, str]:
        """Pull a fresh sharp-betting piece to add to the skill log.

        We keep this dependency-free: a plain urllib GET, 5-second
        timeout, and a very rough title/summary extractor. If every
        source fails we still return a non-empty entry so the daily
        record is always complete.
        """
        for url in LEARNING_FEEDS:
            try:
                req = Request(url, headers={"User-Agent": "sba-daily-learner/1.0"})
                with urlopen(req, timeout=5) as resp:
                    body = resp.read().decode("utf-8", errors="ignore")
            except Exception as exc:
                logger.info("daily_learner feed %s failed: %s", url, exc)
                continue

            title = self._extract_first(body, "<title>", "</title>")
            # Skip the feed title ("Pinnacle Betting Resources" etc.) — we
            # want the first article title, which is the second <title>.
            article_title = self._extract_first(body, "<title>", "</title>", start_after=len(title) + 10 if title else 0)
            summary = self._extract_first(body, "<description>", "</description>", start_after=len(title) + 10 if title else 0)
            headline = article_title or title or "Sharp-betting article"
            return (
                headline.strip()[:200],
                self._strip_tags(summary or "")[:500],
                url,
            )

        return (
            "Self-review only",
            "No external source reachable today — learning from our own CLV and win-rate patterns.",
            "",
        )

    @staticmethod
    def _extract_first(text: str, start_tag: str, end_tag: str, start_after: int = 0) -> str:
        i = text.find(start_tag, start_after)
        if i < 0:
            return ""
        j = text.find(end_tag, i + len(start_tag))
        if j < 0:
            return ""
        return text[i + len(start_tag): j]

    @staticmethod
    def _strip_tags(s: str) -> str:
        out = []
        depth = 0
        for c in s:
            if c == "<":
                depth += 1
            elif c == ">":
                depth = max(0, depth - 1)
            elif depth == 0:
                out.append(c)
        return "".join(out).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").strip()

    # -- notes -------------------------------------------------------

    def _compose_notes(self, review: Dict, clv_stats: Dict, tuned: Dict) -> List[str]:
        notes: List[str] = []
        by_strat = review.get("by_strategy", {})
        if by_strat:
            best = max(by_strat.items(), key=lambda kv: (kv[1].get("win_rate") or -1))
            worst = min(by_strat.items(), key=lambda kv: (kv[1].get("win_rate") or 101))
            if best[1].get("win_rate") is not None:
                notes.append(f"Best strategy so far: {best[0]} @ {best[1]['win_rate']}% ({best[1]['total_settled']} settled).")
            if worst[1].get("win_rate") is not None and worst[0] != best[0]:
                notes.append(f"Weakest strategy: {worst[0]} @ {worst[1]['win_rate']}% — watch for cuts.")
        avg_clv = clv_stats.get("average_clv")
        if avg_clv is not None:
            if avg_clv > 0:
                notes.append(f"CLV is positive ({avg_clv:.2f}%) — picks are beating the close; more volume is safe.")
            else:
                notes.append(f"CLV is {avg_clv:.2f}% — tightening edge thresholds to stop paying the vig.")
        if tuned and any(isinstance(v, dict) and "from" in v for v in tuned.values()):
            for k, v in tuned.items():
                if isinstance(v, dict) and "from" in v:
                    notes.append(f"Auto-tuned {k}: {v['from']:.3f} -> {v['to']:.3f} ({v['reason']}).")
        return notes

    # -- read-only views --------------------------------------------

    def today(self) -> Optional[DailyEntry]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return next((e for e in reversed(self.entries) if e.date == today), None)

    def recent(self, n: int = 30) -> List[DailyEntry]:
        return self.entries[-n:]
