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

from .agents.strategy_auditor import (
    StrategyAuditor,
    run_weekly_audit,
    save_weekly_audit_markdown,
)
from .agents.base import AgentLog
from .agents.self_reflection import (
    SelfReflection,
    VALID_TARGETS as SELF_REFLECTION_TARGETS,
    run_self_reflection,
    save_proposals as save_self_reflection_proposals,
)
from .clv_tracker import CLVTracker
from .config import Settings
from .line_movement import LineMovementStore


logger = logging.getLogger(__name__)


# Expanded rotating pool of high-quality free sources the agent reads
# every morning to pick up new technique. Organized by topic so each
# morning's skill-of-the-day spans sharp strategy, math, weather,
# situational trends, and market microstructure.
LEARNING_FEEDS: List[str] = [
    # Core sharp-betting research (Pinnacle, Boyd, SBR)
    "https://www.pinnacle.com/en/betting-resources/rss",
    "https://www.boydsbets.com/feed/",
    "https://www.sportsbookreview.com/picks/feed/",
    # Advanced analytics + line-shopping
    "https://www.covers.com/rss/nfl",
    "https://www.covers.com/rss/nba",
    "https://www.covers.com/rss/mlb",
    "https://www.actionnetwork.com/rss",
    # Weather + park factors (direct relevance to our totals model)
    "https://www.ballparkpal.com/feed/",
    # DFS / statistical angle (public-side overlay)
    "https://www.fangraphs.com/blogs/feed/",
    # Power ratings + model-based betting
    "https://www.teamrankings.com/blog/feed/",
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
        strategy_auditor: Optional[StrategyAuditor] = None,
        self_reflection: Optional[SelfReflection] = None,
        agent_log: Optional[AgentLog] = None,
    ) -> None:
        self.settings = settings
        self.clv = clv
        self.line_store = line_store
        # Optional weekly-audit agent. Defaults to None so existing
        # call sites / tests don't need to inject it; the Monday run
        # is a no-op when it is absent.
        self.strategy_auditor = strategy_auditor
        # Weekly introspective learner. Iterates the 6 sub-agents
        # once per Sunday UTC and writes proposed prompt/config
        # patches to ``agent_prompt_proposals.json``. No-op when
        # absent (tests, partial wiring).
        self.self_reflection = self_reflection
        self.agent_log = agent_log
        self.data_dir = settings.data_dir
        self.entries: List[DailyEntry] = []
        # Weekly-audit idempotency: record the YYYY-MM-DD of the most
        # recent successful weekly audit so repeated Monday triggers
        # don't spam the filesystem or the agent log.
        self._last_weekly_audit_date: Optional[str] = None
        # Same idempotency guard for the Sunday introspection loop —
        # without it, a mid-day restart would double-append proposals.
        self._last_self_reflection_date: Optional[str] = None
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

    def run(
        self,
        closed_bets: Iterable,
        force: bool = False,
        paper: Optional[object] = None,
    ) -> Optional[DailyEntry]:
        """Execute one daily cycle. Idempotent unless ``force`` is set.

        On Monday (UTC), ALSO runs the weekly strategy audit when a
        :class:`StrategyAuditor` and the ``paper`` trader are wired up.
        The audit is idempotent per calendar date independently of the
        daily cycle.
        """
        # Snapshot once so the daily + weekly paths see the same
        # ledger slice without re-iterating a generator twice.
        closed_list = list(closed_bets)

        # Monday weekly audit (runs BEFORE the daily-idempotent guard so
        # ``force=False`` still kicks off Monday's review on the first
        # hit of the new week).
        self._maybe_run_weekly_audit(paper=paper, closed_bets=closed_list)

        # Sunday self-reflection (same outside-the-idempotent-guard
        # pattern — we want the introspection loop to fire on the
        # first Sunday hit even when the daily entry already exists).
        self._maybe_run_self_reflection()

        if not force and self.already_ran_today():
            return None

        now = datetime.now(timezone.utc)
        review = self._performance_review(closed_list)
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

    # -- weekly audit (Monday) --------------------------------------

    def _maybe_run_weekly_audit(
        self,
        paper: Optional[object],
        closed_bets: List,
    ) -> None:
        """Once-per-Monday narrative audit by :class:`StrategyAuditor`.

        Always safe to call — no-ops unless
        (a) today is Monday (UTC),
        (b) a StrategyAuditor was injected,
        (c) today's audit has not already run.

        The AgentDecision is logged via BaseAgent's agent_log
        contract; in addition we persist the full markdown review to
        ``<data_dir>/weekly_audits/YYYY-MM-DD.md`` so the dashboard
        can surface it without parsing JSON.
        """
        if self.strategy_auditor is None:
            return
        now = datetime.now(timezone.utc)
        if now.weekday() != 0:  # Monday == 0
            return
        today = now.strftime("%Y-%m-%d")
        if self._last_weekly_audit_date == today:
            return
        # A real auditor needs the paper trader for closed bets; if the
        # caller didn't pass one, synthesize a shim so the helper still
        # works — `closed_bets` is already a list in this code path.
        shim_paper = paper if paper is not None else type("_Shim", (), {"closed_bets": closed_bets})()
        try:
            decision = run_weekly_audit(
                self.strategy_auditor,
                shim_paper,
                self.clv,
                self.settings,
            )
        except Exception as exc:
            logger.warning("weekly audit failed: %s", exc)
            return
        if decision is None:
            return
        markdown = (decision.reasoning or "").strip()
        if markdown:
            path = save_weekly_audit_markdown(self.data_dir, today, markdown)
            if path:
                logger.info("weekly audit saved to %s", path)
        self._last_weekly_audit_date = today

    # -- self-reflection (Sunday UTC) -------------------------------

    def _maybe_run_self_reflection(self) -> None:
        """Once-per-Sunday introspective pass over the 6 sub-agents.

        Iterates :data:`SELF_REFLECTION_TARGETS`, pulls the last 50
        AgentLog entries per agent, and appends the resulting
        proposals to ``/data/sba/agent_prompt_proposals.json``.

        Safe no-op when:
        (a) no :class:`SelfReflection` agent was injected,
        (b) no :class:`AgentLog` is available (nothing to reflect on),
        (c) today is not Sunday (weekday == 6) UTC,
        (d) today's pass already ran.
        """
        if self.self_reflection is None or self.agent_log is None:
            return
        now = datetime.now(timezone.utc)
        if now.weekday() != 6:  # Sunday == 6
            return
        today = now.strftime("%Y-%m-%d")
        if self._last_self_reflection_date == today:
            return
        wrote_any = False
        for target in SELF_REFLECTION_TARGETS:
            try:
                decision = run_self_reflection(
                    self.self_reflection,
                    target_name=target,
                    agent_log=self.agent_log,
                    n=50,
                )
            except Exception as exc:
                logger.warning(
                    "self_reflection on %s failed: %s", target, exc
                )
                continue
            if decision is None:
                continue
            try:
                path = save_self_reflection_proposals(
                    self.data_dir, target, decision, audit_date=today
                )
                if path:
                    wrote_any = True
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "save_self_reflection_proposals %s failed: %s",
                    target, exc,
                )
        if wrote_any:
            logger.info(
                "self_reflection pass complete for %s targets on %s",
                len(SELF_REFLECTION_TARGETS), today,
            )
        self._last_self_reflection_date = today

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
        """Pull fresh sharp-betting pieces across ALL sources and
        combine them into a single morning skill summary. Each
        morning the agent now learns from up to 4 articles instead
        of 1 — covering sharp strategy, weather/park factors,
        situational trends, and advanced analytics.
        """
        articles: list[tuple[str, str, str]] = []
        for url in LEARNING_FEEDS:
            try:
                req = Request(url, headers={"User-Agent": "sba-daily-learner/1.0"})
                with urlopen(req, timeout=5) as resp:
                    body = resp.read().decode("utf-8", errors="ignore")
            except Exception as exc:
                logger.info("daily_learner feed %s failed: %s", url, exc)
                continue

            title = self._extract_first(body, "<title>", "</title>")
            article_title = self._extract_first(body, "<title>", "</title>", start_after=len(title) + 10 if title else 0)
            summary = self._extract_first(body, "<description>", "</description>", start_after=len(title) + 10 if title else 0)
            headline = (article_title or title or "").strip()
            if headline:
                articles.append((
                    headline[:200],
                    self._strip_tags(summary or "")[:400],
                    url,
                ))
            if len(articles) >= 4:
                break

        if not articles:
            return (
                "Self-review only",
                "No external sources reachable — learning from our own CLV and win-rate patterns today.",
                "",
            )

        # Combine multiple articles into one morning briefing.
        lead = articles[0]
        combined_summary = "\n".join(
            f"• {a[0]}: {a[1]}" for a in articles
        )
        sources = "; ".join(a[2] for a in articles)
        return (
            f"Morning briefing: {len(articles)} sharp-betting articles",
            combined_summary[:1500],
            sources[:500],
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
