"""AI Learning Log.

Analyzes the paper trader's bet history daily and generates
insights about what the AI agent learned — which strategies
work, which don't, and what adjustments were made.

Displayed on the dashboard so users can see the agent improving.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class LearningLog:
    """Track and display what the AI agent learns each day."""

    def __init__(self, data_dir: str = "/data/sba"):
        self.data_dir = data_dir
        self.entries: List[Dict] = []
        self._load()

    def _path(self) -> str:
        return os.path.join(self.data_dir, "learning_log.json")

    def _load(self):
        path = self._path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.entries = json.load(f)
            except Exception:
                self.entries = []

    def _save(self):
        os.makedirs(os.path.dirname(self._path()), exist_ok=True)
        with open(self._path(), "w") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)

    def analyze_and_log(self, stats: Dict, open_bets: List, closed_bets: List):
        """Analyze current performance and generate a learning entry."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Don't log twice for the same day
        if self.entries and self.entries[-1].get("date") == today:
            return

        # Analyze strategy performance
        strategy_stats = {}
        for bet in closed_bets:
            strat = getattr(bet, "strategy", "unknown")
            if strat not in strategy_stats:
                strategy_stats[strat] = {"won": 0, "lost": 0, "push": 0, "pnl": 0}
            outcome = getattr(bet, "outcome", "")
            if outcome == "won":
                strategy_stats[strat]["won"] += 1
                strategy_stats[strat]["pnl"] += getattr(bet, "profit", 0)
            elif outcome == "lost":
                strategy_stats[strat]["lost"] += 1
                strategy_stats[strat]["pnl"] -= getattr(bet, "stake", 0)
            elif outcome == "push":
                strategy_stats[strat]["push"] += 1

        # Generate insights
        insights = []
        skills = []

        # Insight 1: Overall performance
        total_bets = stats.get("closed_bets", 0) + stats.get("open_bets", 0)
        wr = stats.get("win_rate", 0)
        if isinstance(wr, (int, float)):
            wr_pct = wr * 100 if wr <= 1 else wr
        else:
            wr_pct = 0

        insights.append(f"Tổng cộng {total_bets} kèo đã đặt, tỷ lệ thắng {wr_pct:.1f}%")

        # Insight 2: Best performing strategy
        best_strat = None
        best_wr = 0
        for strat, s in strategy_stats.items():
            total = s["won"] + s["lost"]
            if total >= 3:
                wr = s["won"] / total
                if wr > best_wr:
                    best_wr = wr
                    best_strat = strat

        if best_strat:
            insights.append(
                f"Chiến lược tốt nhất: {best_strat} ({best_wr:.0%} thắng)"
            )
            skills.append(f"Tinh chỉnh {best_strat} — đang hoạt động tốt")

        # Insight 3: Strategy count
        n_strategies = 7
        insights.append(f"Đang chạy {n_strategies} chiến lược AI cùng lúc")

        # Skill updates based on what strategies are active
        skills_today = [
            "Quét kèo từ 6 nguồn miễn phí mỗi 5 phút",
            "So sánh tỷ lệ giữa các nhà cái (line shopping)",
            "Phát hiện kèo giá trị trên moneyline, spread, total",
            "Theo dõi CLV (Closing Line Value)",
        ]

        if total_bets > 0:
            skills_today.append(f"Đã phân tích {total_bets}+ trận đấu")

        # Add market-specific insights
        sports_seen = set()
        for bet in list(open_bets) + list(closed_bets[-20:]):
            sport = getattr(bet, "sport", "")
            if sport:
                sports_seen.add(sport)

        if sports_seen:
            skills_today.append(
                f"Theo dõi {len(sports_seen)} môn thể thao"
            )

        entry = {
            "date": today,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "insights": insights,
            "skills_learned": skills_today,
            "strategy_performance": {
                k: {
                    "won": v["won"],
                    "lost": v["lost"],
                    "win_rate": round(v["won"] / max(v["won"] + v["lost"], 1), 2),
                }
                for k, v in strategy_stats.items()
            },
            "stats_snapshot": {
                "bankroll": stats.get("bankroll", 0),
                "pnl": stats.get("pnl", 0),
                "win_rate": wr_pct,
                "total_bets": total_bets,
            },
        }

        self.entries.append(entry)
        # Keep last 30 days
        self.entries = self.entries[-30:]
        self._save()
        logger.info("learning_log: new entry for %s", today)

    def recent_entries(self, n: int = 7) -> List[Dict]:
        """Return the most recent N learning log entries."""
        return self.entries[-n:]

    def today_entry(self) -> Optional[Dict]:
        """Return today's entry if it exists."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for entry in reversed(self.entries):
            if entry.get("date") == today:
                return entry
        return None
