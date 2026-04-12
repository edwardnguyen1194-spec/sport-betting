"""AI Agent Brain — Self-Learning System.

The brain tracks performance of every strategy and auto-adjusts
confidence weights based on real results. It also reads sports
news to flag relevant games.

Every trade cycle the brain:
1. Analyzes which strategies are winning/losing
2. Adjusts strategy trust scores (strategies that win get boosted)
3. Reads ESPN news for injury/lineup changes
4. Generates daily learning insights
5. Saves everything to disk so it survives restarts

This makes the agent genuinely improve over time — like the
best bettors who track what works and cut what doesn't.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentBrain:
    """Self-learning brain that tracks and improves strategy performance."""

    def __init__(self, data_dir: str = "/data/sba"):
        self.data_dir = data_dir
        # Strategy trust scores (1.0 = neutral, >1 = boosted, <1 = dampened)
        self.strategy_trust: Dict[str, float] = {}
        # Performance tracking per strategy
        self.strategy_stats: Dict[str, Dict] = {}
        # Sport-specific insights
        self.sport_insights: Dict[str, Dict] = {}
        # Skills learned over time
        self.skills_log: List[Dict] = []
        # Research notes from news
        self.research_notes: List[str] = []
        self._load()

    def _path(self) -> str:
        return os.path.join(self.data_dir, "brain.json")

    def _load(self):
        path = self._path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                self.strategy_trust = data.get("strategy_trust", {})
                self.strategy_stats = data.get("strategy_stats", {})
                self.sport_insights = data.get("sport_insights", {})
                self.skills_log = data.get("skills_log", [])
                self.research_notes = data.get("research_notes", [])
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self._path()), exist_ok=True)
        with open(self._path(), "w") as f:
            json.dump({
                "strategy_trust": self.strategy_trust,
                "strategy_stats": self.strategy_stats,
                "sport_insights": self.sport_insights,
                "skills_log": self.skills_log[-30:],
                "research_notes": self.research_notes[-20:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2, ensure_ascii=False)

    def learn_from_results(self, closed_bets: list):
        """Analyze closed bets and update strategy trust scores."""
        stats: Dict[str, Dict] = defaultdict(lambda: {"won": 0, "lost": 0, "push": 0, "pnl": 0.0})

        for bet in closed_bets:
            strat = getattr(bet, "strategy", "unknown")
            status = getattr(bet, "status", "")
            result = getattr(bet, "result", 0) or 0
            sport = getattr(bet, "sport", "unknown")

            if status == "won":
                stats[strat]["won"] += 1
                stats[strat]["pnl"] += result
            elif status == "lost":
                stats[strat]["lost"] += 1
                stats[strat]["pnl"] += result  # result is negative for losses
            elif status == "push":
                stats[strat]["push"] += 1

            # Track sport-specific performance
            sport_key = f"{strat}_{sport}"
            stats[sport_key]["won" if status == "won" else "lost" if status == "lost" else "push"] += 1
            stats[sport_key]["pnl"] += result

        # Update strategy trust scores
        new_skills = []
        for strat, s in stats.items():
            total = s["won"] + s["lost"]
            if total < 3:
                continue

            wr = s["won"] / total
            self.strategy_stats[strat] = {
                "won": s["won"], "lost": s["lost"],
                "win_rate": round(wr, 3),
                "pnl": round(s["pnl"], 2),
            }

            # Adjust trust: winning strategies get boosted
            if wr >= 0.60:
                self.strategy_trust[strat] = min(1.5, self.strategy_trust.get(strat, 1.0) + 0.05)
                new_skills.append(f"Tăng tin tưởng {strat} ({wr:.0%} thắng)")
            elif wr <= 0.40:
                self.strategy_trust[strat] = max(0.5, self.strategy_trust.get(strat, 1.0) - 0.05)
                new_skills.append(f"Giảm tin tưởng {strat} ({wr:.0%} thắng)")
            else:
                self.strategy_trust[strat] = self.strategy_trust.get(strat, 1.0)

        if new_skills:
            self.skills_log.append({
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "type": "strategy_adjustment",
                "skills": new_skills,
            })

        self._save()

    def learn_from_news(self, headlines: List[Dict]):
        """Extract insights from news headlines."""
        insights = []
        for h in headlines:
            title = h.get("title", "").lower()
            sport = h.get("sport", "")

            # Detect injury news
            if any(w in title for w in ["injury", "injured", "out", "ruled out",
                                         "questionable", "doubtful", "surgery",
                                         "concussion", "strain", "sprain"]):
                insights.append(f"[{sport}] Chấn thương: {h['title']}")

            # Detect lineup changes
            if any(w in title for w in ["lineup", "starting", "benched",
                                         "roster", "trade", "signed", "released"]):
                insights.append(f"[{sport}] Đội hình: {h['title']}")

            # Detect streaks
            if any(w in title for w in ["streak", "winning streak", "losing streak",
                                         "consecutive", "hot", "cold"]):
                insights.append(f"[{sport}] Phong độ: {h['title']}")

        if insights:
            self.research_notes = insights[:20]
            self.skills_log.append({
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "type": "news_research",
                "skills": [f"Đọc {len(insights)} tin tức quan trọng"],
            })
            self._save()

    def get_trust(self, strategy: str) -> float:
        """Get trust score for a strategy (default 1.0)."""
        return self.strategy_trust.get(strategy, 1.0)

    def adjust_confidence(self, strategy: str, base_confidence: float) -> float:
        """Adjust confidence based on strategy trust score."""
        trust = self.get_trust(strategy)
        return min(0.95, base_confidence * trust)

    def daily_summary(self) -> Dict:
        """Generate daily learning summary for the dashboard."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Best and worst strategies
        best_strat = None
        worst_strat = None
        best_wr = 0
        worst_wr = 1.0

        for strat, s in self.strategy_stats.items():
            if "_" in strat and strat.count("_") > 1:
                continue  # Skip sport-specific keys
            total = s.get("won", 0) + s.get("lost", 0)
            if total < 3:
                continue
            wr = s.get("win_rate", 0)
            if wr > best_wr:
                best_wr = wr
                best_strat = strat
            if wr < worst_wr:
                worst_wr = wr
                worst_strat = strat

        skills_today = [
            "Quét kèo từ 6 nguồn mỗi 5 phút",
            "Chạy 7 chiến lược AI đồng thời",
            "Theo dõi CLV (Closing Line Value)",
            "Đọc tin tức ESPN 24/7",
            "Tự điều chỉnh chiến lược dựa trên kết quả",
            "So sánh kèo giữa 15+ nhà cái",
            "Phát hiện kèo giữa (middle) tự động",
        ]

        if best_strat:
            skills_today.append(f"Chiến lược tốt nhất: {best_strat} ({best_wr:.0%})")
        if worst_strat and worst_strat != best_strat:
            skills_today.append(f"Cần cải thiện: {worst_strat} ({worst_wr:.0%})")
        if self.research_notes:
            skills_today.append(f"Nghiên cứu {len(self.research_notes)} tin tức hôm nay")

        # Trust adjustments
        boosted = [s for s, t in self.strategy_trust.items() if t > 1.0]
        dampened = [s for s, t in self.strategy_trust.items() if t < 1.0]
        if boosted:
            skills_today.append(f"Tăng cường: {', '.join(boosted)}")
        if dampened:
            skills_today.append(f"Thận trọng hơn: {', '.join(dampened)}")

        return {
            "date": today,
            "skills": skills_today,
            "strategy_stats": self.strategy_stats,
            "trust_scores": self.strategy_trust,
            "research_notes": self.research_notes[:5],
            "total_skills_learned": len(self.skills_log),
        }
