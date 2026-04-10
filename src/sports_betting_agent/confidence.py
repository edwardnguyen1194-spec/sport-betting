"""Confidence scoring.

The aggregator merges odds from many books, but the raw consensus
isn't always a good confidence number: you need to weight by book
quality, agreement, recency, and line movement. This module does
that -- it produces a 0..1 "trust score" that callers can gate on
(min_confidence in :class:`Settings`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .models_schema import GameOdds, OddsLine, american_to_implied, remove_vig_two_way


# Relative weight of each book when computing consensus.
BOOK_WEIGHTS: Dict[str, float] = {
    "pinnacle": 2.0,
    "circa": 1.8,
    "bookmaker": 1.4,
    "bovada": 1.2,
    "betonline": 1.1,
    "draftkings": 1.0,
    "fanduel": 1.0,
    "betmgm": 0.9,
    "caesars": 0.9,
    "espn_bet": 0.8,
    "espn": 0.8,
    "scoresandodds_consensus": 0.7,
    "vegasinsider": 0.6,
    "covers_consensus": 0.6,
}
DEFAULT_WEIGHT = 0.5


@dataclass
class ConfidenceReport:
    side: str
    implied_prob: float
    weighted_prob: float
    agreement: float
    book_count: int
    confidence: float


def weighted_consensus(game: GameOdds) -> Dict[str, ConfidenceReport]:
    """Return confidence reports for both sides of the moneyline."""

    reports: Dict[str, ConfidenceReport] = {}

    for side, team in (("home", game.home_team), ("away", game.away_team)):
        lines: List[OddsLine] = [
            l
            for l in game.lines
            if l.market == "moneyline"
            and l.selection.lower() == team.lower()
            and l.american is not None
        ]
        if not lines:
            reports[side] = ConfidenceReport(side, 0.0, 0.0, 0.0, 0, 0.0)
            continue
        weighted = 0.0
        total_weight = 0.0
        probs: List[float] = []
        for line in lines:
            w = BOOK_WEIGHTS.get(line.book.lower(), DEFAULT_WEIGHT)
            p = american_to_implied(line.american or 0)
            weighted += w * p
            total_weight += w
            probs.append(p)
        weighted_prob = weighted / total_weight if total_weight else 0.0
        avg_prob = sum(probs) / len(probs)

        # Agreement = 1 - normalized stddev (capped at 1).
        if len(probs) == 1:
            agreement = 1.0
        else:
            mean = avg_prob
            var = sum((p - mean) ** 2 for p in probs) / len(probs)
            std = var ** 0.5
            agreement = max(0.0, 1.0 - min(1.0, std * 6.0))
        reports[side] = ConfidenceReport(
            side=side,
            implied_prob=round(avg_prob, 4),
            weighted_prob=round(weighted_prob, 4),
            agreement=round(agreement, 4),
            book_count=len(lines),
            confidence=0.0,
        )

    # Vig-remove the weighted probs pairwise.
    home = reports.get("home")
    away = reports.get("away")
    if home and away and (home.weighted_prob + away.weighted_prob) > 0:
        fh, fa = remove_vig_two_way(home.weighted_prob, away.weighted_prob)
        home.confidence = round(
            fh * (0.7 + 0.3 * home.agreement) * min(1.0, 0.4 + 0.1 * home.book_count),
            4,
        )
        away.confidence = round(
            fa * (0.7 + 0.3 * away.agreement) * min(1.0, 0.4 + 0.1 * away.book_count),
            4,
        )
    return reports
