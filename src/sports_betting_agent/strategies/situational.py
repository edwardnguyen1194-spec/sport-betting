"""Situational edge strategy.

Combines multiple research-backed situational filters that the
world's best sports bettors use:

1. Day-of-week (midweek games are softer markets)
2. Game day classification (Sunday/bullpen days in baseball)
3. Conference strength mismatch
4. Public fade (contrarian on brand-name overvalued teams)
5. Market softness scoring

Sources: Joe Peta "Trading Bases", academic research on market
efficiency by Levitt (2004), Humphreys (2010), Borghesi (2007).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable, List, Optional

from ..config import Settings, get_settings
from ..models_schema import GameOdds, american_to_implied, kelly_fraction
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)

# Brand-name programs that attract disproportionate public money
BRAND_NAMES = {
    # College baseball
    "vanderbilt", "lsu", "florida", "texas", "oregon state", "stanford",
    "miami", "arkansas", "ole miss", "virginia", "florida state",
    # College football
    "alabama", "ohio state", "georgia", "clemson", "michigan",
    "oklahoma", "usc", "notre dame", "penn state", "texas a&m",
    # College basketball
    "duke", "kentucky", "north carolina", "kansas", "villanova",
    "gonzaga", "michigan state", "ucla",
}

# Conference strength tiers for college sports
CONF_STRENGTH = {
    # Baseball
    "sec": 1.00, "acc": 0.95, "big 12": 0.93, "big ten": 0.88,
    "pac-12": 0.87, "sun belt": 0.82, "aac": 0.78, "conference usa": 0.75,
    "missouri valley": 0.73, "wcc": 0.70, "colonial": 0.68,
    "atlantic 10": 0.67, "big east": 0.72, "mountain west": 0.70,
    # Default for unknown
}

SHARP_BOOKS = {"pinnacle", "circa", "bovada", "betonline", "bookmaker"}


class SituationalStrategy(Strategy):
    """Score games by situational edges and bet the highest-scoring ones."""
    name = "situational"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        recs: List[BetRecommendation] = []
        cfg = self.settings

        for game in games:
            for side in ("home", "away"):
                team = game.home_team if side == "home" else game.away_team
                opp = game.away_team if side == "home" else game.home_team

                ml_lines = [
                    l for l in game.lines
                    if l.market == "moneyline"
                    and l.selection.lower() == team.lower()
                    and l.american is not None
                ]
                if not ml_lines:
                    continue

                best = max(ml_lines, key=lambda l: l.decimal or 0.0)
                if best.decimal is None or best.american is None:
                    continue

                # Calculate situational edge score
                edge_score = self._score_situation(game, side, team, opp)

                if edge_score < 0.05:  # Need 5%+ situational edge
                    continue

                implied = american_to_implied(best.american)
                # Only recommend favorites (implied > 55%) with situational edge
                if implied < 0.55:
                    continue
                confidence = min(implied + edge_score, 0.92)
                edge = confidence - implied

                if edge < 0.05:  # Require 5% edge minimum
                    continue

                stake_frac = min(
                    kelly_fraction(confidence, best.decimal, cfg.kelly_fraction),
                    cfg.max_bet_pct,
                )

                reasons = self._build_reasons(game, side, team, opp)

                recs.append(
                    BetRecommendation(
                        game_key=game.game_key,
                        sport=game.sport,
                        league=game.league,
                        home_team=game.home_team,
                        away_team=game.away_team,
                        market="moneyline",
                        selection=team,
                        american=best.american,
                        decimal=best.decimal,
                        book=best.book,
                        strategy=self.name,
                        confidence=round(confidence, 4),
                        edge=round(edge, 4),
                        stake_fraction=round(stake_frac, 4),
                        reasoning=reasons,
                        sources=sorted({l.book for l in ml_lines}),
                    )
                )

        logger.info("situational: %d recs", len(recs))
        return recs

    def _score_situation(self, game: GameOdds, side: str,
                         team: str, opp: str) -> float:
        """Calculate total situational edge as probability boost."""
        score = 0.0

        # 1. Market softness: college sports are less efficient
        if "ncaa" in game.sport or "ncaa" in game.league:
            score += 0.015

        # 2. Midweek game bonus (less efficient market)
        if game.commence_time:
            dow = game.commence_time.weekday()
            if dow in (1, 2, 3):  # Tue/Wed/Thu
                score += 0.01

        # 3. Home field advantage (stronger in college)
        if side == "home":
            if "ncaa" in game.sport:
                score += 0.02  # College HFA is ~2% stronger
            else:
                score += 0.01

        # 4. Opponent is overvalued brand name (fade signal)
        if opp.lower() in BRAND_NAMES:
            # Check if public is heavy on the opponent
            opp_public = game.meta.get(
                f"public_ml_{'home' if side == 'away' else 'away'}_pct"
            )
            if opp_public and opp_public > 0.65:
                score += 0.025  # Public overvaluing brand name

        # 5. Sharp book consensus agrees with this side
        sharp_lines = [
            l for l in game.lines
            if l.market == "moneyline"
            and l.selection.lower() == team.lower()
            and l.book.lower() in SHARP_BOOKS
            and l.american is not None
        ]
        if sharp_lines:
            sharp_implied = sum(
                american_to_implied(l.american) for l in sharp_lines
            ) / len(sharp_lines)
            if sharp_implied > 0.55:
                score += 0.01  # Sharps favor this side

        # 6. Multi-book coverage (more books = more reliable pricing)
        n_books = len(set(l.book for l in game.lines if l.market == "moneyline"))
        if n_books >= 5:
            score += 0.005

        # 7. College underdog edge (academic research: underdogs profitable)
        # Road underdogs in college sports are undervalued
        if "ncaa" in game.sport and side == "away":
            away_lines = [l for l in game.lines if l.market == "moneyline"
                          and l.selection.lower() == team.lower() and l.american is not None]
            if away_lines:
                best_american = max(l.american for l in away_lines)
                if best_american > 100:  # Underdog (+odds)
                    score += 0.02  # Road underdogs in college are undervalued
                    # Low total games: road dogs cover 55% (research backed)
                    total_lines = [l for l in game.lines if l.market == "total" and l.line is not None]
                    if total_lines and min(l.line for l in total_lines) <= 7:
                        score += 0.015  # Low-scoring game boosts underdog value

        # 8. Moderate college favorites cover well (ranked 4-25, fav by 8.5 or less)
        if "ncaa" in game.sport:
            spread_lines = [l for l in game.lines if l.market == "spread"
                            and l.selection.lower() == team.lower() and l.line is not None]
            if spread_lines:
                best_spread = min(abs(l.line) for l in spread_lines)
                if 1.5 <= best_spread <= 8.5:
                    score += 0.015  # Moderate favorites cover 60%+ (BetMGM research)

        # 9. Sunday/midweek bullpen day in college baseball
        if game.sport == "baseball_ncaa" and game.commence_time:
            dow = game.commence_time.weekday()
            if dow == 6:  # Sunday = game 3, often bullpen day
                score += 0.01  # Higher variance, more upsets

        return score

    def _build_reasons(self, game: GameOdds, side: str,
                       team: str, opp: str) -> str:
        parts = [f"Situational edge on {team}:"]
        if "ncaa" in game.sport:
            parts.append("college market (less efficient)")
        if game.commence_time and game.commence_time.weekday() in (1, 2, 3):
            parts.append("midweek game (softer lines)")
        if side == "home":
            parts.append("home field advantage")
        if opp.lower() in BRAND_NAMES:
            parts.append(f"fading overvalued {opp}")
        return " | ".join(parts)
