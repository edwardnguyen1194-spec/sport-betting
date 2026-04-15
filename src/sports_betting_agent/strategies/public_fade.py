"""Fade-the-Public strategy (contrarian betting).

Research-backed edge: underdog teams getting less than 40% of public
bets cover the spread ~63.8% over the last four NFL seasons
(SportsBettingDime / BoydsBets). Similar effect exists in MLB totals
and NBA spreads — the public loves favorites and overs; casual
over-betting pushes soft-book lines away from the sharps.

This strategy extracts the public-betting percentages that Action
Network publishes into ``game.meta`` (``public_spread_home_pct``,
``public_total_over_pct``, etc.) and bets the unpopular side whenever
the split is strongly lopsided (default ≥65% on one side) and the
matching line is available at a sharp book.

Design rules
------------

* Spreads and totals only (Uncle's rule: no moneyline).
* Dog-only on spreads — home dog or away dog has the edge; we don't
  fade the public onto favorites because that's a different (much
  weaker) play.
* Require at least one sharp book to be quoting the unpopular side
  right now — otherwise we're guessing at price.
* Confidence capped at 0.58 so Kelly sizing stays conservative. The
  63.8% historical win rate is on NFL closing-line-value data; real
  future samples will differ.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from ..config import Settings, get_settings
from ..models_schema import GameOdds, american_to_implied, kelly_fraction
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)


SHARP_BOOKS = {"pinnacle", "circa", "bovada", "bookmaker", "betonline"}
# Minimum share of public bets on one side before we fade the other.
PUBLIC_THRESHOLD = 0.65
# Cap confidence conservatively — historical 63.8% win rate is an
# upper bound under ideal data quality, so we sit under that.
CONFIDENCE_CAP = 0.58
# Skip picks where the fair-vig edge is tiny — below this we're just
# paying vig without real signal strength.
MIN_EDGE = 0.02


class PublicFadeStrategy(Strategy):
    name = "public_fade"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        recs: List[BetRecommendation] = []
        cfg = self.settings

        for game in games:
            meta = game.meta or {}
            # Spreads: fade the heavily-backed side.
            recs.extend(self._spread_pick(game, meta, cfg))
            # Totals: fade heavily-backed Over or Under.
            recs.extend(self._total_pick(game, meta, cfg))

        logger.info("public_fade: %d recs", len(recs))
        return recs

    # -- spreads -----------------------------------------------------

    def _spread_pick(self, game: GameOdds, meta: dict, cfg) -> List[BetRecommendation]:
        home_pct = meta.get("public_spread_home_pct")
        away_pct = meta.get("public_spread_away_pct")
        if home_pct is None or away_pct is None:
            return []
        # Only act on lopsided splits.
        fade_side = None
        fade_pct = None
        if home_pct >= PUBLIC_THRESHOLD:
            fade_side = game.away_team  # fade home -> bet away
            fade_pct = home_pct
        elif away_pct >= PUBLIC_THRESHOLD:
            fade_side = game.home_team  # fade away -> bet home
            fade_pct = away_pct
        if fade_side is None:
            return []

        # Find the best available spread line at a sharp book for the
        # unpopular side. Prefer handicap >= 0 (dog) — fading the
        # public ONTO a favorite is the weaker play.
        candidates = [
            l for l in game.lines
            if l.market == "spread"
            and l.american is not None
            and l.decimal is not None
            and l.line is not None
            and l.book.lower() in SHARP_BOOKS
            and l.selection.lower().strip() == fade_side.lower().strip()
            and l.line >= 0  # dog side only
        ]
        if not candidates:
            return []
        best = max(candidates, key=lambda l: l.decimal or 0.0)

        confidence = self._confidence_from_public(fade_pct)
        implied = american_to_implied(best.american)
        edge = confidence - implied
        if edge < MIN_EDGE:
            return []

        stake_frac = min(
            kelly_fraction(confidence, best.decimal, cfg.kelly_fraction),
            cfg.max_bet_pct,
        )

        return [
            BetRecommendation(
                game_key=game.game_key,
                sport=game.sport,
                league=game.league,
                home_team=game.home_team,
                away_team=game.away_team,
                market="spread",
                selection=fade_side,
                american=best.american,
                decimal=best.decimal,
                line=best.line,
                book=best.book,
                strategy=self.name,
                confidence=round(confidence, 4),
                edge=round(edge, 4),
                stake_fraction=round(stake_frac, 4),
                reasoning=(
                    f"Fade public: {fade_pct:.0%} of tickets on the other side, "
                    f"taking {fade_side} {best.line:+.1f} at {best.american:+.0f} "
                    f"on {best.book}."
                ),
                sources=[best.book],
            )
        ]

    # -- totals ------------------------------------------------------

    def _total_pick(self, game: GameOdds, meta: dict, cfg) -> List[BetRecommendation]:
        over_pct = meta.get("public_total_over_pct")
        under_pct = meta.get("public_total_under_pct")
        if over_pct is None or under_pct is None:
            return []
        fade_sel = None
        fade_pct = None
        if over_pct >= PUBLIC_THRESHOLD:
            fade_sel = "Under"
            fade_pct = over_pct
        elif under_pct >= PUBLIC_THRESHOLD:
            fade_sel = "Over"
            fade_pct = under_pct
        if fade_sel is None:
            return []

        candidates = [
            l for l in game.lines
            if l.market == "total"
            and l.american is not None
            and l.decimal is not None
            and l.line is not None
            and l.book.lower() in SHARP_BOOKS
            and l.selection.lower().strip() == fade_sel.lower()
        ]
        if not candidates:
            return []
        best = max(candidates, key=lambda l: l.decimal or 0.0)

        confidence = self._confidence_from_public(fade_pct)
        implied = american_to_implied(best.american)
        edge = confidence - implied
        if edge < MIN_EDGE:
            return []

        stake_frac = min(
            kelly_fraction(confidence, best.decimal, cfg.kelly_fraction),
            cfg.max_bet_pct,
        )

        return [
            BetRecommendation(
                game_key=game.game_key,
                sport=game.sport,
                league=game.league,
                home_team=game.home_team,
                away_team=game.away_team,
                market="total",
                selection=fade_sel,
                american=best.american,
                decimal=best.decimal,
                line=best.line,
                book=best.book,
                strategy=self.name,
                confidence=round(confidence, 4),
                edge=round(edge, 4),
                stake_fraction=round(stake_frac, 4),
                reasoning=(
                    f"Fade public: {fade_pct:.0%} on the other side, taking "
                    f"{fade_sel} {best.line} at {best.american:+.0f} on "
                    f"{best.book}."
                ),
                sources=[best.book],
            )
        ]

    # -- helpers -----------------------------------------------------

    @staticmethod
    def _confidence_from_public(pct: float) -> float:
        """Map the public-share onto a confidence, capped at CONFIDENCE_CAP.

        65% public on the other side -> 0.53 confidence.
        75% public -> 0.55.
        85%+ -> CONFIDENCE_CAP.
        """
        if pct < PUBLIC_THRESHOLD:
            return 0.5
        # Linear ramp from 0.52 at threshold to CONFIDENCE_CAP at 0.90.
        span = max(0.0, pct - PUBLIC_THRESHOLD)
        scaled = 0.52 + span * ((CONFIDENCE_CAP - 0.52) / (0.90 - PUBLIC_THRESHOLD))
        return min(CONFIDENCE_CAP, scaled)
