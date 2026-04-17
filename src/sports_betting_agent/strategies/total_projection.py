"""Model-based total strategy.

Where ``TotalValueStrategy`` finds edges by comparing multiple
bookmakers' prices on the same total, this strategy finds edges by
comparing a single bookmaker's total to an **independent model** of
how many runs/goals/points the game is likely to produce.

The projection is the rolling scored/allowed average for the home
and away teams (see ``team_scoring.TeamScoringTracker``). If the
projected total differs from the posted total by MIN_LINE_EDGE or
more, we flag the value side using the book's posted juice.

This is a standard pillar of professional totals betting: Pinnacle's
own traders compare the posted number to internal pace/scoring
models, bet the discrepancies, and let CLV grade the result. It
complements the market-based strategies instead of replacing them.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, List, Optional

from ..config import Settings, get_settings
from ..models_schema import GameOdds, american_to_implied, kelly_fraction
from ..team_scoring import TeamScoringTracker
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)


# Minimum edge in run/goal/point units before we take the bet. Keeps
# us above measurement noise given we only have ~20 games of rolling
# history per team.
MIN_LINE_EDGE = {
    "baseball_mlb":    0.5,   # half a run — the smallest line increment
    "baseball_ncaa":   0.75,
    "basketball_nba":  2.5,   # NBA totals are noisy; need a real gap
    "basketball_ncaab":3.0,
    "hockey_nhl":      0.4,
    "football_nfl":    2.0,
    "football_ncaaf":  2.5,
}
DEFAULT_MIN_EDGE = 1.0


class TotalProjectionStrategy(Strategy):
    """Flag totals where Bovada's posted number disagrees with the model."""

    name = "total_projection"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        scoring: Optional[TeamScoringTracker] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.scoring = scoring or TeamScoringTracker(self.settings.data_dir)

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        recs: List[BetRecommendation] = []
        cfg = self.settings

        from ..park_factors import park_factor as _park_factor
        from ..weather import total_adjustment as _weather_adj

        for game in games:
            proj = self.scoring.projected_total(
                sport=game.sport, home=game.home_team, away=game.away_team
            )
            if proj is None:
                continue
            projected = proj["projected_total"]
            # Apply ballpark factor for MLB — the market under-weights
            # park extremes like Coors (+15% runs) and Dodger Stadium
            # (-8%), per multi-season BallparkPal / FanGraphs data.
            pf = 1.0
            weather_delta = 0.0
            weather_reason = ""
            if game.sport in ("baseball_mlb", "baseball_ncaa"):
                pf = _park_factor(game.home_team)
                projected = projected * pf
                # Add weather adjustment (wind ≥15 mph, temp extremes).
                # Skipped for indoor stadiums and when forecast fails.
                try:
                    weather_delta, weather_reason = _weather_adj(
                        game.home_team, game.commence_time
                    )
                    projected = projected + weather_delta
                except Exception as exc:
                    logger.debug("weather adj failed: %s", exc)

            # Collect total lines with real juice (no fake -110 defaults).
            totals = [
                l for l in game.lines
                if l.market == "total"
                and l.american is not None
                and l.line is not None
            ]
            if not totals:
                continue

            # Build (line_number, over_price, under_price) views for each
            # book that quotes both sides.
            by_book: dict[tuple[str, float], dict[str, object]] = {}
            for line in totals:
                key = (line.book.lower(), float(line.line))
                entry = by_book.setdefault(
                    key, {"book": line.book, "line": line.line, "over": None, "under": None}
                )
                sel = line.selection.lower()
                if sel == "over":
                    entry["over"] = line
                elif sel == "under":
                    entry["under"] = line

            min_edge = MIN_LINE_EDGE.get(game.sport, DEFAULT_MIN_EDGE)

            # Market-anchored regression: the book line is a MUCH better
            # baseline than an arbitrary league mean (the market already
            # prices in pace, star absences, refs, etc.). We shrink the
            # raw projection 60% toward the book line, so the model only
            # fights the market when its data strongly disagrees. This
            # fixes the "always Over" bias Uncle kept seeing — a game
            # projected at 225 vs book 206 now resolves to ~214 (Over
            # with smaller edge) instead of a runaway +18 Over.
            MARKET_REG = {
                "baseball_mlb":     0.50,
                "baseball_ncaa":    0.55,
                "basketball_nba":   0.60,
                "basketball_ncaab": 0.60,
                "basketball_wnba":  0.60,
                "hockey_nhl":       0.45,
                "football_nfl":     0.40,
                "football_ncaaf":   0.45,
                "soccer_mls":       0.50,
                "soccer_epl":       0.50,
                "soccer_ucl":       0.50,
            }
            market_reg = MARKET_REG.get(game.sport, 0.50)

            for key, entry in by_book.items():
                over = entry["over"]
                under = entry["under"]
                if over is None or under is None:
                    continue
                total_num = float(entry["line"])
                # Blend the raw projection with the book line so the
                # model is "market-aware" instead of blindly trusting
                # its own small-sample averages.
                blended = (1 - market_reg) * projected + market_reg * total_num
                diff = blended - total_num

                if abs(diff) < min_edge:
                    continue

                if diff > 0:
                    chosen = over
                    display_sel = "Over"
                else:
                    chosen = under
                    display_sel = "Under"

                # Convert the run/goal/point gap into a bounded confidence.
                # We deliberately cap at 0.62 — rolling 20-game averages
                # are noisy enough that claiming 99% confidence on a single
                # variable is silly, and Kelly sizing punishes overclaimed
                # confidence by betting aggressively into variance.
                CONFIDENCE_CAP = 0.62
                span = CONFIDENCE_CAP - 0.5
                confidence = 0.5 + span * math.tanh(abs(diff) / (4.0 * min_edge))
                implied = american_to_implied(chosen.american)
                edge = confidence - implied

                if edge <= 0:
                    # Model likes the side but the book is priced above our
                    # confidence — skip rather than pay the vig.
                    continue

                stake_frac = min(
                    kelly_fraction(confidence, chosen.decimal, cfg.kelly_fraction),
                    cfg.max_bet_pct,
                )

                recs.append(
                    BetRecommendation(
                        game_key=game.game_key,
                        sport=game.sport,
                        league=game.league,
                        home_team=game.home_team,
                        away_team=game.away_team,
                        market="total",
                        selection=display_sel,
                        american=chosen.american,
                        decimal=chosen.decimal,
                        line=total_num,
                        book=chosen.book,
                        strategy=self.name,
                        confidence=round(confidence, 4),
                        edge=round(edge, 4),
                        stake_fraction=round(stake_frac, 4),
                        reasoning=(
                            f"Model total projection: {game.away_team} @ "
                            f"{game.home_team} projects {projected:.2f} "
                            f"from {proj['home_games_tracked']}/"
                            f"{proj['away_games_tracked']} home/away games; "
                            f"book line {total_num:.1f} implies {total_num:.1f}. "
                            f"Gap = {diff:+.2f} ({display_sel}) at "
                            f"{chosen.american:+.0f} on {chosen.book}."
                        ),
                        sources=[chosen.book],
                    )
                )

        logger.info("total_projection: %d recs", len(recs))
        return recs
