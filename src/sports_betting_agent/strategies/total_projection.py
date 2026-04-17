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


# Minimum BLENDED-FRAME edge before we take a total bet.
#
# Calibrated from the live team_scoring.json on 2026-04-17 using the
# empirical per-game total standard deviation for each sport, then
# snapped to 0.30 σ (the conventional "not-noise" floor in sports
# modeling literature). See ``docs/research-fleet/04-internal-code-review.md``
# for the derivation script output:
#
#    Sport              Mean  StDev  0.30σ  Old (hand-picked)
#    baseball_mlb        9.16   4.86   1.46   0.50   ← 3× too loose
#    baseball_ncaa      13.13   6.12   1.84   0.75   ← 2.4× too loose
#    basketball_nba    234.14  20.83   6.25   2.50   ← 2.5× too loose
#    hockey_nhl          6.32   2.36   0.71   0.40   ← 1.8× too loose
#    soccer_epl          2.72   1.04   0.31    —
#    soccer_mls          3.36   1.94   0.58    —
#    soccer_ucl          3.58   2.47   0.74    —
#
# Sports with no data yet (NCAAB, NFL, NCAAF, WNBA) use physically
# reasonable estimates from known league stdevs (NFL ~10, NCAAF ~14,
# WNBA ~17, NCAAB ~19) × 0.30.
#
# Because TotalProjectionStrategy now regresses 40-60% toward the book
# line before measuring ``diff``, a threshold of 1.5 in the blended
# frame corresponds to a raw model-vs-book gap of 3.0 runs (for MLB
# market_reg=0.50) — a serious disagreement, not noise. Old 0.5 gap
# translated to a 1-run raw disagreement = within team-mean estimation
# error. We were buying our own variance.
MIN_LINE_EDGE = {
    "baseball_mlb":    1.5,    # 0.30σ from 748 live games
    "baseball_ncaa":   1.75,   # 0.30σ from 2,662 live games
    "basketball_nba":  6.25,   # 0.30σ from 414 live games
    "basketball_ncaab":5.75,   # NCAAB stdev ~19pt × 0.30
    "basketball_wnba": 5.0,    # WNBA stdev ~17pt × 0.30
    "hockey_nhl":      0.75,   # 0.30σ from 472 live games
    "football_nfl":    3.0,    # NFL stdev ~10pt × 0.30
    "football_ncaaf":  4.25,   # NCAAF stdev ~14pt × 0.30
    "soccer_mls":      0.5,    # 0.30σ from 88 live games
    "soccer_epl":      0.25,   # 0.30σ from 36 live games
    "soccer_ucl":      0.75,   # 0.30σ from 24 live games
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
        from ..mlb_umpires import run_adjustment as _ump_adj
        from ..soccer_model import project_soccer_totals as _soccer_dc

        for game in games:
            # Soccer sports use a dedicated Dixon-Coles Poisson model
            # that returns true scoreline probabilities — we branch
            # early and generate soccer recs in their own code path.
            if game.sport in ("soccer_mls", "soccer_epl", "soccer_ucl"):
                recs.extend(self._soccer_picks(game, cfg))
                continue

            proj = self.scoring.projected_total(
                sport=game.sport, home=game.home_team, away=game.away_team
            )
            if proj is None:
                continue
            projected = proj["projected_total"]
            # Park + weather + umpire adjustments. IMPORTANT: we apply
            # them to the BOOK-SIDE of the blend, not the model-side.
            # Applying pf/weather/ump to ``projected`` before the
            # market-regression caused the residual Over bias at
            # Coors/Fenway/summer heat because the inflation survived
            # the 50-60% blend. New approach: compute an "adjusted book
            # line" in the same park/weather/umpire-neutral reference
            # frame as ``projected``, then blend and diff in that neutral
            # frame. The posted bet still uses the actual total_num —
            # only the comparison basis shifts.
            pf = 1.0
            weather_delta = 0.0
            weather_reason = ""
            ump_delta = 0.0
            ump_reason = ""
            if game.sport == "baseball_mlb":
                pf = _park_factor(game.home_team)
                try:
                    weather_delta, weather_reason = _weather_adj(
                        game.home_team, game.commence_time
                    )
                except Exception as exc:
                    logger.debug("weather adj failed: %s", exc)
                try:
                    ump_delta, ump_reason = _ump_adj(
                        game.home_team, game.away_team, game.commence_time
                    )
                except Exception as exc:
                    logger.debug("umpire adj failed: %s", exc)
            elif game.sport == "baseball_ncaa":
                pf = _park_factor(game.home_team)
                try:
                    weather_delta, weather_reason = _weather_adj(
                        game.home_team, game.commence_time
                    )
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
                # Adjusted book line in the same park/weather/umpire-neutral
                # reference frame as ``projected``. Since ``projected``
                # is raw team averages (unaware of the specific park,
                # today's weather, or tonight's HP umpire), we strip
                # those effects out of the book's number before
                # comparing. Apples to apples.
                #   adjusted_book = (total_num / pf) - weather_delta - ump_delta
                # Blend model vs adjusted_book, measure edge in the
                # neutral frame, but BET against the real posted total.
                if pf:
                    adjusted_book = (total_num / pf) - weather_delta - ump_delta
                else:
                    adjusted_book = total_num - weather_delta - ump_delta
                blended = (1 - market_reg) * projected + market_reg * adjusted_book
                diff = blended - adjusted_book

                if abs(diff) < min_edge:
                    continue

                if diff > 0:
                    chosen = over
                    display_sel = "Over"
                else:
                    chosen = under
                    display_sel = "Under"

                # Confidence cap 0.62 → 0.57. With 40-60% market
                # regression admitting the book is mostly right, a 62%
                # claim was inconsistent. Kelly at 0.57 vs 0.62 on +100
                # juice is 14% vs 24% of bankroll per pick — enormous
                # variance reduction, minimal EV loss.
                CONFIDENCE_CAP = 0.57
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
                            f"book {total_num:.1f} — adj {adjusted_book:.2f} "
                            f"(pf={pf:.2f}"
                            + (f", wx {weather_reason}" if weather_reason and weather_reason != "neutral" else "")
                            + (f", {ump_reason}" if ump_reason and ump_reason not in ("unknown", "") and "neutral" not in ump_reason else "")
                            + f"). Gap = {diff:+.2f} ({display_sel}) at "
                            f"{chosen.american:+.0f} on {chosen.book}."
                        ),
                        sources=[chosen.book],
                    )
                )

        logger.info("total_projection: %d recs", len(recs))
        return recs

    # ------------------------------------------------------------------
    # Soccer: Dixon-Coles Poisson model (see ``soccer_model.py``).
    # ------------------------------------------------------------------

    # Confidence ceiling specifically for the soccer model — tighter than
    # the baseball/basketball market-anchored path because the Poisson
    # framework is sharper (true probability, not tanh-mapped gap).
    SOCCER_CONF_CAP = 0.60
    # Minimum edge (probability - implied) before we take the bet. Same
    # thinking as MIN_LINE_EDGE for other sports: stay above estimation
    # noise at our rolling-20 sample sizes.
    SOCCER_MIN_EDGE = 0.03

    def _soccer_picks(self, game, cfg):
        """Emit Dixon-Coles-based totals recs for one soccer game."""
        from ..soccer_model import project_soccer_totals as _soccer_dc
        proj = _soccer_dc(
            self.scoring, game.sport, game.home_team, game.away_team
        )
        if proj is None:
            return []

        prob_over_curve = proj["prob_over"]
        lam_h = proj["lambda_home"]
        lam_a = proj["lambda_away"]
        expected_total = proj["expected_total"]

        # Collect total lines (same shape as the non-soccer path).
        totals = [
            l for l in game.lines
            if l.market == "total"
            and l.american is not None
            and l.line is not None
        ]
        if not totals:
            return []

        by_book: dict[tuple[str, float], dict[str, object]] = {}
        for line in totals:
            key = (line.book.lower(), float(line.line))
            entry = by_book.setdefault(
                key,
                {"book": line.book, "line": line.line, "over": None, "under": None},
            )
            sel = line.selection.lower()
            if sel == "over":
                entry["over"] = line
            elif sel == "under":
                entry["under"] = line

        out: List[BetRecommendation] = []
        for key, entry in by_book.items():
            over = entry["over"]
            under = entry["under"]
            if over is None or under is None:
                continue
            total_num = float(entry["line"])
            # Dixon-Coles gives us P(over X.5). Soccer totals are
            # almost always on the half-line. For integer lines (push
            # risk), we conservatively treat as nearest 0.5 lower.
            line_for_lookup = total_num if total_num % 1 == 0.5 else (int(total_num) - 0.5)
            prob_over = prob_over_curve.get(line_for_lookup)
            if prob_over is None:
                # Line outside our curve (rare — e.g. 6.5+); fall back
                # to the expected-total comparison like non-soccer.
                continue
            prob_under = 1.0 - prob_over

            # Pick the side the model favors, compare to book price.
            if prob_over > prob_under:
                chosen = over
                display_sel = "Over"
                model_prob = prob_over
            else:
                chosen = under
                display_sel = "Under"
                model_prob = prob_under
            # Cap before computing edge — a 68% model prob with Kelly
            # on +120 juice would blow out stake. Confidence cap keeps
            # it honest.
            confidence = min(self.SOCCER_CONF_CAP, model_prob)
            implied = american_to_implied(chosen.american)
            edge = confidence - implied
            if edge < self.SOCCER_MIN_EDGE:
                continue

            stake_frac = min(
                kelly_fraction(confidence, chosen.decimal, cfg.kelly_fraction),
                cfg.max_bet_pct,
            )

            out.append(
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
                        f"Dixon-Coles totals: λ_home={lam_h:.2f}, "
                        f"λ_away={lam_a:.2f} (expected {expected_total:.2f} goals). "
                        f"P(Over {total_num:.1f})={prob_over:.3f}, "
                        f"P(Under)={prob_under:.3f} vs implied "
                        f"{implied:.3f} — taking {display_sel} at "
                        f"{chosen.american:+.0f} on {chosen.book}."
                    ),
                    sources=[chosen.book],
                )
            )
        return out
