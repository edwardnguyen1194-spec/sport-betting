"""Elo-derived spread strategy.

Turns our in-house Elo power ratings (see
:mod:`sports_betting_agent.power_ratings`) into ATS picks by comparing
an Elo-implied point spread to the market's posted main line.

Logic
-----

For each game we:

1. Fetch Elo for home + away. If either rating is missing, skip —
   we will not bet against a market we cannot model.
2. Convert the Elo difference (plus home-field advantage) into an
   expected point spread using a sport-specific Elo-per-point ratio.
   Industry-standard defaults:

     * NFL ~ 1 pt per 25 Elo
     * NBA ~ 1 pt per 28 Elo
     * NCAAF/NCAAB ~ same as pro football/hoops
     * NHL ~ 0.3 goals per 100 Elo
     * MLB ~ 0.3 runs per 50 Elo

3. Compare the Elo-derived spread to the book's posted main-line
   spread. If the absolute gap exceeds a sport-specific minimum
   (e.g. 1.5 NFL pts, 2.0 NBA pts, 0.5 MLB runs), bet whichever side
   Elo prefers.
4. Compute confidence from the logistic Elo win-prob formula,
   capped at 0.58 so a single model signal never overwhelms the
   ensemble.
5. Only emit the pick when the best available price is on a sharp
   book (Pinnacle / Circa / Bovada / BookMaker / BetOnline).
6. Require edge (confidence minus implied-best) >= 0.025, and size
   with fractional Kelly.

Reasoning string shows both Elo ratings and the derived vs posted
spread so the dashboard can explain the pick at a glance.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from ..config import Settings, get_settings
from ..models_schema import (
    GameOdds,
    OddsLine,
    _norm,
    american_to_implied,
    kelly_fraction,
)
from ..power_ratings import EloRatings, SPORT_CONFIG
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)


# Sharp books whose prices we treat as honest. Soft-book prices are
# ignored — they carry promo juice and cross-book bias we cannot
# de-vig cleanly.
SHARP_BOOKS = {"pinnacle", "circa", "bovada", "bookmaker", "betonline"}


# Elo points per one expected scoring unit. A larger value == Elo
# moves less per scored point (i.e. football/basketball are harder to
# move than a single MLB run).
ELO_PER_POINT = {
    "football_nfl": 25.0,
    "football_ncaaf": 25.0,
    "football_cfl": 25.0,
    "basketball_nba": 28.0,
    "basketball_wnba": 28.0,
    "basketball_ncaab": 28.0,
    "basketball_euroleague": 30.0,  # slightly higher — European pace is lower
    # NHL: 0.3 goals per 100 Elo => 100 / 0.3 ~= 333 Elo per goal.
    "hockey_nhl": 333.33,
    "hockey_khl": 333.33,
    # MLB: 0.3 runs per 50 Elo => 50 / 0.3 ~= 166.67 Elo per run.
    "baseball_mlb": 166.67,
    "baseball_ncaa": 166.67,
    # Soccer: ~300 Elo per goal scored margin (tighter than hockey
    # because games are lower-scoring). Covers MLS/EPL/La Liga etc.
    "soccer_mls": 300.0,
    "soccer_epl": 300.0,
    "soccer_ucl": 300.0,
    "soccer_uel": 300.0,
    "soccer_esp": 300.0,
    "soccer_ita": 300.0,
    "soccer_ger": 300.0,
    "soccer_fra": 300.0,
    # Tennis: spreads are ATP/WTA "games spread" (e.g. -4.5, +2.5).
    # Elo diff ~120 per game advantage is a reasonable mapping from
    # published tennis Elo regressions (Sackmann, Riles).
    "tennis_atp": 120.0,
    "tennis_wta": 120.0,
    # MMA: no true spread market — skip (moneyline + method props only
    # which Uncle's rule forbids). MMA gets no elo_spread picks.
}


# Minimum absolute gap (in points/runs/goals, depending on the sport)
# between the Elo-implied spread and the posted market spread before
# we will take the bet. Below this, the edge is noise.
MIN_GAP_BY_SPORT = {
    "baseball_mlb": 0.5,
    "baseball_ncaa": 0.5,
    "hockey_nhl": 0.4,
    "hockey_khl": 0.4,
    "basketball_nba": 2.0,
    "basketball_wnba": 2.0,
    "basketball_ncaab": 2.5,
    "basketball_euroleague": 2.5,
    "football_nfl": 1.5,
    "football_ncaaf": 2.0,
    "football_cfl": 2.0,
    "soccer_mls": 0.4,
    "soccer_epl": 0.4,
    "soccer_ucl": 0.4,
    "soccer_uel": 0.4,
    "soccer_esp": 0.4,
    "soccer_ita": 0.4,
    "soccer_ger": 0.4,
    "soccer_fra": 0.4,
    "tennis_atp": 1.0,   # 1 full game edge minimum before betting
    "tennis_wta": 1.0,
}


# Confidence cap: one-signal Elo belief should never exceed ~58%. The
# ensemble can still push higher when multiple independent strategies
# agree on the same side.
CONFIDENCE_CAP = 0.58


class EloSpreadStrategy(Strategy):
    """Bet ATS when the Elo-implied spread disagrees with the book."""

    name = "elo_spread"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        elo: Optional[EloRatings] = None,
        min_edge: float = 0.025,
    ) -> None:
        self.settings = settings or get_settings()
        # Lazily construct an EloRatings reading from the configured
        # data dir if the caller didn't inject one. Production wires a
        # shared instance in; tests can omit it and get a fresh reader.
        self.elo = elo or EloRatings(self.settings.data_dir)
        self.min_edge = min_edge

    # ------------------------------------------------------------------

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        recs: List[BetRecommendation] = []

        for game in games:
            sport = game.sport
            if sport not in ELO_PER_POINT or sport not in MIN_GAP_BY_SPORT:
                continue

            h_elo = self.elo.get_rating(sport, game.home_team)
            a_elo = self.elo.get_rating(sport, game.away_team)
            if h_elo is None or a_elo is None:
                continue

            # Elo-implied spread from the HOME team's perspective.
            # Positive elo_spread_home = home favored by N points ATS,
            # so the posted handicap for home should be roughly
            # -elo_spread_home (e.g. home favored by 7 => home -7).
            cfg = SPORT_CONFIG.get(sport, SPORT_CONFIG["baseball_mlb"])
            hfa_elo = cfg.get("hfa", 0.0)
            elo_diff_home = (h_elo + hfa_elo) - a_elo
            per_point = ELO_PER_POINT[sport]
            elo_spread_home = elo_diff_home / per_point  # positive = home favored

            # Find the main (median) spread handicap at any sharp book.
            main_home_handicap = _main_sharp_home_handicap(game)
            if main_home_handicap is None:
                continue

            # The book's implied home margin = -(home handicap). If
            # home is -7 on the board, book thinks home wins by 7.
            book_home_margin = -main_home_handicap

            # Gap is comparing two margin estimates on the same scale.
            gap = elo_spread_home - book_home_margin
            min_gap = MIN_GAP_BY_SPORT[sport]
            if abs(gap) < min_gap:
                continue

            # Elo prefers whichever side it thinks has extra margin
            # relative to the book.
            if gap > 0:
                # Elo says home wins by more than book implies -> bet home ATS.
                pick_side = "home"
                pick_team = game.home_team
                pick_handicap = main_home_handicap
            else:
                # Elo says away wins by more (or loses by less) -> bet away ATS.
                pick_side = "away"
                pick_team = game.away_team
                pick_handicap = -main_home_handicap

            # Find the best sharp price on the chosen side at the chosen handicap.
            best_line = _best_sharp_line(game, pick_team, pick_handicap)
            if best_line is None or best_line.decimal is None or best_line.american is None:
                continue

            # Confidence via the standard Elo logistic win-prob formula.
            # Apply it from the picked side's perspective so confidence
            # always refers to the side we are betting.
            if pick_side == "home":
                elo_diff = elo_diff_home
            else:
                # Away has the HFA flipped against them.
                elo_diff = (a_elo - h_elo) - hfa_elo
            win_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            # ATS win prob isn't the same as ML win prob — a favorite
            # giving 7 wins ATS much less often than it wins SU. Pull
            # the estimate toward 0.5 based on how much of the margin
            # is already priced into the spread. When |gap| is small
            # relative to the sport's scoring volatility, the ATS edge
            # collapses; when gap is huge, ATS ~ ML.
            ats_conf = _scale_to_ats(win_prob, gap, sport)
            confidence = min(CONFIDENCE_CAP, ats_conf)

            implied_best = american_to_implied(best_line.american)
            edge = confidence - implied_best
            if edge < self.min_edge:
                continue

            stake_frac = min(
                kelly_fraction(confidence, best_line.decimal, self.settings.kelly_fraction),
                self.settings.max_bet_pct,
            )
            if stake_frac <= 0:
                continue

            elo_spread_display = (
                f"home -{elo_spread_home:.1f}"
                if elo_spread_home > 0
                else f"away -{abs(elo_spread_home):.1f}"
            )
            reason = (
                f"Elo spread: {game.home_team} {h_elo:.0f} vs {game.away_team} "
                f"{a_elo:.0f} (HFA {hfa_elo:.0f}) -> model {elo_spread_display}; "
                f"book main {main_home_handicap:+.1f} home / "
                f"{-main_home_handicap:+.1f} away. Gap {gap:+.1f} "
                f"{'pts' if sport not in ('baseball_mlb','baseball_ncaa','hockey_nhl') else ('runs' if sport.startswith('baseball') else 'goals')}"
                f" -> take {pick_team} {pick_handicap:+.1f} "
                f"at {best_line.american:+.0f} ({best_line.book}). "
                f"Confidence {confidence:.1%}, edge +{edge:.1%}."
            )

            recs.append(
                BetRecommendation(
                    game_key=game.game_key,
                    sport=game.sport,
                    league=game.league,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    market="spread",
                    selection=pick_team,
                    american=best_line.american,
                    decimal=best_line.decimal,
                    line=pick_handicap,
                    book=best_line.book,
                    strategy=self.name,
                    confidence=round(confidence, 4),
                    edge=round(edge, 4),
                    stake_fraction=round(stake_frac, 4),
                    reasoning=reason,
                    sources=sorted({
                        ln.book
                        for ln in game.lines
                        if ln.market == "spread" and ln.book
                    }),
                    meta={
                        "elo_home": round(h_elo, 1),
                        "elo_away": round(a_elo, 1),
                        "elo_spread_home": round(elo_spread_home, 2),
                        "book_home_handicap": main_home_handicap,
                        "gap": round(gap, 2),
                    },
                )
            )

        logger.info("elo_spread: %d recs", len(recs))
        return recs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _main_sharp_home_handicap(game: GameOdds) -> Optional[float]:
    """Return the median home-side handicap across sharp books.

    The main line is the handicap most books agree on. We compute it
    from sharp books only so soft promo lines don't drag the median.
    """

    home_norm = _norm(game.home_team)
    handicaps: List[float] = []
    for ln in game.lines:
        if ln.market != "spread" or ln.line is None or ln.american is None:
            continue
        if ln.book.lower() not in SHARP_BOOKS:
            continue
        if _norm(ln.selection) != home_norm:
            continue
        handicaps.append(float(ln.line))

    if not handicaps:
        # Fall back to median across ALL books for the home side. Still
        # a real line estimate, just less reliable than sharp-only.
        for ln in game.lines:
            if ln.market != "spread" or ln.line is None or ln.american is None:
                continue
            if _norm(ln.selection) != home_norm:
                continue
            handicaps.append(float(ln.line))
    if not handicaps:
        return None

    handicaps.sort()
    mid = len(handicaps) // 2
    if len(handicaps) % 2 == 0:
        return (handicaps[mid - 1] + handicaps[mid]) / 2.0
    return handicaps[mid]


def _best_sharp_line(
    game: GameOdds, team: str, handicap: float, tol: float = 0.01
) -> Optional[OddsLine]:
    """Highest-paying sharp-book line for ``team`` at ``handicap``."""

    team_norm = _norm(team)
    candidates: List[OddsLine] = []
    for ln in game.lines:
        if ln.market != "spread" or ln.line is None or ln.decimal is None:
            continue
        if ln.book.lower() not in SHARP_BOOKS:
            continue
        if _norm(ln.selection) != team_norm:
            continue
        if abs(float(ln.line) - handicap) > tol:
            continue
        candidates.append(ln)
    if not candidates:
        return None
    return max(candidates, key=lambda l: l.decimal or 0.0)


def _scale_to_ats(su_win_prob: float, gap: float, sport: str) -> float:
    """Pull straight-up win prob toward 0.5 based on ATS gap.

    A huge Elo edge vs the book (gap >> min_gap) keeps most of the
    straight-up probability. A thin edge (|gap| just at threshold)
    means we're really only barely beating the spread, so the
    confidence should sit close to 0.5 + a modest bump.
    """

    min_gap = MIN_GAP_BY_SPORT.get(sport, 1.5)
    # Dimensionless "how many thresholds of edge do we have".
    strength = abs(gap) / max(min_gap, 1e-6)
    # Clamp so one absurd reading can't hit 1.0.
    strength = min(1.5, strength)
    # Weight between 0 (use 0.5) and 1 (use full SU prob). strength=1
    # (right at threshold) gives weight ~0.4; strength=1.5 gives ~0.6.
    weight = min(0.6, 0.4 * strength)
    return 0.5 + weight * (su_win_prob - 0.5)
