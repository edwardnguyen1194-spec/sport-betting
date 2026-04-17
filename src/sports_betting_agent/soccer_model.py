"""Dixon-Coles totals model for soccer.

Soccer totals are a textbook Poisson problem: goals are discrete, rare,
and close-to-independent events. But in the low-scoring regime (0-0,
1-0, 0-1, 1-1) draw and low-score outcomes are slightly more correlated
than independent Poisson would predict — the empirical joint frequency
differs from the product of marginals. Dixon & Coles (1997, JRSS C)
introduced a single correlation parameter ρ that corrects exactly those
four cells while leaving the rest of the scoreline matrix alone.

We reuse the rolling goals-for / goals-against data from
``TeamScoringTracker`` (same data the baseball / basketball model
projections use) to estimate per-team attack and defense strengths,
then produce a full P(total = k) distribution and integrate to get
P(Over X.5).

API
---

``project_soccer_totals(tracker, sport, home, away) → dict`` returns::

    {
        "lambda_home": 1.45,       # expected home goals
        "lambda_away": 1.12,
        "expected_total": 2.57,
        "prob_over": {
            1.5: 0.824, 2.5: 0.581, 3.5: 0.306, 4.5: 0.121, ...
        },
    }

Use ``prob_over[line]`` directly as the model's win probability on an
Over bet at that line; ``1 - prob_over[line]`` is the Under probability.
No tanh confidence mapping needed — the model IS the probability.

Literature
----------

- Dixon, M. J. & Coles, S. G. (1997). *Modelling Association Football
  Scores and Inefficiencies in the Football Betting Market.* JRSS C
  46(2): 265-280.
- Karlis, D. & Ntzoufras, I. (2000). On modelling soccer data. ρ
  empirical values: -0.05 to -0.15 for low-scoring leagues, with EPL
  and MLS around -0.10.

We don't run a full MLE fit (would need a season of match-level data
we don't currently store). Instead we estimate attack/defense from the
venue-split rolling averages already tracked, normalize by a stored
league-average per sport, and use a fixed ρ per league (the values
above are stable across seasons).
"""

from __future__ import annotations

import math
from typing import Dict, Optional


# League-average goals-per-team per game + Dixon-Coles rho, 2023-2025.
_LEAGUE_PARAMS: Dict[str, Dict[str, float]] = {
    # MLS averages ~1.45 goals/team/game; home advantage largest in MLS
    # (~0.22 goals/team/game extra) per MLS-Stats public analytics.
    "soccer_mls":  {"mu": 1.45, "home_adv": 0.22, "rho": -0.10},
    "soccer_epl":  {"mu": 1.40, "home_adv": 0.18, "rho": -0.10},
    "soccer_ucl":  {"mu": 1.55, "home_adv": 0.20, "rho": -0.10},
}

# Max goals to sum over when building the scoreline matrix. P(8+) is
# negligible for soccer; truncating at 8 captures >99.99% probability.
MAX_GOALS = 8


def _dc_correction(i: int, j: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles multiplicative correction term for the (i, j) cell.

    Only four cells get adjusted; everything else multiplies by 1.0.
    """
    if i == 0 and j == 0:
        return 1.0 - (lam_h * lam_a * rho)
    if i == 0 and j == 1:
        return 1.0 + (lam_h * rho)
    if i == 1 and j == 0:
        return 1.0 + (lam_a * rho)
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF at k for rate lam. Safe for k=0..MAX_GOALS."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def project_soccer_totals(
    tracker,   # TeamScoringTracker
    sport: str,
    home: str,
    away: str,
) -> Optional[Dict[str, object]]:
    """Return Dixon-Coles expected goals + full P(over X.5) curve.

    Returns ``None`` when either team has too few games tracked or the
    sport is not a configured soccer league.
    """
    params = _LEAGUE_PARAMS.get(sport)
    if params is None:
        return None

    h = tracker.team(sport, home)
    a = tracker.team(sport, away)
    if not h or not a:
        return None

    # Use venue-specific where possible; fall back to combined if a
    # team hasn't played enough games at that venue in our window.
    def _avg(rec, primary, fallback):
        vals = rec.get(primary, []) or []
        if len(vals) >= 3:
            return sum(vals) / len(vals)
        combined = (rec.get(primary, []) or []) + (rec.get(fallback, []) or [])
        if len(combined) >= 3:
            return sum(combined) / len(combined)
        return None

    h_scored = _avg(h, "home_scored", "away_scored")
    h_allowed = _avg(h, "home_allowed", "away_allowed")
    a_scored = _avg(a, "away_scored", "home_scored")
    a_allowed = _avg(a, "away_allowed", "home_allowed")
    if None in (h_scored, h_allowed, a_scored, a_allowed):
        return None

    mu = params["mu"]
    home_adv = params["home_adv"]
    rho = params["rho"]

    # Attack strength = team scoring rate / league average; defense
    # strength = team goals-allowed rate / league average (lower =
    # better). The product-with-league-mean reconstruction is the
    # standard Dixon-Coles parameterization when you don't MLE-fit
    # team-specific alphas.
    attack_h = h_scored / mu
    attack_a = a_scored / mu
    defense_h = h_allowed / mu
    defense_a = a_allowed / mu

    # Expected goals per side: attack × opposing defense × league mean,
    # with home-field bonus applied only to the home team.
    lam_h = (attack_h * defense_a * mu) + home_adv
    lam_a = (attack_a * defense_h * mu)
    # Guard against zero/negative from weird rolling data.
    lam_h = max(0.05, lam_h)
    lam_a = max(0.05, lam_a)

    # Build the full scoreline probability matrix with Dixon-Coles
    # correction on the low-score cells.
    total_prob = 0.0
    total_pmf: Dict[int, float] = {}
    for i in range(MAX_GOALS + 1):
        p_i = _poisson_pmf(i, lam_h)
        for j in range(MAX_GOALS + 1):
            p_j = _poisson_pmf(j, lam_a)
            p_ij = p_i * p_j * _dc_correction(i, j, lam_h, lam_a, rho)
            total_prob += p_ij
            total_goals = i + j
            total_pmf[total_goals] = total_pmf.get(total_goals, 0.0) + p_ij
    # Re-normalize: the DC correction slightly perturbs the total mass,
    # and truncation at MAX_GOALS drops a tiny tail.
    if total_prob > 0:
        for k in total_pmf:
            total_pmf[k] /= total_prob

    # P(over X.5) = P(total >= X+1) = P(total > X).
    # We care about half-integer soccer totals mainly at 1.5, 2.5, 3.5,
    # sometimes 0.5 and 4.5+. Build the full over-curve up through 5.5.
    prob_over: Dict[float, float] = {}
    for threshold_int in range(0, 6):
        line = threshold_int + 0.5
        p = sum(pmf for k, pmf in total_pmf.items() if k > threshold_int)
        prob_over[line] = round(p, 4)

    expected_total = sum(k * v for k, v in total_pmf.items())

    return {
        "lambda_home": round(lam_h, 3),
        "lambda_away": round(lam_a, 3),
        "expected_total": round(expected_total, 3),
        "prob_over": prob_over,
        "rho": rho,
    }
