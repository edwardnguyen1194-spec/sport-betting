"""MLB park factors — a run-scoring multiplier per ballpark.

A park factor greater than 1.0 means a stadium plays more
hitter-friendly than league average (a neutral 1.00); under 1.0
means pitcher-friendly. The values here come from multi-season
Statcast / FanGraphs composites (2021-2025) and are stable from
year to year for non-renovated parks.

Why it matters for totals
-------------------------

The market's posted total already bakes in some of each park's
effect, but research (BetMGM/Pinnacle research blogs, BallparkPal)
consistently shows books under-weight extremes like Coors and
Oakland. Applying a park-factor nudge on top of a rolling scoring
model catches the miss.

API
---

``park_factor(home_team)`` returns a multiplier near 1.0; callers
multiply their projected total by it and compare to the book's
posted total. Teams not in the table return 1.0 (no adjustment).
"""

from __future__ import annotations

from typing import Dict


# Normalized display names as they appear in Action Network /
# Bovada feeds. Values are composite 2024 total-runs park factors.
# Source: FanGraphs / Baseball Savant public splits, 2021-2024.
_PARK_FACTORS: Dict[str, float] = {
    # Extreme hitter parks.
    "Colorado Rockies":       1.15,   # Coors Field — altitude
    "Cincinnati Reds":        1.08,   # Great American Ball Park
    "Boston Red Sox":         1.06,   # Fenway — Green Monster helps LHB
    "Milwaukee Brewers":      1.05,   # Miller/American Family Field
    "Philadelphia Phillies":  1.05,   # Citizens Bank Park
    "Kansas City Royals":     1.04,
    "Baltimore Orioles":      1.03,
    "Chicago White Sox":      1.03,
    "Texas Rangers":          1.03,   # Globe Life Field
    "Atlanta Braves":         1.02,   # Truist Park

    # Mildly hitter-friendly.
    "Houston Astros":         1.02,
    "Arizona Diamondbacks":   1.01,   # depends on roof
    "Chicago Cubs":           1.01,   # Wrigley — wind-dependent, see weather

    # Roughly neutral.
    "Toronto Blue Jays":      1.00,
    "Minnesota Twins":        1.00,
    "Washington Nationals":   1.00,
    "New York Yankees":       1.00,
    "St. Louis Cardinals":    0.99,
    "Los Angeles Angels":     0.99,
    "Cleveland Guardians":    0.99,
    "Tampa Bay Rays":         0.98,
    "Pittsburgh Pirates":     0.97,

    # Pitcher parks.
    "New York Mets":          0.96,   # Citi Field
    "Detroit Tigers":         0.96,
    "Miami Marlins":          0.95,   # loanDepot
    "San Francisco Giants":   0.94,   # Oracle Park — marine layer
    "San Diego Padres":       0.94,   # Petco
    "Athletics":              0.93,   # Oakland Coliseum (historic pitcher park)
    "Oakland Athletics":      0.93,
    "Seattle Mariners":       0.93,   # T-Mobile Park
    "Los Angeles Dodgers":    0.92,   # Dodger Stadium — marine layer at night
}


def park_factor(home_team: str) -> float:
    """Return the run-scoring multiplier for the home team's park.

    Returns 1.0 when the team is not in the table (neutral, no
    adjustment applied). We match on the full display name the
    aggregator uses; teams with multiple names (Athletics / Oakland
    Athletics) have both keys.
    """
    if not home_team:
        return 1.0
    key = home_team.strip()
    return _PARK_FACTORS.get(key, 1.0)


def adjust_total(projected_total: float, home_team: str) -> float:
    """Return the projected total scaled by the park factor."""
    return projected_total * park_factor(home_team)
