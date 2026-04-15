"""MLB stadium coordinates + dome status.

Used to pull weather.gov forecasts for upcoming games. Dome
stadiums (and stadiums that normally close their retractable roofs)
are flagged so we skip weather adjustments — the climate inside
never changes.

Coordinates are home-plate center for each ballpark, accurate to
about 50 meters, good enough for a weather forecast grid cell.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple


# team display name (as it appears in Action Network / Bovada feeds)
# -> (lat, lon, indoor_or_domed)
_STADIUMS: Dict[str, Tuple[float, float, bool]] = {
    "Arizona Diamondbacks":    (33.4453, -112.0667, True),   # Chase Field (retractable, usually closed)
    "Atlanta Braves":          (33.8908,  -84.4678, False),  # Truist Park
    "Baltimore Orioles":       (39.2839,  -76.6217, False),  # Camden Yards
    "Boston Red Sox":          (42.3467,  -71.0972, False),  # Fenway Park
    "Chicago Cubs":            (41.9484,  -87.6553, False),  # Wrigley Field
    "Chicago White Sox":       (41.8300,  -87.6339, False),  # Rate Field / Guaranteed Rate
    "Cincinnati Reds":         (39.0975,  -84.5069, False),  # Great American Ball Park
    "Cleveland Guardians":     (41.4962,  -81.6852, False),  # Progressive Field
    "Colorado Rockies":        (39.7559, -104.9942, False),  # Coors Field (altitude!)
    "Detroit Tigers":          (42.3390,  -83.0485, False),  # Comerica Park
    "Houston Astros":          (29.7573,  -95.3555, True),   # Minute Maid (retractable, often closed)
    "Kansas City Royals":      (39.0517,  -94.4803, False),  # Kauffman Stadium
    "Los Angeles Angels":      (33.8003, -117.8827, False),  # Angel Stadium
    "Los Angeles Dodgers":     (34.0739, -118.2400, False),  # Dodger Stadium
    "Miami Marlins":           (25.7781,  -80.2197, True),   # loanDepot Park (retractable, usually closed)
    "Milwaukee Brewers":       (43.0281,  -87.9711, True),   # American Family Field (retractable, usually closed cold)
    "Minnesota Twins":         (44.9817,  -93.2776, False),  # Target Field
    "New York Mets":           (40.7571,  -73.8458, False),  # Citi Field
    "New York Yankees":        (40.8296,  -73.9262, False),  # Yankee Stadium
    "Athletics":               (37.7516, -122.2005, False),  # Oakland Coliseum
    "Oakland Athletics":       (37.7516, -122.2005, False),
    "Philadelphia Phillies":   (39.9061,  -75.1665, False),  # Citizens Bank Park
    "Pittsburgh Pirates":      (40.4469,  -80.0057, False),  # PNC Park
    "San Diego Padres":        (32.7076, -117.1570, False),  # Petco Park
    "San Francisco Giants":    (37.7786, -122.3893, False),  # Oracle Park
    "Seattle Mariners":        (47.5914, -122.3325, True),   # T-Mobile Park (retractable, often closed)
    "St. Louis Cardinals":     (38.6226,  -90.1928, False),  # Busch Stadium
    "Tampa Bay Rays":          (27.7683,  -82.6534, True),   # Tropicana Field (dome)
    "Texas Rangers":           (32.7473,  -97.0847, True),   # Globe Life Field (retractable, usually closed)
    "Toronto Blue Jays":       (43.6414,  -79.3894, True),   # Rogers Centre (retractable, often closed)
    "Washington Nationals":    (38.8730,  -77.0074, False),  # Nationals Park
}


def stadium(home_team: str) -> Optional[Tuple[float, float, bool]]:
    """Return (lat, lon, indoor) tuple or None if team unknown."""
    if not home_team:
        return None
    return _STADIUMS.get(home_team.strip())


def is_indoor(home_team: str) -> bool:
    """True if the stadium is a dome or typically closed retractable."""
    entry = stadium(home_team)
    return bool(entry and entry[2])
