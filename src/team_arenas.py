"""NBA team arena coordinates for travel distance."""

from __future__ import annotations

import math

# Approximate arena city coordinates (lat, lon).
ARENA_COORDS: dict[str, tuple[float, float]] = {
    "ATL": (33.757, -84.396),
    "BOS": (42.366, -71.062),
    "BKN": (40.683, -73.975),
    "BRK": (40.683, -73.975),
    "CHA": (35.225, -80.839),
    "CHO": (35.225, -80.839),
    "CHI": (41.881, -87.674),
    "CLE": (41.497, -81.688),
    "DAL": (32.790, -96.810),
    "DEN": (39.748, -105.007),
    "DET": (42.341, -83.055),
    "GSW": (37.768, -122.387),
    "GS": (37.768, -122.387),
    "HOU": (29.751, -95.362),
    "IND": (39.764, -86.155),
    "LAC": (34.043, -118.267),
    "LAL": (34.043, -118.267),
    "MEM": (35.138, -90.051),
    "MIA": (25.781, -80.188),
    "MIL": (43.045, -87.917),
    "MIN": (44.979, -93.276),
    "NOP": (29.949, -90.082),
    "NO": (29.949, -90.082),
    "NYK": (40.751, -73.993),
    "OKC": (35.463, -97.515),
    "ORL": (28.539, -81.384),
    "PHI": (39.901, -75.172),
    "PHX": (33.446, -112.071),
    "PHO": (33.446, -112.071),
    "POR": (45.532, -122.667),
    "SAC": (38.580, -121.499),
    "SAS": (29.427, -98.438),
    "SA": (29.427, -98.438),
    "TOR": (43.643, -79.379),
    "UTA": (40.768, -111.901),
    "UTH": (40.768, -111.901),
    "WAS": (38.898, -77.021),
}

ALTITUDE_TEAMS = {"DEN"}


def travel_miles(from_team: str, to_team: str) -> float:
    """Great-circle distance between two team cities in miles."""
    a = ARENA_COORDS.get(from_team.upper())
    b = ARENA_COORDS.get(to_team.upper())
    if a is None or b is None:
        return 0.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))


def is_altitude_home(team: str) -> bool:
    return team.upper() in ALTITUDE_TEAMS
