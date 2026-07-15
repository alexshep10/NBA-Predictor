"""NBA team metadata."""

from __future__ import annotations

# Current and historical abbreviations from stats.nba.com game logs.
EAST_TEAMS = {
    "ATL",
    "BOS",
    "BKN",
    "BRK",
    "CHA",
    "CHO",
    "CHI",
    "CLE",
    "DET",
    "IND",
    "MIA",
    "MIL",
    "NJN",
    "NYK",
    "ORL",
    "PHI",
    "TOR",
    "WAS",
}

WEST_TEAMS = {
    "DAL",
    "DEN",
    "GSW",
    "GS",
    "HOU",
    "LAC",
    "LAL",
    "MEM",
    "MIN",
    "NO",
    "NOP",
    "OKC",
    "PHO",
    "PHX",
    "POR",
    "SAC",
    "SA",
    "SAS",
    "SEA",
    "UTA",
    "UTH",
}


def team_conference(team_abbr: str) -> str:
    abbr = team_abbr.upper()
    if abbr in EAST_TEAMS:
        return "east"
    if abbr in WEST_TEAMS:
        return "west"
    raise ValueError(f"Unknown conference for team: {team_abbr}")
