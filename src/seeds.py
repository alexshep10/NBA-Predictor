"""Playoff seed computation from regular-season results."""

from __future__ import annotations

import pandas as pd

from .teams import team_conference


def _team_records(games: pd.DataFrame, season: str) -> pd.DataFrame:
    season_games = games[games["season"] == season]
    records: dict[str, dict[str, int]] = {}
    for _, row in season_games.iterrows():
        for team, opp, pts, opp_pts in [
            (row["home_team"], row["away_team"], row["home_pts"], row["away_pts"]),
            (row["away_team"], row["home_team"], row["away_pts"], row["home_pts"]),
        ]:
            entry = records.setdefault(team, {"wins": 0, "losses": 0})
            if pts > opp_pts:
                entry["wins"] += 1
            else:
                entry["losses"] += 1
    rows = []
    for team, rec in records.items():
        total = rec["wins"] + rec["losses"]
        rows.append(
            {
                "team": team,
                "wins": rec["wins"],
                "losses": rec["losses"],
                "win_pct": rec["wins"] / total if total else 0.0,
                "conference": team_conference(team),
            }
        )
    return pd.DataFrame(rows)


def compute_seeds(games: pd.DataFrame, season: str) -> dict[str, int]:
    """Return team -> seed (1-8 per conference) for a season."""
    records = _team_records(games, season)
    seeds: dict[str, int] = {}
    for conf in ("east", "west"):
        conf_teams = records[records["conference"] == conf].sort_values(
            ["wins", "win_pct"], ascending=False
        )
        for rank, team in enumerate(conf_teams["team"].tolist()[:8], start=1):
            seeds[team] = rank
    return seeds


def compute_win_pcts(games: pd.DataFrame, season: str) -> dict[str, float]:
    records = _team_records(games, season)
    return dict(zip(records["team"], records["win_pct"]))
