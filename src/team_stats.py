"""Derive team efficiency stats from game log box-score columns."""

from __future__ import annotations

import pandas as pd


def _team_possessions(fga: float, fta: float, oreb: float, tov: float) -> float:
    return max(fga + 0.44 * fta - oreb + tov, 1.0)


def _team_stats_row(
    pts: float,
    fgm: float,
    fga: float,
    fg3m: float,
    fg3a: float,
    fta: float,
    oreb: float,
    tov: float,
    pf: float,
    opp_pts: float,
) -> dict[str, float]:
    poss = _team_possessions(fga, fta, oreb, tov)
    efg = (fgm + 0.5 * fg3m) / fga if fga > 0 else 0.54
    return {
        "ortg": 100.0 * pts / poss,
        "drtg": 100.0 * opp_pts / poss,
        "pace": poss,
        "efg_pct": efg,
        "tov_pct": tov / poss if poss > 0 else 0.13,
        "reb_pct": 0.5,
        "three_pt_rate": fg3a / fga if fga > 0 else 0.35,
        "foul_rate": pf / poss if poss > 0 else 0.2,
        "pts": pts,
        "pts_allowed": opp_pts,
    }


def team_box_from_games(games: pd.DataFrame) -> pd.DataFrame:
    """Build team-level box score table from enriched game logs."""
    if "home_fga" not in games.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, g in games.iterrows():
        gid = g["game_id"]
        for side, opp in [("home", "away"), ("away", "home")]:
            stats = _team_stats_row(
                pts=float(g[f"{side}_pts"]),
                fgm=float(g[f"{side}_fgm"]),
                fga=float(g[f"{side}_fga"]),
                fg3m=float(g[f"{side}_fg3m"]),
                fg3a=float(g[f"{side}_fg3a"]),
                fta=float(g[f"{side}_fta"]),
                oreb=float(g[f"{side}_oreb"]),
                tov=float(g[f"{side}_tov"]),
                pf=float(g[f"{side}_pf"]),
                opp_pts=float(g[f"{opp}_pts"]),
            )
            rows.append({"game_id": gid, "team": g[f"{side}_team"], **stats})
    return pd.DataFrame(rows)
