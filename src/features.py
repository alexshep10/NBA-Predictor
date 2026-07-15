"""Feature engineering for per-game score prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .seeds import compute_seeds, compute_win_pcts
from .team_arenas import is_altitude_home, travel_miles

ROLLING_WINDOW = 10
PLAYER_WINDOW = 15
TOP_PLAYERS = 5
BLOWOUT_MARGIN = 15
KEY_PLAYER_MIN_THRESHOLD = 18.0

FEATURE_COLUMNS = [
    # efficiency
    "home_roll_ortg",
    "home_roll_drtg",
    "home_roll_pace",
    "away_roll_ortg",
    "away_roll_drtg",
    "away_roll_pace",
    "home_roll_efg_pct",
    "away_roll_efg_pct",
    "home_roll_tov_pct",
    "away_roll_tov_pct",
    "home_roll_reb_pct",
    "away_roll_reb_pct",
    # player strength + injuries
    "home_strength_index",
    "away_strength_index",
    "home_key_players_out",
    "away_key_players_out",
    # travel / home environment
    "away_travel_miles",
    "home_back_to_back",
    "away_back_to_back",
    "is_altitude",
    # playoff context
    "is_playoffs",
    "home_playoff_games",
    "away_playoff_games",
    "seed_diff",
    "home_rs_win_pct",
    "away_rs_win_pct",
    # style matchup
    "pace_mismatch",
    "three_pt_rate_diff",
    "foul_rate_diff",
    # game context
    "home_rest_days",
    "away_rest_days",
    "is_elimination",
    "home_after_blowout",
    "away_after_blowout",
    "h2h_margin",
]

TEAM_STAT_COLS = [
    "ortg",
    "drtg",
    "pace",
    "efg_pct",
    "tov_pct",
    "reb_pct",
    "three_pt_rate",
    "foul_rate",
    "pts",
    "pts_allowed",
]


def _empty_team_state() -> dict:
    return {
        "stats": {col: [] for col in TEAM_STAT_COLS},
        "last_game_date": None,
        "last_location": None,
        "last_margin": 0.0,
        "playoff_games": 0,
        "player_minutes": {},
        "player_pts": {},
        "strength_index": 110.0,
        "key_players_out": 0,
    }


def _roll_mean(values: list[float], default: float) -> float:
    if len(values) >= 3:
        return float(np.mean(values[-ROLLING_WINDOW:]))
    return default


def _compute_strength_index(player_minutes: dict[int, list[float]], player_pts: dict[int, list[float]]) -> float:
    scores: list[float] = []
    for pid, mins in player_minutes.items():
        if len(mins) < 3:
            continue
        avg_min = float(np.mean(mins[-PLAYER_WINDOW:]))
        avg_pts = float(np.mean(player_pts.get(pid, [0.0])[-PLAYER_WINDOW:]))
        if avg_min >= 10:
            scores.append(avg_pts * avg_min / 48.0)
    scores.sort(reverse=True)
    return float(sum(scores[:TOP_PLAYERS])) if scores else 110.0


def _infer_key_players_out(
    player_minutes: dict[int, list[float]],
    current_minutes: dict[int, float],
) -> int:
    candidates: list[tuple[float, int]] = []
    for pid, mins in player_minutes.items():
        if len(mins) < 5:
            continue
        avg_min = float(np.mean(mins[-PLAYER_WINDOW:]))
        if avg_min >= KEY_PLAYER_MIN_THRESHOLD:
            candidates.append((avg_min, pid))
    candidates.sort(reverse=True)
    top_ids = [pid for _, pid in candidates[:TOP_PLAYERS]]
    out = 0
    for pid in top_ids:
        if current_minutes.get(pid, 0.0) <= 0.0:
            out += 1
    return out


BOX_STAT_COLS = [
    "ortg",
    "drtg",
    "pace",
    "efg_pct",
    "tov_pct",
    "reb_pct",
    "three_pt_rate",
    "foul_rate",
]


def _attach_team_box_stats(games: pd.DataFrame, team_box: pd.DataFrame) -> pd.DataFrame:
    box_cols = ["game_id", "team"] + [c for c in BOX_STAT_COLS if c in team_box.columns]
    home_box = team_box[box_cols].rename(
        columns=lambda c: f"home_{c}" if c not in ("game_id", "team") else c
    ).rename(columns={"team": "home_team"})
    away_box = team_box[box_cols].rename(
        columns=lambda c: f"away_{c}" if c not in ("game_id", "team") else c
    ).rename(columns={"team": "away_team"})
    out = games.merge(home_box, on=["game_id", "home_team"], how="left")
    out = out.merge(away_box, on=["game_id", "away_team"], how="left")
    out["home_pts_allowed"] = out["away_pts"]
    out["away_pts_allowed"] = out["home_pts"]
    return out


def _team_long_history(games: pd.DataFrame, team_box: pd.DataFrame) -> pd.DataFrame:
    """Chronological per-team rows with box score stats."""
    enriched = _attach_team_box_stats(games, team_box)
    home = enriched[
        [
            "game_id",
            "game_date",
            "season",
            "home_team",
            "home_pts",
            "home_pts_allowed",
            "home_ortg",
            "home_drtg",
            "home_pace",
            "home_efg_pct",
            "home_tov_pct",
            "home_reb_pct",
            "home_three_pt_rate",
            "home_foul_rate",
        ]
    ].rename(
        columns={
            "home_team": "team",
            "home_pts": "pts",
            "home_pts_allowed": "pts_allowed",
            "home_ortg": "ortg",
            "home_drtg": "drtg",
            "home_pace": "pace",
            "home_efg_pct": "efg_pct",
            "home_tov_pct": "tov_pct",
            "home_reb_pct": "reb_pct",
            "home_three_pt_rate": "three_pt_rate",
            "home_foul_rate": "foul_rate",
        }
    )
    home["is_home"] = 1
    home["location"] = home["team"]
    home["opponent"] = enriched["away_team"].values

    away = enriched[
        [
            "game_id",
            "game_date",
            "season",
            "away_team",
            "away_pts",
            "away_pts_allowed",
            "away_ortg",
            "away_drtg",
            "away_pace",
            "away_efg_pct",
            "away_tov_pct",
            "away_reb_pct",
            "away_three_pt_rate",
            "away_foul_rate",
        ]
    ].rename(
        columns={
            "away_team": "team",
            "away_pts": "pts",
            "away_pts_allowed": "pts_allowed",
            "away_ortg": "ortg",
            "away_drtg": "drtg",
            "away_pace": "pace",
            "away_efg_pct": "efg_pct",
            "away_tov_pct": "tov_pct",
            "away_reb_pct": "reb_pct",
            "away_three_pt_rate": "three_pt_rate",
            "away_foul_rate": "foul_rate",
        }
    )
    away["is_home"] = 0
    away["location"] = enriched["home_team"].values
    away["opponent"] = enriched["home_team"].values

    history = pd.concat([home, away], ignore_index=True)
    return history.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)


def _add_rolling_team_features(history: pd.DataFrame) -> pd.DataFrame:
    history = history.copy()
    grouped = history.groupby("team", sort=False)
    for col in TEAM_STAT_COLS:
        history[f"roll_{col}"] = grouped[col].transform(
            lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean()
        )
    history["last_game_date"] = grouped["game_date"].shift(1)
    history["last_location"] = grouped["location"].shift(1)
    history["last_margin"] = grouped["pts"].transform(lambda s: s.shift(1)) - grouped[
        "pts_allowed"
    ].transform(lambda s: s.shift(1))
    history["rest_days"] = (history["game_date"] - history["last_game_date"]).dt.days
    history["back_to_back"] = (history["rest_days"] == 1).astype(int)
    history["after_blowout"] = (history["last_margin"].abs() >= BLOWOUT_MARGIN).astype(int)
    history["travel_miles"] = 0.0
    away_mask = history["is_home"] == 0
    history.loc[away_mask, "travel_miles"] = history.loc[away_mask].apply(
        lambda r: travel_miles(r["team"], r["location"])
        if pd.notna(r.get("last_location"))
        else 0.0,
        axis=1,
    )
    return history


def _build_player_game_features(
    games: pd.DataFrame,
    player_box: pd.DataFrame,
) -> pd.DataFrame:
    """Per game/team: strength index and inferred injuries before tip."""
    if player_box.empty:
        return pd.DataFrame(columns=["game_id", "team", "strength_index", "key_players_out"])

    meta = games[["game_id", "game_date"]].drop_duplicates()
    pb = player_box.merge(meta, on="game_id", how="inner")
    pb = pb.sort_values(["team", "game_date", "game_id"])

    results: list[dict] = []
    for team, team_games in pb.groupby("team", sort=False):
        history_mins: dict[int, list[float]] = {}
        history_pts: dict[int, list[float]] = {}
        seen_games: list[str] = []
        game_ids = team_games["game_id"].drop_duplicates().tolist()

        for gid in game_ids:
            gplayers = team_games[team_games["game_id"] == gid]
            current_minutes = dict(zip(gplayers["player_id"], gplayers["minutes"]))
            results.append(
                {
                    "game_id": gid,
                    "team": team,
                    "strength_index": _compute_strength_index(history_mins, history_pts),
                    "key_players_out": _infer_key_players_out(history_mins, current_minutes),
                }
            )
            for _, row in gplayers.iterrows():
                pid = int(row["player_id"])
                history_mins.setdefault(pid, []).append(float(row["minutes"]))
                history_pts.setdefault(pid, []).append(float(row["pts"]))
                history_mins[pid] = history_mins[pid][-PLAYER_WINDOW:]
                history_pts[pid] = history_pts[pid][-PLAYER_WINDOW:]

    return pd.DataFrame(results)


def _playoff_exp_before(
    games: pd.DataFrame,
    playoff_history: pd.DataFrame | None,
) -> pd.DataFrame:
    if playoff_history is None or playoff_history.empty:
        games = games.copy()
        games["home_playoff_games"] = 0
        games["away_playoff_games"] = 0
        return games

    team_dates: dict[str, list[pd.Timestamp]] = {}
    for _, row in playoff_history.sort_values("game_date").iterrows():
        for team in (row["home_team"], row["away_team"]):
            team_dates.setdefault(team, []).append(row["game_date"])

    def count_before(team: str, dt: pd.Timestamp) -> int:
        dates = team_dates.get(team, [])
        return sum(1 for d in dates if d < dt)

    out = games.copy()
    out["home_playoff_games"] = [
        count_before(t, d) for t, d in zip(out["home_team"], out["game_date"])
    ]
    out["away_playoff_games"] = [
        count_before(t, d) for t, d in zip(out["away_team"], out["game_date"])
    ]
    return out


def _head_to_head_margin(games: pd.DataFrame) -> pd.Series:
    margins: list[float] = []
    seen: dict[tuple[str, str], list[int]] = {}
    for _, row in games.iterrows():
        key = tuple(sorted([row["home_team"], row["away_team"]]))
        prior = seen.get(key, [])
        margins.append(float(np.mean(prior)) if prior else 0.0)
        margin = int(row["home_pts"]) - int(row["away_pts"])
        seen.setdefault(key, []).append(margin)
    return pd.Series(margins, index=games.index)


def build_game_features(
    games: pd.DataFrame,
    team_box: pd.DataFrame,
    player_box: pd.DataFrame,
    *,
    is_playoffs: bool = False,
    playoff_history: pd.DataFrame | None = None,
    seeds: dict[str, int] | None = None,
    win_pcts: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build model features for each game without future leakage."""
    games = games.sort_values("game_date").reset_index(drop=True).copy()
    history = _add_rolling_team_features(_team_long_history(games, team_box))

    home_stats = history[history["is_home"] == 1].copy()
    away_stats = history[history["is_home"] == 0].copy()

    rename_roll = {f"roll_{c}": f"home_roll_{c}" for c in TEAM_STAT_COLS}
    rename_roll.update(
        {
            "rest_days": "home_rest_days",
            "back_to_back": "home_back_to_back",
            "after_blowout": "home_after_blowout",
            "travel_miles": "home_travel_miles",
        }
    )
    home_stats = home_stats.rename(columns=rename_roll)

    away_rename = {f"roll_{c}": f"away_roll_{c}" for c in TEAM_STAT_COLS}
    away_rename.update(
        {
            "rest_days": "away_rest_days",
            "back_to_back": "away_back_to_back",
            "after_blowout": "away_after_blowout",
            "travel_miles": "away_travel_miles",
        }
    )
    away_stats = away_stats.rename(columns=away_rename)

    featured = games.merge(
        home_stats[
            ["game_id"]
            + [f"home_roll_{c}" for c in TEAM_STAT_COLS]
            + ["home_rest_days", "home_back_to_back", "home_after_blowout", "home_travel_miles"]
        ],
        on="game_id",
    ).merge(
        away_stats[
            ["game_id"]
            + [f"away_roll_{c}" for c in TEAM_STAT_COLS]
            + ["away_rest_days", "away_back_to_back", "away_after_blowout", "away_travel_miles"]
        ],
        on="game_id",
    )

    player_feats = _build_player_game_features(games, player_box)
    if not player_feats.empty:
        home_pf = player_feats.rename(
            columns={
                "team": "home_team",
                "strength_index": "home_strength_index",
                "key_players_out": "home_key_players_out",
            }
        )
        away_pf = player_feats.rename(
            columns={
                "team": "away_team",
                "strength_index": "away_strength_index",
                "key_players_out": "away_key_players_out",
            }
        )
        featured = featured.merge(
            home_pf[["game_id", "home_team", "home_strength_index", "home_key_players_out"]],
            on=["game_id", "home_team"],
            how="left",
        )
        featured = featured.merge(
            away_pf[["game_id", "away_team", "away_strength_index", "away_key_players_out"]],
            on=["game_id", "away_team"],
            how="left",
        )
    else:
        featured["home_strength_index"] = 110.0
        featured["away_strength_index"] = 110.0
        featured["home_key_players_out"] = 0
        featured["away_key_players_out"] = 0

    featured = _playoff_exp_before(featured, playoff_history if is_playoffs else None)
    featured["h2h_margin"] = _head_to_head_margin(games)
    featured["is_playoffs"] = int(is_playoffs)
    featured["is_altitude"] = featured["home_team"].apply(is_altitude_home).astype(int)
    featured["is_elimination"] = 0

    season = games["season"].iloc[0] if "season" in games.columns and len(games) else None
    if seeds is None and season:
        seeds = compute_seeds(games, season)
    if win_pcts is None and season:
        win_pcts = compute_win_pcts(games, season)

    featured["home_rs_win_pct"] = featured["home_team"].map(win_pcts or {}).fillna(0.5)
    featured["away_rs_win_pct"] = featured["away_team"].map(win_pcts or {}).fillna(0.5)
    featured["seed_diff"] = featured.apply(
        lambda r: (seeds or {}).get(r["home_team"], 5) - (seeds or {}).get(r["away_team"], 5),
        axis=1,
    )

    featured["pace_mismatch"] = (
        featured["home_roll_pace"].fillna(100) - featured["away_roll_pace"].fillna(100)
    ).abs()
    featured["three_pt_rate_diff"] = (
        featured["home_roll_three_pt_rate"].fillna(0.35)
        - featured["away_roll_three_pt_rate"].fillna(0.35)
    )
    featured["foul_rate_diff"] = (
        featured["home_roll_foul_rate"].fillna(0.2)
        - featured["away_roll_foul_rate"].fillna(0.2)
    )

    defaults = {
        "home_roll_ortg": 112.0,
        "away_roll_ortg": 112.0,
        "home_roll_drtg": 112.0,
        "away_roll_drtg": 112.0,
        "home_roll_pace": 100.0,
        "away_roll_pace": 100.0,
        "home_roll_efg_pct": 0.54,
        "away_roll_efg_pct": 0.54,
        "home_roll_tov_pct": 0.13,
        "away_roll_tov_pct": 0.13,
        "home_roll_reb_pct": 0.5,
        "away_roll_reb_pct": 0.5,
        "home_strength_index": 110.0,
        "away_strength_index": 110.0,
        "home_key_players_out": 0,
        "away_key_players_out": 0,
        "home_travel_miles": 0.0,
        "away_travel_miles": 0.0,
        "home_rest_days": 3.0,
        "away_rest_days": 3.0,
        "home_back_to_back": 0,
        "away_back_to_back": 0,
        "home_after_blowout": 0,
        "away_after_blowout": 0,
        "home_playoff_games": 0,
        "away_playoff_games": 0,
    }
    for col, val in defaults.items():
        if col in featured.columns:
            featured[col] = featured[col].fillna(val)

    return featured


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].astype(float)


def update_team_state(
    state: dict[str, dict],
    team: str,
    *,
    pts_for: float,
    pts_against: float,
    game_date: pd.Timestamp,
    location: str,
    box: dict[str, float] | None = None,
    player_minutes: dict[int, float] | None = None,
    player_pts: dict[int, float] | None = None,
    is_playoff: bool = False,
) -> None:
    entry = state.setdefault(team, _empty_team_state())
    for col in TEAM_STAT_COLS:
        val = box.get(col, pts_for if col == "pts" else pts_against if col == "pts_allowed" else 0.0) if box else (
            pts_for if col == "pts" else pts_against if col == "pts_allowed" else 0.0
        )
        entry["stats"][col].append(float(val))
        entry["stats"][col] = entry["stats"][col][-ROLLING_WINDOW:]

    entry["last_game_date"] = game_date
    entry["last_location"] = location
    entry["last_margin"] = pts_for - pts_against
    if is_playoff:
        entry["playoff_games"] += 1

    if player_minutes:
        for pid, mins in player_minutes.items():
            entry["player_minutes"].setdefault(pid, []).append(mins)
            entry["player_minutes"][pid] = entry["player_minutes"][pid][-PLAYER_WINDOW:]
            entry["player_pts"].setdefault(pid, []).append(player_pts.get(pid, 0.0) if player_pts else 0.0)
            entry["player_pts"][pid] = entry["player_pts"][pid][-PLAYER_WINDOW:]
        entry["strength_index"] = _compute_strength_index(entry["player_minutes"], entry["player_pts"])
        entry["key_players_out"] = _infer_key_players_out(entry["player_minutes"], player_minutes)


def build_matchup_features(
    home_team: str,
    away_team: str,
    state: dict[str, dict],
    *,
    game_date: pd.Timestamp | None = None,
    h2h_margin: float = 0.0,
    is_playoffs: bool = True,
    seeds: dict[str, int] | None = None,
    win_pcts: dict[str, float] | None = None,
    is_elimination: bool = False,
    league_defaults: dict[str, float] | None = None,
) -> dict[str, float]:
    defaults = league_defaults or {}
    home = state.get(home_team, _empty_team_state())
    away = state.get(away_team, _empty_team_state())

    def roll(entry: dict, col: str, default: float) -> float:
        return _roll_mean(entry["stats"].get(col, []), default)

    def rest_days(entry: dict, default: float = 3.0) -> float:
        if game_date is None or entry.get("last_game_date") is None:
            return default
        return max((game_date - entry["last_game_date"]).days, 0)

    home_rest = rest_days(home)
    away_rest = rest_days(away)
    home_roll_pace = roll(home, "pace", defaults.get("pace", 100.0))
    away_roll_pace = roll(away, "pace", defaults.get("pace", 100.0))

    last_away_loc = away.get("last_location") or away_team
    away_travel = travel_miles(last_away_loc, home_team) if last_away_loc != home_team else 0.0

    home_seed = (seeds or {}).get(home_team, 5)
    away_seed = (seeds or {}).get(away_team, 5)

    return {
        "home_roll_ortg": roll(home, "ortg", defaults.get("ortg", 112.0)),
        "home_roll_drtg": roll(home, "drtg", defaults.get("drtg", 112.0)),
        "home_roll_pace": home_roll_pace,
        "away_roll_ortg": roll(away, "ortg", defaults.get("ortg", 112.0)),
        "away_roll_drtg": roll(away, "drtg", defaults.get("drtg", 112.0)),
        "away_roll_pace": away_roll_pace,
        "home_roll_efg_pct": roll(home, "efg_pct", defaults.get("efg_pct", 0.54)),
        "away_roll_efg_pct": roll(away, "efg_pct", defaults.get("efg_pct", 0.54)),
        "home_roll_tov_pct": roll(home, "tov_pct", defaults.get("tov_pct", 0.13)),
        "away_roll_tov_pct": roll(away, "tov_pct", defaults.get("tov_pct", 0.13)),
        "home_roll_reb_pct": roll(home, "reb_pct", defaults.get("reb_pct", 0.5)),
        "away_roll_reb_pct": roll(away, "reb_pct", defaults.get("reb_pct", 0.5)),
        "home_strength_index": home.get("strength_index", 110.0),
        "away_strength_index": away.get("strength_index", 110.0),
        "home_key_players_out": float(home.get("key_players_out", 0)),
        "away_key_players_out": float(away.get("key_players_out", 0)),
        "away_travel_miles": away_travel,
        "home_back_to_back": float(home_rest == 1),
        "away_back_to_back": float(away_rest == 1),
        "is_altitude": float(is_altitude_home(home_team)),
        "is_playoffs": float(is_playoffs),
        "home_playoff_games": float(home.get("playoff_games", 0)),
        "away_playoff_games": float(away.get("playoff_games", 0)),
        "seed_diff": float(home_seed - away_seed),
        "home_rs_win_pct": float((win_pcts or {}).get(home_team, 0.5)),
        "away_rs_win_pct": float((win_pcts or {}).get(away_team, 0.5)),
        "pace_mismatch": abs(home_roll_pace - away_roll_pace),
        "three_pt_rate_diff": roll(home, "three_pt_rate", 0.35) - roll(away, "three_pt_rate", 0.35),
        "foul_rate_diff": roll(home, "foul_rate", 0.2) - roll(away, "foul_rate", 0.2),
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "is_elimination": float(is_elimination),
        "home_after_blowout": float(abs(home.get("last_margin", 0)) >= BLOWOUT_MARGIN),
        "away_after_blowout": float(abs(away.get("last_margin", 0)) >= BLOWOUT_MARGIN),
        "h2h_margin": h2h_margin,
    }


def init_team_state_from_games(
    games: pd.DataFrame,
    team_box: pd.DataFrame,
    player_box: pd.DataFrame,
    season: str,
) -> dict[str, dict]:
    """Seed team state from regular-season games before playoffs."""
    season_games = games[games["season"] == season].sort_values("game_date")
    box_by_game = {
        (row["game_id"], row["team"]): row
        for _, row in team_box.iterrows()
    }
    players_by_game_team: dict[tuple[str, str], pd.DataFrame] = {}
    if not player_box.empty:
        for (gid, team), grp in player_box.groupby(["game_id", "team"]):
            players_by_game_team[(gid, team)] = grp

    state: dict[str, dict] = {}
    for _, row in season_games.iterrows():
        game_date = row["game_date"]
        gid = row["game_id"]
        for team, opp, pts, opp_pts, loc in [
            (row["home_team"], row["away_team"], row["home_pts"], row["away_pts"], row["home_team"]),
            (row["away_team"], row["home_team"], row["away_pts"], row["home_pts"], row["home_team"]),
        ]:
            box_row = box_by_game.get((gid, team))
            box = None
            if box_row is not None:
                box = {
                    "ortg": float(box_row.get("ortg", 0)),
                    "drtg": float(box_row.get("drtg", 0)),
                    "pace": float(box_row.get("pace", 0)),
                    "efg_pct": float(box_row.get("efg_pct", 0)),
                    "tov_pct": float(box_row.get("tov_pct", 0)),
                    "reb_pct": float(box_row.get("reb_pct", 0)),
                    "three_pt_rate": float(box_row.get("three_pt_rate", 0)),
                    "foul_rate": float(box_row.get("foul_rate", 0)),
                    "pts": float(pts),
                    "pts_allowed": float(opp_pts),
                }
            pgrp = players_by_game_team.get((gid, team))
            p_mins = dict(zip(pgrp["player_id"], pgrp["minutes"])) if pgrp is not None else None
            p_pts = dict(zip(pgrp["player_id"], pgrp["pts"])) if pgrp is not None else None
            update_team_state(
                state,
                team,
                pts_for=pts,
                pts_against=opp_pts,
                game_date=game_date,
                location=loc,
                box=box,
                player_minutes=p_mins,
                player_pts=p_pts,
                is_playoff=False,
            )
    return state
