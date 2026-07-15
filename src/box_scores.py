"""Fetch and cache per-game box scores for team and player stats."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import boxscoreadvancedv3, boxscoretraditionalv3
from tqdm import tqdm

from .fetch_data import DATA_DIR, load_playoff_games, load_training_games

REQUEST_DELAY_SEC = 0.6
TEAM_BOX_PATH = DATA_DIR / "box_scores_team.parquet"
PLAYER_BOX_PATH = DATA_DIR / "box_scores_player.parquet"


def _parse_minutes(min_str: str | float) -> float:
    if pd.isna(min_str) or min_str in ("", "0", 0):
        return 0.0
    if isinstance(min_str, (int, float)):
        return float(min_str)
    text = str(min_str).strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2:
            return int(parts[0]) + int(parts[1]) / 60.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _fetch_one_game(game_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    time.sleep(REQUEST_DELAY_SEC)
    adv = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id)
    trad = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
    team_adv = adv.team_stats.get_data_frame()
    team_trad = trad.team_stats.get_data_frame()
    players = trad.player_stats.get_data_frame()

    trad_by_team = {row["teamTricode"]: row for _, row in team_trad.iterrows()}

    team_rows: list[dict] = []
    for _, row in team_adv.iterrows():
        team = row["teamTricode"]
        t = trad_by_team.get(team)
        if t is None:
            continue
        fga = float(t.get("fieldGoalsAttempted", 0) or 0)
        fg3a = float(t.get("threePointersAttempted", 0) or 0)
        fta = float(t.get("freeThrowsAttempted", 0) or 0)
        pf = float(t.get("foulsPersonal", 0) or 0)
        team_rows.append(
            {
                "game_id": game_id,
                "team": team,
                "ortg": float(row.get("offensiveRating", 0) or 0),
                "drtg": float(row.get("defensiveRating", 0) or 0),
                "pace": float(row.get("pace", 0) or 0),
                "efg_pct": float(row.get("effectiveFieldGoalPercentage", 0) or 0),
                "tov_pct": float(row.get("turnoverRatio", 0) or 0) / 100.0,
                "reb_pct": float(row.get("reboundPercentage", 0) or 0) / 100.0,
                "fg3a": fg3a,
                "fga": fga,
                "fta": fta,
                "pts": float(t.get("points", 0) or 0),
                "pf": pf,
                "three_pt_rate": fg3a / fga if fga > 0 else 0.0,
                "foul_rate": pf / (fga + 0.44 * fta) if (fga + 0.44 * fta) > 0 else 0.0,
            }
        )

    player_rows: list[dict] = []
    for _, p in players.iterrows():
        team = p.get("teamTricode")
        if team is None:
            continue
        player_rows.append(
            {
                "game_id": game_id,
                "team": team,
                "player_id": int(p["personId"]),
                "player_name": p.get("nameI", ""),
                "minutes": _parse_minutes(p.get("minutes", 0)),
                "pts": float(p.get("points", 0) or 0),
            }
        )

    return pd.DataFrame(team_rows), pd.DataFrame(player_rows)


def _load_cached() -> tuple[pd.DataFrame, pd.DataFrame]:
    team = pd.read_parquet(TEAM_BOX_PATH) if TEAM_BOX_PATH.exists() else pd.DataFrame()
    player = pd.read_parquet(PLAYER_BOX_PATH) if PLAYER_BOX_PATH.exists() else pd.DataFrame()
    return team, player


def _save_cached(team: pd.DataFrame, player: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not team.empty:
        team.to_parquet(TEAM_BOX_PATH, index=False)
    if not player.empty:
        player.to_parquet(PLAYER_BOX_PATH, index=False)


def fetch_box_scores_for_games(
    games: pd.DataFrame,
    *,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch team/player box scores for all game_ids in games DataFrame."""
    cached_team, cached_player = (
        _load_cached() if use_cache and not force_refresh else (pd.DataFrame(), pd.DataFrame())
    )
    cached_ids = set(cached_team["game_id"].unique()) if not cached_team.empty else set()

    needed = [gid for gid in games["game_id"].unique() if gid not in cached_ids]
    game_ids = set(games["game_id"].unique())

    if not needed and not cached_team.empty:
        return (
            cached_team[cached_team["game_id"].isin(game_ids)].copy(),
            cached_player[cached_player["game_id"].isin(game_ids)].copy(),
        )

    new_team_frames: list[pd.DataFrame] = []
    new_player_frames: list[pd.DataFrame] = []
    for game_id in tqdm(needed, desc="Fetching box scores"):
        try:
            team_df, player_df = _fetch_one_game(str(game_id))
            if not team_df.empty:
                new_team_frames.append(team_df)
            if not player_df.empty:
                new_player_frames.append(player_df)
        except Exception as exc:
            tqdm.write(f"Warning: box score failed for {game_id}: {exc}")

    team_parts = [df for df in [cached_team, *new_team_frames] if not df.empty]
    player_parts = [df for df in [cached_player, *new_player_frames] if not df.empty]
    team = pd.concat(team_parts, ignore_index=True) if team_parts else pd.DataFrame()
    player = pd.concat(player_parts, ignore_index=True) if player_parts else pd.DataFrame()

    if not team.empty:
        team = team.drop_duplicates(subset=["game_id", "team"], keep="last")
    if not player.empty:
        player = player.drop_duplicates(subset=["game_id", "team", "player_id"], keep="last")

    _save_cached(team, player)

    if team.empty:
        return team, player
    return (
        team[team["game_id"].isin(game_ids)].copy(),
        player[player["game_id"].isin(game_ids)].copy() if not player.empty else player,
    )


def fetch_box_scores_range(
    start_year: int,
    end_year: int,
    *,
    include_playoffs: bool = True,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = load_training_games(start_year, end_year, use_cache=use_cache)
    if include_playoffs:
        playoffs = load_playoff_games(start_year, end_year, use_cache=use_cache)
        games = pd.concat([games, playoffs], ignore_index=True)
    return fetch_box_scores_for_games(games, use_cache=use_cache)


def load_box_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _load_cached()
