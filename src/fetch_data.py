"""Fetch NBA game logs via nba_api with local parquet cache."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REGULAR_SEASON = "Regular Season"
PLAYOFFS = "Playoffs"
REQUEST_DELAY_SEC = 0.6


def season_label(year: int) -> str:
    """Convert start year to NBA season string, e.g. 2023 -> '2023-24'."""
    return f"{year}-{str(year + 1)[-2:]}"


def _cache_path(season: str, season_type: str) -> Path:
    slug = season_type.lower().replace(" ", "_")
    return DATA_DIR / f"games_{season.replace('-', '_')}_{slug}.parquet"


def _fetch_raw_log(season: str, season_type: str) -> pd.DataFrame:
    time.sleep(REQUEST_DELAY_SEC)
    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="T",
    )
    return log.get_data_frames()[0]


def _parse_matchup(matchup: str) -> tuple[str, str, bool]:
    """Return (team_abbr, opponent_abbr, is_home)."""
    if " vs. " in matchup:
        team, opp = matchup.split(" vs. ")
        return team.strip(), opp.strip(), True
    if " @ " in matchup:
        team, opp = matchup.split(" @ ")
        return team.strip(), opp.strip(), False
    raise ValueError(f"Unrecognized MATCHUP format: {matchup}")


def _team_rows_to_games(df: pd.DataFrame) -> pd.DataFrame:
    """Convert two-rows-per-game team log into one row per game."""
    records: list[dict] = []
    for game_id, group in df.groupby("GAME_ID"):
        if len(group) != 2:
            continue
        row_a, row_b = group.iloc[0], group.iloc[1]
        team_a, opp_a, home_a = _parse_matchup(row_a["MATCHUP"])
        team_b, opp_b, home_b = _parse_matchup(row_b["MATCHUP"])
        if home_a:
            home_row, away_row = row_a, row_b
            home_team, away_team = team_a, team_b
        elif home_b:
            home_row, away_row = row_b, row_a
            home_team, away_team = team_b, team_a
        else:
            continue
        if home_team != team_a and away_team != team_a:
            home_row, away_row = row_b, row_a
            home_team, away_team = team_b, team_a
        records.append(
            {
                "game_id": game_id,
                "game_date": pd.to_datetime(home_row["GAME_DATE"]),
                "season_id": home_row["SEASON_ID"],
                "home_team": home_team,
                "away_team": away_team,
                "home_team_id": int(home_row["TEAM_ID"]),
                "away_team_id": int(away_row["TEAM_ID"]),
                "home_pts": int(home_row["PTS"]),
                "away_pts": int(away_row["PTS"]),
                "home_win": int(home_row["WL"] == "W"),
                "home_fgm": float(home_row.get("FGM", 0) or 0),
                "home_fga": float(home_row.get("FGA", 0) or 0),
                "home_fg3m": float(home_row.get("FG3M", 0) or 0),
                "home_fg3a": float(home_row.get("FG3A", 0) or 0),
                "home_ftm": float(home_row.get("FTM", 0) or 0),
                "home_fta": float(home_row.get("FTA", 0) or 0),
                "home_oreb": float(home_row.get("OREB", 0) or 0),
                "home_tov": float(home_row.get("TOV", 0) or 0),
                "home_pf": float(home_row.get("PF", 0) or 0),
                "away_fgm": float(away_row.get("FGM", 0) or 0),
                "away_fga": float(away_row.get("FGA", 0) or 0),
                "away_fg3m": float(away_row.get("FG3M", 0) or 0),
                "away_fg3a": float(away_row.get("FG3A", 0) or 0),
                "away_ftm": float(away_row.get("FTM", 0) or 0),
                "away_fta": float(away_row.get("FTA", 0) or 0),
                "away_oreb": float(away_row.get("OREB", 0) or 0),
                "away_tov": float(away_row.get("TOV", 0) or 0),
                "away_pf": float(away_row.get("PF", 0) or 0),
            }
        )
    games = pd.DataFrame(records)
    if games.empty:
        return games
    return games.sort_values("game_date").reset_index(drop=True)


def fetch_season_games(
    season: str,
    season_type: str = REGULAR_SEASON,
    *,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch or load cached games for one season and season type."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(season, season_type)
    if use_cache and path.exists() and not force_refresh:
        return pd.read_parquet(path)
    raw = _fetch_raw_log(season, season_type)
    games = _team_rows_to_games(raw)
    games["season"] = season
    games["season_type"] = season_type
    games.to_parquet(path, index=False)
    return games


def fetch_season_range(
    start_year: int,
    end_year: int,
    season_type: str = REGULAR_SEASON,
    *,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch multiple seasons and concatenate."""
    frames: list[pd.DataFrame] = []
    for year in tqdm(range(start_year, end_year + 1), desc=f"Fetching {season_type}"):
        season = season_label(year)
        try:
            games = fetch_season_games(
                season,
                season_type,
                use_cache=use_cache,
                force_refresh=force_refresh,
            )
            if not games.empty:
                frames.append(games)
        except Exception as exc:
            tqdm.write(f"Warning: failed {season} {season_type}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_training_games(
    start_year: int = 2015,
    end_year: int = 2023,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load regular-season games for model training."""
    return fetch_season_range(
        start_year, end_year, REGULAR_SEASON, use_cache=use_cache
    )


def load_playoff_games(
    start_year: int = 2015,
    end_year: int = 2023,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load playoff games for backtesting."""
    return fetch_season_range(start_year, end_year, PLAYOFFS, use_cache=use_cache)


if __name__ == "__main__":
    sample = fetch_season_games("2023-24", REGULAR_SEASON)
    print(f"Fetched {len(sample)} games for 2023-24 regular season")
    print(sample.head())
