"""Train home/away score regression models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from .box_scores import load_box_scores
from .features import FEATURE_COLUMNS, build_game_features, feature_matrix
from .fetch_data import load_training_games
from .seeds import compute_seeds, compute_win_pcts
from .team_stats import team_box_from_games

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
HOME_MODEL_PATH = MODELS_DIR / "home_score_model.joblib"
AWAY_MODEL_PATH = MODELS_DIR / "away_score_model.joblib"
METADATA_PATH = MODELS_DIR / "training_metadata.json"


@dataclass
class ScoreModels:
    home_model: XGBRegressor
    away_model: XGBRegressor
    home_mae: float
    away_mae: float
    feature_columns: list[str]


def _default_regressor() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )


def _require_shooting_columns(games: pd.DataFrame) -> None:
    if "home_fga" not in games.columns:
        raise FileNotFoundError(
            "Game logs missing shooting stats. Re-run: python -m src.predict fetch"
        )


def _resolve_team_player_box(
    games: pd.DataFrame,
    team_box: pd.DataFrame | None,
    player_box: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_shooting_columns(games)
    if team_box is None or team_box.empty:
        team_box = team_box_from_games(games)
    if player_box is None:
        _, player_box = load_box_scores()
    if player_box is None:
        player_box = pd.DataFrame()
    return team_box, player_box


def train_models(
    games: pd.DataFrame | None = None,
    team_box: pd.DataFrame | None = None,
    player_box: pd.DataFrame | None = None,
    *,
    start_year: int = 2015,
    end_year: int = 2023,
    test_season: str | None = None,
) -> ScoreModels:
    """Train dual regressors on regular-season games."""
    if games is None:
        games = load_training_games(start_year, end_year)
    if games.empty:
        raise ValueError("No training games available.")

    team_box, player_box = _resolve_team_player_box(games, team_box, player_box)

    featured_frames: list[pd.DataFrame] = []
    for season, season_games in games.groupby("season"):
        seeds, win_pcts = compute_seeds(season_games, season), compute_win_pcts(season_games, season)
        feat = build_game_features(
            season_games,
            team_box,
            player_box,
            is_playoffs=False,
            seeds=seeds,
            win_pcts=win_pcts,
        )
        featured_frames.append(feat)

    featured = pd.concat(featured_frames, ignore_index=True)
    X = feature_matrix(featured)
    y_home = featured["home_pts"].astype(float)
    y_away = featured["away_pts"].astype(float)

    if test_season:
        train_mask = featured["season"] != test_season
        X_train, X_test = X[train_mask], X[~train_mask]
        y_home_train, y_home_test = y_home[train_mask], y_home[~train_mask]
        y_away_train, y_away_test = y_away[train_mask], y_away[~train_mask]
    else:
        X_train, X_test, y_home_train, y_home_test = train_test_split(
            X, y_home, test_size=0.15, random_state=42
        )
        _, _, y_away_train, y_away_test = train_test_split(
            X, y_away, test_size=0.15, random_state=42
        )

    home_model = _default_regressor()
    away_model = _default_regressor()
    home_model.fit(X_train, y_home_train)
    away_model.fit(X_train, y_away_train)

    home_mae = float(mean_absolute_error(y_home_test, home_model.predict(X_test)))
    away_mae = float(mean_absolute_error(y_away_test, away_model.predict(X_test)))
    home_rmse = float(np.sqrt(mean_squared_error(y_home_test, home_model.predict(X_test))))
    away_rmse = float(np.sqrt(mean_squared_error(y_away_test, away_model.predict(X_test))))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(home_model, HOME_MODEL_PATH)
    joblib.dump(away_model, AWAY_MODEL_PATH)
    metadata = {
        "home_mae": home_mae,
        "away_mae": away_mae,
        "home_rmse": home_rmse,
        "away_rmse": away_rmse,
        "feature_columns": FEATURE_COLUMNS,
        "train_games": int(len(X_train)),
        "test_games": int(len(X_test)),
        "start_year": start_year,
        "end_year": end_year,
        "test_season": test_season,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    return ScoreModels(
        home_model=home_model,
        away_model=away_model,
        home_mae=home_mae,
        away_mae=away_mae,
        feature_columns=FEATURE_COLUMNS,
    )


def load_models() -> ScoreModels:
    if not HOME_MODEL_PATH.exists() or not AWAY_MODEL_PATH.exists():
        raise FileNotFoundError("Models not found. Run train first.")
    metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    return ScoreModels(
        home_model=joblib.load(HOME_MODEL_PATH),
        away_model=joblib.load(AWAY_MODEL_PATH),
        home_mae=float(metadata.get("home_mae", 8.0)),
        away_mae=float(metadata.get("away_mae", 8.0)),
        feature_columns=metadata.get("feature_columns", FEATURE_COLUMNS),
    )
