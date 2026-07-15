"""Backtest score MAE and series-winner accuracy on historical playoffs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error

from .box_scores import load_box_scores
from .features import build_game_features, feature_matrix, init_team_state_from_games
from .fetch_data import load_playoff_games, load_training_games
from .seeds import compute_seeds, compute_win_pcts
from .simulate import (
    actual_series_winners,
    infer_first_round_matchups,
    simulate_playoff_bracket,
    simulate_series,
)
from .team_stats import team_box_from_games
from .train import ScoreModels, train_models

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

ROUND_NAMES = [
    "west_r1",
    "east_r1",
    "west_r2",
    "east_r2",
    "west_conf_finals",
    "east_conf_finals",
    "finals",
]


def evaluate_playoff_scores(
    models: ScoreModels,
    playoff_games: pd.DataFrame,
    team_box: pd.DataFrame,
    player_box: pd.DataFrame,
    regular_season_games: pd.DataFrame,
    *,
    seasons: list[str] | None = None,
) -> dict[str, float]:
    games = playoff_games.copy()
    if seasons:
        games = games[games["season"].isin(seasons)]

    featured_frames: list[pd.DataFrame] = []
    for season, season_games in games.groupby("season"):
        rs = regular_season_games[regular_season_games["season"] == season]
        seeds = compute_seeds(rs, season)
        win_pcts = compute_win_pcts(rs, season)
        prior_po = playoff_games[playoff_games["season"] < season]
        feat = build_game_features(
            season_games,
            team_box,
            player_box,
            is_playoffs=True,
            playoff_history=pd.concat([prior_po, season_games], ignore_index=True),
            seeds=seeds,
            win_pcts=win_pcts,
        )
        featured_frames.append(feat)

    featured = pd.concat(featured_frames, ignore_index=True)
    X = feature_matrix(featured)
    home_pred = models.home_model.predict(X)
    away_pred = models.away_model.predict(X)
    return {
        "playoff_home_mae": float(mean_absolute_error(featured["home_pts"], home_pred)),
        "playoff_away_mae": float(mean_absolute_error(featured["away_pts"], away_pred)),
        "playoff_games": int(len(featured)),
    }


def _series_win_prob(
    models: ScoreModels,
    higher_seed: str,
    lower_seed: str,
    state: dict,
    *,
    seeds: dict[str, int],
    win_pcts: dict[str, float],
    n_sims: int = 30,
    seed: int = 0,
) -> float:
    wins = 0
    for i in range(n_sims):
        sim_state = copy.deepcopy(state)
        result = simulate_series(
            higher_seed,
            lower_seed,
            models,
            sim_state,
            seeds=seeds,
            win_pcts=win_pcts,
            rng=np.random.default_rng(seed + i),
        )
        if result.winner == higher_seed:
            wins += 1
    return wins / n_sims


def evaluate_series_by_round(
    models: ScoreModels,
    regular_season_games: pd.DataFrame,
    playoff_games: pd.DataFrame,
    team_box: pd.DataFrame,
    player_box: pd.DataFrame,
    *,
    seasons: list[str] | None = None,
    n_sims: int = 10,
    seed: int = 42,
) -> dict:
    if seasons is None:
        seasons = sorted(playoff_games["season"].unique())

    round_correct: dict[str, int] = {r: 0 for r in ROUND_NAMES}
    round_total: dict[str, int] = {r: 0 for r in ROUND_NAMES}
    brier_probs: list[float] = []
    brier_labels: list[float] = []

    for season in seasons:
        actual = actual_series_winners(playoff_games, season)
        if not actual:
            continue

        rs = regular_season_games[regular_season_games["season"] == season]
        seeds = compute_seeds(rs, season)
        win_pcts = compute_win_pcts(rs, season)
        state = init_team_state_from_games(regular_season_games, team_box, player_box, season)

        for i in range(n_sims):
            rng = np.random.default_rng(seed + i)
            try:
                bracket = simulate_playoff_bracket(
                    season,
                    models,
                    regular_season_games,
                    playoff_games,
                    team_box=team_box,
                    player_box=player_box,
                    rng=rng,
                )
            except (ValueError, IndexError):
                continue

            for round_name in ROUND_NAMES:
                for series in bracket.rounds.get(round_name, []):
                    key = tuple(sorted([series.higher_seed, series.lower_seed]))
                    if key not in actual:
                        continue
                    round_total[round_name] += 1
                    if series.winner == actual[key]:
                        round_correct[round_name] += 1

        for higher, lower in infer_first_round_matchups(
            playoff_games, season, regular_season_games
        ):
            key = tuple(sorted([higher, lower]))
            if key not in actual:
                continue
            prob = _series_win_prob(
                models,
                higher,
                lower,
                state,
                seeds=seeds,
                win_pcts=win_pcts,
                n_sims=20,
                seed=seed,
            )
            label = 1.0 if actual[key] == higher else 0.0
            brier_probs.append(prob)
            brier_labels.append(label)

    by_round = {
        r: {
            "series_accuracy": round_correct[r] / round_total[r] if round_total[r] else 0.0,
            "series_predictions": round_total[r],
        }
        for r in ROUND_NAMES
    }

    calibration = None
    if brier_probs:
        calibration = {
            "brier_score": float(brier_score_loss(brier_labels, brier_probs)),
            "mean_predicted_prob": float(np.mean(brier_probs)),
            "actual_win_rate": float(np.mean(brier_labels)),
            "samples": len(brier_probs),
        }

    return {"by_round": by_round, "calibration": calibration}


def evaluate_bracket(
    models: ScoreModels,
    regular_season_games: pd.DataFrame,
    playoff_games: pd.DataFrame,
    *,
    seasons: list[str] | None = None,
    n_sims: int = 1,
    seed: int = 42,
) -> dict:
    if seasons is None:
        seasons = sorted(playoff_games["season"].unique())

    season_results: list[dict] = []
    for season in seasons:
        rs = regular_season_games[regular_season_games["season"] == season]
        po = playoff_games[playoff_games["season"] == season]
        if rs.empty or po.empty:
            continue

        last_game = po.sort_values("game_date").iloc[-1]
        actual_champion = (
            last_game["home_team"]
            if last_game["home_pts"] > last_game["away_pts"]
            else last_game["away_team"]
        )

        champion_hits = 0
        sim_champions: list[str] = []
        for i in range(n_sims):
            rng = np.random.default_rng(seed + i) if n_sims > 1 else None
            try:
                bracket = simulate_playoff_bracket(
                    season, models, regular_season_games, playoff_games, rng=rng
                )
            except (ValueError, IndexError):
                continue
            sim_champions.append(bracket.champion)
            if bracket.champion == actual_champion:
                champion_hits += 1

        if not sim_champions:
            continue

        most_common = pd.Series(sim_champions).mode().iloc[0]
        season_results.append(
            {
                "season": season,
                "actual_champion": actual_champion,
                "predicted_champion": most_common,
                "champion_correct": most_common == actual_champion,
                "champion_hit_rate": champion_hits / n_sims,
            }
        )

    return {
        "seasons_evaluated": len(season_results),
        "champion_accuracy": float(np.mean([r["champion_correct"] for r in season_results]))
        if season_results
        else 0.0,
        "mean_champion_hit_rate": float(np.mean([r["champion_hit_rate"] for r in season_results]))
        if season_results
        else 0.0,
        "by_season": season_results,
    }


def run_backtest(
    *,
    train_start: int = 2015,
    train_end: int = 2022,
    test_start: int = 2022,
    test_end: int = 2023,
    n_sims: int = 25,
) -> dict:
    regular = load_training_games(train_start, test_end)
    playoffs = load_playoff_games(test_start, test_end)
    team_box, player_box = load_box_scores()
    if team_box.empty:
        team_box = team_box_from_games(regular)
    if team_box.empty:
        raise FileNotFoundError("Re-run fetch for shooting stats: python -m src.predict fetch")

    models = train_models(
        start_year=train_start,
        end_year=train_end,
        test_season="2022-23",
        team_box=team_box,
        player_box=player_box,
    )
    test_seasons = sorted(playoffs["season"].unique())

    report = {
        "training": {
            "home_mae": models.home_mae,
            "away_mae": models.away_mae,
            "train_years": [train_start, train_end],
            "feature_count": len(models.feature_columns),
        },
        "playoff_score_metrics": evaluate_playoff_scores(
            models, playoffs, team_box, player_box, regular, seasons=list(test_seasons)
        ),
        "bracket_metrics": evaluate_bracket(
            models, regular, playoffs, seasons=list(test_seasons), n_sims=n_sims
        ),
        "series_metrics": evaluate_series_by_round(
            models,
            regular,
            playoffs,
            team_box,
            player_box,
            seasons=list(test_seasons),
            n_sims=min(n_sims, 10),
        ),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "backtest_report.json").write_text(json.dumps(report, indent=2))
    return report
