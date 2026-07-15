"""CLI entry point for training and playoff bracket prediction."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .backtest import run_backtest
from .box_scores import fetch_box_scores_range
from .fetch_data import load_playoff_games, load_training_games, season_label
from .simulate import monte_carlo_champion_probs, simulate_playoff_bracket
from .train import load_models, train_models


def _print_bracket(result) -> None:
    print(f"\n=== {result.season} Playoff Simulation ===")
    for round_name, series_list in result.rounds.items():
        print(f"\n{round_name.replace('_', ' ').title()}:")
        for s in series_list:
            loser = s.lower_seed if s.winner == s.higher_seed else s.higher_seed
            scoreline = f"{s.winner} defeats {loser}"
            games = ", ".join(
                f"G{g.game_number}: {g.home_team} {g.home_score:.0f}-{g.away_score:.0f} {g.away_team}"
                for g in s.games
            )
            print(f"  {scoreline} ({len(s.games)} games)")
            print(f"    {games}")
    print(f"\nChampion: {result.champion}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA Playoff Predictor")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train score models on regular-season data")
    train_p.add_argument("--start-year", type=int, default=2015)
    train_p.add_argument("--end-year", type=int, default=2023)
    train_p.add_argument("--test-season", type=str, default="2022-23")

    fetch_p = sub.add_parser("fetch", help="Download and cache NBA game logs")
    fetch_p.add_argument("--start-year", type=int, default=2015)
    fetch_p.add_argument("--end-year", type=int, default=2023)

    box_p = sub.add_parser("fetch-boxscores", help="Download box scores for cached games")
    box_p.add_argument("--start-year", type=int, default=2015)
    box_p.add_argument("--end-year", type=int, default=2023)

    predict_p = sub.add_parser("predict", help="Simulate playoff bracket for a season")
    predict_p.add_argument("--season-year", type=int, default=2023, help="Start year, e.g. 2023 for 2023-24")
    predict_p.add_argument("--sims", type=int, default=1, help="Monte Carlo simulations")
    predict_p.add_argument("--seed", type=int, default=42)

    backtest_p = sub.add_parser("backtest", help="Train and backtest on historical playoffs")
    backtest_p.add_argument("--train-start", type=int, default=2015)
    backtest_p.add_argument("--train-end", type=int, default=2022)
    backtest_p.add_argument("--test-start", type=int, default=2022)
    backtest_p.add_argument("--test-end", type=int, default=2023)
    backtest_p.add_argument("--sims", type=int, default=25)

    args = parser.parse_args()

    if args.command == "train":
        models = train_models(
            start_year=args.start_year,
            end_year=args.end_year,
            test_season=args.test_season,
        )
        print(f"Training complete. Home MAE: {models.home_mae:.2f}, Away MAE: {models.away_mae:.2f}")

    elif args.command == "fetch":
        regular = load_training_games(args.start_year, args.end_year)
        playoffs = load_playoff_games(args.start_year, args.end_year)
        print(f"Cached {len(regular)} regular-season games and {len(playoffs)} playoff games.")

    elif args.command == "fetch-boxscores":
        team, player = fetch_box_scores_range(args.start_year, args.end_year)
        print(f"Cached {len(team)} team box scores and {len(player)} player rows.")

    elif args.command == "predict":
        season = season_label(args.season_year)
        models = load_models()
        regular = load_training_games(args.season_year, args.season_year)
        playoffs = load_playoff_games(args.season_year, args.season_year)
        if args.sims == 1:
            rng = np.random.default_rng(args.seed)
            result = simulate_playoff_bracket(
                season, models, regular, playoffs, rng=rng
            )
            _print_bracket(result)
        else:
            probs = monte_carlo_champion_probs(
                season,
                models,
                regular,
                playoffs,
                n_sims=args.sims,
                seed=args.seed,
            )
            print(f"\nChampion probabilities ({args.sims} simulations):")
            for team, prob in probs.sort_values(ascending=False).items():
                print(f"  {team}: {prob:.1%}")

    elif args.command == "backtest":
        report = run_backtest(
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            n_sims=args.sims,
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
