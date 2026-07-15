"""Best-of-7 series and full playoff bracket simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .box_scores import load_box_scores
from .features import build_matchup_features, init_team_state_from_games, update_team_state
from .seeds import compute_seeds, compute_win_pcts
from .team_stats import team_box_from_games
from .train import ScoreModels
from .teams import team_conference

HOME_PATTERN = [True, True, False, False, True, False, True]


@dataclass
class GameResult:
    game_number: int
    home_team: str
    away_team: str
    home_score: float
    away_score: float
    winner: str


@dataclass
class SeriesResult:
    higher_seed: str
    lower_seed: str
    winner: str
    games: list[GameResult] = field(default_factory=list)


@dataclass
class BracketResult:
    season: str
    champion: str
    rounds: dict[str, list[SeriesResult]]


def _rng_score(pred: float, mae: float, rng: np.random.Generator) -> float:
    noisy = pred + rng.normal(0.0, mae * 0.75)
    return max(80.0, round(noisy, 1))


def predict_game_scores(
    models: ScoreModels,
    features: dict[str, float],
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    X = pd.DataFrame([features])[models.feature_columns]
    home_pred = float(models.home_model.predict(X)[0])
    away_pred = float(models.away_model.predict(X)[0])
    if rng is None:
        return home_pred, away_pred
    return (
        _rng_score(home_pred, models.home_mae, rng),
        _rng_score(away_pred, models.away_mae, rng),
    )


def _is_elimination_game(wins: dict[str, int], home_team: str, away_team: str) -> bool:
    """True if the losing team would be eliminated."""
    for team in (home_team, away_team):
        other = away_team if team == home_team else home_team
        if wins[team] == 3 and wins[other] < 3:
            return True
    return False


def simulate_series(
    higher_seed: str,
    lower_seed: str,
    models: ScoreModels,
    state: dict[str, dict],
    *,
    game_date: pd.Timestamp | None = None,
    seeds: dict[str, int] | None = None,
    win_pcts: dict[str, float] | None = None,
    league_defaults: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> SeriesResult:
    """Simulate a best-of-7 series; higher seed has home court."""
    wins = {higher_seed: 0, lower_seed: 0}
    games: list[GameResult] = []
    game_num = 0
    series_margins: list[float] = []

    while max(wins.values()) < 4 and game_num < 7:
        home_is_higher = HOME_PATTERN[game_num]
        home_team = higher_seed if home_is_higher else lower_seed
        away_team = lower_seed if home_is_higher else higher_seed

        h2h_margin = float(np.mean(series_margins)) if series_margins else 0.0
        if home_team == higher_seed:
            margin_feature = h2h_margin
        else:
            margin_feature = -h2h_margin

        features = build_matchup_features(
            home_team,
            away_team,
            state,
            game_date=game_date,
            h2h_margin=margin_feature,
            is_playoffs=True,
            seeds=seeds,
            win_pcts=win_pcts,
            is_elimination=_is_elimination_game(wins, home_team, away_team),
            league_defaults=league_defaults,
        )
        home_score, away_score = predict_game_scores(models, features, rng)
        if home_score == away_score:
            home_score += 0.5
        winner = home_team if home_score > away_score else away_team
        wins[winner] += 1
        game_num += 1
        games.append(
            GameResult(
                game_number=game_num,
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                winner=winner,
            )
        )
        margin = home_score - away_score
        series_margins.append(margin if home_team == higher_seed else -margin)

        if game_date is not None:
            game_date = game_date + pd.Timedelta(days=2)

        for team, opp, scored, allowed, loc in [
            (home_team, away_team, home_score, away_score, home_team),
            (away_team, home_team, away_score, home_score, home_team),
        ]:
            update_team_state(
                state,
                team,
                pts_for=scored,
                pts_against=allowed,
                game_date=game_date or pd.Timestamp.now(),
                location=loc,
                is_playoff=True,
            )

    return SeriesResult(
        higher_seed=higher_seed,
        lower_seed=lower_seed,
        winner=higher_seed if wins[higher_seed] == 4 else lower_seed,
        games=games,
    )


def _play_round(
    matchups: list[tuple[str, str]],
    models: ScoreModels,
    state: dict[str, dict],
    *,
    game_date: pd.Timestamp | None,
    seeds: dict[str, int] | None,
    win_pcts: dict[str, float] | None,
    league_defaults: dict[str, float] | None,
    rng: np.random.Generator | None,
) -> list[SeriesResult]:
    results: list[SeriesResult] = []
    for higher_seed, lower_seed in matchups:
        results.append(
            simulate_series(
                higher_seed,
                lower_seed,
                models,
                state,
                game_date=game_date,
                seeds=seeds,
                win_pcts=win_pcts,
                league_defaults=league_defaults,
                rng=rng,
            )
        )
    return results


def _series_first_dates(playoff_games: pd.DataFrame, season: str) -> list[tuple[str, str, pd.Timestamp]]:
    season_games = playoff_games[playoff_games["season"] == season]
    series_dates: dict[tuple[str, str], pd.Timestamp] = {}
    for _, row in season_games.iterrows():
        key = tuple(sorted([row["home_team"], row["away_team"]]))
        game_date = row["game_date"]
        if key not in series_dates or game_date < series_dates[key]:
            series_dates[key] = game_date
    return [(a, b, d) for (a, b), d in series_dates.items()]


def infer_first_round_matchups(
    playoff_games: pd.DataFrame,
    season: str,
    regular_season_games: pd.DataFrame | None = None,
) -> list[tuple[str, str]]:
    """Infer eight round-1 pairings ordered as (higher_seed, lower_seed)."""
    seeds = (
        compute_seeds(regular_season_games, season)
        if regular_season_games is not None
        else {}
    )
    all_series = sorted(_series_first_dates(playoff_games, season), key=lambda x: x[2])
    picked: list[tuple[str, str]] = []
    used_teams: set[str] = set()

    for team_a, team_b, _ in all_series:
        if team_a in used_teams or team_b in used_teams:
            continue
        seed_a = seeds.get(team_a, 9)
        seed_b = seeds.get(team_b, 9)
        if seed_a <= seed_b:
            picked.append((team_a, team_b))
        else:
            picked.append((team_b, team_a))
        used_teams.update([team_a, team_b])
        if len(picked) == 8:
            break
    return picked


def _split_conferences(
    pairings: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if len(pairings) != 8:
        raise ValueError(f"Expected 8 first-round series, found {len(pairings)}")
    west: list[tuple[str, str]] = []
    east: list[tuple[str, str]] = []
    for higher_seed, lower_seed in pairings:
        if team_conference(higher_seed) == "west":
            west.append((higher_seed, lower_seed))
        else:
            east.append((higher_seed, lower_seed))
    if len(west) != 4 or len(east) != 4:
        raise ValueError(f"Expected 4 series per conference, found west={len(west)} east={len(east)}")
    return west, east


def _bracket_next_round(results: list[SeriesResult]) -> list[tuple[str, str]]:
    if len(results) % 2 != 0:
        raise ValueError(f"Cannot pair an odd number of series winners: {len(results)}")
    winners = [r.winner for r in results]
    return [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]


def _league_defaults(season_rs: pd.DataFrame) -> dict[str, float]:
    if season_rs.empty:
        return {"ortg": 112.0, "drtg": 112.0, "pace": 100.0, "efg_pct": 0.54, "tov_pct": 0.13, "reb_pct": 0.5}
    return {
        "ortg": 112.0,
        "drtg": 112.0,
        "pace": 100.0,
        "efg_pct": 0.54,
        "tov_pct": 0.13,
        "reb_pct": 0.5,
    }


def simulate_playoff_bracket(
    season: str,
    models: ScoreModels,
    regular_season_games: pd.DataFrame,
    playoff_games: pd.DataFrame,
    team_box: pd.DataFrame | None = None,
    player_box: pd.DataFrame | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> BracketResult:
    """Simulate full playoffs for a season using inferred bracket."""
    if team_box is None or team_box.empty:
        season_rs = regular_season_games[regular_season_games["season"] == season]
        team_box = team_box_from_games(season_rs)
    if player_box is None:
        _, player_box = load_box_scores()
        if player_box is None:
            player_box = pd.DataFrame()

    state = init_team_state_from_games(
        regular_season_games, team_box, player_box, season
    )
    season_rs = regular_season_games[regular_season_games["season"] == season]
    seeds = compute_seeds(season_rs, season)
    win_pcts = compute_win_pcts(season_rs, season)
    league_defaults = _league_defaults(season_rs)

    first_round = infer_first_round_matchups(playoff_games, season, regular_season_games)
    if not first_round:
        raise ValueError(f"No playoff data to infer bracket for {season}")

    west_r1, east_r1 = _split_conferences(first_round)
    game_date = pd.Timestamp(f"{season[:4]}-04-20")
    rounds: dict[str, list[SeriesResult]] = {}
    round_args = dict(
        models=models,
        state=state,
        game_date=game_date,
        seeds=seeds,
        win_pcts=win_pcts,
        league_defaults=league_defaults,
        rng=rng,
    )

    west_r1_results = _play_round(west_r1, **round_args)
    east_r1_results = _play_round(east_r1, **round_args)
    rounds["west_r1"] = west_r1_results
    rounds["east_r1"] = east_r1_results

    west_r2_results = _play_round(_bracket_next_round(west_r1_results), **round_args)
    east_r2_results = _play_round(_bracket_next_round(east_r1_results), **round_args)
    rounds["west_r2"] = west_r2_results
    rounds["east_r2"] = east_r2_results

    west_final_results = _play_round(_bracket_next_round(west_r2_results), **round_args)
    east_final_results = _play_round(_bracket_next_round(east_r2_results), **round_args)
    rounds["west_conf_finals"] = west_final_results
    rounds["east_conf_finals"] = east_final_results

    finals = _play_round(
        [(west_final_results[0].winner, east_final_results[0].winner)],
        **round_args,
    )
    rounds["finals"] = finals

    return BracketResult(season=season, champion=finals[0].winner, rounds=rounds)


def monte_carlo_champion_probs(
    season: str,
    models: ScoreModels,
    regular_season_games: pd.DataFrame,
    playoff_games: pd.DataFrame,
    *,
    n_sims: int = 200,
    seed: int = 42,
) -> pd.Series:
    counts: dict[str, int] = {}
    for i in range(n_sims):
        rng = np.random.default_rng(seed + i)
        result = simulate_playoff_bracket(
            season, models, regular_season_games, playoff_games, rng=rng
        )
        counts[result.champion] = counts.get(result.champion, 0) + 1
    return pd.Series(counts) / n_sims


def actual_series_winners(playoff_games: pd.DataFrame, season: str) -> dict[tuple[str, str], str]:
    """Map sorted team pair -> series winner for completed series."""
    season_games = playoff_games[playoff_games["season"] == season]
    pairs: dict[tuple[str, str], dict[str, int]] = {}
    for _, row in season_games.iterrows():
        key = tuple(sorted([row["home_team"], row["away_team"]]))
        wins = pairs.setdefault(key, {})
        winner = row["home_team"] if row["home_pts"] > row["away_pts"] else row["away_team"]
        wins[winner] = wins.get(winner, 0) + 1
    result: dict[tuple[str, str], str] = {}
    for key, wins in pairs.items():
        for team, count in wins.items():
            if count == 4:
                result[key] = team
                break
    return result
