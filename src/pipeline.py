"""End-to-end orchestration for the season review.

league shots -> features -> league xG model (graded team's matches held out)
-> score the team's shots for and against -> grading and season tests;
plus the separate average-position team-shape comparison.

Run as `python -m src.pipeline`. Caches intermediate results to
data/processed/ as parquet, since re-fetching ~25-137 matches from the
StatsBomb API on every run is slow and unnecessary network load. Delete
the cached files if you need to force a refresh (e.g., after changing
season_windows in config.yaml).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path
from src.data import statsbomb_loader
from src.data.cleaning import statsbomb_to_meters
from src.features import formation
from src.features.geometry import add_geometry_features
from src.features.shot_context import add_context_features
from src.models import grading, league_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASIC_FEATURES = ["distance_to_goal", "angle_to_goal"]
ENHANCED_FEATURES = [
    "distance_to_goal",
    "angle_to_goal",
    "defenders_in_cone",
    "goalkeeper_distance",
]


def _cache_path(name: str) -> Path:
    cfg = load_config()
    processed_dir = resolve_path(cfg["paths"]["data_processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir / name


LEAGUE_SHOT_COLUMNS = [
    "id", "match_id", "match_date", "home_team", "away_team", "is_gotham_match",
    "team", "opponent", "player", "period", "minute", "play_pattern",
    "shot_type", "shot_body_part", "shot_outcome", "shot_statsbomb_xg",
    "x", "y", "is_goal", "has_freeze_frame",
    "distance_to_goal", "angle_to_goal", "defenders_in_cone", "goalkeeper_distance",
]


def build_league_shots_dataset(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch (or load cached) every 2023 NWSL shot with geometry + context features.

    About 137 API calls on the first run; cached to
    league_shots_with_features.parquet afterwards. Only the flat columns in
    LEAGUE_SHOT_COLUMNS are kept, not the raw nested event fields.
    """
    cache = _cache_path("league_shots_with_features.parquet")
    if cache.exists() and not force_refresh:
        logger.info(f"Loading cached league shots dataset from {cache}")
        return pd.read_parquet(cache)

    logger.info("Fetching every 2023 NWSL shot from StatsBomb (live, ~137 matches)...")
    shots = statsbomb_loader.get_all_nwsl_shots()
    shots["has_freeze_frame"] = shots["shot_freeze_frame"].apply(
        lambda ff: isinstance(ff, list) and len(ff) > 0
    )
    shots = statsbomb_to_meters(shots, "x", "y")
    shots = add_geometry_features(shots)

    freeze = statsbomb_loader.get_shot_context(shots)
    freeze = statsbomb_to_meters(freeze, "x", "y")
    shots = add_context_features(shots, freeze)

    shots = shots[LEAGUE_SHOT_COLUMNS].reset_index(drop=True)
    shots.to_parquet(cache)
    logger.info(f"Cached {len(shots)} league shots to {cache}")
    return shots


def build_formation_comparison(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch (or load cached) the early-season vs. title-run formation comparison.

    Also caches each window's per-player average positions to
    formation_avg_positions.parquet, so the Streamlit app can draw the
    shape maps without re-fetching touch events.
    """
    cache = _cache_path("formation_comparison.parquet")
    positions_cache = _cache_path("formation_avg_positions.parquet")
    if cache.exists() and positions_cache.exists() and not force_refresh:
        logger.info(f"Loading cached formation comparison from {cache}")
        return pd.read_parquet(cache)

    cfg = load_config()
    logger.info("Fetching Gotham FC touch events from StatsBomb (live)...")
    matches = statsbomb_loader.get_gotham_matches()
    touches = statsbomb_loader.get_touch_events(matches)

    windows = cfg["season_windows"]
    result = formation.compare_windows(
        touches,
        window_a=(windows["early_season"]["start_date"], windows["early_season"]["end_date"]),
        window_b=(windows["title_run"]["start_date"], windows["title_run"]["end_date"]),
        window_a_label="early_season",
        window_b_label="title_run",
    )
    result.to_parquet(cache)

    positions = pd.concat(
        [
            formation.compute_average_positions(
                touches, windows[label]["start_date"], windows[label]["end_date"]
            ).assign(window=label)
            for label in ("early_season", "title_run")
        ],
        ignore_index=True,
    )
    positions.to_parquet(positions_cache)
    logger.info(f"Cached formation comparison to {cache} and positions to {positions_cache}")
    return result


def build_league_review(force_refresh: bool = False) -> dict:
    """Fit the league models (graded team's matches held out) and score its shots.

    Caches:
    - scored_shots.parquet: the team's non-penalty shots for and against,
      with a 'side' column ('for' / 'against') and basic_xg / enhanced_xg.
    - holdout_metrics.parquet: AUC, log-loss, Brier and predicted vs.
      actual goals for each model and StatsBomb's own xG, per side.
    - calibration.parquet: enhanced-model calibration bins on all held-out shots.
    """
    cfg = load_config()
    team = cfg["statsbomb"]["team_name"]
    league = build_league_shots_dataset(force_refresh=force_refresh)
    sets = league_model.split_league_shots(league, team, ENHANCED_FEATURES)
    models = league_model.fit_league_models(sets["train"], BASIC_FEATURES, ENHANCED_FEATURES)
    logger.info(
        f"League models fit on {len(sets['train'])} shots "
        f"({int(sets['train']['is_goal'].sum())} goals) from matches {team} did not play"
    )

    scored = pd.concat(
        [
            league_model.score_shots(sets[side], models, BASIC_FEATURES, ENHANCED_FEATURES)
            .assign(side=side)
            for side in ("for", "against")
        ],
        ignore_index=True,
    )
    scored.to_parquet(_cache_path("scored_shots.parquet"))

    metric_rows = []
    for side, group in [("for", scored[scored["side"] == "for"]),
                        ("against", scored[scored["side"] == "against"]),
                        ("both", scored)]:
        for model_col in ("basic_xg", "enhanced_xg", "shot_statsbomb_xg"):
            metrics = league_model.holdout_metrics(group["is_goal"], group[model_col])
            metric_rows.append({"side": side, "model": model_col, **metrics})
    holdout = pd.DataFrame(metric_rows)
    holdout.to_parquet(_cache_path("holdout_metrics.parquet"))
    logger.info(f"Held-out metrics on {team}:\n{holdout}")

    calibration = league_model.calibration_table(
        scored["is_goal"].to_numpy(), scored["enhanced_xg"].to_numpy()
    )
    calibration.to_parquet(_cache_path("calibration.parquet"))

    # Grading. Only the graded team's row of the league table is fully
    # out-of-sample; other teams' shots were mostly in the training set, so
    # their rows are league context, not held-out estimates.
    sides = grading.side_summary(scored)
    sides.to_parquet(_cache_path("side_summary.parquet"))
    players = grading.player_finishing(scored[scored["side"] == "for"])
    players.to_parquet(_cache_path("player_finishing.parquet"))
    league_scored = league_model.score_shots(
        pd.concat(list(sets.values()), ignore_index=True),
        models, BASIC_FEATURES, ENHANCED_FEATURES,
    )
    teams = grading.team_table(league_scored)
    teams.to_parquet(_cache_path("team_table.parquet"))
    logger.info(f"{team} goals vs. xG:\n{sides}")

    series = grading.per_match_series(scored)
    series.to_parquet(_cache_path("match_series.parquet"))
    windows = cfg["season_windows"]
    season_tests = grading.compare_season_windows(
        series,
        (windows["early_season"]["start_date"], windows["early_season"]["end_date"]),
        (windows["title_run"]["start_date"], windows["title_run"]["end_date"]),
    )
    season_tests.to_parquet(_cache_path("season_tests.parquet"))
    logger.info(f"Early season vs. title run, per match:\n{season_tests}")

    return {
        "scored_shots": scored,
        "holdout_metrics": holdout,
        "calibration": calibration,
        "side_summary": sides,
        "player_finishing": players,
        "team_table": teams,
        "match_series": series,
        "season_tests": season_tests,
    }


def run_full_pipeline(force_refresh: bool = False) -> dict:
    """Run everything and return a summary dict. This is the main entry point."""
    league_review = build_league_review(force_refresh=force_refresh)
    formation_comparison = build_formation_comparison(force_refresh=force_refresh)
    logger.info(f"Formation comparison:\n{formation_comparison}")
    return {"league_review": league_review, "formation_comparison": formation_comparison}


if __name__ == "__main__":
    run_full_pipeline()
