"""League-trained xG model, held out from the team being graded.

The model is fit on every 2023 NWSL non-penalty shot from matches the graded
team did NOT play in (either side), then applied to that team's own shots and
to the shots it conceded. Nothing from the graded team's 25 matches is ever
seen in training, so metrics computed on those shots are out-of-sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.data.cleaning import exclude_penalties
from src.models.basic_xg_model import fit_basic_model
from src.models.enhanced_xg_model import fit_enhanced_model

MIN_GOALS_FOR_METRICS = 10


def split_league_shots(
    shots: pd.DataFrame, team: str, feature_cols: list[str]
) -> dict[str, pd.DataFrame]:
    """Split league shots into training, graded-team 'for' and 'against' sets.

    Parameters
    ----------
    shots : pd.DataFrame
        Output of pipeline.build_league_shots_dataset(): needs 'team',
        'is_gotham_match', 'shot_type', 'is_goal' and feature_cols.
    team : str
        Exact team name being graded (config.yaml statsbomb.team_name).
    feature_cols : list[str]
        Rows missing any of these are dropped from every set, so all three
        sets use the same complete-feature rule.

    Returns
    -------
    dict[str, pd.DataFrame]
        'train': non-penalty shots from matches the team did not play in.
        'for': the team's own non-penalty shots.
        'against': non-penalty shots the team conceded.
    """
    clean = exclude_penalties(shots).dropna(subset=feature_cols + ["is_goal"])
    in_team_match = clean["is_gotham_match"].astype(bool)
    return {
        "train": clean[~in_team_match].reset_index(drop=True),
        "for": clean[in_team_match & (clean["team"] == team)].reset_index(drop=True),
        "against": clean[in_team_match & (clean["team"] != team)].reset_index(drop=True),
    }


def fit_league_models(
    train: pd.DataFrame, basic_features: list[str], enhanced_features: list[str]
) -> dict[str, LogisticRegression]:
    """Fit the basic and enhanced models on the league training set."""
    return {
        "basic": fit_basic_model(train, basic_features),
        "enhanced": fit_enhanced_model(train, enhanced_features),
    }


def calibration_table(
    y: np.ndarray, predicted: np.ndarray, n_bins: int = 5
) -> pd.DataFrame:
    """Mean predicted vs. observed goal rate, in equal-count bins of prediction.

    Equal-count (quantile) bins rather than fixed-width ones, because most
    shots have low xG and fixed-width bins would leave the top bins nearly
    empty.

    Returns
    -------
    pd.DataFrame
        One row per bin: n_shots, n_goals, mean_predicted, observed_rate.
    """
    df = pd.DataFrame({"y": y, "p": predicted})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=n_bins, labels=False)
    return (
        df.groupby("bin")
        .agg(
            n_shots=("y", "size"),
            n_goals=("y", "sum"),
            mean_predicted=("p", "mean"),
            observed_rate=("y", "mean"),
        )
        .reset_index(drop=True)
    )


def holdout_metrics(y: np.ndarray, predicted: np.ndarray) -> dict:
    """AUC, log-loss and Brier score with sample size, or a note if too few goals.

    Also reports total predicted vs. actual goals, the simplest calibration
    check for a set of shots.
    """
    y = np.asarray(y)
    predicted = np.asarray(predicted)
    n_goals = int(y.sum())
    result = {
        "n_shots": int(len(y)),
        "n_goals": n_goals,
        "predicted_goals": float(predicted.sum()),
    }
    if n_goals < MIN_GOALS_FOR_METRICS:
        result["note"] = (
            f"Only {n_goals} goals, so AUC/log-loss are omitted as unreliable "
            f"below {MIN_GOALS_FOR_METRICS} positive cases."
        )
        return result
    result["auc"] = float(roc_auc_score(y, predicted))
    result["log_loss"] = float(log_loss(y, predicted))
    result["brier"] = float(brier_score_loss(y, predicted))
    return result


def score_shots(
    shots: pd.DataFrame,
    models: dict[str, LogisticRegression],
    basic_features: list[str],
    enhanced_features: list[str],
) -> pd.DataFrame:
    """Add 'basic_xg' and 'enhanced_xg' columns from the league models."""
    out = shots.copy()
    out["basic_xg"] = models["basic"].predict_proba(out[basic_features].to_numpy())[:, 1]
    out["enhanced_xg"] = models["enhanced"].predict_proba(
        out[enhanced_features].to_numpy()
    )[:, 1]
    return out
