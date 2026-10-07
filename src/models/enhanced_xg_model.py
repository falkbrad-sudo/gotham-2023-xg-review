"""The enhanced xG model: geometry plus freeze-frame shot context
(defenders in the shooting cone, goalkeeper distance from the goal center).

Fit on the league training set in src/models/league_model.py: 2,920
non-penalty shots (249 goals) from matches the graded team did not play in.
Compare it with the basic model on held-out shots, with sample sizes shown
(league_model.holdout_metrics).
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import load_config


def fit_enhanced_model(
    shots: pd.DataFrame,
    feature_cols: list[str] = [
        "distance_to_goal",
        "angle_to_goal",
        "defenders_in_cone",
        "goalkeeper_distance",
    ],
    target_col: str = "is_goal",
) -> LogisticRegression:
    """Fit a logistic regression xG model on geometry + shot-context features.

    Parameters
    ----------
    shots : pd.DataFrame
        Must contain all feature_cols and target_col. Rows with NaN in any
        feature (e.g., shots where the goalkeeper didn't appear in the
        freeze frame) are dropped. Check how many are dropped and report
        it, since a large drop rate would undermine the comparison to the
        basic model (which can use every shot).
    feature_cols : list[str]
    target_col : str

    Returns
    -------
    LogisticRegression
    """
    cfg = load_config()
    clean = shots.dropna(subset=feature_cols + [target_col])
    X = clean[feature_cols].to_numpy()
    y = clean[target_col].to_numpy()
    model = LogisticRegression(random_state=cfg["model"]["random_state"], C=0.5)
    model.fit(X, y)
    return model
