"""The basic xG model: logistic regression on distance and angle alone.

The baseline the enhanced model is compared against. It is fit on the league
training set in src/models/league_model.py.
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import load_config


def fit_basic_model(
    shots: pd.DataFrame,
    feature_cols: list[str] = ["distance_to_goal", "angle_to_goal"],
    target_col: str = "is_goal",
) -> LogisticRegression:
    """Fit a logistic regression xG model on geometry features only.

    Parameters
    ----------
    shots : pd.DataFrame
        Must contain feature_cols and target_col, with no NaNs in either
        (drop or impute upstream: this function does not silently handle
        missing values, to avoid masking a data-quality issue).
    feature_cols : list[str]
    target_col : str

    Returns
    -------
    LogisticRegression
        Fitted model. Access .predict_proba(X)[:, 1] for goal probability.
    """
    cfg = load_config()
    clean = shots.dropna(subset=feature_cols + [target_col])
    X = clean[feature_cols].to_numpy()
    y = clean[target_col].to_numpy()
    model = LogisticRegression(random_state=cfg["model"]["random_state"])
    model.fit(X, y)
    return model
