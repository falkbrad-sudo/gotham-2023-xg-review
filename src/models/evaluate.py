"""Find the shots where the basic and enhanced xG models disagree most."""
from __future__ import annotations

import pandas as pd


def find_biggest_disagreements(compared_shots: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the N shots where the two models disagree most (by absolute diff).

    Includes enough context columns (distance, angle, defenders_in_cone,
    goalkeeper_distance, both predictions, actual outcome) to look at each
    one and understand why the models diverged, without having to
    re-query the raw shot data.
    """
    display_cols = [
        c
        for c in [
            "player",
            "match_date",
            "distance_to_goal",
            "angle_to_goal",
            "defenders_in_cone",
            "goalkeeper_distance",
            "basic_xg_pred",
            "enhanced_xg_pred",
            "prediction_diff",
            "is_goal",
        ]
        if c in compared_shots.columns
    ]
    compared_shots = compared_shots.copy()
    compared_shots["abs_diff"] = compared_shots["prediction_diff"].abs()
    return compared_shots.nlargest(n, "abs_diff")[display_cols]
