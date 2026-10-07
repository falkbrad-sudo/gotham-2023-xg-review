"""Shot-context features computed from StatsBomb shot freeze-frame data:
defenders positioned between the shooter and goal, and goalkeeper positioning.

Uses per-shot freeze-frame data (see
src/data/cleaning.py:parse_all_freeze_frames), NOT a tracking-based
continuous calculation. Each shot's freeze frame is a single static snapshot
at the moment of the shot, which is exactly what's needed for these
"was the lane blocked" / "was the keeper positioned well" questions, even
without full continuous tracking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.geometry import GOAL_POST_Y, GOAL_X


def _cross2d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """2D cross product (z-component), vectorized over rows."""
    return v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]


def count_defenders_in_cone(
    shot_x: float, shot_y: float, defender_positions: pd.DataFrame
) -> int:
    """Count defenders inside the triangular cone from shot to both goalposts.

    A defender is "in the cone" if their (x, y) position falls within the
    angular wedge subtended by the two goalposts as seen from the shot
    location, AND they are positioned between the shooter and the goal
    line (not behind the shooter, which wouldn't block anything).

    Parameters
    ----------
    shot_x, shot_y : float
        Shot location in standard meter coordinates.
    defender_positions : pd.DataFrame
        Must contain 'x', 'y' columns (standard meter coordinates) for
        opposing (non-teammate) players only. Filter to `teammate == False`
        before calling this (and typically exclude the goalkeeper, handled
        separately by goalkeeper_distance_from_goal_center below).

    Returns
    -------
    int
        Count of defenders meeting both criteria.
    """
    if len(defender_positions) == 0:
        return 0

    post_top = np.array([GOAL_X, GOAL_POST_Y])
    post_bottom = np.array([GOAL_X, -GOAL_POST_Y])
    shot = np.array([shot_x, shot_y])

    v_top = post_top - shot
    v_bottom = post_bottom - shot
    cross_ref = _cross2d(v_top, v_bottom)

    defenders = defender_positions[["x", "y"]].to_numpy(dtype=float)
    v_defenders = defenders - shot  # shape (n, 2)

    cross_top = _cross2d(np.broadcast_to(v_top, v_defenders.shape), v_defenders)
    cross_bottom = _cross2d(v_defenders, np.broadcast_to(v_bottom, v_defenders.shape))

    in_wedge = (np.sign(cross_top) == np.sign(cross_ref)) & (
        np.sign(cross_bottom) == np.sign(cross_ref)
    )
    # Must also be positioned further toward the goal than the shooter,
    # not behind them (a defender "in the cone" geometrically but standing
    # behind the shooter isn't blocking anything).
    in_front = defenders[:, 0] > shot_x

    return int(np.sum(in_wedge & in_front))


def goalkeeper_distance_from_goal_center(
    defender_positions: pd.DataFrame,
) -> float | None:
    """Distance of the defending goalkeeper from the center of their goal.

    Parameters
    ----------
    defender_positions : pd.DataFrame
        Must contain 'x', 'y' (standard meters) and 'position_name', and
        filters to the row where position_name == 'Goalkeeper'.

    Returns
    -------
    float or None
        Distance in meters, or None if no goalkeeper appears in this
        shot's freeze frame. Freeze frames only include players in camera
        view, so an unusually advanced or deep goalkeeper can fall outside
        it; the gap is reported as missing rather than estimated.
    """
    gk_rows = defender_positions[defender_positions["position_name"] == "Goalkeeper"]
    if len(gk_rows) == 0:
        return None
    gk = gk_rows.iloc[0]
    return float(np.sqrt((GOAL_X - gk["x"]) ** 2 + gk["y"] ** 2))


def add_context_features(shots: pd.DataFrame, freeze_frames_meters: pd.DataFrame) -> pd.DataFrame:
    """Add defenders_in_cone and goalkeeper_distance columns to a shots DataFrame.

    Parameters
    ----------
    shots : pd.DataFrame
        Must contain 'id' (matching freeze_frames_meters' 'shot_id') and
        'x', 'y' (standard meter coordinates: apply geometry conversion
        before calling this).
    freeze_frames_meters : pd.DataFrame
        Output of cleaning.parse_all_freeze_frames(), ALREADY converted to
        standard meter coordinates (via cleaning.statsbomb_to_meters).
        this function does not convert units itself, to keep unit handling
        centralized in one place.

    Returns
    -------
    pd.DataFrame
        Copy of shots with 'defenders_in_cone' and 'goalkeeper_distance'
        columns added. Shots with no matching freeze-frame rows (shouldn't
        happen given ~100% coverage observed in this dataset, but checked
        rather than assumed) get NaN, not a placeholder 0.
    """
    out = shots.copy()
    defenders_counts = []
    gk_distances = []

    for _, shot in out.iterrows():
        frame = freeze_frames_meters[freeze_frames_meters["shot_id"] == shot["id"]]
        if len(frame) == 0:
            defenders_counts.append(np.nan)
            gk_distances.append(np.nan)
            continue
        opponents = frame[frame["teammate"] == False]  # noqa: E712
        defenders_counts.append(
            count_defenders_in_cone(shot["x"], shot["y"], opponents)
        )
        gk_distances.append(goalkeeper_distance_from_goal_center(opponents))

    out["defenders_in_cone"] = defenders_counts
    out["goalkeeper_distance"] = gk_distances
    return out
