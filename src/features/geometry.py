"""Shot geometry: distance and angle to goal.

Hand-verified against known cases in tests/test_geometry.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.cleaning import GOAL_WIDTH_M, PITCH_LENGTH_M

GOAL_X = PITCH_LENGTH_M / 2
GOAL_POST_Y = GOAL_WIDTH_M / 2


def distance_to_goal(x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    """Euclidean distance from a shot location to the center of the goal."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dist = np.sqrt((GOAL_X - x) ** 2 + y**2)
    return float(dist) if dist.ndim == 0 else dist


def angle_to_goal(x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    """Angle (radians) subtended by the goal mouth from a shot location."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    scalar_input = x_arr.ndim == 0
    x_flat = np.atleast_1d(x_arr)
    y_flat = np.atleast_1d(y_arr)

    dx = GOAL_X - x_flat
    y_near = GOAL_POST_Y - y_flat
    y_far = -GOAL_POST_Y - y_flat

    dot = dx * dx + y_near * y_far
    norm_near = np.sqrt(dx**2 + y_near**2)
    norm_far = np.sqrt(dx**2 + y_far**2)
    cos_angle = np.clip(dot / (norm_near * norm_far), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return float(angle[0]) if scalar_input else angle


def add_geometry_features(shots: pd.DataFrame, x_col: str = "x", y_col: str = "y") -> pd.DataFrame:
    """Add distance_to_goal and angle_to_goal columns to a shots DataFrame."""
    out = shots.copy()
    out["distance_to_goal"] = distance_to_goal(out[x_col].values, out[y_col].values)
    out["angle_to_goal"] = angle_to_goal(out[x_col].values, out[y_col].values)
    return out
