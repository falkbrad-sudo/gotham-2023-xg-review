"""Aggregate team-shape analysis from Gotham FC touch-event data.

Computes each player's average position from their touches within a date
window, and summary compactness metrics from that. The intended use case
is comparing Gotham FC's shape early in their 2023 season vs. during their
"worst-to-first" title run (see config.yaml: season_windows).

This is an AGGREGATE proxy, not frame-by-frame tracking-based analysis.
StatsBomb's freeze-frame data (used in shot_context.py) only covers the
moment of shots, not continuous play, so it can't support frame-by-frame
shape metrics the way continuous tracking data would. Don't present this
module's output as equivalent to that.
"""
from __future__ import annotations

import pandas as pd

from src.data.cleaning import statsbomb_to_meters


def compute_average_positions(
    touches: pd.DataFrame,
    start_date: str,
    end_date: str,
    min_touches: int = 10,
    exclude_goalkeepers: bool = True,
) -> pd.DataFrame:
    """Average (x, y) position per player, from touches in a date window.

    Parameters
    ----------
    touches : pd.DataFrame
        Output of statsbomb_loader.get_touch_events(). Must contain
        'player', 'x', 'y' (raw StatsBomb units), 'match_date'.
    start_date, end_date : str
        ISO date strings (inclusive), matching config.yaml's
        season_windows entries.
    min_touches : int
        Players with fewer touches than this in the window are excluded;
        a handful of touches isn't a meaningful "average position," and
        including them (e.g., a substitute who played 5 minutes) would
        distort the compactness metrics without saying anything real about
        team shape.
    exclude_goalkeepers : bool
        Drop touches made while playing as goalkeeper (StatsBomb 'position'
        column == "Goalkeeper"), so the compactness metrics describe the
        outfield shape. Including the keeper would make team_length_m
        mostly a measure of how far the back line sits from its own goal.

    Returns
    -------
    pd.DataFrame
        One row per player (with at least min_touches), with mean x
        (meters, standard coordinates), mean y (meters), and touch_count.
    """
    window = touches[
        (touches["match_date"] >= start_date) & (touches["match_date"] <= end_date)
    ].copy()
    window = window.dropna(subset=["x", "y"])
    if exclude_goalkeepers and "position" in window.columns:
        window = window[window["position"] != "Goalkeeper"]
    window = statsbomb_to_meters(window, "x", "y")

    grouped = (
        window.groupby("player")
        .agg(x=("x", "mean"), y=("y", "mean"), touch_count=("x", "size"))
        .reset_index()
    )
    return grouped[grouped["touch_count"] >= min_touches].reset_index(drop=True)


def compute_compactness(average_positions: pd.DataFrame) -> dict[str, float]:
    """Summary compactness metrics from a set of average player positions.

    Parameters
    ----------
    average_positions : pd.DataFrame
        Output of compute_average_positions(): one row per player, 'x'/'y'
        columns in standard meter coordinates.

    Returns
    -------
    dict[str, float]
        - team_length_m: spread of average positions along the attacking
          axis (max x - min x), i.e. how "stretched" the team is end to end.
        - team_width_m: spread across the pitch width (max y - min y).
        - centroid_spread_m: mean distance of each player's average
          position from the team's overall centroid, an overall
          compactness scalar, computed from window-long averages rather
          than a single instant.
        - n_players: how many players met the min_touches threshold and
          were included. Report this alongside the other numbers, since a
          window with very few qualifying players (e.g., due to injuries/
          rotation) makes the shape metrics less meaningful.
    """
    if len(average_positions) == 0:
        raise ValueError("No players met the touch-count threshold in this window.")

    centroid_x = average_positions["x"].mean()
    centroid_y = average_positions["y"].mean()
    distances = (
        (average_positions["x"] - centroid_x) ** 2 + (average_positions["y"] - centroid_y) ** 2
    ) ** 0.5

    return {
        "team_length_m": float(average_positions["x"].max() - average_positions["x"].min()),
        "team_width_m": float(average_positions["y"].max() - average_positions["y"].min()),
        "centroid_spread_m": float(distances.mean()),
        "n_players": int(len(average_positions)),
    }


def compare_windows(
    touches: pd.DataFrame,
    window_a: tuple[str, str],
    window_b: tuple[str, str],
    window_a_label: str = "window_a",
    window_b_label: str = "window_b",
) -> pd.DataFrame:
    """Compare Gotham FC's compactness between two date windows.

    Parameters
    ----------
    touches : pd.DataFrame
        Output of statsbomb_loader.get_touch_events(), across the full set
        of matches spanning both windows.
    window_a, window_b : tuple[str, str]
        Each an (start_date, end_date) pair.
    window_a_label, window_b_label : str
        Human-readable labels for the output (e.g., "early_season",
        "title_run"), pulled from config.yaml's season_windows keys by
        the pipeline, not hardcoded here.

    Returns
    -------
    pd.DataFrame
        Two rows (one per window), with all compute_compactness() metrics
        as columns, ready for direct display or plotting.
    """
    results = []
    for (start, end), label in [(window_a, window_a_label), (window_b, window_b_label)]:
        avg_pos = compute_average_positions(touches, start, end)
        compactness = compute_compactness(avg_pos)
        compactness["window"] = label
        compactness["start_date"] = start
        compactness["end_date"] = end
        results.append(compactness)
    return pd.DataFrame(results)
