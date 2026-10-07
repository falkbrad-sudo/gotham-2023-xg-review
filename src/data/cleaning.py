"""Shared cleaning utilities: coordinate conversion + freeze-frame parsing.

Coordinate convention: standard meters, origin at pitch center, x toward the
attacking goal.
"""
from __future__ import annotations

import pandas as pd

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GOAL_WIDTH_M = 7.32

STATSBOMB_LENGTH_UNITS = 120.0
STATSBOMB_WIDTH_UNITS = 80.0


def statsbomb_to_meters(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Convert StatsBomb's 120x80 pitch-unit coordinates to standard meters."""
    out = df.copy()
    out[x_col] = (out[x_col] / STATSBOMB_LENGTH_UNITS - 0.5) * PITCH_LENGTH_M
    out[y_col] = (out[y_col] / STATSBOMB_WIDTH_UNITS - 0.5) * PITCH_WIDTH_M
    return out


def validate_coordinates(df: pd.DataFrame, x_col: str, y_col: str) -> None:
    """Raise if any (x, y) pair falls meaningfully outside pitch bounds."""
    margin = 1.0
    bad_x = (df[x_col].abs() > PITCH_LENGTH_M / 2 + margin).any()
    bad_y = (df[y_col].abs() > PITCH_WIDTH_M / 2 + margin).any()
    if bad_x or bad_y:
        raise ValueError(
            f"Coordinates in columns '{x_col}'/'{y_col}' fall outside "
            f"expected pitch bounds after conversion; check units."
        )


def exclude_penalties(shots: pd.DataFrame, shot_type_col: str = "shot_type") -> pd.DataFrame:
    """Drop penalty kicks before fitting or comparing xG models.

    Penalties are a fixed-location set piece with their own conversion rate,
    so standard xG practice models them separately. In this dataset it also
    matters for the comparison itself: 4 of Gotham's 5 2023 penalties (all
    goals) have no freeze frame, so keeping penalties would leave only the
    one missed penalty in the complete-feature sample used to compare models.
    """
    return shots[shots[shot_type_col] != "Penalty"].reset_index(drop=True)


def parse_freeze_frame(freeze_frame: list[dict], shot_id) -> pd.DataFrame:
    """Parse one shot's StatsBomb freeze-frame data into a tidy DataFrame.

    StatsBomb's shot_freeze_frame column contains a list of dicts, one per
    nearby player, shaped like:
        {"location": [x, y], "player": {"id": ..., "name": ...},
         "position": {"id": ..., "name": ...}, "teammate": bool}
    (the shooter themselves is NOT included in the freeze frame, only
    other players visible at that moment).

    Parameters
    ----------
    freeze_frame : list[dict]
        One shot's raw shot_freeze_frame value (a Python list after
        statsbombpy's JSON parsing, not a JSON string).
    shot_id : Any
        Identifier for the shot this freeze frame belongs to (e.g., the
        event id), attached as a column so results can be joined back to
        the parent shots DataFrame after processing many shots at once.

    Returns
    -------
    pd.DataFrame
        One row per player in the freeze frame, with columns: shot_id, x,
        y (raw StatsBomb units: convert with statsbomb_to_meters after
        concatenating multiple shots' frames), player_id, player_name,
        position_name, teammate (bool).
    """
    rows = []
    for entry in freeze_frame:
        loc = entry.get("location", [None, None])
        player = entry.get("player", {}) or {}
        position = entry.get("position", {}) or {}
        rows.append(
            {
                "shot_id": shot_id,
                "x": loc[0],
                "y": loc[1],
                "player_id": player.get("id"),
                "player_name": player.get("name"),
                "position_name": position.get("name"),
                "teammate": entry.get("teammate"),
            }
        )
    return pd.DataFrame(rows)


def parse_all_freeze_frames(shots: pd.DataFrame) -> pd.DataFrame:
    """Parse every shot's freeze frame in a shots DataFrame into one long table.

    Parameters
    ----------
    shots : pd.DataFrame
        Must contain 'shot_freeze_frame' (list or NaN) and an 'id' column
        (StatsBomb's event id, used as shot_id). statsbombpy's sb.events()
        output has both.

    Returns
    -------
    pd.DataFrame
        Concatenated output of parse_freeze_frame() across all shots with
        non-null freeze frame data. Shots with no freeze frame (rare in
        this dataset, but check, see notebooks/00_exploration.ipynb) are
        skipped, not filled with estimated positions.
    """
    frames = []
    for _, row in shots.iterrows():
        ff = row.get("shot_freeze_frame")
        if isinstance(ff, list) and len(ff) > 0:
            frames.append(parse_freeze_frame(ff, shot_id=row["id"]))
    if not frames:
        return pd.DataFrame(
            columns=["shot_id", "x", "y", "player_id", "player_name", "position_name", "teammate"]
        )
    return pd.concat(frames, ignore_index=True)
