"""Shot maps and model-comparison visualizations using mplsoccer."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch


def make_pitch() -> Pitch:
    # "skillcorner" pitch type uses the exact coordinate convention this
    # project standardizes on: centered origin, x in [-52.5, 52.5], y in
    # [-34, 34] (meters), confirmed against mplsoccer's dimension source,
    # not assumed. Do NOT use pitch_type="custom" here: that type expects
    # (0, 0) at a corner, not the pitch center, and silently misrenders
    # this project's data without raising any error.
    return Pitch(
        pitch_type="skillcorner", pitch_length=105, pitch_width=68, line_color="black"
    )


def plot_comparison_shot_maps(shots: pd.DataFrame) -> plt.Figure:
    """Side-by-side shot maps: basic_xg_pred vs. enhanced_xg_pred, shared color scale."""
    vmax = max(shots["basic_xg_pred"].max(), shots["enhanced_xg_pred"].max())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, col, label in zip(
        axes, ["basic_xg_pred", "enhanced_xg_pred"], ["Basic model", "Enhanced model"]
    ):
        make_pitch().draw(ax=ax)
        goals = shots[shots.get("is_goal", 0) == 1]
        misses = shots[shots.get("is_goal", 0) != 1]
        points = ax.scatter(
            misses["x"], misses["y"], c=misses[col], cmap="Reds", s=50, marker="o",
            edgecolors="black", linewidths=0.4, vmin=0, vmax=vmax,
        )
        if len(goals) > 0:
            ax.scatter(
                goals["x"], goals["y"], c=goals[col], cmap="Reds", s=130, marker="*",
                edgecolors="black", linewidths=0.7, vmin=0, vmax=vmax,
            )
        ax.set_title(label)
    fig.subplots_adjust(left=0.01, right=0.9, wspace=0.04)
    colorbar_axis = fig.add_axes((0.92, 0.2, 0.012, 0.6))
    fig.colorbar(points, cax=colorbar_axis, label="xG (goal probability)")
    return fig
