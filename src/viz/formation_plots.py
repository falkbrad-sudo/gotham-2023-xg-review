"""Visualize Gotham FC's average-position shape for a given time window."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.viz.shot_maps import make_pitch


def _uncrowded(positions: pd.DataFrame, min_gap_m: float = 4.0) -> pd.DataFrame:
    """Rows to label, most-involved players first, skipping any within min_gap_m
    of a player already labeled, so names don't pile on top of each other."""
    order = positions.sort_values("touch_count", ascending=False)
    kept: list = []
    for i, row in order.iterrows():
        if all(((row["x"] - order.loc[j, "x"]) ** 2 + (row["y"] - order.loc[j, "y"]) ** 2) ** 0.5
               >= min_gap_m for j in kept):
            kept.append(i)
    return positions.loc[kept]


def plot_formation_comparison(
    early_positions: pd.DataFrame,
    title_run_positions: pd.DataFrame,
) -> plt.Figure:
    """Side-by-side average-position maps: early season vs. title run."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, positions, label in [
        (axes[0], early_positions, "Early season"),
        (axes[1], title_run_positions, "Title run"),
    ]:
        make_pitch().draw(ax=ax)
        ax.scatter(positions["x"], positions["y"], s=180, color="#1F4E5F", zorder=3)
        for _, row in _uncrowded(positions).iterrows():
            ax.annotate(
                row["player"].split()[-1],
                (row["x"], row["y"]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        ax.set_title(label)
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.04)
    return fig
