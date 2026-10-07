"""Visualize Gotham FC's average-position shape for a given time window."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.viz.shot_maps import make_pitch


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
        for _, row in positions.iterrows():
            ax.annotate(
                row["player"].split()[-1],
                (row["x"], row["y"]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        ax.set_title(label)
    fig.tight_layout()
    return fig
