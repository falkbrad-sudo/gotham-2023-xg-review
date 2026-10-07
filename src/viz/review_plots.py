"""Season-review charts: per-match xG, league context, finishing, calibration.

Colors follow a fixed role assignment (reference data-viz palette, light
mode): slot 1 blue for the graded team / chances created, slot 2 orange for
chances conceded, neutral grays for everything else. Text always uses text
tokens, never a series color, and every chart has a table view in the app.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
WINDOW_FILL = "#f0efec"
FOR_COLOR = "#2a78d6"
AGAINST_COLOR = "#eb6834"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)


def _figure(figsize: tuple[float, float], ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=figsize, facecolor=SURFACE)
    for ax in np.atleast_1d(axes):
        _style(ax)
    return fig, axes


def _title(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=TEXT_PRIMARY)
    if subtitle:
        fig.text(0.01, 0.905, subtitle, ha="left", fontsize=9, color=TEXT_SECONDARY)


def plot_match_xg(
    series: pd.DataFrame, windows: dict[str, tuple[str, str]], team: str
) -> plt.Figure:
    """xG created and conceded in each match, with the season windows shaded.

    Parameters
    ----------
    series : pd.DataFrame
        grading.per_match_series(): match_date, xg_for, xg_against.
    windows : dict[str, tuple[str, str]]
        Label -> (start_date, end_date); each window is shaded and its
        per-match means drawn as thin dashed lines.
    """
    fig, ax = _figure((10, 4.6))
    x = np.arange(len(series))
    dates = series["match_date"].astype(str)
    for label, (start, end) in windows.items():
        idx = x[(dates >= start) & (dates <= end)]
        if len(idx) == 0:
            continue
        ax.axvspan(idx[0] - 0.5, idx[-1] + 0.5, color=WINDOW_FILL, zorder=0)
        ax.text((idx[0] + idx[-1]) / 2, 2.32, f"{label} ({len(idx)} matches)",
                ha="center", fontsize=9, color=TEXT_SECONDARY)
        for col, color in (("xg_for", FOR_COLOR), ("xg_against", AGAINST_COLOR)):
            mean = series[col].iloc[idx].mean()
            ax.hlines(mean, idx[0] - 0.4, idx[-1] + 0.4, color=color, linewidth=1,
                      linestyles="--", alpha=0.8)

    for col, color, name in (("xg_for", FOR_COLOR, "xG created"),
                             ("xg_against", AGAINST_COLOR, "xG conceded")):
        ax.plot(x, series[col], color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
        ax.annotate(name, (x[-1], series[col].iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT_PRIMARY)

    ax.set_xticks(x[::2])
    ax.set_xticklabels([d[5:] for d in dates.iloc[::2]], fontsize=8)
    ax.set_xlim(-0.7, len(series) + 1.5)
    ax.set_ylim(0, 2.5)
    ax.set_xlabel("Match date (2023)")
    ax.set_ylabel("Non-penalty xG")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)
    _title(fig, f"{team}: chances created and conceded, match by match",
           "League xG model (trained without Gotham's matches). Dashed lines: window means.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def _labels_without_collisions(
    table: pd.DataFrame, x_col: str, y_col: str, team: str, min_gap: float = 0.09
) -> pd.DataFrame:
    """Rows to direct-label: the graded team first, then outliers, skipping crowding.

    Points are compared in axis-normalized units; a label is skipped if its
    point sits within min_gap of an already-labeled point. Every team stays
    named in the app's table view, so skipped labels lose no information.
    """
    norm = table[[x_col, y_col]].apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    centre = norm.mean()
    order = ((norm - centre) ** 2).sum(axis=1).sort_values(ascending=False).index
    order = [i for i in table.index if table.loc[i, "team"] == team] + [
        i for i in order if table.loc[i, "team"] != team
    ]
    placed: list[int] = []
    for i in order:
        if all(np.hypot(*(norm.loc[i] - norm.loc[j])) >= min_gap for j in placed):
            placed.append(i)
    return table.loc[placed]


def plot_volume_vs_quality(team_table: pd.DataFrame, team: str) -> plt.Figure:
    """Shots per match vs. xG per shot for every team, attack and defense side by side.

    Faint curves mark constant xG per match (shots x xG per shot), so a team's
    position shows whether its xG comes from volume or from chance quality.
    """
    fig, axes = _figure((11, 4.8), ncols=2)
    panels = [
        (axes[0], "shots_for_per_match", "xg_per_shot_for", "Attack: chances created"),
        (axes[1], "shots_against_per_match", "xg_per_shot_against", "Defense: chances conceded"),
    ]
    for ax, x_col, y_col, label in panels:
        others = team_table[team_table["team"] != team]
        graded = team_table[team_table["team"] == team]
        x_lo, x_hi = team_table[x_col].min() - 1, team_table[x_col].max() + 1
        y_lo, y_hi = team_table[y_col].min() - 0.008, team_table[y_col].max() + 0.008
        xs = np.linspace(x_lo, x_hi, 100)
        for level in (0.9, 1.1, 1.3, 1.5):
            ax.plot(xs, level / xs, color=GRID, linewidth=1, zorder=1)
            y_end = level / x_hi
            if y_lo < y_end < y_hi:
                ax.text(x_hi, y_end, f" {level} xG/match", fontsize=7, color=MUTED,
                        va="center")
        ax.scatter(others[x_col], others[y_col], s=60, color=MUTED,
                   edgecolor=SURFACE, linewidth=1.5, zorder=2)
        ax.scatter(graded[x_col], graded[y_col], s=110, color=FOR_COLOR,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        for _, row in _labels_without_collisions(team_table, x_col, y_col, team).iterrows():
            is_graded = row["team"] == team
            ax.annotate(row["team"], (row[x_col], row[y_col]), xytext=(6, 4),
                        textcoords="offset points", fontsize=8 if not is_graded else 9,
                        color=TEXT_PRIMARY if is_graded else TEXT_SECONDARY,
                        fontweight="bold" if is_graded else "normal")
        ax.set_xlim(x_lo, x_hi + 1.6)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Shots per match")
        ax.set_ylabel("xG per shot")
        ax.set_title(label, loc="left", fontsize=10, color=TEXT_PRIMARY)
    _title(fig, "2023 NWSL: shot volume vs. chance quality (non-penalty)",
           f"{team} highlighted; its values are out-of-sample. Other teams' shots were "
           "mostly in the training data.")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def plot_player_finishing(players: pd.DataFrame, team: str) -> plt.Figure:
    """Goals vs. the plausible range implied by xG, per player.

    The gray bar is the central 95% range of goal counts the player's xG makes
    plausible; the open tick is total xG; the dot is goals actually scored.
    """
    table = players.sort_values("xg").reset_index(drop=True)
    fig, ax = _figure((8, 0.55 * len(table) + 1.6))
    y = np.arange(len(table))
    ax.hlines(y, table["range_low"], table["range_high"], color=GRID, linewidth=7,
              zorder=1, label="Plausible range (95%)")
    ax.scatter(table["xg"], y, marker="|", s=220, color=TEXT_SECONDARY, linewidth=2,
               zorder=2, label="xG")
    ax.scatter(table["goals"], y, s=70, color=FOR_COLOR, edgecolor=SURFACE,
               linewidth=1.5, zorder=3, label="Goals")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p} ({n} shots)" for p, n in zip(table["player"], table["n_shots"])],
                       fontsize=9, color=TEXT_PRIMARY)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Non-penalty goals")
    ax.set_xlim(left=-0.5)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)
    inside = int(((table["goals"] >= table["range_low"])
                  & (table["goals"] <= table["range_high"])).sum())
    _title(fig, f"{team} finishing vs. xG, players with 15+ shots",
           f"{inside} of {len(table)} players scored within the range their xG makes plausible.")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def plot_calibration(calibration: pd.DataFrame) -> plt.Figure:
    """Mean predicted xG vs. observed goal rate per bin, against the diagonal."""
    fig, ax = _figure((5, 4.6))
    top = max(calibration["mean_predicted"].max(), calibration["observed_rate"].max()) * 1.15
    ax.plot([0, top], [0, top], color=BASELINE, linewidth=1, linestyle="--", zorder=1)
    ax.plot(calibration["mean_predicted"], calibration["observed_rate"], color=FOR_COLOR,
            linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=2)
    for _, row in calibration.iterrows():
        ax.annotate(f"{int(row['n_goals'])}/{int(row['n_shots'])}",
                    (row["mean_predicted"], row["observed_rate"]), xytext=(6, -10),
                    textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Mean predicted xG")
    ax.set_ylabel("Observed goal rate")
    _title(fig, "Calibration on held-out Gotham shots",
           "Equal-count bins; labels are goals/shots. Dashed: perfect calibration.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def add_source(fig: plt.Figure, logo: np.ndarray | None, text: str,
               logo_height_in: float = 0.22) -> None:
    """Data-source credit: provider logo bottom-left, text credit bottom-right.

    StatsBomb's open-data terms ask published work to name StatsBomb as the
    source and show their logo. The logo keeps a fixed physical height so it
    is the same size on every figure.
    """
    fig_w, fig_h = fig.get_size_inches()
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, (logo_height_in + 0.45) / fig_h))
    if logo is not None:
        h = logo_height_in / fig_h
        w = h * (logo.shape[1] / logo.shape[0]) * (fig_h / fig_w)
        ax = fig.add_axes((0.01, 0.015, w, h))
        ax.imshow(logo)
        ax.axis("off")
    fig.text(0.99, 0.02, text, ha="right", va="bottom", fontsize=8, color=MUTED)
