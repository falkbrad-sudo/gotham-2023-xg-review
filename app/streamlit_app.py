"""Interactive season review: NJ/NY Gotham FC's 2023 season through league xG.

Run with: streamlit run app/streamlit_app.py

Reads only the cached outputs of src/pipeline.py. The tables it needs are
committed to the repo; `python -m src.pipeline` regenerates them (if any are
missing, the app shows a message saying so).
Every sentence that states a result is computed from those cached tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit puts app/ on sys.path, not the repo root; add it so `src` imports
# work both locally and on Streamlit Community Cloud.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path  # noqa: E402
from src.models.evaluate import find_biggest_disagreements  # noqa: E402
from src.viz import review_plots  # noqa: E402
from src.viz.formation_plots import plot_formation_comparison  # noqa: E402
from src.viz.shot_maps import plot_comparison_shot_maps  # noqa: E402

st.set_page_config(page_title="Gotham FC 2023: an xG review", layout="wide")

CFG = load_config()
TEAM = CFG["statsbomb"]["team_name"]
WINDOWS = {
    "Early season": (CFG["season_windows"]["early_season"]["start_date"],
                     CFG["season_windows"]["early_season"]["end_date"]),
    "Title run": (CFG["season_windows"]["title_run"]["start_date"],
                  CFG["season_windows"]["title_run"]["end_date"]),
}
REQUIRED = ["scored_shots", "holdout_metrics", "calibration", "side_summary",
            "player_finishing", "team_table", "match_series", "season_tests"]

METRIC_NAMES = {
    "xg_for": "xG created per match",
    "xg_against": "xG conceded per match",
    "xg_diff": "xG difference per match",
    "shots_for": "shots per match",
    "shots_against": "shots conceded per match",
}
MODEL_NAMES = {
    "basic_xg": "Basic (distance, angle)",
    "enhanced_xg": "Enhanced (+ defenders, keeper)",
    "shot_statsbomb_xg": "StatsBomb xG (reference)",
}


@st.cache_data
def load(name: str) -> pd.DataFrame | None:
    path = resolve_path(f"data/processed/{name}.parquet")
    return pd.read_parquet(path) if path.exists() else None


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def sentence_case(label: str) -> str:
    """Capitalize the first letter unless the label starts with 'xG'."""
    return label if label.startswith("xG") else label[:1].upper() + label[1:]


def ranked(position: int, superlative: str) -> str:
    """'the fewest' for 1st place, otherwise e.g. 'the 3rd-fewest'."""
    return f"the {superlative}" if position == 1 else f"the {ordinal(position)}-{superlative}"


def rank(table: pd.DataFrame, col: str, ascending: bool) -> int:
    ranks = table[col].rank(ascending=ascending, method="min")
    return int(ranks[table["team"] == TEAM].iloc[0])


def season_tab(data: dict[str, pd.DataFrame]) -> None:
    teams = data["team_table"]
    me = teams[teams["team"] == TEAM].iloc[0]
    n_teams = len(teams)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shots conceded per match", f"{me['shots_against_per_match']:.1f}",
              help=f"{ordinal(rank(teams, 'shots_against_per_match', True))} fewest of {n_teams}")
    c2.metric("xG per shot conceded", f"{me['xg_per_shot_against']:.3f}",
              help=f"{ordinal(rank(teams, 'xg_per_shot_against', False))} highest of {n_teams}")
    c3.metric("xG per shot created", f"{me['xg_per_shot_for']:.3f}",
              help=f"{ordinal(rank(teams, 'xg_per_shot_for', False))} highest of {n_teams}")
    c4.metric("xG difference per match", f"{me['xg_diff_per_match']:+.2f}",
              help=f"{ordinal(rank(teams, 'xg_diff_per_match', False))} of {n_teams}")
    st.caption(
        f"Gotham conceded {ranked(rank(teams, 'shots_against_per_match', True), 'fewest')} "
        f"shots per match in the league, but those shots were "
        f"{ranked(rank(teams, 'xg_per_shot_against', False), 'highest')} quality. In attack "
        f"its shots were {ranked(rank(teams, 'xg_per_shot_for', True), 'lowest')} quality. "
        f"Non-penalty shots only; {int(me['matches'])} matches including playoffs."
    )

    st.subheader("Shot volume vs. chance quality")
    st.pyplot(review_plots.plot_volume_vs_quality(teams, TEAM))
    with st.expander("Table: every team"):
        cols = {"team": "Team", "matches": "Matches",
                "shots_for_per_match": "Shots/match", "xg_per_shot_for": "xG/shot",
                "xg_for_per_match": "xG/match",
                "shots_against_per_match": "Shots against/match",
                "xg_per_shot_against": "xG/shot against",
                "xg_against_per_match": "xG against/match",
                "xg_diff_per_match": "xG diff/match"}
        st.dataframe(teams[list(cols)].rename(columns=cols).round(3), hide_index=True,
                     width="stretch")
        st.caption(f"{TEAM}'s row is out-of-sample. Other teams' shots were mostly in the "
                   "model's training data, so their rows are league context.")

    st.subheader("Match by match")
    series = data["match_series"]
    st.pyplot(review_plots.plot_match_xg(series, WINDOWS, TEAM))

    tests = data["season_tests"]
    threshold = 0.05 / len(tests)
    n_pass = int(tests["clears_bonferroni"].sum())
    smallest = tests.loc[tests["p_value"].idxmin()]
    verdict = (", which is suggestive but not established." if not smallest["clears_bonferroni"]
               else ".")
    st.markdown(
        f"**Early season vs. title run** ({int(tests['n_a'].iloc[0])} vs. "
        f"{int(tests['n_b'].iloc[0])} matches, Welch's t-test on per-match values). "
        f"{n_pass} of {len(tests)} comparisons clear the multiple-comparison threshold "
        f"(p < {threshold:.3f}). The largest shift is in "
        f"{METRIC_NAMES[smallest['metric']]}: {smallest['mean_a']:.2f} to "
        f"{smallest['mean_b']:.2f} (p = {smallest['p_value']:.3f}){verdict}"
    )
    labels = tests["metric"].map(METRIC_NAMES).map(sentence_case)
    shown = tests.assign(metric=labels).rename(columns={
        "metric": "Per match", "mean_a": "Early season", "mean_b": "Title run",
        "difference": "Change", "p_value": "p", "clears_bonferroni": "Clears threshold"})
    st.dataframe(shown[["Per match", "Early season", "Title run", "Change", "p",
                        "Clears threshold"]].round(3), hide_index=True, width="stretch")
    with st.expander("Table: every match"):
        st.dataframe(series.drop(columns=["match_id"]).round(2), hide_index=True,
                     width="stretch")


def finishing_tab(data: dict[str, pd.DataFrame]) -> None:
    sides = data["side_summary"].set_index("side")
    c1, c2 = st.columns(2)
    for col, side, label in ((c1, "for", "scored"), (c2, "against", "conceded")):
        row = sides.loc[side]
        col.metric(f"Goals {label} vs. xG", f"{int(row['goals'])} vs. {row['xg']:.1f}",
                   delta=f"{row['goals_minus_xg'] + 0.0:+.1f}".replace("-0.0", "+0.0"),
                   delta_color="off")
        col.caption(
            f"{int(row['n_shots'])} non-penalty shots. Plausible range from xG: "
            f"{int(row['range_low'])} to {int(row['range_high'])} goals (95%), "
            f"p = {row['p_value']:.2f}. StatsBomb's xG for the same shots: "
            f"{row['statsbomb_xg']:.1f}."
        )
    st.caption("Goals are non-penalty shot goals, so penalties and own goals are not "
               "included and totals differ from the official score.")

    players = data["player_finishing"]
    st.subheader("Players with 15+ shots")
    st.pyplot(review_plots.plot_player_finishing(players, TEAM))
    n_sig = int(players["clears_bonferroni"].sum())
    st.caption(
        f"{n_sig} of {len(players)} players clear a Bonferroni-corrected threshold. "
        "With this many players, a few will look hot or cold by chance alone."
    )
    cols = {"player": "Player", "n_shots": "Shots", "goals": "Goals", "xg": "xG",
            "goals_minus_xg": "Goals - xG", "range_low": "Range low", "range_high": "Range high",
            "p_value": "p", "xg_per_shot": "xG/shot", "statsbomb_xg": "StatsBomb xG"}
    st.dataframe(players[list(cols)].rename(columns=cols).round(3), hide_index=True,
                 width="stretch")


def shots_tab(data: dict[str, pd.DataFrame]) -> None:
    metrics = data["holdout_metrics"]
    enhanced = metrics[(metrics["side"] == "both")
                       & (metrics["model"] == "enhanced_xg")].iloc[0]
    st.markdown(
        f"The xG model is trained on every non-penalty shot from matches {TEAM} did not "
        f"play in, so every Gotham shot below is out-of-sample. On "
        f"{int(enhanced['n_shots'])} Gotham shots for and against ({int(enhanced['n_goals'])} "
        f"goals) it predicts {enhanced['predicted_goals']:.1f} goals."
    )
    shown = metrics.assign(model=metrics["model"].map(MODEL_NAMES)).rename(columns={
        "side": "Shots", "model": "Model", "n_shots": "n", "n_goals": "Goals",
        "predicted_goals": "Predicted goals", "auc": "AUC", "log_loss": "Log-loss",
        "brier": "Brier"})
    cols = ["Shots", "Model", "n", "Goals", "Predicted goals", "AUC", "Log-loss", "Brier"]
    st.dataframe(shown[[c for c in cols if c in shown.columns]].round(3), hide_index=True,
                 width="stretch")

    left, right = st.columns([1, 2])
    with left:
        st.pyplot(review_plots.plot_calibration(data["calibration"]))

    scored = data["scored_shots"]
    own = scored[scored["side"] == "for"].assign(
        basic_xg_pred=lambda d: d["basic_xg"],
        enhanced_xg_pred=lambda d: d["enhanced_xg"],
        prediction_diff=lambda d: d["enhanced_xg"] - d["basic_xg"],
    )
    with right:
        st.caption("Gotham's own shots. Color is goal probability; stars are goals.")
        st.pyplot(plot_comparison_shot_maps(own))

    st.subheader("Where defender and keeper positions changed the assessment most")
    cols = {"player": "Player", "match_date": "Date", "distance_to_goal": "Distance (m)",
            "defenders_in_cone": "Defenders in cone", "goalkeeper_distance": "GK off center (m)",
            "basic_xg_pred": "Basic xG", "enhanced_xg_pred": "Enhanced xG",
            "prediction_diff": "Difference", "is_goal": "Goal"}
    disagreements = find_biggest_disagreements(own, n=10)
    st.dataframe(disagreements[[c for c in cols if c in disagreements.columns]]
                 .rename(columns=cols).round(3), hide_index=True, width="stretch")


def shape_tab() -> None:
    comparison = load("formation_comparison")
    positions = load("formation_avg_positions")
    st.caption(
        "Average position of each outfield player (10+ touches) across all their touches in "
        "each window. This is an aggregate over event locations, not continuous tracking, "
        "and the comparison is descriptive: no significance test is run."
    )
    if comparison is None:
        st.info("Run `python -m src.pipeline` to build the team-shape comparison.")
        return
    cols = {"window": "Window", "team_length_m": "Team length (m)",
            "team_width_m": "Team width (m)", "centroid_spread_m": "Spread from centroid (m)",
            "n_players": "Players"}
    st.dataframe(comparison[list(cols)].rename(columns=cols).round(1), hide_index=True,
                 width="stretch")
    if positions is not None:
        st.pyplot(plot_formation_comparison(
            positions[positions["window"] == "early_season"],
            positions[positions["window"] == "title_run"],
        ))


def main() -> None:
    st.title(f"{TEAM}, 2023: a season through league xG")
    st.caption(
        "How good were the chances Gotham created and allowed in its title-winning season, "
        "graded by an xG model trained on the rest of the league?"
    )
    st.info(
        "StatsBomb's free, public 2023 NWSL open data (competition_id=49, season_id=107). "
        "Shot context comes from per-shot freeze frames (one snapshot at the moment of each "
        "shot), not continuous tracking. Penalties are excluded.",
        icon="ℹ️",
    )

    data = {name: load(name) for name in REQUIRED}
    missing = [name for name, df in data.items() if df is None]
    if missing:
        st.error(
            "Cached results not found (" + ", ".join(missing) + "). Run "
            "`python -m src.pipeline` from the repo root first. It fetches the data from "
            "StatsBomb and caches it to data/processed/."
        )
        return

    season, finishing, shots, shape = st.tabs(
        ["Season", "Finishing", "Shots & model", "Team shape"]
    )
    with season:
        season_tab(data)
    with finishing:
        finishing_tab(data)
    with shots:
        shots_tab(data)
    with shape:
        shape_tab()

    logo = resolve_path("reports/assets/hudl-statsbomb-logo-default.png")
    if logo.exists():
        st.image(str(logo), width=180)
    st.caption("Data: StatsBomb open data, 2023 NWSL.")


if __name__ == "__main__":
    main()
