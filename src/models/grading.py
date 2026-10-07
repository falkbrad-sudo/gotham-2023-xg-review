"""Grade a team's shots with league xG: chance quality, finishing, chances conceded.

Goals minus xG on a few dozen shots is mostly noise, so every total here comes
with the exact distribution of goals implied by the shot-level xG values (a
Poisson-binomial: each shot is an independent coin flip with its own
probability). That gives a plausible range and a two-sided p-value for the
observed goal count, instead of a bare over/under number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA = 0.05


def poisson_binomial_pmf(probabilities: np.ndarray) -> np.ndarray:
    """Exact distribution of the number of successes among independent trials.

    Parameters
    ----------
    probabilities : np.ndarray
        One success probability per trial (here, one xG value per shot).

    Returns
    -------
    np.ndarray
        pmf[k] = P(exactly k successes), for k = 0..n. Computed by adding one
        trial at a time (dynamic programming), which is exact and fast for a
        few hundred shots.
    """
    pmf = np.array([1.0])
    for p in np.asarray(probabilities, dtype=float):
        pmf = np.append(pmf * (1 - p), 0.0) + np.insert(pmf * p, 0, 0.0)
    return pmf


def goals_vs_xg(goals: int, probabilities: np.ndarray, alpha: float = ALPHA) -> dict:
    """Compare an observed goal count with the distribution its xG implies.

    Returns
    -------
    dict
        - n_shots, goals, xg (sum of probabilities), goals_minus_xg
        - range_low, range_high: central (1 - alpha) range of goal counts the
          xG values make plausible
        - p_value: two-sided, 2 * min(P(X <= goals), P(X >= goals)), capped at 1
    """
    probabilities = np.asarray(probabilities, dtype=float)
    pmf = poisson_binomial_pmf(probabilities)
    cdf = np.cumsum(pmf)
    goals = int(goals)
    p_low = cdf[goals]
    p_high = 1.0 - (cdf[goals - 1] if goals > 0 else 0.0)
    xg = float(probabilities.sum())
    return {
        "n_shots": int(len(probabilities)),
        "goals": goals,
        "xg": xg,
        "goals_minus_xg": goals - xg,
        "range_low": int(np.searchsorted(cdf, alpha / 2)),
        "range_high": int(np.searchsorted(cdf, 1 - alpha / 2)),
        "p_value": float(min(1.0, 2 * min(p_low, p_high))),
    }


def side_summary(scored: pd.DataFrame, xg_col: str = "enhanced_xg") -> pd.DataFrame:
    """Goals vs. xG for the team's own shots and the shots it conceded.

    Parameters
    ----------
    scored : pd.DataFrame
        pipeline's scored_shots: 'side', 'is_goal', 'match_id', xg_col and
        'shot_statsbomb_xg'.

    Returns
    -------
    pd.DataFrame
        One row per side with goals_vs_xg() fields plus matches, shots and
        xG per match, xG per shot, and StatsBomb's xG for the same shots.
    """
    rows = []
    for side, group in scored.groupby("side", sort=False):
        n_matches = group["match_id"].nunique()
        stats = goals_vs_xg(group["is_goal"].sum(), group[xg_col])
        rows.append(
            {
                "side": side,
                "matches": n_matches,
                **stats,
                "shots_per_match": len(group) / n_matches,
                "xg_per_match": stats["xg"] / n_matches,
                "xg_per_shot": stats["xg"] / len(group),
                "statsbomb_xg": float(group["shot_statsbomb_xg"].sum()),
            }
        )
    return pd.DataFrame(rows)


def player_finishing(
    shots_for: pd.DataFrame, min_shots: int = 15, xg_col: str = "enhanced_xg"
) -> pd.DataFrame:
    """Per-player goals vs. xG for players with at least min_shots shots.

    Ranking many players will always turn up a few extreme ones by chance, so
    a Bonferroni-corrected flag is included: 'clears_bonferroni' is True only
    if p_value < ALPHA / (number of players listed). Treat the rest as
    descriptive.
    """
    rows = []
    for player, group in shots_for.groupby("player"):
        if len(group) < min_shots:
            continue
        rows.append(
            {
                "player": player,
                **goals_vs_xg(group["is_goal"].sum(), group[xg_col]),
                "xg_per_shot": float(group[xg_col].mean()),
                "statsbomb_xg": float(group["shot_statsbomb_xg"].sum()),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["clears_bonferroni"] = table["p_value"] < ALPHA / len(table)
    return table.sort_values("goals_minus_xg", ascending=False).reset_index(drop=True)


def team_table(league_scored: pd.DataFrame, xg_col: str = "enhanced_xg") -> pd.DataFrame:
    """Every team's chances created and conceded per match, for league context.

    Parameters
    ----------
    league_scored : pd.DataFrame
        All non-penalty league shots scored by the league model: 'team',
        'opponent', 'match_id', 'is_goal', xg_col.

    Returns
    -------
    pd.DataFrame
        One row per team: matches, shots/xG/goals per match for and against,
        and xG per shot for and against, sorted by xG difference per match.
    """
    created = league_scored.groupby("team").agg(
        matches=("match_id", "nunique"),
        shots_for=("is_goal", "size"),
        goals_for=("is_goal", "sum"),
        xg_for=(xg_col, "sum"),
    )
    conceded = league_scored.groupby("opponent").agg(
        shots_against=("is_goal", "size"),
        goals_against=("is_goal", "sum"),
        xg_against=(xg_col, "sum"),
    )
    table = created.join(conceded).reset_index().rename(columns={"index": "team"})
    for col in ["shots_for", "goals_for", "xg_for", "shots_against", "goals_against",
                "xg_against"]:
        table[f"{col}_per_match"] = table[col] / table["matches"]
    table["xg_per_shot_for"] = table["xg_for"] / table["shots_for"]
    table["xg_per_shot_against"] = table["xg_against"] / table["shots_against"]
    table["xg_diff_per_match"] = table["xg_for_per_match"] - table["xg_against_per_match"]
    return table.sort_values("xg_diff_per_match", ascending=False).reset_index(drop=True)


SEASON_METRICS = ["xg_for", "xg_against", "xg_diff", "shots_for", "shots_against"]


def per_match_series(scored: pd.DataFrame, xg_col: str = "enhanced_xg") -> pd.DataFrame:
    """One row per match: shots, goals and xG for and against, in date order.

    Goals here are non-penalty shot goals only (no penalties, no own goals),
    so they will not always match the official score.
    """
    rows = []
    for (match_id, match_date), group in scored.groupby(["match_id", "match_date"]):
        own = group[group["side"] == "for"]
        opp = group[group["side"] == "against"]
        rows.append(
            {
                "match_id": match_id,
                "match_date": str(match_date),
                "opponent": own["opponent"].iloc[0] if len(own) else opp["team"].iloc[0],
                "shots_for": len(own),
                "shots_against": len(opp),
                "goals_for": int(own["is_goal"].sum()),
                "goals_against": int(opp["is_goal"].sum()),
                "xg_for": float(own[xg_col].sum()),
                "xg_against": float(opp[xg_col].sum()),
            }
        )
    series = pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)
    series["xg_diff"] = series["xg_for"] - series["xg_against"]
    return series


def compare_season_windows(
    series: pd.DataFrame,
    window_a: tuple[str, str],
    window_b: tuple[str, str],
    metrics: list[str] = SEASON_METRICS,
) -> pd.DataFrame:
    """Welch's t-test per metric on per-match values, between two date windows.

    Each match is one observation, so n is the number of matches per window
    (reported). A Bonferroni threshold over all metrics tested is included as
    'clears_bonferroni'.
    """
    from scipy.stats import ttest_ind

    def in_window(window: tuple[str, str]) -> pd.DataFrame:
        start, end = window
        return series[(series["match_date"] >= start) & (series["match_date"] <= end)]

    a, b = in_window(window_a), in_window(window_b)
    rows = []
    for metric in metrics:
        test = ttest_ind(a[metric], b[metric], equal_var=False)
        rows.append(
            {
                "metric": metric,
                "n_a": len(a),
                "mean_a": float(a[metric].mean()),
                "n_b": len(b),
                "mean_b": float(b[metric].mean()),
                "difference": float(b[metric].mean() - a[metric].mean()),
                "p_value": float(test.pvalue),
            }
        )
    table = pd.DataFrame(rows)
    table["clears_bonferroni"] = table["p_value"] < ALPHA / len(table)
    return table
