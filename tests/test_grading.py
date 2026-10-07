"""Tests for src/models/grading.py, against hand-computed expected values."""
import numpy as np
import pandas as pd
import pytest

from src.models.grading import goals_vs_xg, player_finishing, poisson_binomial_pmf, team_table


class TestPoissonBinomialPmf:
    def test_two_fair_coins(self):
        # P(0, 1, 2 goals) from two 0.5 shots = 0.25, 0.5, 0.25.
        assert poisson_binomial_pmf(np.array([0.5, 0.5])) == pytest.approx([0.25, 0.5, 0.25])

    def test_mixed_probabilities(self):
        # Shots at 0.1 and 0.3: P(0) = 0.9*0.7, P(2) = 0.1*0.3, P(1) = the rest.
        pmf = poisson_binomial_pmf(np.array([0.1, 0.3]))
        assert pmf == pytest.approx([0.63, 0.34, 0.03])

    def test_sums_to_one_and_mean_is_total_xg(self):
        p = np.random.default_rng(0).uniform(0, 0.4, 200)
        pmf = poisson_binomial_pmf(p)
        assert pmf.sum() == pytest.approx(1.0)
        assert (np.arange(len(pmf)) * pmf).sum() == pytest.approx(p.sum())


class TestGoalsVsXg:
    def test_two_fair_coins_one_goal(self):
        # P(X <= 1) = 0.75, P(X >= 1) = 0.75, so p = min(1, 1.5) = 1.
        result = goals_vs_xg(1, np.array([0.5, 0.5]))
        assert result["xg"] == pytest.approx(1.0)
        assert result["goals_minus_xg"] == pytest.approx(0.0)
        assert result["p_value"] == pytest.approx(1.0)

    def test_scoring_on_every_long_shot_is_significant(self):
        # 10 shots at 0.1 xG, all scored: P(X >= 10) = 0.1**10.
        result = goals_vs_xg(10, np.full(10, 0.1))
        assert result["p_value"] == pytest.approx(2 * 0.1**10)
        assert result["range_high"] < 10

    def test_range_contains_expected_goals(self):
        result = goals_vs_xg(5, np.full(100, 0.05))
        assert result["range_low"] <= 5 <= result["range_high"]


class TestPlayerFinishing:
    def test_min_shots_and_bonferroni(self):
        shots = pd.DataFrame(
            {
                "player": ["A"] * 20 + ["B"] * 20 + ["C"] * 5,
                "is_goal": [1] * 10 + [0] * 10 + [0] * 20 + [1] * 5,
                "enhanced_xg": [0.1] * 45,
                "shot_statsbomb_xg": [0.1] * 45,
            }
        )
        table = player_finishing(shots, min_shots=15)
        assert list(table["player"]) == ["A", "B"]  # C has only 5 shots
        assert table.loc[0, "goals_minus_xg"] == pytest.approx(10 - 2.0)
        assert bool(table.loc[0, "clears_bonferroni"])  # 10 goals from 2.0 xG
        assert not bool(table.loc[1, "clears_bonferroni"])  # 0 from 2.0 is plausible


class TestTeamTable:
    def test_for_and_against_per_match(self):
        shots = pd.DataFrame(
            {
                "team": ["X", "X", "Y"],
                "opponent": ["Y", "Y", "X"],
                "match_id": [1, 1, 1],
                "is_goal": [1, 0, 0],
                "enhanced_xg": [0.3, 0.1, 0.2],
            }
        )
        table = team_table(shots).set_index("team")
        assert table.loc["X", "xg_for_per_match"] == pytest.approx(0.4)
        assert table.loc["X", "xg_against_per_match"] == pytest.approx(0.2)
        assert table.loc["Y", "goals_against"] == 1
        assert table.loc["X", "xg_per_shot_for"] == pytest.approx(0.2)


class TestSeason:
    def _scored(self):
        return pd.DataFrame(
            {
                "match_id": [1, 1, 1, 2, 2],
                "match_date": ["2023-04-01"] * 3 + ["2023-10-01"] * 2,
                "side": ["for", "for", "against", "for", "against"],
                "team": ["G", "G", "X", "G", "Y"],
                "opponent": ["X", "X", "G", "Y", "G"],
                "is_goal": [1, 0, 0, 0, 1],
                "enhanced_xg": [0.3, 0.1, 0.2, 0.5, 0.4],
            }
        )

    def test_per_match_series(self):
        from src.models.grading import per_match_series

        series = per_match_series(self._scored())
        first = series.iloc[0]
        assert (first["opponent"], first["shots_for"], first["shots_against"]) == ("X", 2, 1)
        assert first["xg_for"] == pytest.approx(0.4)
        assert first["xg_diff"] == pytest.approx(0.2)
        assert series.iloc[1]["goals_against"] == 1

    def test_compare_season_windows_counts_matches_per_window(self):
        from src.models.grading import compare_season_windows

        series = pd.DataFrame(
            {
                "match_date": ["2023-04-01", "2023-04-08", "2023-04-15",
                               "2023-10-01", "2023-10-08", "2023-10-15"],
                "xg_for": [0.5, 0.6, 0.7, 1.5, 1.6, 1.7],
            }
        )
        table = compare_season_windows(
            series, ("2023-03-01", "2023-06-30"), ("2023-09-01", "2023-11-12"),
            metrics=["xg_for"],
        )
        row = table.iloc[0]
        assert (row["n_a"], row["n_b"]) == (3, 3)
        assert row["difference"] == pytest.approx(1.0)
        assert row["p_value"] < 0.001
