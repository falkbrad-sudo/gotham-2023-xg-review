"""Tests for src/models/league_model.py on small synthetic shot tables (no network)."""
import numpy as np
import pandas as pd
import pytest

from src.models.league_model import calibration_table, holdout_metrics, split_league_shots

TEAM = "NJ/NY Gotham FC"
FEATURES = ["distance_to_goal", "goalkeeper_distance"]


def _league_shots() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team": [TEAM, "Opp A", "Opp B", "Opp C", "Opp D", TEAM],
            "is_gotham_match": [True, True, False, False, False, True],
            "shot_type": ["Open Play"] * 3 + ["Penalty"] + ["Open Play"] * 2,
            "is_goal": [1, 0, 0, 1, 1, 0],
            "distance_to_goal": [10.0, 12.0, 15.0, 10.4, 20.0, 8.0],
            "goalkeeper_distance": [1.0, 2.0, 1.5, 0.5, 3.0, np.nan],
        }
    )


class TestSplitLeagueShots:
    def test_training_set_never_contains_graded_team_matches(self):
        sets = split_league_shots(_league_shots(), TEAM, FEATURES)
        assert not sets["train"]["is_gotham_match"].any()

    def test_penalties_and_incomplete_rows_are_dropped_everywhere(self):
        sets = split_league_shots(_league_shots(), TEAM, FEATURES)
        assert len(sets["train"]) == 2  # Opp B and Opp D; Opp C's penalty dropped
        assert len(sets["for"]) == 1  # second Gotham shot has no goalkeeper distance
        assert list(sets["against"]["team"]) == ["Opp A"]


class TestCalibrationTable:
    def test_bins_partition_all_shots(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 0.5, 100)
        y = (rng.uniform(0, 1, 100) < p).astype(int)
        table = calibration_table(y, p, n_bins=5)
        assert table["n_shots"].sum() == 100
        assert table["n_goals"].sum() == y.sum()
        assert table["mean_predicted"].is_monotonic_increasing

    def test_perfectly_calibrated_constant_prediction(self):
        y = np.array([1, 0, 0, 0] * 5)
        table = calibration_table(y, np.full(20, 0.25), n_bins=1)
        assert table["observed_rate"].iloc[0] == pytest.approx(0.25)
        assert table["mean_predicted"].iloc[0] == pytest.approx(0.25)


class TestHoldoutMetrics:
    def test_reports_predicted_goals_and_metrics(self):
        y = np.array([1] * 12 + [0] * 88)
        p = np.concatenate([np.full(12, 0.4), np.full(88, 0.05)])
        result = holdout_metrics(y, p)
        assert result["n_goals"] == 12
        assert result["predicted_goals"] == pytest.approx(12 * 0.4 + 88 * 0.05)
        assert result["auc"] == pytest.approx(1.0)

    def test_omits_metrics_below_ten_goals(self):
        result = holdout_metrics(np.array([1] * 5 + [0] * 95), np.full(100, 0.05))
        assert "auc" not in result
        assert "note" in result
