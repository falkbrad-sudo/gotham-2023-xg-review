"""Tests for src/features/formation.py using small synthetic touch data
(not live network calls; see test_loaders.py for the live-data checks)."""
import pandas as pd
import pytest

from src.features.formation import compute_average_positions, compute_compactness


def _make_touches():
    # Two players, StatsBomb raw units (0-120, 0-80). Player A's touches
    # cluster around (60, 40) -- center of pitch in raw units, which is
    # (0, 0) after conversion. Player B's touches cluster around (100, 60).
    return pd.DataFrame(
        {
            "player": ["A"] * 12 + ["B"] * 12 + ["C"] * 3,  # C has too few touches
            "x": [58, 60, 62, 59, 61, 60, 58, 60, 62, 59, 61, 60]
            + [98, 100, 102, 99, 101, 100, 98, 100, 102, 99, 101, 100]
            + [10, 10, 10],
            "y": [38, 40, 42, 39, 41, 40, 38, 40, 42, 39, 41, 40]
            + [58, 60, 62, 59, 61, 60, 58, 60, 62, 59, 61, 60]
            + [70, 70, 70],
            "match_date": ["2023-04-01"] * 27,
        }
    )


class TestComputeAveragePositions:
    def test_excludes_players_below_min_touches(self):
        touches = _make_touches()
        result = compute_average_positions(touches, "2023-01-01", "2023-12-31", min_touches=10)
        assert set(result["player"]) == {"A", "B"}  # C excluded (only 3 touches)

    def test_average_position_is_approximately_correct(self):
        touches = _make_touches()
        result = compute_average_positions(touches, "2023-01-01", "2023-12-31", min_touches=10)
        player_a = result[result["player"] == "A"].iloc[0]
        # Raw (60, 40) -> standard meters (0, 0) after statsbomb_to_meters.
        assert player_a["x"] == pytest.approx(0.0, abs=0.5)
        assert player_a["y"] == pytest.approx(0.0, abs=0.5)

    def test_date_window_filtering(self):
        touches = _make_touches()
        result = compute_average_positions(touches, "2024-01-01", "2024-12-31", min_touches=10)
        assert len(result) == 0  # no touches fall in this window


class TestComputeCompactness:
    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError):
            compute_compactness(pd.DataFrame({"x": [], "y": []}))

    def test_known_two_player_spread(self):
        # Two players at (0, 0) and (10, 0) in standard meters.
        avg_pos = pd.DataFrame({"player": ["A", "B"], "x": [0.0, 10.0], "y": [0.0, 0.0]})
        result = compute_compactness(avg_pos)
        assert result["team_length_m"] == pytest.approx(10.0)
        assert result["team_width_m"] == pytest.approx(0.0)
        assert result["n_players"] == 2


class TestExcludeGoalkeepers:
    def _touches_with_keeper(self):
        touches = _make_touches()
        touches["position"] = "Center Forward"
        keeper = pd.DataFrame(
            {"player": ["GK"] * 12, "x": [5] * 12, "y": [40] * 12,
             "match_date": ["2023-04-01"] * 12, "position": ["Goalkeeper"] * 12}
        )
        return pd.concat([touches, keeper], ignore_index=True)

    def test_goalkeeper_excluded_by_default(self):
        result = compute_average_positions(
            self._touches_with_keeper(), "2023-01-01", "2023-12-31"
        )
        assert "GK" not in set(result["player"])

    def test_goalkeeper_kept_when_requested(self):
        result = compute_average_positions(
            self._touches_with_keeper(), "2023-01-01", "2023-12-31", exclude_goalkeepers=False
        )
        assert "GK" in set(result["player"])
