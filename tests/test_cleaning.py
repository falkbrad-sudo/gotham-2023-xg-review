"""Tests for src/data/cleaning.py."""
import pandas as pd
import pytest

from src.data.cleaning import (
    exclude_penalties,
    parse_all_freeze_frames,
    parse_freeze_frame,
    statsbomb_to_meters,
    validate_coordinates,
)


class TestStatsbombToMeters:
    def test_center_maps_to_origin(self):
        df = pd.DataFrame({"x": [60.0], "y": [40.0]})
        out = statsbomb_to_meters(df, "x", "y")
        assert out["x"].iloc[0] == pytest.approx(0.0)
        assert out["y"].iloc[0] == pytest.approx(0.0)


class TestValidateCoordinates:
    def test_out_of_bounds_raises(self):
        df = pd.DataFrame({"x": [1000.0], "y": [0.0]})
        with pytest.raises(ValueError):
            validate_coordinates(df, "x", "y")


class TestParseFreezeFrame:
    def test_parses_expected_fields(self):
        freeze_frame = [
            {
                "location": [50.0, 30.0],
                "player": {"id": 1, "name": "Test Player"},
                "position": {"id": 1, "name": "Goalkeeper"},
                "teammate": False,
            }
        ]
        result = parse_freeze_frame(freeze_frame, shot_id="abc123")
        assert len(result) == 1
        row = result.iloc[0]
        assert row["shot_id"] == "abc123"
        assert row["x"] == 50.0
        assert row["y"] == 30.0
        assert row["player_id"] == 1
        assert row["player_name"] == "Test Player"
        assert row["position_name"] == "Goalkeeper"
        assert bool(row["teammate"]) is False

    def test_empty_freeze_frame_returns_empty_dataframe(self):
        result = parse_freeze_frame([], shot_id="abc123")
        assert len(result) == 0


class TestParseAllFreezeFrames:
    def test_skips_shots_with_no_freeze_frame(self):
        shots = pd.DataFrame(
            {
                "id": ["shot1", "shot2"],
                "shot_freeze_frame": [
                    [
                        {
                            "location": [1, 2],
                            "player": {"id": 1, "name": "A"},
                            "position": {"id": 1, "name": "GK"},
                            "teammate": False,
                        }
                    ],
                    None,
                ],
            }
        )
        result = parse_all_freeze_frames(shots)
        assert len(result) == 1
        assert result.iloc[0]["shot_id"] == "shot1"

    def test_all_missing_returns_empty_with_correct_columns(self):
        shots = pd.DataFrame({"id": ["shot1"], "shot_freeze_frame": [None]})
        result = parse_all_freeze_frames(shots)
        assert len(result) == 0
        assert "shot_id" in result.columns


class TestExcludePenalties:
    def test_drops_only_penalties(self):
        shots = pd.DataFrame({"shot_type": ["Open Play", "Penalty", "Free Kick", "Penalty"]})
        out = exclude_penalties(shots)
        assert list(out["shot_type"]) == ["Open Play", "Free Kick"]
