"""Tests for src/features/geometry.py, against hand-computed expected values."""
import math

import pandas as pd
import pytest

from src.features.geometry import (
    GOAL_POST_Y,
    GOAL_X,
    add_geometry_features,
    angle_to_goal,
    distance_to_goal,
)


class TestDistanceToGoal:
    def test_on_goal_line_centered_is_zero(self):
        assert distance_to_goal(GOAL_X, 0) == pytest.approx(0.0)

    def test_penalty_spot_is_11m(self):
        assert distance_to_goal(GOAL_X - 11, 0) == pytest.approx(11.0)

    def test_pythagorean_off_center_shot(self):
        assert distance_to_goal(GOAL_X - 4, 3) == pytest.approx(5.0)


class TestAngleToGoal:
    def test_standing_on_goal_line_centered_gives_pi(self):
        assert angle_to_goal(GOAL_X, 0) == pytest.approx(math.pi, rel=1e-4)

    def test_penalty_spot_matches_hand_calculation(self):
        expected = 2 * math.atan(GOAL_POST_Y / 11)
        assert angle_to_goal(GOAL_X - 11, 0) == pytest.approx(expected, rel=1e-6)

    def test_symmetric_about_the_centerline(self):
        left = angle_to_goal(GOAL_X - 11, 5)
        right = angle_to_goal(GOAL_X - 11, -5)
        assert left == pytest.approx(right, rel=1e-9)


class TestAddGeometryFeatures:
    def test_adds_expected_columns(self):
        shots = pd.DataFrame({"x": [GOAL_X - 11, GOAL_X - 20], "y": [0, 5]})
        out = add_geometry_features(shots)
        assert "distance_to_goal" in out.columns
        assert "angle_to_goal" in out.columns
