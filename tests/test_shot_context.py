"""Tests for src/features/shot_context.py: hand-verified geometric cases
(the same checks run live during development, see the module's docstring).
"""
import pandas as pd
import pytest

from src.features.geometry import GOAL_X
from src.features.shot_context import count_defenders_in_cone, goalkeeper_distance_from_goal_center


class TestCountDefendersInCone:
    def test_direct_blocker_counts(self):
        defenders = pd.DataFrame({"x": [GOAL_X - 5], "y": [0.0]})
        assert count_defenders_in_cone(GOAL_X - 11, 0, defenders) == 1

    def test_wide_defender_does_not_count(self):
        defenders = pd.DataFrame({"x": [GOAL_X - 5], "y": [30.0]})
        assert count_defenders_in_cone(GOAL_X - 11, 0, defenders) == 0

    def test_defender_behind_shooter_does_not_count(self):
        """Same angle as a real blocker, but positioned further from goal
        than the shooter -- shouldn't count, since they can't be blocking
        a shot they're behind.
        """
        defenders = pd.DataFrame({"x": [GOAL_X - 20], "y": [0.0]})
        assert count_defenders_in_cone(GOAL_X - 11, 0, defenders) == 0

    def test_mixed_defenders_counts_only_the_blocker(self):
        defenders = pd.DataFrame({"x": [GOAL_X - 5, GOAL_X - 5], "y": [0.0, 30.0]})
        assert count_defenders_in_cone(GOAL_X - 11, 0, defenders) == 1

    def test_no_defenders_returns_zero(self):
        defenders = pd.DataFrame({"x": [], "y": []})
        assert count_defenders_in_cone(GOAL_X - 11, 0, defenders) == 0


class TestGoalkeeperDistance:
    def test_goalkeeper_on_goal_line_center_is_zero(self):
        gk = pd.DataFrame({"x": [GOAL_X], "y": [0.0], "position_name": ["Goalkeeper"]})
        assert goalkeeper_distance_from_goal_center(gk) == pytest.approx(0.0)

    def test_goalkeeper_off_line_matches_distance(self):
        gk = pd.DataFrame({"x": [GOAL_X - 3], "y": [0.0], "position_name": ["Goalkeeper"]})
        assert goalkeeper_distance_from_goal_center(gk) == pytest.approx(3.0)

    def test_no_goalkeeper_in_frame_returns_none(self):
        no_gk = pd.DataFrame({"x": [10], "y": [10], "position_name": ["Center Back"]})
        assert goalkeeper_distance_from_goal_center(no_gk) is None
