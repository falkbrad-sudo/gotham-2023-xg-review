"""Integration tests for src/data/statsbomb_loader.py. They require network
access to StatsBomb's open API. Run explicitly with `pytest -m integration`.

Each assertion documents a property of the data that the analysis relies on
(see notebooks/00_exploration.ipynb). If one fails, the data has changed.
"""
import pytest

from src.data.statsbomb_loader import (
    get_2023_nwsl_matches,
    get_gotham_matches,
    get_gotham_shots,
    get_shot_context,
)


@pytest.mark.integration
def test_gotham_team_name_is_exact():
    """StatsBomb's exact naming is 'NJ/NY Gotham FC', not 'Gotham FC',
    confirmed live. If this ever fails, StatsBomb has changed their naming
    and config.yaml's statsbomb.team_name needs updating.
    """
    matches = get_2023_nwsl_matches()
    all_teams = set(matches["home_team"]) | set(matches["away_team"])
    assert "NJ/NY Gotham FC" in all_teams
    assert "Gotham FC" not in all_teams


@pytest.mark.integration
def test_all_matches_are_from_2023():
    """Executable scope guard: the analysis is about the 2023 season only.
    If StatsBomb ever adds later seasons under this season_id, this fails
    rather than silently mixing seasons (or teams that did not exist yet).
    """
    matches = get_2023_nwsl_matches()
    assert matches["match_date"].astype(str).str.startswith("2023").all()


@pytest.mark.integration
def test_gotham_matches_returns_25_matches():
    """Confirmed live during development: exactly 25 matches (22 regular
    season + playoffs) in the free 2023 release. If StatsBomb's data
    changes, this test documents the change rather than silently producing
    different downstream results.
    """
    matches = get_gotham_matches()
    assert len(matches) == 25


@pytest.mark.integration
def test_gotham_shots_have_near_complete_freeze_frame_coverage():
    """Confirmed live: freeze-frame coverage was 100% in every match sampled
    during development. Assert a high threshold (not exactly 100%) so a
    single edge-case shot somewhere doesn't break CI unnecessarily, while
    still catching a real regression (e.g., StatsBomb changing what's
    included in the free release).
    """
    matches = get_gotham_matches()
    shots = get_gotham_shots(match_ids=matches["match_id"].head(5).tolist())
    coverage = shots["shot_freeze_frame"].notna().mean()
    assert coverage > 0.9


@pytest.mark.integration
def test_shot_context_parses_to_nonempty_dataframe():
    matches = get_gotham_matches()
    shots = get_gotham_shots(match_ids=matches["match_id"].head(3).tolist())
    context = get_shot_context(shots)
    assert len(context) > 0
    assert {"shot_id", "x", "y", "teammate", "position_name"}.issubset(context.columns)
