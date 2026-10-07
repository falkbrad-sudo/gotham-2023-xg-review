"""Loads StatsBomb open data for NJ/NY Gotham FC's 2023 NWSL season.

Verified against live data during development (see
notebooks/00_exploration.ipynb for the checks that were run). Every
function here makes network calls via statsbombpy; consider caching
results to data/processed/ (see src/pipeline.py) rather than re-fetching on
every run, especially for get_all_nwsl_shots() which loops over ~137 matches.

Source: https://github.com/statsbomb/open-data (competition_id=49,
season_id=107: the free 2023 NWSL season, the only one currently
available). StatsBomb's exact team name string is "NJ/NY Gotham FC", not
"Gotham FC" (confirmed live). get_gotham_matches() uses the confirmed
string, don't guess a different one elsewhere in this codebase.
"""
from __future__ import annotations

import warnings

import pandas as pd
from statsbombpy import sb

from src.config import load_config
from src.data.cleaning import parse_all_freeze_frames

# statsbombpy prints noisy credential warnings for the free/open API even
# though no credentials are needed for open data, so suppress at the source
# rather than letting every caller of this module deal with it.
warnings.filterwarnings("ignore", module="statsbombpy")


def get_2023_nwsl_matches() -> pd.DataFrame:
    """Return the full 2023 NWSL match list (all 12 teams, 137 matches)."""
    cfg = load_config()
    return sb.matches(
        competition_id=cfg["statsbomb"]["competition_id"],
        season_id=cfg["statsbomb"]["season_id"],
    )


def get_gotham_matches() -> pd.DataFrame:
    """Return only NJ/NY Gotham FC's 2023 matches (25 matches, confirmed live)."""
    cfg = load_config()
    team = cfg["statsbomb"]["team_name"]
    matches = get_2023_nwsl_matches()
    mask = (matches["home_team"] == team) | (matches["away_team"] == team)
    return matches[mask].sort_values("match_date").reset_index(drop=True)


def get_match_events(match_id: int) -> pd.DataFrame:
    """Return full event data for one match."""
    return sb.events(match_id=match_id)


def get_gotham_shots(match_ids: list[int] | None = None) -> pd.DataFrame:
    """Return all of Gotham FC's own shots (not their opponents') across
    the given matches, including freeze-frame data.

    Parameters
    ----------
    match_ids : list[int], optional
        If None, uses every Gotham FC match from get_gotham_matches().

    Returns
    -------
    pd.DataFrame
        One row per Gotham shot: includes 'id' (event id, used to join
        freeze-frame data), 'location' (raw [x, y] list), 'shot_statsbomb_xg',
        'shot_outcome', 'shot_body_part', match_id, and match_date (joined
        in from the match list, useful for the early-season vs. title-run
        window comparison).
    """
    cfg = load_config()
    team = cfg["statsbomb"]["team_name"]

    if match_ids is None:
        gotham_matches = get_gotham_matches()
        match_ids = gotham_matches["match_id"].tolist()
        date_lookup = dict(zip(gotham_matches["match_id"], gotham_matches["match_date"]))
    else:
        all_matches = get_2023_nwsl_matches()
        date_lookup = dict(zip(all_matches["match_id"], all_matches["match_date"]))

    all_shots = []
    for match_id in match_ids:
        events = get_match_events(match_id)
        shots = events[(events["type"] == "Shot") & (events["team"] == team)].copy()
        shots["match_id"] = match_id
        shots["match_date"] = date_lookup.get(match_id)
        all_shots.append(shots)

    if not all_shots:
        return pd.DataFrame()

    result = pd.concat(all_shots, ignore_index=True)
    # Unpack the raw [x, y] location list into separate columns up front:
    # every downstream feature module expects flat x/y columns, not a
    # nested list.
    result["x"] = result["location"].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
    result["y"] = result["location"].apply(lambda loc: loc[1] if isinstance(loc, list) else None)
    result["is_goal"] = (result["shot_outcome"] == "Goal").astype(int)
    return result


def get_all_nwsl_shots() -> pd.DataFrame:
    """Return shots from EVERY 2023 NWSL team, tagged with match context.

    This is the training sample for the league xG model (fit on matches
    Gotham did not play in) and the source of the shots Gotham conceded.
    Adds 'home_team', 'away_team', 'is_gotham_match' (either side is the
    configured team) and 'opponent' (the team that did not take the shot).

    Warning
    -------
    This makes ~137 API calls (one per match), which is slow, and re-running it
    repeatedly is unnecessary network load. Cache the result (see
    src/pipeline.py) rather than calling this from an app or notebook cell
    that might re-run casually.
    """
    cfg = load_config()
    team = cfg["statsbomb"]["team_name"]
    all_matches = get_2023_nwsl_matches()

    all_shots = []
    for match in all_matches.itertuples():
        events = get_match_events(match.match_id)
        shots = events[events["type"] == "Shot"].copy()
        shots["match_id"] = match.match_id
        shots["match_date"] = match.match_date
        shots["home_team"] = match.home_team
        shots["away_team"] = match.away_team
        shots["is_gotham_match"] = team in (match.home_team, match.away_team)
        shots["opponent"] = shots["team"].map(
            lambda t, m=match: m.away_team if t == m.home_team else m.home_team
        )
        all_shots.append(shots)

    result = pd.concat(all_shots, ignore_index=True)
    result["x"] = result["location"].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
    result["y"] = result["location"].apply(lambda loc: loc[1] if isinstance(loc, list) else None)
    result["is_goal"] = (result["shot_outcome"] == "Goal").astype(int)
    return result


def get_shot_context(shots: pd.DataFrame) -> pd.DataFrame:
    """Parse freeze-frame data for a shots DataFrame into a long player-position table.

    Parameters
    ----------
    shots : pd.DataFrame
        Output of get_gotham_shots() or get_all_nwsl_shots(). Must contain
        'id' and 'shot_freeze_frame'.

    Returns
    -------
    pd.DataFrame
        One row per (shot, nearby player) pair. See
        src.data.cleaning.parse_all_freeze_frames for the exact schema.
        Join back to `shots` on shot_id == id to combine with shot outcome/
        location.
    """
    return parse_all_freeze_frames(shots)


def get_touch_events(team_matches: pd.DataFrame) -> pd.DataFrame:
    """Return all of Gotham FC's touch events (passes, carries, shots, etc.)
    across a set of matches, for average-position/formation-shape analysis.

    Parameters
    ----------
    team_matches : pd.DataFrame
        Subset of get_gotham_matches() output (e.g., filtered to a date
        window) to pull events for.

    Returns
    -------
    pd.DataFrame
        One row per touch event with a location, including player name,
        x, y (raw StatsBomb units), event type, and match_date.
    """
    cfg = load_config()
    team = cfg["statsbomb"]["team_name"]

    all_events = []
    for match_id, match_date in zip(team_matches["match_id"], team_matches["match_date"]):
        events = get_match_events(match_id)
        # Restrict to event types that have a meaningful single location
        # for the acting player. This excludes things like substitutions,
        # which have no location, or duels, which are less representative
        # of "where this player normally is."
        touch_types = ["Pass", "Carry", "Shot", "Pressure", "Ball Receipt*"]
        touches = events[
            (events["team"] == team)
            & (events["type"].isin(touch_types))
            & events["location"].notna()
        ].copy()
        touches["match_id"] = match_id
        touches["match_date"] = match_date
        all_events.append(touches)

    if not all_events:
        return pd.DataFrame()

    result = pd.concat(all_events, ignore_index=True)
    result["x"] = result["location"].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
    result["y"] = result["location"].apply(lambda loc: loc[1] if isinstance(loc, list) else None)
    return result
