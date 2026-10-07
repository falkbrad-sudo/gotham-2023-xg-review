# Methodology and data principles

This project reviews one team's season (NJ/NY Gotham FC, 2023 NWSL) with an xG model trained on the rest of the league: chance quality, finishing, chances conceded, and change across the season. One team's season is a small sample, so what each result can claim is set by its sample size. This document sets out the principles the analysis follows and the pitfalls a future change could easily reintroduce.

## Principles

1. **Public data only, nothing fabricated.** Only StatsBomb's free 2023 NWSL release (competition_id=49, season_id=107) is used. Synthetic data appears only inside unit tests, where it is labelled as such. Missing values (for example a goalkeeper outside the freeze frame) stay missing; they are never filled with a made-up position.

2. **One season, checked in code.** `tests/test_loaders.py::test_all_matches_are_from_2023` fails if the configured season ever contains matches from another year. If StatsBomb releases newer seasons, change `config.yaml` deliberately and re-run the tests.

3. **The team name is exactly `"NJ/NY Gotham FC"`**, not `"Gotham FC"`. This is StatsBomb's string, enforced by `tests/test_loaders.py::test_gotham_team_name_is_exact`. Tidying it up breaks every downstream filter.

4. **Freeze frames are not tracking.** `shot_context.py` features come from one snapshot per shot (`shot_freeze_frame`), and `formation.py` averages event locations. Neither is continuous tracking, and docs, comments and app text should not describe them that way.

5. **The graded team is never in the training data.** `league_model.split_league_shots` trains on non-penalty shots from matches the team did not play in, by either side. `tests/test_league_model.py` checks this. Every Gotham metric is therefore out-of-sample. Other teams' rows in the league table are partly in-sample and are labelled as league context.

6. **Uncertainty and sample size next to every claim.** Goals vs. xG totals use the exact Poisson-binomial distribution (`grading.goals_vs_xg`): a 95% plausible range and a two-sided p-value. Player tables need 15+ shots and carry a Bonferroni flag. Season comparisons use Welch's t-tests on per-match values with a Bonferroni threshold over all metrics tested. AUC and log-loss are omitted below 10 goals. A result that misses its threshold is reported as suggestive, never as a finding.

7. **App text is computed, not written.** Every sentence in the app that states a result is built from the cached tables, so it cannot drift from the numbers.

## Results as they stand

- Training: 2,920 non-penalty shots, 249 goals, from 112 matches. Held out: Gotham's 341 shots (26 goals) and the 227 it conceded (24 goals).
- Held-out on Gotham, enhanced model: AUC 0.759, log-loss 0.258, 49.8 predicted goals vs. 50 actual; well calibrated in 5 equal-count bins.
- Fewest shots conceded per match in the league (9.1) at the highest xG per shot conceded (0.105). In attack, 2nd-lowest xG per shot (0.076).
- Goals vs. xG: 26 vs. 26.0 scored, 24 vs. 23.8 conceded; all 7 players with 15+ shots within their plausible range.
- Early season vs. title run (13 vs. 8 matches): xG conceded per match 1.07 to 0.78 (p = 0.013), xG difference -0.16 to +0.32 (p = 0.027). Neither clears the 0.01 threshold.
- Outfield shape (pooled averages, descriptive only, no significance test): length 41.5m to 33.5m, width 37.4m to 49.6m. A per-match test in the companion `entropy-of-transition` project finds no robust change, so do not present these numbers as a demonstrated change in shape.

## Code conventions

- Python 3.11+, type hints, numpy-style docstrings.
- One coordinate convention after `src/data/cleaning.py`: meters, origin at the center of the pitch, x positive toward the attacking goal. StatsBomb event data is already oriented so the shooting team attacks toward x = 120, which maps to x = +52.5.
- Every feature with hand-verifiable geometry (distance, angle, cone membership) has a test with a known expected value; see `tests/test_geometry.py` and `tests/test_shot_context.py`. Statistical helpers follow the same pattern (for example, two 0.5 xG shots give P(0, 1, 2 goals) = 0.25, 0.5, 0.25 in `tests/test_grading.py`).
- Cache expensive StatsBomb calls the way `src/pipeline.py` does (parquet in `data/processed/`, gitignored) instead of re-fetching in loops, notebooks or the app. The league dataset keeps only flat columns; raw nested event fields are not cached.
- Charts use fixed color roles from a validated palette (blue for the graded team and chances created, orange for chances conceded, grays otherwise) and carry the StatsBomb logo and a source credit (`review_plots.add_source`).
- `pytest -m "not integration"` and `ruff check .` must pass before a change is finished.

## Pitfalls a future change could reintroduce

- **mplsoccer pitch type must be `"skillcorner"`, not `"custom"`.** `"custom"` puts the origin at a corner and silently misrenders this project's centered coordinates with no error. See the comment in `src/viz/shot_maps.py:_make_pitch`.
- **Penalties.** 4 of Gotham's 5 penalties have no freeze frame, and all 4 were goals; league-wide there are 40. Penalties are a fixed-location set piece with their own conversion rate. `cleaning.exclude_penalties` is applied inside `split_league_shots`, before any fitting or grading.
- **Goal totals are non-penalty shot goals.** Penalties and own goals are not shots in this sample, so totals differ from the official score. Say so wherever totals are shown.
- **Training on the graded team.** Fitting the model on all league shots and then grading Gotham would leak Gotham's own outcomes into its grades. Keep the split by match, not by shooting team: opponents' shots against Gotham are held out too.
- **Goalkeepers in the shape metrics.** Including the keeper turns team length into mostly a measure of how deep the back line sits relative to its own goal. `formation.compute_average_positions` drops goalkeeper touches by default.
- **Testing the Streamlit app.** `AppTest` runs with the repo root already importable, so it can miss import-path problems in a real `streamlit run`. `app/streamlit_app.py` adds the repo root to `sys.path` itself; check a real launch after changing imports.
