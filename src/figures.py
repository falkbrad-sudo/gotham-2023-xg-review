"""Render the README figures from the pipeline's cached outputs.

Run as `python -m src.figures` after `python -m src.pipeline`. Writes PNGs to
reports/figures/. Every figure carries a source credit and the StatsBomb logo
(reports/assets/), as StatsBomb's open-data terms ask.
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402
from src.viz import review_plots  # noqa: E402
from src.viz.formation_plots import plot_formation_comparison  # noqa: E402
from src.viz.shot_maps import plot_comparison_shot_maps  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LOGO = PROJECT_ROOT / "reports/assets/hudl-statsbomb-logo-default.png"
SOURCE = "Data: StatsBomb open data, 2023 NWSL. Non-penalty shots."
SHAPE_SOURCE = "Data: StatsBomb open data, 2023 NWSL. Event locations, goalkeepers excluded."


def main() -> None:
    cfg = load_config()
    team = cfg["statsbomb"]["team_name"]
    w = cfg["season_windows"]
    windows = {
        "Early season": (w["early_season"]["start_date"], w["early_season"]["end_date"]),
        "Title run": (w["title_run"]["start_date"], w["title_run"]["end_date"]),
    }

    def cached(name: str) -> pd.DataFrame:
        return pd.read_parquet(resolve_path(f"data/processed/{name}.parquet"))

    out = PROJECT_ROOT / "reports/figures"
    out.mkdir(parents=True, exist_ok=True)
    logo = mpimg.imread(LOGO)
    scored = cached("scored_shots")
    own = scored[scored["side"] == "for"].assign(
        basic_xg_pred=lambda d: d["basic_xg"], enhanced_xg_pred=lambda d: d["enhanced_xg"]
    )
    shot_maps = plot_comparison_shot_maps(own)
    shot_maps.suptitle(f"{team} shots, colored by xG: basic vs. enhanced model (stars are goals)",
                       x=0.01, ha="left", fontsize=12)
    positions = cached("formation_avg_positions")
    team_shape = plot_formation_comparison(
        positions[positions["window"] == "early_season"],
        positions[positions["window"] == "title_run"],
    )
    team_shape.suptitle(f"{team} average outfield positions (descriptive, 10+ touches)",
                        x=0.01, ha="left", fontsize=12)

    figures = {
        "shot_maps": shot_maps,
        "team_shape": team_shape,
        "volume_vs_quality": review_plots.plot_volume_vs_quality(cached("team_table"), team),
        "match_xg": review_plots.plot_match_xg(cached("match_series"), windows, team),
        "player_finishing": review_plots.plot_player_finishing(cached("player_finishing"), team),
        "calibration": review_plots.plot_calibration(cached("calibration")),
    }
    for fig in (shot_maps, team_shape):
        fig.subplots_adjust(top=0.84)
    for name, fig in figures.items():
        review_plots.add_source(fig, logo, SHAPE_SOURCE if name == "team_shape" else SOURCE)
        fig.savefig(out / f"{name}.png", dpi=130, facecolor=review_plots.SURFACE)
        logger.info(f"Wrote {out / name}.png")


if __name__ == "__main__":
    main()
