"""
Generate all publication figures for the ABS challenge strategy framework.

Figures produced:
  1. p* heatmap by count × base state (batter, k=1)
  2. p* heatmap by count × base state (defense, k=1)
  3. p* faceted heatmap (3 outs panels)
  4. p* by inning × score differential
  5. Simplified vs. DP comparison scatter
  6. Location threshold curves by count
  7. EV gain curve (example state)

Usage
-----
    python scripts/generate_figures.py [--dp-csv results/dp_thresholds.csv]
                                       [--simplified-csv results/simplified_thresholds.csv]
                                       [--output figures/]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend.
import matplotlib.pyplot as plt
import pandas as pd

from abs.config import FIGURE_DPI, FIGURE_FORMAT
from abs.models.overturn_model import OverturnModel
from abs.viz.heatmaps import (
    plot_threshold_by_count_bases,
    plot_threshold_by_inning_score,
    plot_threshold_faceted,
    plot_simplified_vs_dp,
)
from abs.viz.threshold_curves import plot_location_threshold_by_count


def _load_or_run_simplified(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        print(f"  Loading simplified thresholds from {csv_path}")
        return pd.read_csv(csv_path)
    print("  Simplified CSV not found — running model now...")
    from abs.models.simplified import SimplifiedThresholdModel
    from abs.state import enumerate_all_states
    model = SimplifiedThresholdModel()
    df = model.threshold_table(enumerate_all_states())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def _load_or_run_dp(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        print(f"  Loading DP thresholds from {csv_path}")
        return pd.read_csv(csv_path)
    print("  DP CSV not found — running solver now (this may take a few minutes)...")
    from abs.models.dp_solver import DPSolver
    solver = DPSolver()
    solver.solve(verbose=True)
    df = solver.threshold_table()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def save(fig: plt.Figure, output_dir: Path, name: str, fmt: str, dpi: int) -> None:
    path = output_dir / f"{name}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ABS strategy figures.")
    parser.add_argument("--dp-csv", default="results/dp_thresholds.csv")
    parser.add_argument("--simplified-csv", default="results/simplified_thresholds.csv")
    parser.add_argument("--output", default="figures/")
    parser.add_argument("--format", default=FIGURE_FORMAT, choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=FIGURE_DPI)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading threshold tables...")
    dp_df = _load_or_run_dp(Path(args.dp_csv))
    simplified_df = _load_or_run_simplified(Path(args.simplified_csv))

    overturn_model = OverturnModel(method="parametric")

    def s(fig, name):
        save(fig, output_dir, name, args.format, args.dpi)

    print("\nGenerating figures...")

    # Figure 1: Batter p* by count × base state (0 outs, k=1)
    fig = plot_threshold_by_count_bases(dp_df, outs=0, k=1, challenger_type="batter")
    s(fig, "fig1_pstar_count_bases_batter_0outs_k1")

    # Figure 2: Defense p* by count × base state (0 outs, k=1)
    fig = plot_threshold_by_count_bases(dp_df, outs=0, k=1, challenger_type="defense")
    s(fig, "fig2_pstar_count_bases_defense_0outs_k1")

    # Figure 3: Faceted (all outs) batter heatmap
    fig = plot_threshold_faceted(dp_df, k=1, challenger_type="batter")
    s(fig, "fig3_pstar_faceted_batter_k1")

    # Figure 4: p* by inning × score diff (0 outs, bases empty, 0-0 count, k=1)
    fig = plot_threshold_by_inning_score(dp_df, outs=0, base_state=0, k=1,
                                          challenger_type="batter", balls=0, strikes=0)
    s(fig, "fig4_pstar_inning_score_batter")

    # Figure 5: Simplified vs. DP scatter
    # Merge the two tables for comparison.
    key_cols = ["inning", "half", "score_diff", "outs", "base_state",
                "balls", "strikes", "challenges_remaining", "challenger_type"]
    if "p_star_simplified" in simplified_df.columns and "p_star_dp" in dp_df.columns:
        merged = simplified_df.merge(dp_df, on=key_cols, how="inner")
        fig = plot_simplified_vs_dp(merged, challenger_type="batter", sample_frac=0.05)
        s(fig, "fig5_simplified_vs_dp")
    else:
        print("  Skipping fig5 (columns mismatch).")

    # Figure 6: Location threshold curves by count (batter, k=1)
    # Extract mean p* by (balls, strikes) for 0 outs, bases empty, k=1, batter.
    sub = dp_df[
        (dp_df["outs"] == 0) & (dp_df["base_state"] == 0) &
        (dp_df["challenges_remaining"] == 1) & (dp_df["challenger_type"] == "batter")
    ]
    p_star_by_count = {
        (int(r.balls), int(r.strikes)): float(r.p_star_dp)
        for r in sub[sub["score_diff"] == 0].itertuples()
        if not pd.isna(r.p_star_dp)
    }
    highlight_counts = [(0, 0), (0, 2), (3, 0), (3, 2), (1, 1), (2, 2)]
    p_star_subset = {k: v for k, v in p_star_by_count.items() if k in highlight_counts}
    if p_star_subset:
        fig = plot_location_threshold_by_count(
            p_star_subset, overturn_model, called_type="strike", initiator_type="batter"
        )
        s(fig, "fig6_location_curves_by_count")
    else:
        print("  Skipping fig6 (no matching p* data).")

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
