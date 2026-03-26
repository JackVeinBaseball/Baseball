"""
Run the simplified (single-period) ABS challenge threshold model.

Usage
-----
    python scripts/run_simplified.py [--output results/simplified_thresholds.csv]
                                     [--wp-csv path/to/wp.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the package root is on sys.path when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from abs.models.simplified import SimplifiedThresholdModel
from abs.state import enumerate_all_states
from abs.tables.win_probability import WPTable


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simplified ABS threshold model.")
    parser.add_argument("--output", default="results/simplified_thresholds.csv",
                        help="Output CSV path.")
    parser.add_argument("--wp-csv", default=None,
                        help="Path to WP table CSV. Uses bundled table if omitted.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N states (for quick testing).")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading win probability table...")
    wp_table = WPTable(Path(args.wp_csv) if args.wp_csv else None)

    print("Enumerating game states...")
    states = enumerate_all_states()
    if args.sample:
        import random
        random.seed(42)
        states = random.sample(states, min(args.sample, len(states)))
    print(f"  {len(states):,} states to evaluate.")

    print("Computing simplified thresholds...")
    model = SimplifiedThresholdModel(wp_table)
    df = model.threshold_table(states)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows → {output_path}")

    # Print a quick summary.
    k1_states = df[df["challenges_remaining"] >= 1]
    valid = k1_states["p_star_simplified"].dropna()
    n_none = k1_states["p_star_simplified"].isna().sum()
    print(f"\nSummary (k≥1 states, n={len(k1_states):,}):")
    print(f"  States with no meaningful threshold (None): {n_none:,}")
    print(f"    (count change doesn't affect WP table — WP table is count-agnostic)")
    print(f"  States with p*=0 (challenge whenever p>0):  {(valid == 0.0).sum():,}")
    print(f"    (simplified model ignores opportunity cost of using a challenge)")
    if len(valid[valid > 0]) > 0:
        print(f"  States with p*>0: {(valid > 0).sum():,}")
        print(f"  Mean p* (>0):   {valid[valid > 0].mean():.4f}")
    print(f"\nNote: The simplified model correctly gives p*=0 for terminal-count")
    print(f"  situations (e.g. 3-2 called strike) because it ignores the future")
    print(f"  value of saving a challenge. Run run_dp.py for meaningful thresholds.")


if __name__ == "__main__":
    main()
