"""
Run the full dynamic-programming ABS challenge threshold solver.

Usage
-----
    python scripts/run_dp.py [--output results/dp_thresholds.csv]
                             [--wp-csv path/to/wp.csv]
                             [--quiet]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from abs.models.dp_solver import DPSolver
from abs.tables.run_expectancy import RE24Table
from abs.tables.win_probability import WPTable


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full DP ABS threshold solver.")
    parser.add_argument("--output", default="results/dp_thresholds.csv",
                        help="Output CSV path.")
    parser.add_argument("--wp-csv", default=None,
                        help="Path to WP table CSV. Uses bundled table if omitted.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress bar.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading tables...")
    wp_table = WPTable(Path(args.wp_csv) if args.wp_csv else None)
    re_table = RE24Table()

    print("Initializing DP solver...")
    solver = DPSolver(wp_table, re_table)

    print("Running backward induction (this may take 30–120 seconds)...")
    t0 = time.time()
    solver.solve(verbose=not args.quiet)
    elapsed = time.time() - t0
    print(f"Solve complete in {elapsed:.1f}s.")

    print("Extracting threshold table...")
    df = solver.threshold_table()

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows → {output_path}")

    valid = df["p_star_dp"].dropna()
    print(f"\nSummary:")
    print(f"  Mean p*:   {valid.mean():.4f}")
    print(f"  Median p*: {valid.median():.4f}")
    print(f"  Min p*:    {valid.min():.4f}")
    print(f"  Max p*:    {valid.max():.4f}")
    print(f"  States where challenging is always optimal (p*=0): {(valid == 0.0).sum():,}")
    print(f"  States where saving challenge is optimal (p*=1):   {(valid == 1.0).sum():,}")


if __name__ == "__main__":
    main()
