"""
Fit and evaluate the overturn probability model (p̂).

Usage
-----
    python scripts/fit_overturn_model.py [--data path/to/challenges.csv]
                                         [--method parametric|logistic|xgb]
                                         [--output results/overturn_model_eval.csv]

If no --data is provided, runs the parametric model with default coefficients
and prints a calibration curve summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from abs.models.overturn_model import OverturnModel


def demo_parametric_model() -> None:
    """Print overturn probability by location_delta for the default model."""
    model = OverturnModel(method="parametric")
    print("\nParametric overturn model — P(overturn) by distance from ABS boundary:")
    print(f"{'delta (in)':>12}  {'called_strike':>14}  {'called_ball':>12}")
    print("-" * 42)
    for delta in [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]:
        p_cs = model.predict_single(delta, "strike", 0, 0, "batter")
        p_cb = model.predict_single(delta, "ball", 0, 0, "catcher")
        print(f"{delta:>12.1f}  {p_cs:>14.4f}  {p_cb:>12.4f}")

    print("\nOptimal cutoffs for selected p* values:")
    print(f"{'p*':>6}  {'CS cutoff (in)':>16}  {'CB cutoff (in)':>16}")
    print("-" * 42)
    for p_star in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        cut_cs = model.optimal_cutoff(p_star, "strike", 0, 0, "batter")
        cut_cb = model.optimal_cutoff(p_star, "ball", 0, 0, "catcher")
        cs_str = f"{cut_cs:.3f}" if cut_cs is not None else "N/A"
        cb_str = f"{cut_cb:.3f}" if cut_cb is not None else "N/A"
        print(f"{p_star:>6.2f}  {cs_str:>16}  {cb_str:>16}")


def fit_from_data(data_path: Path, method: str, output_path: Path) -> None:
    """Fit model to labeled data and evaluate."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} challenge records from {data_path}")

    model = OverturnModel(method=method)
    model.fit(df)
    print(f"Fitted {method} model.")

    # Evaluate on full dataset (for calibration; use cross-val for production).
    proba = model.predict_proba(df)
    y_true = df["overturned"].astype(int).values

    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    print(f"\nModel evaluation (in-sample):")
    print(f"  AUC:         {roc_auc_score(y_true, proba):.4f}")
    print(f"  Log loss:    {log_loss(y_true, proba):.4f}")
    print(f"  Brier score: {brier_score_loss(y_true, proba):.4f}")
    print(f"  Overturn rate: {y_true.mean():.4f}")
    print(f"  Mean p̂:        {proba.mean():.4f}")

    df["p_hat"] = proba
    df.to_csv(output_path, index=False)
    print(f"\nSaved predictions → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit/evaluate overturn probability model.")
    parser.add_argument("--data", default=None,
                        help="Path to labeled challenge CSV.")
    parser.add_argument("--method", default="parametric",
                        choices=["parametric", "logistic", "xgb"],
                        help="Model method.")
    parser.add_argument("--output", default="results/overturn_model_eval.csv")
    args = parser.parse_args()

    if args.data is None:
        print("No data provided — running parametric model demo.")
        demo_parametric_model()
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fit_from_data(Path(args.data), args.method, output_path)


if __name__ == "__main__":
    main()
