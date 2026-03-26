"""
Win Probability (WP) table and logistic fallback model.

Resolution order when looking up WP for a state:
  1. Exact match in loaded CSV table (fast dict lookup).
  2. LogisticWPModel (always available; used for states absent from table,
     including extra innings and extreme score differentials).

All values are from the *home team's perspective* (WP = P(home wins)).
The DP and analysis layers work in the *offensive team's perspective*;
conversion is handled by wp_for_state().
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid

from abs.config import INNINGS_MAX, LOGISTIC_WP_COEFFS, SCORE_DIFF_CLAMP


# ── Logistic WP model ─────────────────────────────────────────────────────────


class LogisticWPModel:
    """
    Parametric logistic win-probability model.

    Features
    --------
    - score_diff     : run differential (clamped, from home team POV)
    - score_diff_sq  : captures non-linearity (diminishing returns of runs)
    - inning_urgency : max(0, inning - 6) / 3  (ramps up in innings 7–9)
    - is_bottom      : 1 if home team is batting, 0 if away
    - outs           : number of outs
    - runners        : number of runners on base (popcount of base_state)
    - score_x_inning : interaction: score_diff × inning_urgency

    Coefficients default to LOGISTIC_WP_COEFFS from config; can be updated
    by calling fit() against a WP DataFrame.
    """

    def __init__(self, coeffs: Optional[dict[str, float]] = None) -> None:
        self.coeffs = dict(LOGISTIC_WP_COEFFS if coeffs is None else coeffs)

    def _features(
        self,
        inning: int,
        half: str,
        score_diff_home: int,
        outs: int,
        base_state: int,
    ) -> float:
        """Return log-odds (pre-sigmoid) for the given state."""
        c = self.coeffs
        inning_urgency = max(0, inning - 6) / 3.0
        is_bottom = 1.0 if half == "bottom" else 0.0
        runners = bin(base_state).count("1")
        return (
            c["intercept"]
            + c["score_diff"] * score_diff_home
            + c["score_diff_sq"] * score_diff_home ** 2
            + c["inning_urgency"] * inning_urgency
            + c["is_bottom"] * is_bottom
            + c["outs"] * outs
            + c["runners"] * runners
            + c["score_x_inning"] * score_diff_home * inning_urgency
        )

    def predict(
        self,
        inning: int,
        half: str,
        score_diff_home: int,
        outs: int,
        base_state: int,
    ) -> float:
        """Return P(home wins) ∈ [0, 1]."""
        return float(expit(self._features(inning, half, score_diff_home, outs, base_state)))

    def fit(self, df: pd.DataFrame) -> "LogisticWPModel":
        """
        Fit coefficients to a WP DataFrame.

        Expected columns: inning, half, score_diff (home POV), outs,
        base_state, wp (home win probability).

        Uses scikit-learn LogisticRegression under the hood.
        """
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            raise ImportError("scikit-learn is required for LogisticWPModel.fit().")

        X, y = [], []
        for row in df.itertuples(index=False):
            inning = min(int(row.inning), INNINGS_MAX)
            inning_urgency = max(0, inning - 6) / 3.0
            is_bottom = 1.0 if row.half == "bottom" else 0.0
            runners = bin(int(row.base_state)).count("1")
            sd = int(row.score_diff)
            X.append([
                sd,
                sd ** 2,
                inning_urgency,
                is_bottom,
                int(row.outs),
                runners,
                sd * inning_urgency,
            ])
            y.append(float(row.wp) >= 0.5)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_scaled, y)

        names = ["score_diff", "score_diff_sq", "inning_urgency",
                 "is_bottom", "outs", "runners", "score_x_inning"]
        self.coeffs["intercept"] = float(clf.intercept_[0])
        for name, coef in zip(names, clf.coef_[0]):
            self.coeffs[name] = float(coef)
        self._scaler = scaler
        return self


# ── WP Table ──────────────────────────────────────────────────────────────────


class WPTable:
    """
    Win probability lookup table with logistic fallback.

    Parameters
    ----------
    csv_path : Path, optional
        Path to a Tango/MGL-style WP CSV with columns:
        ``inning, half, score_diff, outs, base_state, wp``.
        score_diff is from the *home team's perspective*:
        positive = home leading.
    """

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        self._index: dict[tuple, float] = {}
        self._fallback = LogisticWPModel()

        # Try to load the bundled table automatically.
        bundled = Path(__file__).parent / "data" / "wp_tango.csv"
        if csv_path is not None:
            self.load(csv_path)
        elif bundled.exists():
            self.load(bundled)

    def load(self, csv_path: Path) -> None:
        """Load a WP CSV and index it for O(1) lookup."""
        df = pd.read_csv(csv_path)
        required = {"inning", "half", "score_diff", "outs", "base_state", "wp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"WP CSV missing columns: {missing}")

        df["score_diff"] = df["score_diff"].clip(-SCORE_DIFF_CLAMP, SCORE_DIFF_CLAMP).astype(int)
        df["inning"] = df["inning"].clip(1, INNINGS_MAX).astype(int)

        self._index = {
            (int(r.inning), str(r.half), int(r.score_diff), int(r.outs), int(r.base_state)): float(r.wp)
            for r in df.itertuples(index=False)
        }

    def get(
        self,
        inning: int,
        half: str,
        score_diff_home: int,
        outs: int,
        base_state: int,
    ) -> float:
        """
        Return P(home wins) for the given state.

        Falls back to LogisticWPModel if the exact state is not in the table.
        """
        key = (
            min(inning, INNINGS_MAX),
            half,
            max(-SCORE_DIFF_CLAMP, min(SCORE_DIFF_CLAMP, score_diff_home)),
            outs,
            base_state,
        )
        if key in self._index:
            return self._index[key]
        return self._fallback.predict(inning, half, score_diff_home, outs, base_state)

    def wp_for_state(self, s: "GameState") -> float:  # noqa: F821
        """
        Return WP from the *offensive team's perspective*.

        The WP table stores values from the home team's POV.  When the away
        team is batting (half="top"), the offensive team's WP = 1 − WP(home).
        When the home team is batting (half="bottom"), offensive WP = WP(home).

        score_diff in GameState is from the offensive team's POV (positive =
        offense leads).  We convert to home POV before lookup:
        - top (away bats): home score_diff = −s.score_diff
        - bottom (home bats): home score_diff = +s.score_diff
        """
        if s.half == "top":
            sd_home = -s.score_diff
            wp_home = self.get(s.inning, s.half, sd_home, s.outs, s.base_state)
            return 1.0 - wp_home   # away team wins = 1 − P(home wins)
        else:
            sd_home = s.score_diff
            return self.get(s.inning, s.half, sd_home, s.outs, s.base_state)

    def fit_fallback(self, df: pd.DataFrame) -> None:
        """Fit the logistic fallback model to a WP DataFrame."""
        self._fallback.fit(df)
