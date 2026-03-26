"""
Overturn probability model: p̂(pitch).

Estimates the probability that a challenged pitch will be overturned by the
ABS system, given pitch-level features.

Two implementations are available:

1. **Parametric** (default): logistic model with hand-tuned coefficients.
   Works out of the box with no data.  Primary predictor is
   ``location_delta`` — signed distance from the ABS boundary in inches;
   negative values mean the pitch was clearly mis-called (higher overturn
   probability), positive values mean it was correctly called (lower prob).

2. **Logistic** (data-driven): scikit-learn LogisticRegression trained on
   user-supplied challenge records.

3. **XGB** (optional): XGBClassifier for higher accuracy with large datasets.
   Requires xgboost to be installed.

Decision rule
-------------
Challenge when p̂ > p*(s).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit

from abs.config import OVERTURN_COEFFS


# ── Feature engineering ───────────────────────────────────────────────────────


def _count_leverage(balls: int, strikes: int) -> float:
    """
    Proxy for count leverage: how much does this count matter?
    Higher on 3-2 and 0-2; lower on 0-0.
    Uses a simple heuristic: |balls - strikes| / 5 + full_count_bonus.
    """
    full_count = 1.0 if (balls == 3 and strikes == 2) else 0.0
    return abs(balls - strikes) / 5.0 + 0.2 * full_count


def _build_features(
    location_delta: float,
    called_type: str,
    balls: int,
    strikes: int,
    initiator_type: str,
) -> np.ndarray:
    """
    Build feature vector for the overturn model.

    Parameters
    ----------
    location_delta : float
        Signed distance from the ABS strike-zone boundary in inches.
        Negative = pitch is outside the zone when called a strike
        (or inside the zone when called a ball) — favorable to challenger.
        Positive = pitch appears correctly called — unfavorable to challenger.
    called_type : str
        "strike" (batter challenges) or "ball" (defense challenges).
    balls : int
        Balls before this pitch.
    strikes : int
        Strikes before this pitch.
    initiator_type : str
        "batter", "catcher", or "pitcher".

    Returns
    -------
    np.ndarray, shape (5,)
        [location_delta, is_called_strike, count_leverage, is_batter, 1]
    """
    is_cs = 1.0 if called_type == "strike" else 0.0
    clev = _count_leverage(balls, strikes)
    is_batter = 1.0 if initiator_type == "batter" else 0.0
    return np.array([location_delta, is_cs, clev, is_batter, 1.0], dtype=np.float64)


# ── Model class ───────────────────────────────────────────────────────────────


class OverturnModel:
    """
    Pitch-level overturn probability estimator.

    Parameters
    ----------
    method : str
        "parametric" (default), "logistic", or "xgb".
    """

    def __init__(self, method: str = "parametric") -> None:
        if method not in {"parametric", "logistic", "xgb"}:
            raise ValueError(f"method must be 'parametric', 'logistic', or 'xgb'; got {method!r}")
        self.method = method
        self._coeffs: np.ndarray = self._default_parametric_coeffs()
        self._sklearn_model = None
        self._fitted = False

    # ── Parametric model ──────────────────────────────────────────────────────

    def _default_parametric_coeffs(self) -> np.ndarray:
        """
        Default coefficients for the parametric model.
        Feature order: [location_delta, is_called_strike, count_leverage,
                         is_batter, intercept].
        """
        c = OVERTURN_COEFFS
        return np.array([
            c["location_delta"],
            c["is_called_strike"],
            c["count_leverage"],
            0.0,                 # is_batter (symmetric by default)
            c["intercept"],
        ], dtype=np.float64)

    def _parametric_predict(self, X: np.ndarray) -> np.ndarray:
        """X shape: (N, 5). Returns probabilities shape (N,)."""
        return expit(X @ self._coeffs)

    # ── Data-driven fitting ───────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "OverturnModel":
        """
        Fit the model to labeled challenge data.

        Parameters
        ----------
        df : pd.DataFrame
            Required columns: location_delta, called_type, balls, strikes,
            initiator_type, overturned (bool/int: 1 = overturned).

        Returns self for chaining.
        """
        required = {"location_delta", "called_type", "balls", "strikes",
                    "initiator_type", "overturned"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        X = np.vstack([
            _build_features(
                row.location_delta, row.called_type,
                row.balls, row.strikes, row.initiator_type,
            )
            for row in df.itertuples(index=False)
        ])
        y = df["overturned"].astype(int).values

        if self.method == "logistic":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=500)
            clf.fit(X[:, :-1], y)  # sklearn adds its own intercept
            # Rebuild coeffs array to match our manual dot-product format.
            self._coeffs = np.append(clf.coef_[0], clf.intercept_[0])
            self._sklearn_model = clf
        elif self.method == "xgb":
            try:
                from xgboost import XGBClassifier
            except ImportError:
                raise ImportError("xgboost is required for method='xgb'. Install with: pip install xgboost")
            clf = XGBClassifier(n_estimators=100, max_depth=4, eval_metric="logloss", random_state=42)
            clf.fit(X, y)
            self._sklearn_model = clf
        else:
            # Parametric: re-fit via logistic regression on the features.
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=500, fit_intercept=True)
            clf.fit(X[:, :-1], y)
            self._coeffs = np.append(clf.coef_[0], clf.intercept_[0])

        self._fitted = True
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Batch inference.

        Parameters
        ----------
        features_df : pd.DataFrame
            Columns: location_delta, called_type, balls, strikes, initiator_type.

        Returns
        -------
        np.ndarray, shape (N,)
            Estimated overturn probabilities.
        """
        X = np.vstack([
            _build_features(
                row.location_delta, row.called_type,
                row.balls, row.strikes, row.initiator_type,
            )
            for row in features_df.itertuples(index=False)
        ])

        if self.method == "xgb" and self._sklearn_model is not None:
            return self._sklearn_model.predict_proba(X)[:, 1]
        return self._parametric_predict(X)

    def predict_single(
        self,
        location_delta: float,
        called_type: str,
        balls: int,
        strikes: int,
        initiator_type: str,
    ) -> float:
        """
        Single-pitch inference.

        Parameters
        ----------
        location_delta : float
            Signed inches from ABS boundary (negative = favorable to challenger).
        called_type : str
            "strike" or "ball".
        balls : int
            Ball count (0–3).
        strikes : int
            Strike count (0–2).
        initiator_type : str
            "batter", "catcher", or "pitcher".

        Returns
        -------
        float in [0, 1]
        """
        x = _build_features(location_delta, called_type, balls, strikes, initiator_type)
        if self.method == "xgb" and self._sklearn_model is not None:
            return float(self._sklearn_model.predict_proba(x[np.newaxis, :])[:, 1][0])
        return float(self._parametric_predict(x[np.newaxis, :])[0])

    def location_curve(
        self,
        called_type: str = "strike",
        balls: int = 0,
        strikes: int = 0,
        initiator_type: str = "batter",
        delta_range: tuple[float, float] = (-4.0, 4.0),
        n_points: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (location_deltas, overturn_probs) curve for plotting.

        Useful for overlaying with p* threshold lines.
        """
        deltas = np.linspace(delta_range[0], delta_range[1], n_points)
        probs = np.array([
            self.predict_single(d, called_type, balls, strikes, initiator_type)
            for d in deltas
        ])
        return deltas, probs

    def optimal_cutoff(
        self,
        p_star: float,
        called_type: str = "strike",
        balls: int = 0,
        strikes: int = 0,
        initiator_type: str = "batter",
    ) -> Optional[float]:
        """
        Find the location_delta cutoff where p̂ = p*.

        Pitches with location_delta < cutoff should be challenged (p̂ > p*).
        Returns None if no crossing exists.
        """
        deltas, probs = self.location_curve(called_type, balls, strikes, initiator_type)
        crossings = np.where(np.diff(np.sign(probs - p_star)))[0]
        if len(crossings) == 0:
            return None
        # Return the first (most negative) crossing — the boundary of the
        # "challenge zone".
        idx = crossings[0]
        # Linear interpolation.
        d0, d1 = deltas[idx], deltas[idx + 1]
        p0, p1 = probs[idx], probs[idx + 1]
        if abs(p1 - p0) < 1e-12:
            return float(d0)
        return float(d0 + (p_star - p0) * (d1 - d0) / (p1 - p0))
