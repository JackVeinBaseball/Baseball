"""Tests for abs/models/overturn_model.py."""

import pytest
import numpy as np
import pandas as pd
from abs.models.overturn_model import OverturnModel


class TestOverturnModelParametric:
    def setup_method(self):
        self.model = OverturnModel(method="parametric")

    def test_predict_single_in_01(self):
        p = self.model.predict_single(-1.0, "strike", 0, 0, "batter")
        assert 0.0 <= p <= 1.0

    def test_favorable_location_high_prob(self):
        # Very favorable to challenger (pitch clearly mis-called).
        p = self.model.predict_single(-3.0, "strike", 0, 0, "batter")
        assert p > 0.5

    def test_unfavorable_location_low_prob(self):
        # Pitch clearly correctly called.
        p = self.model.predict_single(3.0, "strike", 0, 0, "batter")
        assert p < 0.5

    def test_monotone_in_location_delta(self):
        # More negative delta → higher overturn prob (monotone decreasing in delta).
        deltas = np.linspace(-3.0, 3.0, 20)
        probs = [self.model.predict_single(d, "strike", 0, 0, "batter") for d in deltas]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1] - 0.01, (
                f"Non-monotone at delta={deltas[i]:.2f}: {probs[i]:.4f} > {probs[i+1]:.4f}"
            )

    def test_predict_proba_batch(self):
        df = pd.DataFrame({
            "location_delta": [-1.0, 0.0, 1.0],
            "called_type": ["strike", "strike", "ball"],
            "balls": [0, 1, 2],
            "strikes": [0, 1, 2],
            "initiator_type": ["batter", "batter", "catcher"],
        })
        probs = self.model.predict_proba(df)
        assert len(probs) == 3
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_location_curve_shape(self):
        deltas, probs = self.model.location_curve("strike", 0, 0, "batter")
        assert len(deltas) == len(probs)
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_optimal_cutoff_returns_float(self):
        cutoff = self.model.optimal_cutoff(0.3, "strike", 0, 0, "batter")
        # Should find a crossing.
        if cutoff is not None:
            assert isinstance(cutoff, float)

    def test_works_without_fit(self):
        # Parametric model should work out-of-the-box without fit().
        model = OverturnModel(method="parametric")
        p = model.predict_single(0.0, "strike", 1, 1, "batter")
        assert 0.0 <= p <= 1.0


class TestOverturnModelLogistic:
    def test_invalid_method(self):
        with pytest.raises(ValueError):
            OverturnModel(method="invalid")

    def test_fit_returns_self(self):
        model = OverturnModel(method="logistic")
        df = pd.DataFrame({
            "location_delta": np.random.uniform(-3, 3, 100),
            "called_type": np.random.choice(["strike", "ball"], 100),
            "balls": np.random.randint(0, 4, 100),
            "strikes": np.random.randint(0, 3, 100),
            "initiator_type": np.random.choice(["batter", "catcher"], 100),
            "overturned": np.random.randint(0, 2, 100),
        })
        result = model.fit(df)
        assert result is model

    def test_fit_missing_column_raises(self):
        model = OverturnModel(method="logistic")
        df = pd.DataFrame({"location_delta": [0.0]})
        with pytest.raises(ValueError):
            model.fit(df)

    def test_fit_predict_in_01(self):
        model = OverturnModel(method="logistic")
        np.random.seed(0)
        n = 100
        df = pd.DataFrame({
            "location_delta": np.random.uniform(-3, 3, n),
            "called_type": np.random.choice(["strike", "ball"], n),
            "balls": np.random.randint(0, 4, n),
            "strikes": np.random.randint(0, 3, n),
            "initiator_type": np.random.choice(["batter", "catcher"], n),
            "overturned": np.random.randint(0, 2, n),
        })
        model.fit(df)
        probs = model.predict_proba(df)
        assert all(0.0 <= p <= 1.0 for p in probs)
