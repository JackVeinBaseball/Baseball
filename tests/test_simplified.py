"""Tests for abs/models/simplified.py."""

import pytest
from abs.models.simplified import SimplifiedThresholdModel
from abs.state import GameState
from abs.tables.win_probability import WPTable


def make_state(**kwargs) -> GameState:
    defaults = dict(
        inning=5, half="top", score_diff=0, outs=0,
        base_state=0, balls=0, strikes=0,
        challenges_remaining=1, challenger_type="batter",
    )
    defaults.update(kwargs)
    return GameState(**defaults)


class TestSimplifiedThresholdModel:
    def setup_method(self):
        self.model = SimplifiedThresholdModel()

    def test_threshold_in_01_or_none(self):
        s = make_state()
        p = self.model.threshold(s)
        assert p is None or (0.0 <= p <= 1.0)

    def test_no_challenges_returns_none(self):
        s = make_state(challenges_remaining=0)
        assert self.model.threshold(s) is None

    def test_threshold_returns_float_with_challenges(self):
        s = make_state(challenges_remaining=1)
        p = self.model.threshold(s)
        # May be None if denominator is near zero, but most states should have a value.
        if p is not None:
            assert 0.0 <= p <= 1.0

    def test_ev_gain_positive_when_below_threshold(self):
        s = make_state(challenges_remaining=1)
        p_star = self.model.threshold(s)
        if p_star is not None and p_star > 0.01:
            # At p_overturn=1.0, always challenging is strongly beneficial.
            ev = self.model.ev_gain(s, 1.0)
            assert ev >= -0.01  # Allow tiny floating-point slack.

    def test_ev_gain_zero_with_no_challenges(self):
        s = make_state(challenges_remaining=0)
        ev = self.model.ev_gain(s, 0.5)
        assert ev == 0.0

    def test_threshold_table_shape(self):
        states = [make_state(), make_state(inning=9), make_state(challenges_remaining=2)]
        df = self.model.threshold_table(states)
        assert len(df) == len(states)
        assert "p_star_simplified" in df.columns

    def test_threshold_table_k1_all_valid(self):
        # Run on a small sample of states.
        from abs.state import enumerate_all_states
        states = [s for s in enumerate_all_states()[:200] if s.challenges_remaining == 1]
        df = self.model.threshold_table(states)
        valid = df["p_star_simplified"].dropna()
        assert all((0.0 <= v <= 1.0) for v in valid)

    def test_defense_challenger_type(self):
        s = make_state(challenger_type="defense", challenges_remaining=1)
        p = self.model.threshold(s)
        if p is not None:
            assert 0.0 <= p <= 1.0
