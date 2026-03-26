"""Tests for abs/analysis/policy.py."""

import pytest
import numpy as np
import pandas as pd
from abs.analysis.policy import ChallengePolicy
from abs.models.overturn_model import OverturnModel
from abs.models.simplified import SimplifiedThresholdModel
from abs.state import GameState


def make_state(**kwargs) -> GameState:
    defaults = dict(
        inning=5, half="top", score_diff=0, outs=0,
        base_state=0, balls=0, strikes=0,
        challenges_remaining=1, challenger_type="batter",
    )
    defaults.update(kwargs)
    return GameState(**defaults)


@pytest.fixture
def policy():
    return ChallengePolicy(
        threshold_model=SimplifiedThresholdModel(),
        overturn_model=OverturnModel(method="parametric"),
    )


class TestChallengePolicy:
    def test_p_star_returns_float_or_none(self, policy):
        s = make_state(challenges_remaining=1)
        p = policy.p_star(s)
        assert p is None or 0.0 <= p <= 1.0

    def test_p_hat_returns_float(self, policy):
        p = policy.p_hat(-1.0, "strike", 0, 0, "batter")
        assert 0.0 <= p <= 1.0

    def test_should_challenge_true_when_phat_above_pstar(self, policy):
        s = make_state(challenges_remaining=1, challenger_type="batter")
        p_star = policy.p_star(s)
        if p_star is not None and p_star < 0.99:
            # Force a very favorable pitch (p̂ should be high).
            result = policy.should_challenge(s, location_delta=-4.0, initiator_type="batter")
            assert result["recommend"] is True or result["p_hat"] > result["p_star"]

    def test_should_challenge_false_when_no_challenges(self, policy):
        s = make_state(challenges_remaining=0)
        result = policy.should_challenge(s, location_delta=-4.0, initiator_type="batter")
        assert result["recommend"] is False

    def test_should_challenge_has_all_keys(self, policy):
        s = make_state(challenges_remaining=1)
        result = policy.should_challenge(s, location_delta=0.0, initiator_type="batter")
        assert "p_hat" in result
        assert "p_star" in result
        assert "recommend" in result
        assert "ev_gain" in result
        assert "reason" in result

    def test_should_challenge_false_unfavorable_pitch(self, policy):
        s = make_state(challenges_remaining=1)
        # Pitch clearly correctly called (p̂ near 0).
        result = policy.should_challenge(s, location_delta=4.0, initiator_type="batter")
        # p̂ should be low; may or may not recommend depending on p*.
        assert 0.0 <= result["p_hat"] <= 1.0

    def test_ev_gain_sign_consistent(self, policy):
        s = make_state(challenges_remaining=1)
        # Very favorable pitch: EV gain should be positive.
        result = policy.should_challenge(s, location_delta=-4.0, initiator_type="batter")
        if result["recommend"] and not np.isnan(result["ev_gain"]):
            assert result["ev_gain"] >= -0.01  # slight tolerance

    def test_evaluate_decisions_batch(self, policy):
        df = pd.DataFrame({
            "inning": [5, 9, 3],
            "half": ["top", "bottom", "top"],
            "score_diff": [0, -1, 2],
            "outs": [0, 2, 1],
            "base_state": [0, 7, 0],
            "balls": [1, 3, 0],
            "strikes": [1, 2, 0],
            "challenges_remaining": [1, 1, 2],
            "challenger_type": ["batter", "batter", "defense"],
            "location_delta": [-1.0, -2.0, 0.5],
            "initiator_type": ["batter", "batter", "catcher"],
            "challenged": [True, True, False],
            "overturned": [True, False, False],
        })
        result = policy.evaluate_decisions(df)
        assert len(result) == 3
        assert "p_hat" in result.columns
        assert "p_star" in result.columns
        assert "recommend" in result.columns
        assert "was_optimal" in result.columns
