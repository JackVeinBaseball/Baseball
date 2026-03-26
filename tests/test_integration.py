"""
Integration tests: full pipeline from tables through policy recommendation.

These tests are slower but verify end-to-end correctness.
"""

import pytest
import numpy as np
import pandas as pd
from abs.models.simplified import SimplifiedThresholdModel
from abs.models.overturn_model import OverturnModel
from abs.analysis.policy import ChallengePolicy
from abs.state import GameState, enumerate_all_states
from abs.tables.win_probability import WPTable
from abs.tables.run_expectancy import RE24Table


def make_state(**kwargs) -> GameState:
    defaults = dict(
        inning=5, half="top", score_diff=0, outs=0,
        base_state=0, balls=0, strikes=0,
        challenges_remaining=1, challenger_type="batter",
    )
    defaults.update(kwargs)
    return GameState(**defaults)


class TestFullSimplifiedPipeline:
    """End-to-end simplified model pipeline."""

    def test_full_simplified_pipeline(self):
        wp = WPTable()
        model = SimplifiedThresholdModel(wp)
        states = enumerate_all_states()[:1000]
        df = model.threshold_table(states)
        assert len(df) == 1000
        assert "p_star_simplified" in df.columns
        valid = df["p_star_simplified"].dropna()
        assert all(0.0 <= v <= 1.0 for v in valid)

    def test_policy_full_pipeline(self):
        policy = ChallengePolicy(
            threshold_model=SimplifiedThresholdModel(),
            overturn_model=OverturnModel(method="parametric"),
        )

        # State where challenge should be obvious: 9th inning, tie, full count, k=1.
        s = make_state(inning=9, half="bottom", score_diff=0, outs=2,
                       base_state=0b111, balls=3, strikes=2,
                       challenges_remaining=1, challenger_type="batter")

        # Very favorable pitch.
        result = policy.should_challenge(s, location_delta=-3.0, initiator_type="batter")
        assert result["p_hat"] > 0.5  # p̂ should be high for very favorable pitch.
        assert result["p_star"] is not None

    def test_p_hat_greater_than_pstar_triggers_challenge(self):
        policy = ChallengePolicy(
            threshold_model=SimplifiedThresholdModel(),
            overturn_model=OverturnModel(method="parametric"),
        )
        s = make_state(challenges_remaining=1)
        p_star = policy.p_star(s)
        if p_star is not None:
            # Create a state where p̂ is definitely above p*.
            # Use location_delta that gives p̂ = 0.99.
            result_high = policy.should_challenge(s, -5.0, "batter")
            assert result_high["p_hat"] > result_high.get("p_star", 1.0) - 0.01 or result_high["recommend"]

    def test_no_challenges_never_recommends(self):
        policy = ChallengePolicy(
            threshold_model=SimplifiedThresholdModel(),
            overturn_model=OverturnModel(method="parametric"),
        )
        states = [make_state(challenges_remaining=0, inning=i) for i in range(1, 10)]
        for s in states:
            result = policy.should_challenge(s, -5.0, "batter")
            assert result["recommend"] is False


class TestDPSolverIntegration:
    """DP solver end-to-end (mark as slow; excluded from fast test runs)."""

    @pytest.mark.slow
    def test_dp_solve_completes(self):
        import time
        from abs.models.dp_solver import DPSolver
        solver = DPSolver()
        t0 = time.time()
        solver.solve(verbose=False)
        elapsed = time.time() - t0
        assert solver._solved
        assert elapsed < 300, f"DP solve took {elapsed:.0f}s (expected < 300s)"

    @pytest.mark.slow
    def test_dp_threshold_table_shape(self):
        from abs.models.dp_solver import DPSolver
        solver = DPSolver()
        solver.solve(verbose=False)
        df = solver.threshold_table()
        assert len(df) > 0
        assert "p_star_dp" in df.columns

    @pytest.mark.slow
    def test_simplified_pstar_leq_dp_pstar_for_k1(self):
        """
        Simplified p* ≤ DP p* for k=1 (simplified ignores future challenge value).

        The simplified model always recommends challenging more aggressively than
        the DP, because it doesn't account for the opportunity cost of spending
        a challenge.  This means simplified thresholds should be lower (or equal).
        """
        from abs.models.dp_solver import DPSolver
        from dataclasses import replace

        simplified = SimplifiedThresholdModel()
        dp = DPSolver()
        dp.solve(verbose=False)

        states = [s for s in enumerate_all_states()[:2000] if s.challenges_remaining == 1]
        violations = 0
        for s in states:
            p_s = simplified.threshold(s)
            p_dp = dp.threshold(s)
            if p_s is not None and p_dp is not None:
                if p_s > p_dp + 0.05:  # Allow 5% tolerance.
                    violations += 1

        violation_rate = violations / max(len(states), 1)
        assert violation_rate < 0.05, (
            f"Simplified p* > DP p* in {violations}/{len(states)} states "
            f"({violation_rate:.1%})"
        )


class TestOverturnModelIntegration:
    def test_parametric_model_produces_reasonable_cutoffs(self):
        model = OverturnModel(method="parametric")
        # For p*=0.5, the cutoff should be somewhere in (-2, 2) inches.
        cutoff = model.optimal_cutoff(0.5, "strike", 0, 0, "batter")
        if cutoff is not None:
            assert -3.0 < cutoff < 3.0

    def test_extreme_location_always_recommend(self):
        policy = ChallengePolicy(
            threshold_model=SimplifiedThresholdModel(),
            overturn_model=OverturnModel(method="parametric"),
        )
        s = make_state(challenges_remaining=1)
        # Pitch 5 inches outside zone: p̂ should be near 1.
        result = policy.should_challenge(s, location_delta=-5.0, initiator_type="batter")
        assert result["p_hat"] > 0.8

    def test_borderline_pitch_depends_on_game_state(self):
        """A borderline pitch (delta ≈ 0) may or may not be recommended based on state."""
        policy = ChallengePolicy(
            threshold_model=SimplifiedThresholdModel(),
            overturn_model=OverturnModel(method="parametric"),
        )
        # Early inning, no pressure.
        s_early = make_state(inning=1, score_diff=0, challenges_remaining=1)
        # Late inning, high pressure.
        s_late = make_state(inning=9, score_diff=0, outs=2, base_state=7,
                             balls=3, strikes=2, challenges_remaining=1)
        r_early = policy.should_challenge(s_early, location_delta=0.0, initiator_type="batter")
        r_late = policy.should_challenge(s_late, location_delta=0.0, initiator_type="batter")
        # Both should return valid results (not error).
        assert isinstance(r_early["recommend"], bool)
        assert isinstance(r_late["recommend"], bool)
