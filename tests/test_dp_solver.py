"""Tests for abs/models/dp_solver.py."""

import pytest
import numpy as np
from abs.models.dp_solver import DPSolver
from abs.state import GameState
from abs.transitions import is_terminal


def make_state(**kwargs) -> GameState:
    defaults = dict(
        inning=5, half="top", score_diff=0, outs=0,
        base_state=0, balls=0, strikes=0,
        challenges_remaining=1, challenger_type="batter",
    )
    defaults.update(kwargs)
    return GameState(**defaults)


@pytest.fixture(scope="module")
def solved_solver():
    """Shared solver instance (expensive to create)."""
    solver = DPSolver()
    solver.solve(verbose=False)
    return solver


class TestDPSolverBasics:
    def test_raises_before_solve(self):
        solver = DPSolver()
        s = make_state()
        with pytest.raises(RuntimeError):
            solver.threshold(s)

    def test_solve_runs(self):
        solver = DPSolver()
        solver.solve(verbose=False)
        assert solver._solved


class TestTerminalValues:
    def test_terminal_state_has_value(self, solved_solver):
        # Terminal state: start of 9th inning top, winning.
        s = make_state(inning=9, half="top", score_diff=2, outs=0, balls=0, strikes=0,
                       challenges_remaining=1)
        v = solved_solver.get_value(s)
        assert 0.0 <= v <= 1.0

    def test_terminal_values_all_k(self, solved_solver):
        # Terminal state values should not vary by k (challenges don't help after game).
        s_traj = make_state(inning=9, half="top", score_diff=3, outs=0, balls=0, strikes=0)
        from dataclasses import replace
        v0 = solved_solver.get_value(replace(s_traj, challenges_remaining=0))
        v1 = solved_solver.get_value(replace(s_traj, challenges_remaining=1))
        v2 = solved_solver.get_value(replace(s_traj, challenges_remaining=2))
        # All should be equal (or very close) at terminal states.
        assert abs(v0 - v1) < 0.05
        assert abs(v1 - v2) < 0.05


class TestValueMonotonicity:
    def test_more_challenges_weakly_better(self, solved_solver):
        """V(s, k=1) >= V(s, k=0) for all states."""
        from dataclasses import replace
        from abs.state import enumerate_all_states
        states = enumerate_all_states()[:300]
        violations = 0
        for s in states:
            s0 = replace(s, challenges_remaining=0)
            s1 = replace(s, challenges_remaining=1)
            v0 = solved_solver.get_value(s0)
            v1 = solved_solver.get_value(s1)
            if v1 < v0 - 0.01:
                violations += 1
        assert violations == 0, f"{violations} states violate V(k=1)>=V(k=0)"

    def test_two_challenges_better_than_one(self, solved_solver):
        from dataclasses import replace
        from abs.state import enumerate_all_states
        states = enumerate_all_states()[:300]
        violations = 0
        for s in states:
            s1 = replace(s, challenges_remaining=1)
            s2 = replace(s, challenges_remaining=2)
            v1 = solved_solver.get_value(s1)
            v2 = solved_solver.get_value(s2)
            if v2 < v1 - 0.01:
                violations += 1
        assert violations == 0, f"{violations} states violate V(k=2)>=V(k=1)"


class TestThresholds:
    def test_no_challenges_returns_none(self, solved_solver):
        s = make_state(challenges_remaining=0)
        assert solved_solver.threshold(s) is None

    def test_threshold_in_01(self, solved_solver):
        from dataclasses import replace
        from abs.state import enumerate_all_states
        states = [s for s in enumerate_all_states()[:500] if s.challenges_remaining >= 1]
        for s in states:
            p = solved_solver.threshold(s)
            if p is not None:
                assert 0.0 <= p <= 1.0, f"p={p} out of [0,1] for state {s}"

    def test_regression_ninth_inning_tie_full_count(self, solved_solver):
        """9th inning, tie game, full count, 2 outs, bases loaded, k=1 should have low p*."""
        s = make_state(
            inning=9, half="bottom", score_diff=0, outs=2,
            base_state=0b111, balls=3, strikes=2,
            challenges_remaining=1, challenger_type="batter",
        )
        p = solved_solver.threshold(s)
        # Extremely high leverage: should have low threshold (challenge aggressively).
        if p is not None:
            assert p < 0.9, f"Expected low p* in high leverage, got {p:.3f}"

    def test_threshold_table_columns(self, solved_solver):
        df = solved_solver.threshold_table()
        assert "p_star_dp" in df.columns
        assert "challenges_remaining" in df.columns
        assert "challenger_type" in df.columns

    def test_threshold_table_has_rows(self, solved_solver):
        df = solved_solver.threshold_table()
        assert len(df) > 0
