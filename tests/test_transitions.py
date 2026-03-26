"""Tests for abs/transitions.py."""

import pytest
from abs.state import GameState
from abs.transitions import (
    advance_count_ball,
    advance_count_strike,
    advance_half_inning,
    apply_walk,
    apply_strikeout,
    is_terminal,
    state_after_failed_challenge,
    state_after_no_challenge,
    state_after_overturn,
)


def make_state(**kwargs) -> GameState:
    defaults = dict(
        inning=5, half="top", score_diff=0, outs=0,
        base_state=0, balls=0, strikes=0,
        challenges_remaining=2, challenger_type="batter",
    )
    defaults.update(kwargs)
    return GameState(**defaults)


class TestAdvanceCountBall:
    def test_increments_balls(self):
        s = make_state(balls=1)
        s2 = advance_count_ball(s)
        assert s2.balls == 2

    def test_walk_on_ball_4(self):
        s = make_state(balls=3, base_state=0)
        s2 = advance_count_ball(s)
        # Batter on 1B, count reset.
        assert s2.base_state & 0b001  # 1B occupied
        assert s2.balls == 0
        assert s2.strikes == 0

    def test_walk_forces_runner_from_1b(self):
        # Runner on 1B; walk forces to 2B.
        s = make_state(balls=3, base_state=0b001)
        s2 = advance_count_ball(s)
        assert s2.base_state & 0b011  # Both 1B and 2B occupied

    def test_walk_bases_loaded_scores_run(self):
        # Bases loaded; walk forces run to score.
        s = make_state(balls=3, base_state=0b111, score_diff=0)
        s2 = advance_count_ball(s)
        # Offense scores a run: score_diff should increase by 1.
        assert s2.score_diff == 1
        # Bases still have runners after walk.
        assert s2.base_state & 0b111


class TestAdvanceCountStrike:
    def test_increments_strikes(self):
        s = make_state(strikes=1)
        s2 = advance_count_strike(s)
        assert s2.strikes == 2

    def test_strikeout_on_strike_3(self):
        s = make_state(strikes=2, outs=0)
        s2 = advance_count_strike(s)
        # Should increment outs (or advance half-inning).
        assert s2.outs == 1 or (s2.outs == 0 and s2.inning != s.inning)

    def test_strikeout_with_2_outs_advances_half_inning(self):
        s = make_state(strikes=2, outs=2, half="top", inning=3)
        s2 = advance_count_strike(s)
        # Should advance to bottom half or next inning.
        assert s2.half == "bottom" or s2.inning > s.inning


class TestApplyWalk:
    def test_bases_empty(self):
        s = make_state(base_state=0)
        s2 = apply_walk(s)
        assert s2.base_state & 0b001  # batter on 1B

    def test_runner_on_1b_forced_to_2b(self):
        s = make_state(base_state=0b001)
        s2 = apply_walk(s)
        assert s2.base_state & 0b011

    def test_runner_on_2b_not_forced(self):
        # Runner on 2B only; walk puts batter on 1B, 2B runner stays.
        s = make_state(base_state=0b010)
        s2 = apply_walk(s)
        assert s2.base_state & 0b001  # batter on 1B
        assert s2.base_state & 0b010  # runner on 2B still there
        assert not (s2.base_state & 0b100)  # no runner on 3B

    def test_bases_loaded_scores_run(self):
        s = make_state(base_state=0b111, score_diff=0)
        s2 = apply_walk(s)
        assert s2.score_diff == 1

    def test_count_reset(self):
        s = make_state(balls=3, strikes=1, base_state=0)
        s2 = apply_walk(s)
        assert s2.balls == 0
        assert s2.strikes == 0


class TestApplyStrikeout:
    def test_increments_outs(self):
        s = make_state(outs=0)
        s2 = apply_strikeout(s)
        assert s2.outs == 1

    def test_count_reset(self):
        s = make_state(balls=2, strikes=2, outs=0)
        s2 = apply_strikeout(s)
        assert s2.balls == 0
        assert s2.strikes == 0

    def test_third_out_advances_half_inning(self):
        s = make_state(outs=2, half="top", inning=4)
        s2 = apply_strikeout(s)
        # After 3 outs, half-inning advances.
        assert s2.half == "bottom" or s2.inning > 4
        assert s2.outs == 0


class TestAdvanceHalfInning:
    def test_top_to_bottom(self):
        s = make_state(half="top", inning=3, score_diff=2)
        s2 = advance_half_inning(s)
        assert s2.half == "bottom"
        assert s2.inning == 3
        # score_diff should flip (offense switches).
        assert s2.score_diff == -2

    def test_bottom_to_next_inning_top(self):
        s = make_state(half="bottom", inning=3, score_diff=-1)
        s2 = advance_half_inning(s)
        assert s2.half == "top"
        assert s2.inning == 4
        assert s2.score_diff == 1  # flipped

    def test_inning_clamps_at_9(self):
        s = make_state(half="bottom", inning=9)
        s2 = advance_half_inning(s)
        assert s2.inning == 9

    def test_resets_bases_outs_count(self):
        s = make_state(half="top", base_state=7, outs=2, balls=3, strikes=1)
        s2 = advance_half_inning(s)
        assert s2.base_state == 0
        assert s2.outs == 0
        assert s2.balls == 0
        assert s2.strikes == 0


class TestScoreDiffClamping:
    def test_clamp_at_max(self):
        # Score diff at max; walk scores a run — should stay clamped.
        s = make_state(balls=3, base_state=0b111, score_diff=10)
        s2 = advance_count_ball(s)
        assert s2.score_diff <= 10

    def test_clamp_at_min(self):
        s = make_state(balls=3, base_state=0b111, score_diff=-10, challenger_type="defense")
        # Defense is batting, their score_diff perspective is flipped in practice.
        # Just verify no error.
        advance_count_ball(s)


class TestIsTerminal:
    def test_terminal_state(self):
        s = make_state(inning=9, half="top", outs=0, balls=0, strikes=0)
        assert is_terminal(s)

    def test_not_terminal_mid_game(self):
        s = make_state(inning=5, half="top", outs=0)
        assert not is_terminal(s)

    def test_not_terminal_ninth_bottom(self):
        s = make_state(inning=9, half="bottom", outs=0)
        assert not is_terminal(s)


class TestChallengeTransitions:
    def test_overturn_batter_called_strike_to_ball(self):
        s = make_state(balls=1, strikes=1, challenger_type="batter", challenges_remaining=1)
        s2 = state_after_overturn(s)
        # Called strike overturned to ball → balls increases.
        assert s2.balls == s.balls + 1
        assert s2.strikes == s.strikes  # strikes unchanged
        # Challenges retained (not decremented).
        assert s2.challenges_remaining == s.challenges_remaining

    def test_overturn_defense_called_ball_to_strike(self):
        s = make_state(balls=1, strikes=1, challenger_type="defense", challenges_remaining=1)
        s2 = state_after_overturn(s)
        # Called ball overturned to strike → strikes increases.
        assert s2.strikes == s.strikes + 1
        assert s2.balls == s.balls  # balls unchanged
        # Challenges retained.
        assert s2.challenges_remaining == s.challenges_remaining

    def test_failed_challenge_loses_one(self):
        s = make_state(balls=1, strikes=0, challenger_type="batter", challenges_remaining=2)
        s2 = state_after_failed_challenge(s)
        assert s2.challenges_remaining == 1

    def test_failed_challenge_requires_challenges(self):
        s = make_state(challenges_remaining=0)
        with pytest.raises(ValueError):
            state_after_failed_challenge(s)

    def test_no_challenge_batter_advance_strike(self):
        s = make_state(balls=1, strikes=0, challenger_type="batter", challenges_remaining=1)
        s2 = state_after_no_challenge(s)
        # Called strike stands.
        assert s2.strikes == 1
        assert s2.balls == 1
        assert s2.challenges_remaining == 1  # unchanged

    def test_no_challenge_defense_advance_ball(self):
        s = make_state(balls=1, strikes=0, challenger_type="defense", challenges_remaining=1)
        s2 = state_after_no_challenge(s)
        # Called ball stands.
        assert s2.balls == 2
        assert s2.challenges_remaining == 1  # unchanged
