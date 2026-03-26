"""Tests for abs/state.py."""

import pytest
from abs.state import GameState, enumerate_all_states, build_index


def make_state(**kwargs) -> GameState:
    defaults = dict(
        inning=1, half="top", score_diff=0, outs=0,
        base_state=0, balls=0, strikes=0,
        challenges_remaining=1, challenger_type="batter",
    )
    defaults.update(kwargs)
    return GameState(**defaults)


class TestGameStateConstruction:
    def test_valid_construction(self):
        s = make_state()
        assert s.inning == 1
        assert s.half == "top"

    def test_all_valid_ranges(self):
        # Boundary values should not raise.
        make_state(inning=9, score_diff=10, outs=2, base_state=7, balls=3, strikes=2, challenges_remaining=2)
        make_state(inning=1, score_diff=-10, outs=0, base_state=0, balls=0, strikes=0, challenges_remaining=0)

    def test_invalid_inning_low(self):
        with pytest.raises(ValueError):
            make_state(inning=0)

    def test_invalid_inning_high(self):
        with pytest.raises(ValueError):
            make_state(inning=10)

    def test_invalid_half(self):
        with pytest.raises(ValueError):
            make_state(half="middle")

    def test_invalid_score_diff_high(self):
        with pytest.raises(ValueError):
            make_state(score_diff=11)

    def test_invalid_score_diff_low(self):
        with pytest.raises(ValueError):
            make_state(score_diff=-11)

    def test_invalid_outs(self):
        with pytest.raises(ValueError):
            make_state(outs=3)

    def test_invalid_base_state(self):
        with pytest.raises(ValueError):
            make_state(base_state=8)

    def test_invalid_balls(self):
        with pytest.raises(ValueError):
            make_state(balls=4)

    def test_invalid_strikes(self):
        with pytest.raises(ValueError):
            make_state(strikes=3)

    def test_invalid_challenges(self):
        with pytest.raises(ValueError):
            make_state(challenges_remaining=3)

    def test_invalid_challenger_type(self):
        with pytest.raises(ValueError):
            make_state(challenger_type="umpire")

    def test_defense_challenger_type(self):
        s = make_state(challenger_type="defense")
        assert s.challenger_type == "defense"


class TestGameStateProperties:
    def test_base_out_index(self):
        s = make_state(outs=1, base_state=3)
        assert s.base_out_index == 1 * 8 + 3

    def test_base_out_index_zero(self):
        s = make_state(outs=0, base_state=0)
        assert s.base_out_index == 0

    def test_count_index(self):
        s = make_state(balls=3, strikes=2)
        assert s.count_index == 3 * 3 + 2

    def test_is_offense_challenging_batter(self):
        s = make_state(challenger_type="batter")
        assert s.is_offense_challenging is True

    def test_is_offense_challenging_defense(self):
        s = make_state(challenger_type="defense")
        assert s.is_offense_challenging is False

    def test_runners_on_base(self):
        assert make_state(base_state=0b000).runners_on_base == 0
        assert make_state(base_state=0b001).runners_on_base == 1
        assert make_state(base_state=0b111).runners_on_base == 3
        assert make_state(base_state=0b110).runners_on_base == 2


class TestGameStateHashability:
    def test_hashable(self):
        s = make_state()
        d = {s: 42}
        assert d[s] == 42

    def test_equal_states_same_hash(self):
        s1 = make_state(inning=3, outs=1)
        s2 = make_state(inning=3, outs=1)
        assert s1 == s2
        assert hash(s1) == hash(s2)

    def test_different_states_different_hash(self):
        s1 = make_state(inning=1)
        s2 = make_state(inning=2)
        assert s1 != s2


class TestEnumerateAllStates:
    def test_returns_list(self):
        states = enumerate_all_states()
        assert isinstance(states, list)
        assert len(states) > 0

    def test_no_duplicates(self):
        states = enumerate_all_states()
        assert len(set(states)) == len(states)

    def test_all_states_valid(self):
        states = enumerate_all_states()
        for s in states[:100]:  # Check a sample.
            assert 1 <= s.inning <= 9
            assert s.half in ("top", "bottom")
            assert -10 <= s.score_diff <= 10

    def test_build_index_bijection(self):
        states = enumerate_all_states()
        index = build_index(states)
        assert len(index) == len(states)
        for s, i in index.items():
            assert states[i] == s
