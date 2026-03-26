"""Tests for abs/tables/win_probability.py."""

import pytest
from abs.tables.win_probability import WPTable, LogisticWPModel
from abs.state import GameState


def make_state(**kwargs) -> GameState:
    defaults = dict(
        inning=5, half="top", score_diff=0, outs=0,
        base_state=0, balls=0, strikes=0,
        challenges_remaining=1, challenger_type="batter",
    )
    defaults.update(kwargs)
    return GameState(**defaults)


class TestWPTable:
    def test_loads_bundled_table(self):
        table = WPTable()
        # Should load without error.
        assert table is not None

    def test_returns_value_in_01(self):
        table = WPTable()
        wp = table.get(inning=1, half="top", score_diff_home=0, outs=0, base_state=0)
        assert 0.0 <= wp <= 1.0

    def test_tie_game_near_half(self):
        table = WPTable()
        wp = table.get(inning=1, half="top", score_diff_home=0, outs=0, base_state=0)
        # Tie game should be near 0.5.
        assert 0.3 <= wp <= 0.7

    def test_large_lead_near_one(self):
        table = WPTable()
        wp = table.get(inning=9, half="top", score_diff_home=10, outs=0, base_state=0)
        assert wp > 0.85

    def test_large_deficit_near_zero(self):
        table = WPTable()
        wp = table.get(inning=9, half="top", score_diff_home=-10, outs=0, base_state=0)
        assert wp < 0.15

    def test_wp_for_state_offense_pov_top(self):
        table = WPTable()
        # Away team batting (top), tied game.
        s = make_state(half="top", score_diff=0)
        wp = table.wp_for_state(s)
        # Away team's WP in a tie game should be near 0.5.
        assert 0.3 <= wp <= 0.7

    def test_wp_for_state_offense_pov_bottom(self):
        table = WPTable()
        # Home team batting (bottom), home team leading.
        s = make_state(half="bottom", score_diff=3, inning=8)
        wp = table.wp_for_state(s)
        assert wp > 0.5  # Home team is leading and batting.

    def test_monotone_in_score_diff(self):
        table = WPTable()
        # Higher home score_diff → higher home WP.
        wps = [table.get(5, "top", sd, 0, 0) for sd in range(-5, 6)]
        assert all(wps[i] <= wps[i + 1] + 0.01 for i in range(len(wps) - 1))

    def test_fallback_for_missing_state(self):
        # Even if no CSV loaded, fallback should work.
        from abs.tables.win_probability import LogisticWPModel
        model = LogisticWPModel()
        wp = model.predict(9, "top", 5, 0, 0)
        assert 0.0 <= wp <= 1.0


class TestLogisticWPModel:
    def test_returns_value_in_01(self):
        model = LogisticWPModel()
        wp = model.predict(5, "top", 0, 0, 0)
        assert 0.0 <= wp <= 1.0

    def test_monotone_in_score_diff_home(self):
        model = LogisticWPModel()
        wps = [model.predict(5, "top", sd, 0, 0) for sd in range(-10, 11)]
        for i in range(len(wps) - 1):
            assert wps[i] <= wps[i + 1] + 0.01, f"Non-monotone at sd={i-10}"

    def test_symmetry_tie_game(self):
        model = LogisticWPModel()
        wp = model.predict(1, "top", 0, 0, 0)
        # With 0 intercept and symmetric features, should be near 0.5.
        assert abs(wp - 0.5) < 0.1
