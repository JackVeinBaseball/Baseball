"""Tests for abs/tables/run_expectancy.py."""

import pytest
import numpy as np
from abs.tables.run_expectancy import RE24Table


class TestRE24Table:
    def test_loads_without_csv(self):
        table = RE24Table()
        assert table is not None

    def test_all_values_positive(self):
        table = RE24Table()
        for outs in range(3):
            for base_state in range(8):
                assert table.get(outs, base_state) >= 0.0

    def test_bases_loaded_higher_than_empty(self):
        table = RE24Table()
        re_loaded = table.get(0, 0b111)
        re_empty = table.get(0, 0b000)
        assert re_loaded > re_empty

    def test_more_outs_lower_re(self):
        table = RE24Table()
        for base_state in range(8):
            re_0 = table.get(0, base_state)
            re_1 = table.get(1, base_state)
            re_2 = table.get(2, base_state)
            assert re_0 >= re_1 >= re_2

    def test_re_delta_positive_event(self):
        table = RE24Table()
        # Going from bases empty 0 outs to runner on 1B 0 outs (hit) should be positive.
        delta = table.re_delta(0, 0b000, 0, 0b001)
        assert delta > 0

    def test_re_delta_out_is_negative(self):
        table = RE24Table()
        # Recording an out (0→1) with no runners should decrease RE.
        delta = table.re_delta(0, 0b000, 1, 0b000)
        assert delta < 0

    def test_re_delta_includes_runs_scored(self):
        table = RE24Table()
        # Runner on 3B scores (bases loaded → runner scores, now 2 on 2 outs).
        delta_no_run = table.re_delta(1, 0b100, 1, 0b000)
        delta_with_run = table.re_delta(1, 0b100, 1, 0b000, runs_scored=1)
        assert delta_with_run > delta_no_run

    def test_as_dataframe(self):
        table = RE24Table()
        df = table.as_dataframe()
        assert len(df) == 24
        assert "outs" in df.columns
        assert "base_state" in df.columns
        assert "re" in df.columns

    def test_table_property(self):
        table = RE24Table()
        arr = table.table
        assert arr.shape == (3, 8)
        assert arr.dtype == np.float64

    def test_canonical_values_spot_check(self):
        table = RE24Table()
        # Canonical RE24 values (2010-2019 MLB).
        assert abs(table.get(0, 0) - 0.461) < 0.01  # bases empty, 0 outs
        assert abs(table.get(0, 7) - 1.798) < 0.01  # bases loaded, 0 outs
        assert abs(table.get(2, 0) - 0.095) < 0.01  # bases empty, 2 outs
