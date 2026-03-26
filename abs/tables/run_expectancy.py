"""
Run Expectancy (RE24) table.

Provides expected runs from a given base-out state to the end of the
half-inning.  Values are the canonical 2010–2019 MLB averages from the
standard RE24 reference.

The builtin 24-cell table (3 outs × 8 base states) is hardcoded and works
with no external files.  An optional count-conditioned 288-cell CSV
(3 × 8 × 4 × 3) can be loaded for higher-resolution analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── Builtin 24-cell RE24 values (2010-2019 MLB averages) ─────────────────────
# Rows = outs (0, 1, 2); columns = base_state (0b000 through 0b111).
# Source: Tango/MGL "The Book" and FanGraphs RE24 reference page.
_BUILTIN_RE24 = np.array(
    [
        # 000     001     010     011     100     101     110     111
        [0.461,  0.831,  1.068,  1.373,  0.859,  1.211,  1.425,  1.798],  # 0 outs
        [0.243,  0.509,  0.644,  0.908,  0.519,  0.771,  0.950,  1.243],  # 1 out
        [0.095,  0.214,  0.305,  0.429,  0.214,  0.348,  0.438,  0.569],  # 2 outs
    ],
    dtype=np.float64,
)


class RE24Table:
    """
    Run expectancy table indexed by (outs, base_state).

    Parameters
    ----------
    csv_path : Path, optional
        Path to a count-conditioned RE CSV with columns:
        ``outs, base_state, balls, strikes, run_expectancy``.
        If omitted the builtin 24-cell table is used.
    """

    def __init__(self, csv_path: Path | None = None) -> None:
        self._table: np.ndarray = _BUILTIN_RE24.copy()  # shape (3, 8)
        self._count_table: np.ndarray | None = None      # shape (3, 8, 4, 3) if loaded
        self._count_conditioned = False

        if csv_path is not None:
            self.load(csv_path)

    def load(self, csv_path: Path) -> None:
        """Load a count-conditioned RE CSV and update internal tables."""
        df = pd.read_csv(csv_path)
        required = {"outs", "base_state", "run_expectancy"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"RE CSV missing columns: {missing}")

        has_count = {"balls", "strikes"}.issubset(df.columns)
        if has_count:
            arr = np.zeros((3, 8, 4, 3), dtype=np.float64)
            for row in df.itertuples(index=False):
                arr[row.outs, row.base_state, row.balls, row.strikes] = (
                    row.run_expectancy
                )
            self._count_table = arr
            self._count_conditioned = True
            # Also rebuild the 24-cell marginal from count-conditioned data.
            self._table = arr.mean(axis=(2, 3))
        else:
            arr = np.zeros((3, 8), dtype=np.float64)
            for row in df.itertuples(index=False):
                arr[row.outs, row.base_state] = row.run_expectancy
            self._table = arr

    def get(
        self,
        outs: int,
        base_state: int,
        balls: int | None = None,
        strikes: int | None = None,
    ) -> float:
        """
        Expected runs from (outs, base_state) to end of half-inning.

        If count-conditioned data is loaded and ``balls``/``strikes`` are
        provided, returns the count-specific value; otherwise falls back to
        the 24-cell marginal.
        """
        if self._count_conditioned and balls is not None and strikes is not None:
            return float(self._count_table[outs, base_state, balls, strikes])
        return float(self._table[outs, base_state])

    def re_delta(
        self,
        from_outs: int,
        from_bases: int,
        to_outs: int,
        to_bases: int,
        runs_scored: int = 0,
    ) -> float:
        """
        Change in run expectancy: RE(to) + runs_scored − RE(from).

        A positive value means the event was *good* for the offense.
        """
        return self.get(to_outs, to_bases) + runs_scored - self.get(from_outs, from_bases)

    @property
    def table(self) -> np.ndarray:
        """The 24-cell (3 × 8) RE array."""
        return self._table.copy()

    def as_dataframe(self) -> pd.DataFrame:
        """Return RE24 as a tidy DataFrame with columns outs, base_state, re."""
        rows = []
        for outs in range(3):
            for base_state in range(8):
                rows.append(
                    {"outs": outs, "base_state": base_state, "re": self._table[outs, base_state]}
                )
        return pd.DataFrame(rows)
