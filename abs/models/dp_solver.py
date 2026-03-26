"""
Full dynamic-programming solver for the ABS challenge strategy problem.

Algorithm
---------
Exact backward induction over the finite game horizon (innings 1–9).

State space
-----------
Trajectory dimensions: inning × half × score_diff × outs × base_state ×
  balls × strikes — roughly 108,864 states.
Challenge dimension: k ∈ {0, 1, 2} challenges remaining.
Challenger type: batter | defense (evaluated at each trajectory state).

Value function
--------------
V[s, k] = expected win probability (offensive team) under optimal challenge
           strategy from state s with k challenges remaining.

Bellman equation (at a challenge decision point)
-------------------------------------------------
If k == 0:
    V[s, 0] = WP(s_no_challenge, 0)     # can't challenge

If k >= 1:
    V[s, k] = max(
        WP(s_no_challenge, k),                          # don't challenge
        EV_challenge(s, k)                              # challenge
    )

where:
    EV_challenge(s, k) = p*(s) * V[s_overturn, k]
                       + (1 - p*(s)) * V[s_fail, k-1]

But p*(s) is the *actual* overturn probability for a specific pitch, which
the DP doesn't know ahead of time.  So the DP instead stores:

    V_no_challenge[s, k]   — value of *not* challenging
    V_overturn[s, k]       — value of state after overturn
    V_fail[s, k-1]         — value of state after failed challenge

These are sufficient to compute the threshold at inference time:

    p*(s, k) = [V_no_challenge(s,k) - V_fail(s,k-1)]
               / [V_overturn(s,k)   - V_fail(s,k-1)]

The optimal V[s,k] integrates over the distribution of p values actually
encountered (not needed for threshold computation).

Backward induction ordering
-----------------------------
We process states from the *end* of the game backward:
  - inning 9 → 1
  - within inning: bottom half → top half
  - within half-inning: outs 2 → 0
  - within out state: all (base_state, balls, strikes) combinations

Terminal boundary: when the DP reaches inning=INNINGS_MAX at the top of a
new inning (i.e., after the bottom of inning 9 completes), WP is read from
the WP table directly.
"""

from __future__ import annotations

from itertools import product
from typing import Optional

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from abs.config import (
    BALLS_MAX,
    CHALLENGER_BATTER,
    CHALLENGER_DEFENSE,
    DP_CONVERGENCE_TOL,
    INNINGS_MAX,
    MAX_CHALLENGES,
    OUTS_MAX,
    SCORE_DIFF_CLAMP,
    STRIKES_MAX,
)
from abs.state import GameState
from abs.tables.run_expectancy import RE24Table
from abs.tables.win_probability import WPTable
from abs.transitions import (
    is_terminal,
    state_after_failed_challenge,
    state_after_no_challenge,
    state_after_overturn,
)


# ── State indexing ────────────────────────────────────────────────────────────

def _build_trajectory_index() -> tuple[list[GameState], dict[GameState, int]]:
    """
    Build an ordered list and reverse-lookup dict for all trajectory states.

    Trajectory state = (inning, half, score_diff, outs, base_state, balls,
    strikes) with challenges_remaining=0 and challenger_type=CHALLENGER_BATTER
    as placeholders.  The DP value array V has shape (N_traj, 3, 2) where
    the last two axes are (k, challenger_type_idx).
    """
    states = []
    for inning, half, sd, outs, bs, balls, strikes in product(
        range(1, INNINGS_MAX + 1),
        ["top", "bottom"],
        range(-SCORE_DIFF_CLAMP, SCORE_DIFF_CLAMP + 1),
        range(OUTS_MAX + 1),
        range(8),
        range(BALLS_MAX + 1),
        range(STRIKES_MAX + 1),
    ):
        s = GameState(
            inning=inning, half=half, score_diff=sd, outs=outs,
            base_state=bs, balls=balls, strikes=strikes,
            challenges_remaining=0, challenger_type=CHALLENGER_BATTER,
        )
        states.append(s)

    index = {s: i for i, s in enumerate(states)}
    return states, index


_CTYPES = [CHALLENGER_BATTER, CHALLENGER_DEFENSE]
_CTYPE_IDX = {CHALLENGER_BATTER: 0, CHALLENGER_DEFENSE: 1}


def _traj_key(s: GameState) -> GameState:
    """Return the trajectory-key version of s (k=0, batter challenger)."""
    from dataclasses import replace
    return replace(s, challenges_remaining=0, challenger_type=CHALLENGER_BATTER)


# ── DP Solver ─────────────────────────────────────────────────────────────────


class DPSolver:
    """
    Full backward-induction solver for the ABS challenge decision problem.

    Parameters
    ----------
    wp_table : WPTable, optional
    re_table : RE24Table, optional
    """

    def __init__(
        self,
        wp_table: Optional[WPTable] = None,
        re_table: Optional[RE24Table] = None,
    ) -> None:
        self.wp = wp_table if wp_table is not None else WPTable()
        self.re = re_table if re_table is not None else RE24Table()
        self._solved = False

        # Build state index.
        self._states, self._index = _build_trajectory_index()
        N = len(self._states)

        # V[i, k, ct] = E[WP | state i, k challenges remaining, ct challenger type]
        # ct: 0=batter, 1=defense
        self._V = np.full((N, MAX_CHALLENGES + 1, 2), np.nan, dtype=np.float64)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _idx(self, s: GameState) -> int:
        """Integer index for a trajectory state."""
        return self._index[_traj_key(s)]

    def _get_v(self, s: GameState, k: int, ct: str) -> float:
        """Look up V[s, k, ct] after solving."""
        try:
            return float(self._V[self._idx(s), k, _CTYPE_IDX[ct]])
        except KeyError:
            # State outside our grid (shouldn't happen after solve).
            return self.wp.wp_for_state(s)

    def _terminal_wp(self, s: GameState) -> float:
        """Win probability at a terminal boundary state (read from WP table)."""
        return self.wp.wp_for_state(s)

    # ── Solve ─────────────────────────────────────────────────────────────────

    def solve(self, verbose: bool = True) -> None:
        """
        Run full backward induction.  Sets self._solved = True when done.

        Processing order:
          inning 9 → 1, half [bottom → top], outs 2 → 0,
          base_state 0 → 7, (balls, strikes) from full count back to 0-0.
        """
        V = self._V
        idx = self._idx

        # ── Step 1: Terminal boundary values ──────────────────────────────────
        # States at the start of inning INNINGS_MAX top (after bottom of 9 done)
        # are treated as terminal: assign WP from table for all k and ct.
        for i, s in enumerate(self._states):
            if is_terminal(s):
                wp = self._terminal_wp(s)
                V[i, :, :] = wp

        # ── Step 2: Backward induction ────────────────────────────────────────
        from dataclasses import replace as _replace_dc

        inning_range = range(INNINGS_MAX, 0, -1)
        # Temporal order: top of I → bottom of I → top of I+1.
        # Backward: bottom first (its successor is already processed top of I+1),
        # then top (its successor is bottom of same I, just processed).
        halves = ["bottom", "top"]
        outs_range = range(OUTS_MAX, -1, -1)
        # Base state: process in REVERSE order (7→0) so walk successors
        # (which tend to have higher bitmasks) are computed first.
        base_range = range(7, -1, -1)
        # Count order: process from full count back to 0-0 so terminal-count
        # successors are always available when we reach earlier counts.
        count_order = [
            (b, s)
            for b in range(BALLS_MAX, -1, -1)
            for s in range(STRIKES_MAX, -1, -1)
        ]

        total = INNINGS_MAX * 2 * (OUTS_MAX + 1) * 8
        iterator = product(inning_range, halves, outs_range, base_range)
        if verbose and _HAS_TQDM:
            iterator = tqdm(
                list(iterator), total=total, desc="DP solve", unit="states"
            )

        for inning, half, outs, base_state in iterator:
            for balls, strikes in count_order:
                s_base_args = dict(
                    inning=inning, half=half, score_diff=0,
                    outs=outs, base_state=base_state,
                    balls=balls, strikes=strikes,
                    challenges_remaining=0, challenger_type=CHALLENGER_BATTER,
                )

                for sd in range(-SCORE_DIFF_CLAMP, SCORE_DIFF_CLAMP + 1):
                    s_args = {**s_base_args, "score_diff": sd}
                    s0 = GameState(**s_args)
                    i0 = idx(s0)

                    if is_terminal(s0):
                        continue  # Already assigned.

                    for ct_str in _CTYPES:
                        ct_i = _CTYPE_IDX[ct_str]
                        s = GameState(**{**s_args, "challenger_type": ct_str})

                        # Compute V for each k in ascending order.
                        # V(s, k=0) must be set before V(s, k=1) can reference it.
                        for k in range(MAX_CHALLENGES + 1):
                            s_k = _replace_dc(s, challenges_remaining=k)
                            s_nc = state_after_no_challenge(s_k)
                            v_nc = self._lookup_successor(s_nc, k, ct_str, V)

                            if k == 0:
                                V[i0, k, ct_i] = v_nc
                            else:
                                # Successor after overturn (challenge retained).
                                s_ov = state_after_overturn(s_k)
                                v_ov = self._lookup_successor(s_ov, k, ct_str, V)

                                # Successor after failed challenge (k-1).
                                s_fail = state_after_failed_challenge(s_k)
                                v_fail = self._lookup_successor(s_fail, k - 1, ct_str, V)

                                # Compute optimal V assuming p̂ ~ Uniform[0,1].
                                # Under this distribution, V(s,k) = optimal expected WP.
                                #
                                # Let a=V_overturn, b=V_fail, c=V_nc.
                                # Challenge threshold: p* = (c-b)/(a-b)  if a>b.
                                #
                                # V(s,k) = (a+b)/2 + (c-b)^2 / (2*(a-b))  if 0 < p* < 1
                                #        = (a+b)/2                          if p* ≤ 0
                                #        = c                                if p* ≥ 1
                                #        = c                                if a ≤ b
                                a, b, c = v_ov, v_fail, v_nc
                                denom = a - b
                                if denom <= DP_CONVERGENCE_TOL:
                                    # Overturn not beneficial.
                                    V[i0, k, ct_i] = c
                                else:
                                    p_star_local = (c - b) / denom
                                    if p_star_local >= 1.0:
                                        V[i0, k, ct_i] = c
                                    elif p_star_local <= 0.0:
                                        V[i0, k, ct_i] = (a + b) / 2.0
                                    else:
                                        V[i0, k, ct_i] = (a + b) / 2.0 + (c - b) ** 2 / (2.0 * denom)

        self._solved = True

    def _lookup_successor(
        self, s: GameState, k: int, ct: str, V: np.ndarray
    ) -> float:
        """
        Look up V for a successor state.  Handles terminal states and states
        outside the grid by falling back to the WP table.
        """
        k_clamped = max(0, min(MAX_CHALLENGES, k))
        if is_terminal(s):
            return self._terminal_wp(s)
        try:
            i = self._idx(s)
            v = V[i, k_clamped, _CTYPE_IDX[ct]]
            if np.isnan(v):
                # Not yet solved (should not happen in correct ordering).
                return self.wp.wp_for_state(s)
            return float(v)
        except KeyError:
            return self.wp.wp_for_state(s)

    # ── Threshold computation ─────────────────────────────────────────────────

    def threshold(self, s: GameState) -> Optional[float]:
        """
        p*(s, k): minimum overturn probability to justify a challenge.

        p*(s, k) = [V(s_no_challenge, k) - V(s_fail, k-1)]
                   / [V(s_overturn, k)   - V(s_fail, k-1)]

        Returns None if k == 0 or the denominator is negligible.
        """
        if not self._solved:
            raise RuntimeError("Call solve() first.")
        k = s.challenges_remaining
        if k == 0:
            return None

        ct = s.challenger_type
        ct_i = _CTYPE_IDX[ct]
        k1 = max(0, k - 1)

        s_nc = state_after_no_challenge(s)
        s_ov = state_after_overturn(s)
        s_fail = state_after_failed_challenge(s)

        def _v(state: GameState, kk: int) -> float:
            if is_terminal(state):
                return self._terminal_wp(state)
            try:
                return float(self._V[self._idx(state), kk, ct_i])
            except KeyError:
                return self.wp.wp_for_state(state)

        v_nc = _v(s_nc, k)
        v_ov = _v(s_ov, k)
        v_fail = _v(s_fail, k1)

        denom = v_ov - v_fail
        if abs(denom) < DP_CONVERGENCE_TOL:
            return None

        p_star = (v_nc - v_fail) / denom
        return max(0.0, min(1.0, p_star))

    def get_value(self, s: GameState) -> float:
        """V(s, k) under optimal play (no-challenge baseline stored in V)."""
        if not self._solved:
            raise RuntimeError("Call solve() first.")
        k = min(s.challenges_remaining, MAX_CHALLENGES)
        ct_i = _CTYPE_IDX.get(s.challenger_type, 0)
        try:
            v = float(self._V[self._idx(s), k, ct_i])
            return v if not np.isnan(v) else self.wp.wp_for_state(s)
        except KeyError:
            return self.wp.wp_for_state(s)

    def threshold_table(self) -> pd.DataFrame:
        """
        Extract all p*(s, k) into a tidy DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: inning, half, score_diff, outs, base_state, balls,
            strikes, challenges_remaining, challenger_type, p_star_dp.
        """
        if not self._solved:
            raise RuntimeError("Call solve() first.")

        records = []
        for s_traj in self._states:
            for k in range(1, MAX_CHALLENGES + 1):
                for ct in _CTYPES:
                    from dataclasses import replace
                    s = replace(s_traj, challenges_remaining=k, challenger_type=ct)
                    p = self.threshold(s)
                    records.append(
                        {
                            "inning": s.inning,
                            "half": s.half,
                            "score_diff": s.score_diff,
                            "outs": s.outs,
                            "base_state": s.base_state,
                            "balls": s.balls,
                            "strikes": s.strikes,
                            "challenges_remaining": k,
                            "challenger_type": ct,
                            "p_star_dp": p,
                        }
                    )
        return pd.DataFrame(records)
