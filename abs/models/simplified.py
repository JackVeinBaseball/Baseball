"""
Simplified (single-period) ABS challenge threshold model.

This is the *presentation* version: it ignores the option value of future
challenges and computes the minimum overturn probability p*(s) that makes
an immediate challenge EV-positive.

Formula
-------
    p*(s) = [WP(s_no_challenge) - WP(s_fail)] /
            [WP(s_overturn)     - WP(s_fail)]

where:
  s_no_challenge : state when no challenge is taken (original call stands).
  s_fail         : state when challenge fails (original call stands, k -= 1).
  s_overturn     : state when challenge succeeds (call reversed, k unchanged).

Because this model ignores the *continuation value* of saving a challenge,
it systematically recommends lower thresholds (i.e., it over-recommends
challenging) compared to the DP solution.  The gap between simplified and DP
thresholds quantifies the shadow cost of using up a challenge.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from abs.state import GameState, enumerate_all_states
from abs.tables.win_probability import WPTable
from abs.transitions import (
    state_after_failed_challenge,
    state_after_no_challenge,
    state_after_overturn,
)


class SimplifiedThresholdModel:
    """
    Single-period challenge threshold model.

    Parameters
    ----------
    wp_table : WPTable
        Win probability table.
    """

    def __init__(self, wp_table: Optional[WPTable] = None) -> None:
        self.wp = wp_table if wp_table is not None else WPTable()

    def threshold(self, s: GameState) -> Optional[float]:
        """
        Compute p*(s): minimum overturn probability to justify a challenge.

        Returns
        -------
        float in [0, 1], or None if:
          - k == 0 (no challenges remaining), or
          - the WP gain from overturning is negligible (denominator < tol).
        """
        if s.challenges_remaining == 0:
            return None

        s_overturn = state_after_overturn(s)
        s_fail = state_after_failed_challenge(s)
        s_no_challenge = state_after_no_challenge(s)

        wp_overturn = self.wp.wp_for_state(s_overturn)
        wp_fail = self.wp.wp_for_state(s_fail)
        wp_no_challenge = self.wp.wp_for_state(s_no_challenge)

        denominator = wp_overturn - wp_fail
        if abs(denominator) < 1e-9:
            return None

        p_star = (wp_no_challenge - wp_fail) / denominator
        return max(0.0, min(1.0, p_star))

    def ev_gain(self, s: GameState, p_overturn: float) -> float:
        """
        Expected WP gain from challenging at overturn probability p_overturn.

        EV_challenge − EV_no_challenge.
        Positive → challenging is beneficial.
        """
        if s.challenges_remaining == 0:
            return 0.0

        s_overturn = state_after_overturn(s)
        s_fail = state_after_failed_challenge(s)
        s_no_challenge = state_after_no_challenge(s)

        wp_overturn = self.wp.wp_for_state(s_overturn)
        wp_fail = self.wp.wp_for_state(s_fail)
        wp_no_challenge = self.wp.wp_for_state(s_no_challenge)

        ev_challenge = p_overturn * wp_overturn + (1 - p_overturn) * wp_fail
        return ev_challenge - wp_no_challenge

    def threshold_table(
        self,
        states: Optional[list[GameState]] = None,
    ) -> pd.DataFrame:
        """
        Compute p*(s) for each state and return a tidy DataFrame.

        Parameters
        ----------
        states : list[GameState], optional
            States to evaluate.  Defaults to all states from
            enumerate_all_states().

        Returns
        -------
        pd.DataFrame
            Columns: inning, half, score_diff, outs, base_state, balls,
            strikes, challenges_remaining, challenger_type, p_star_simplified.
        """
        if states is None:
            states = enumerate_all_states()

        records = []
        for s in states:
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
                    "challenges_remaining": s.challenges_remaining,
                    "challenger_type": s.challenger_type,
                    "p_star_simplified": p,
                }
            )
        return pd.DataFrame(records)
