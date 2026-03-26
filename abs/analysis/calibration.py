"""
Calibration analysis: compare actual challenge behavior to the optimal policy.

Identifies overchallenging (p̂ < p* but challenged) and underchallenging
(p̂ > p* but did not challenge) at the team, count, and game-state level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Data loading ──────────────────────────────────────────────────────────────


def load_challenge_data(csv_path: Path) -> pd.DataFrame:
    """
    Load user-supplied MLB challenge data.

    Expected columns (at minimum):
      inning, half, score_diff, outs, base_state, balls, strikes,
      challenges_remaining, challenger_type, location_delta, initiator_type,
      challenged (bool), overturned (bool).

    Optional: team, game_id, date, player_id.
    """
    df = pd.read_csv(csv_path)
    required = {
        "inning", "half", "outs", "base_state", "balls", "strikes",
        "challenges_remaining", "challenger_type",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Challenge data CSV missing columns: {missing}")
    return df


# ── Overchallenging / underchallenging ────────────────────────────────────────


def challenge_rate_by_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute actual challenge rate grouped by (balls, strikes).

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: balls, strikes, challenged.

    Returns
    -------
    pd.DataFrame
        Columns: balls, strikes, n_pitches, n_challenged, challenge_rate.
    """
    agg = (
        df.groupby(["balls", "strikes"])
        .agg(n_pitches=("challenged", "count"), n_challenged=("challenged", "sum"))
        .reset_index()
    )
    agg["challenge_rate"] = agg["n_challenged"] / agg["n_pitches"]
    return agg


def overchallenging_by_team(
    evaluated_df: pd.DataFrame,
    team_col: str = "team",
) -> pd.DataFrame:
    """
    Compute overchallenging and underchallenging rates by team.

    Parameters
    ----------
    evaluated_df : pd.DataFrame
        Output of ChallengePolicy.evaluate_decisions() with a team column.

    Returns
    -------
    pd.DataFrame
        Columns: team, n_decisions, n_overchallenged, n_underchallenged,
        overchallenge_rate, underchallenge_rate, net_error_rate.
        Sorted by net_error_rate descending.
    """
    if team_col not in evaluated_df.columns:
        raise ValueError(f"Column {team_col!r} not found in DataFrame.")

    required = {"challenged", "overchallenged", "underchallenged"}
    missing = required - set(evaluated_df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}. "
                         "Run ChallengePolicy.evaluate_decisions() first.")

    agg = (
        evaluated_df.groupby(team_col)
        .agg(
            n_decisions=("challenged", "count"),
            n_overchallenged=("overchallenged", "sum"),
            n_underchallenged=("underchallenged", "sum"),
        )
        .reset_index()
    )
    agg["overchallenge_rate"] = agg["n_overchallenged"] / agg["n_decisions"]
    agg["underchallenge_rate"] = agg["n_underchallenged"] / agg["n_decisions"]
    agg["net_error_rate"] = agg["overchallenge_rate"] - agg["underchallenge_rate"]
    return agg.sort_values("net_error_rate", ascending=False).reset_index(drop=True)


def threshold_vs_actual_by_count(
    evaluated_df: pd.DataFrame,
    challenger_type: str = "batter",
) -> pd.DataFrame:
    """
    Compare average p* vs. average actual challenge rate by count.

    Parameters
    ----------
    evaluated_df : pd.DataFrame
        Must have columns: balls, strikes, challenger_type, p_star, challenged.

    Returns
    -------
    pd.DataFrame
        Columns: balls, strikes, mean_p_star, actual_challenge_rate, gap.
        Gap > 0 means teams are overchallenging on this count.
    """
    sub = evaluated_df[evaluated_df["challenger_type"] == challenger_type]
    agg = (
        sub.groupby(["balls", "strikes"])
        .agg(
            mean_p_star=("p_star", "mean"),
            actual_challenge_rate=("challenged", "mean"),
        )
        .reset_index()
    )
    agg["gap"] = agg["actual_challenge_rate"] - agg["mean_p_star"]
    return agg


def ev_impact_summary(evaluated_df: pd.DataFrame) -> dict:
    """
    Summarize total expected WP gained/lost from suboptimal challenge decisions.

    Returns
    -------
    dict with keys:
        total_decisions, optimal_decisions, suboptimal_decisions,
        total_ev_gain_optimal, total_ev_gain_actual, ev_left_on_table.
    """
    n = len(evaluated_df)
    if "was_optimal" not in evaluated_df.columns:
        return {"total_decisions": n, "note": "was_optimal column not present."}

    n_opt = evaluated_df["was_optimal"].sum()
    n_sub = n - n_opt

    ev_actual = evaluated_df["ev_gain"].fillna(0).sum() if "ev_gain" in evaluated_df.columns else np.nan

    return {
        "total_decisions": n,
        "optimal_decisions": int(n_opt),
        "suboptimal_decisions": int(n_sub),
        "optimality_rate": float(n_opt / n) if n > 0 else np.nan,
        "total_ev_gain_actual": float(ev_actual),
    }
