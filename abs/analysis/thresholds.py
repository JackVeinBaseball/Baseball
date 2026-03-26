"""
Threshold extraction and aggregation utilities.

Slices the p*(s) output from either the simplified or DP model into
presentation-ready DataFrames and pivot tables.
"""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from abs.config import BASE_STATE_LABELS


# ── Helpers ───────────────────────────────────────────────────────────────────


def _count_label(balls: int, strikes: int) -> str:
    return f"{balls}-{strikes}"


def _base_label(base_state: int) -> str:
    return BASE_STATE_LABELS.get(base_state, str(base_state))


# ── Aggregation functions ─────────────────────────────────────────────────────


def pivot_by_count_and_bases(
    df: pd.DataFrame,
    outs: int,
    k: int,
    challenger_type: str = "batter",
    value_col: str = "p_star_dp",
) -> pd.DataFrame:
    """
    Return a 12 × 8 pivot table of p* indexed by count (rows) and
    base_state (columns).

    Parameters
    ----------
    df : pd.DataFrame
        Threshold table with columns matching the output of
        DPSolver.threshold_table() or SimplifiedThresholdModel.threshold_table().
    outs : int
        Filter to this number of outs.
    k : int
        Filter to this number of challenges remaining.
    challenger_type : str
        "batter" or "defense".
    value_col : str
        Column name for the threshold values.
    """
    sub = df[
        (df["outs"] == outs)
        & (df["challenges_remaining"] == k)
        & (df["challenger_type"] == challenger_type)
    ].copy()

    sub["count"] = sub.apply(lambda r: _count_label(r["balls"], r["strikes"]), axis=1)
    sub["base_label"] = sub["base_state"].map(_base_label)

    # Define row order for counts.
    count_order = [f"{b}-{s}" for b in range(4) for s in range(3)]

    pivot = sub.pivot_table(
        index="count",
        columns="base_label",
        values=value_col,
        aggfunc="mean",
    )
    # Reorder rows and columns.
    pivot = pivot.reindex(
        [c for c in count_order if c in pivot.index]
    )
    base_col_order = [BASE_STATE_LABELS[i] for i in range(8)]
    pivot = pivot.reindex(columns=[c for c in base_col_order if c in pivot.columns])
    return pivot


def pivot_by_inning_and_score(
    df: pd.DataFrame,
    outs: int = 0,
    base_state: int = 0,
    k: int = 1,
    challenger_type: str = "batter",
    balls: int = 0,
    strikes: int = 0,
    value_col: str = "p_star_dp",
) -> pd.DataFrame:
    """
    Return a 9 × 21 pivot table of p* indexed by inning (rows) and
    score_diff (columns).
    """
    sub = df[
        (df["outs"] == outs)
        & (df["base_state"] == base_state)
        & (df["challenges_remaining"] == k)
        & (df["challenger_type"] == challenger_type)
        & (df["balls"] == balls)
        & (df["strikes"] == strikes)
    ]

    pivot = sub.pivot_table(
        index="inning",
        columns="score_diff",
        values=value_col,
        aggfunc="mean",
    )
    return pivot


def summarize_by_count(
    df: pd.DataFrame,
    value_col: str = "p_star_dp",
    challenger_type: str = "batter",
) -> pd.DataFrame:
    """
    Average p* by count, averaging over all other state dimensions.

    Returns a 12-row DataFrame with columns: balls, strikes, count_label,
    mean_p_star, std_p_star.
    """
    sub = df[df["challenger_type"] == challenger_type].copy()
    sub["count"] = sub.apply(lambda r: _count_label(r["balls"], r["strikes"]), axis=1)
    agg = (
        sub.groupby(["balls", "strikes", "count"])[value_col]
        .agg(mean_p_star="mean", std_p_star="std")
        .reset_index()
        .sort_values(["balls", "strikes"])
    )
    return agg


def compare_simplified_vs_dp(
    simplified_df: pd.DataFrame,
    dp_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge simplified and DP threshold tables for comparison.

    Returns a DataFrame with both p_star_simplified and p_star_dp,
    plus a delta column showing the opportunity cost of using a challenge.
    """
    key_cols = [
        "inning", "half", "score_diff", "outs", "base_state",
        "balls", "strikes", "challenges_remaining", "challenger_type",
    ]
    merged = simplified_df.merge(dp_df, on=key_cols, how="inner", suffixes=("_s", "_dp"))

    # Handle column name conflicts when both tables have same value column.
    if "p_star_simplified" in merged.columns and "p_star_dp" in merged.columns:
        pass  # Already have both.
    elif "p_star_dp_s" in merged.columns:
        merged = merged.rename(columns={"p_star_dp_s": "p_star_simplified",
                                         "p_star_dp_dp": "p_star_dp"})

    if "p_star_simplified" in merged.columns and "p_star_dp" in merged.columns:
        merged["delta"] = merged["p_star_dp"] - merged["p_star_simplified"]
    return merged


def high_leverage_states(
    df: pd.DataFrame,
    value_col: str = "p_star_dp",
    k: int = 1,
    challenger_type: str = "batter",
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Return the top_n states with the *lowest* p* (highest urgency to challenge).

    Low p* means you should challenge even if overturn probability is modest —
    indicating a high-leverage situation.
    """
    sub = df[
        (df["challenges_remaining"] == k)
        & (df["challenger_type"] == challenger_type)
        & df[value_col].notna()
    ]
    return (
        sub.nsmallest(top_n, value_col)
        .reset_index(drop=True)
    )


def conservative_states(
    df: pd.DataFrame,
    value_col: str = "p_star_dp",
    k: int = 1,
    challenger_type: str = "batter",
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Return the top_n states with the *highest* p* (most conservative threshold).

    High p* means only challenge if overturn is near-certain — indicating
    the challenge is better saved for later.
    """
    sub = df[
        (df["challenges_remaining"] == k)
        & (df["challenger_type"] == challenger_type)
        & df[value_col].notna()
    ]
    return (
        sub.nlargest(top_n, value_col)
        .reset_index(drop=True)
    )
