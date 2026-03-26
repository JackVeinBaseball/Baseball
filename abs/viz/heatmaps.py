"""
Heatmap visualizations for p* threshold tables.

All functions return matplotlib Figure objects.
Callers are responsible for saving (fig.savefig(...)).
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abs.analysis.thresholds import pivot_by_count_and_bases, pivot_by_inning_and_score
from abs.config import BASE_STATE_LABELS, FIGURE_DPI


def plot_threshold_by_count_bases(
    df: pd.DataFrame,
    outs: int = 0,
    k: int = 1,
    challenger_type: str = "batter",
    value_col: str = "p_star_dp",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of p* by count (rows) × base state (columns).

    Parameters
    ----------
    df : pd.DataFrame
        Threshold table from DPSolver.threshold_table() or simplified model.
    outs : int
        Number of outs to filter on.
    k : int
        Challenges remaining to filter on.
    challenger_type : str
        "batter" or "defense".
    value_col : str
        Column containing the threshold values.
    title : str, optional
        Custom figure title.

    Returns
    -------
    matplotlib.Figure
    """
    pivot = pivot_by_count_and_bases(df, outs, k, challenger_type, value_col)

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="lightgray",
        cbar_kws={"label": "p* (min. overturn prob. to challenge)"},
    )
    ctype_label = "Batter (called strike)" if challenger_type == "batter" else "Defense (called ball)"
    default_title = (
        f"Challenge Threshold p* | {outs} outs, {k} challenge(s) remaining\n"
        f"{ctype_label}"
    )
    ax.set_title(title or default_title, fontsize=13, pad=12)
    ax.set_xlabel("Base State", fontsize=11)
    ax.set_ylabel("Count (balls-strikes)", fontsize=11)
    fig.tight_layout()
    return fig


def plot_threshold_faceted(
    df: pd.DataFrame,
    k: int = 1,
    challenger_type: str = "batter",
    value_col: str = "p_star_dp",
) -> plt.Figure:
    """
    3-panel heatmap (one per outs value: 0, 1, 2) of p* by count × base state.
    """
    fig, axes = plt.subplots(1, 3, figsize=(33, 7), sharey=True)
    for outs, ax in zip(range(3), axes):
        pivot = pivot_by_count_and_bases(df, outs, k, challenger_type, value_col)
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="RdYlGn_r",
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
            linewidths=0.3,
            linecolor="lightgray",
            cbar=(outs == 2),
            cbar_kws={"label": "p*"} if outs == 2 else {},
        )
        ax.set_title(f"{outs} outs", fontsize=12)
        ax.set_xlabel("Base State", fontsize=10)
        if outs == 0:
            ax.set_ylabel("Count", fontsize=10)
    ctype_label = "batter" if challenger_type == "batter" else "defense"
    fig.suptitle(
        f"p* Threshold by Count × Base State — {ctype_label} challenger, "
        f"k={k} challenge(s) remaining",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_threshold_by_inning_score(
    df: pd.DataFrame,
    outs: int = 0,
    base_state: int = 0,
    k: int = 1,
    challenger_type: str = "batter",
    balls: int = 0,
    strikes: int = 0,
    value_col: str = "p_star_dp",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of p* by inning (rows) × score differential (columns).
    """
    pivot = pivot_by_inning_and_score(
        df, outs, base_state, k, challenger_type, balls, strikes, value_col
    )

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        linecolor="lightgray",
        cbar_kws={"label": "p*"},
    )
    base_lbl = BASE_STATE_LABELS.get(base_state, str(base_state))
    default_title = (
        f"p* by Inning × Score Differential | {outs} outs, {base_lbl}, "
        f"{balls}-{strikes} count, k={k}"
    )
    ax.set_title(title or default_title, fontsize=12, pad=10)
    ax.set_xlabel("Score Differential (offense perspective)", fontsize=10)
    ax.set_ylabel("Inning", fontsize=10)
    fig.tight_layout()
    return fig


def plot_simplified_vs_dp(
    df: pd.DataFrame,
    challenger_type: str = "batter",
    sample_frac: float = 0.1,
    seed: int = 42,
) -> plt.Figure:
    """
    Scatter plot comparing simplified p* vs. DP p*.

    Shows the shadow cost of ignoring future challenge value.
    Points above the diagonal: DP recommends a higher threshold
    (more conservative) than the simplified model.
    """
    sub = df[df["challenger_type"] == challenger_type].dropna(
        subset=["p_star_simplified", "p_star_dp"]
    )
    if sample_frac < 1.0:
        sub = sub.sample(frac=sample_frac, random_state=seed)

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(
        sub["p_star_simplified"],
        sub["p_star_dp"],
        c=sub["challenges_remaining"],
        cmap="viridis",
        alpha=0.4,
        s=10,
        rasterized=True,
    )
    lims = [0, 1]
    ax.plot(lims, lims, "k--", lw=1, label="y = x (equivalence)")
    plt.colorbar(scatter, ax=ax, label="Challenges remaining (k)")
    ax.set_xlabel("p* simplified (single-period)", fontsize=11)
    ax.set_ylabel("p* DP (full backward induction)", fontsize=11)
    ax.set_title(
        "Simplified vs. DP Challenge Threshold\n"
        "Points above diagonal: DP is more conservative (values future challenges)",
        fontsize=11,
    )
    ax.legend()
    fig.tight_layout()
    return fig
