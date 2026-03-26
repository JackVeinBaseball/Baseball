"""
Calibration visualizations: overchallenging / underchallenging analysis.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abs.config import BASE_STATE_LABELS, FIGURE_DPI


def plot_overchallenging_by_team(
    team_summary: pd.DataFrame,
    title: Optional[str] = None,
    top_n: int = 30,
) -> plt.Figure:
    """
    Horizontal bar chart showing overchallenging/underchallenging rate by team.

    Parameters
    ----------
    team_summary : pd.DataFrame
        Output of calibration.overchallenging_by_team().
        Expected columns: team, overchallenge_rate, underchallenge_rate,
        net_error_rate.
    top_n : int
        Maximum number of teams to display.
    """
    df = team_summary.head(top_n).copy()
    colors = ["#c0392b" if v > 0 else "#27ae60" for v in df["net_error_rate"]]

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.35)))
    bars = ax.barh(df["team"], df["net_error_rate"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Net error rate (overchallenge − underchallenge)", fontsize=10)
    ax.set_title(title or "Overchallenging (+) / Underchallenging (−) by Team", fontsize=12)
    ax.invert_yaxis()

    for bar, val in zip(bars, df["net_error_rate"]):
        x_pos = val + (0.003 if val >= 0 else -0.003)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=8)

    fig.tight_layout()
    return fig


def plot_challenge_rate_vs_threshold(
    comparison_df: pd.DataFrame,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side heatmaps: actual challenge rate vs. p* by count.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of calibration.threshold_vs_actual_by_count().
        Expected columns: balls, strikes, mean_p_star, actual_challenge_rate, gap.
    """
    count_order = [f"{b}-{s}" for b in range(4) for s in range(3)]

    def _to_pivot(col):
        tmp = comparison_df.copy()
        tmp["count"] = tmp.apply(lambda r: f"{int(r.balls)}-{int(r.strikes)}", axis=1)
        p = tmp.pivot(index="count", columns=None, values=col)
        return tmp.set_index("count")[col].reindex(count_order).to_frame()

    df_pstar = _to_pivot("mean_p_star")
    df_actual = _to_pivot("actual_challenge_rate")
    df_gap = _to_pivot("gap")

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
    datasets = [
        (df_pstar, "mean_p_star", "p* (optimal threshold)", "Blues"),
        (df_actual, "actual_challenge_rate", "Actual challenge rate", "Oranges"),
        (df_gap, "gap", "Gap (actual − p*)", "RdBu_r"),
    ]

    for ax, (data, col, label, cmap) in zip(axes, datasets):
        vmin = -0.3 if "gap" in col else 0.0
        vmax = 0.3 if "gap" in col else 1.0
        sns.heatmap(
            data,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            linecolor="lightgray",
            cbar_kws={"label": label},
        )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Count" if ax == axes[0] else "")

    fig.suptitle(
        title or "Challenge Rate: Actual vs. Optimal Threshold (p*) by Count",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_roc_challenge(
    evaluated_df: pd.DataFrame,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    ROC-style curve for challenge accuracy.

    x-axis: false challenge rate (challenged but call was correct, i.e., not overturned).
    y-axis: true challenge rate (challenged and call was wrong, i.e., overturned).

    Parameterized by p̂ cutoff.  Team-specific points overlaid if 'team' column
    present.

    Requires 'overturned' column in evaluated_df.
    """
    if "overturned" not in evaluated_df.columns or "p_hat" not in evaluated_df.columns:
        raise ValueError("evaluated_df must have 'overturned' and 'p_hat' columns.")

    thresholds = np.linspace(0.0, 1.0, 200)
    tprs, fprs = [], []

    y_true = evaluated_df["overturned"].astype(int).values
    y_score = evaluated_df["p_hat"].values

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fprs, tprs, color="steelblue", lw=2, label="p̂ model ROC curve")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.5)")

    # If team column present, plot team-level points.
    if "team" in evaluated_df.columns and "challenged" in evaluated_df.columns:
        for team, grp in evaluated_df.groupby("team"):
            challenged = grp[grp["challenged"].astype(bool)]
            if len(challenged) == 0:
                continue
            tp = challenged["overturned"].astype(int).sum()
            fp = len(challenged) - tp
            total_ov = grp["overturned"].astype(int).sum()
            total_no_ov = len(grp) - total_ov
            tpr_t = tp / total_ov if total_ov > 0 else 0.0
            fpr_t = fp / total_no_ov if total_no_ov > 0 else 0.0
            ax.scatter(fpr_t, tpr_t, s=50, zorder=5, alpha=0.7)
            ax.annotate(team, (fpr_t, tpr_t), fontsize=7, alpha=0.7)

    ax.set_xlabel("False Challenge Rate (challenged correct calls)", fontsize=11)
    ax.set_ylabel("True Challenge Rate (challenged incorrect calls)", fontsize=11)
    ax.set_title(title or "Challenge Accuracy: ROC-Style Curve", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ev_by_state_bucket(
    evaluated_df: pd.DataFrame,
    bucket_col: str = "inning",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart showing average EV gain / loss from actual challenge decisions,
    grouped by a state dimension (inning, outs, count, etc.).
    """
    if "ev_gain" not in evaluated_df.columns:
        raise ValueError("evaluated_df must have 'ev_gain' column.")

    agg = (
        evaluated_df.groupby(bucket_col)["ev_gain"]
        .agg(mean_ev="mean", se=lambda x: x.std() / np.sqrt(len(x)))
        .reset_index()
    )

    colors = ["#27ae60" if v >= 0 else "#c0392b" for v in agg["mean_ev"]]
    fig, ax = plt.subplots(figsize=(max(7, len(agg)), 5))
    ax.bar(agg[bucket_col].astype(str), agg["mean_ev"], color=colors, edgecolor="white")
    ax.axhline(0, color="black", lw=1)
    ax.errorbar(
        range(len(agg)), agg["mean_ev"], yerr=agg["se"],
        fmt="none", color="black", capsize=3, lw=1,
    )
    ax.set_xlabel(bucket_col.replace("_", " ").title(), fontsize=10)
    ax.set_ylabel("Mean EV gain per challenge opportunity", fontsize=10)
    ax.set_title(title or f"Expected WP Gain by {bucket_col}", fontsize=12)
    fig.tight_layout()
    return fig
