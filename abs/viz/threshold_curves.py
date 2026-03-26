"""
Threshold curve visualizations.

Shows p̂(location_delta) overlaid with p* threshold lines, revealing the
optimal distance cutoff for each game state and count.
"""

from __future__ import annotations

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from abs.config import FIGURE_DPI
from abs.models.overturn_model import OverturnModel
from abs.state import GameState


def plot_location_threshold(
    p_star: float,
    overturn_model: OverturnModel,
    called_type: str = "strike",
    balls: int = 0,
    strikes: int = 0,
    initiator_type: str = "batter",
    delta_range: tuple[float, float] = (-4.0, 4.0),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot the p̂(location_delta) curve with the p* threshold overlaid.

    The shaded region shows where challenging is recommended (p̂ > p*).
    The x-axis crossing marks the optimal distance cutoff in inches.

    Parameters
    ----------
    p_star : float
        Optimal threshold for this game state.
    overturn_model : OverturnModel
        Fitted or parametric overturn probability model.
    called_type : str
        "strike" or "ball".
    balls, strikes : int
        Current count.
    initiator_type : str
        "batter", "catcher", or "pitcher".
    delta_range : tuple
        (min, max) range of location_delta to plot (inches from boundary).
    title : str, optional
        Custom title.
    ax : plt.Axes, optional
        Axes to draw on (creates a new figure if None).
    """
    deltas, probs = overturn_model.location_curve(
        called_type=called_type,
        balls=balls,
        strikes=strikes,
        initiator_type=initiator_type,
        delta_range=delta_range,
    )
    cutoff = overturn_model.optimal_cutoff(p_star, called_type, balls, strikes, initiator_type)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.get_figure()

    ax.plot(deltas, probs, color="steelblue", lw=2.5, label="p̂(location_delta)")
    ax.axhline(p_star, color="crimson", linestyle="--", lw=1.8,
               label=f"p* = {p_star:.3f}")

    # Shade challenge zone (p̂ > p*).
    ax.fill_between(
        deltas, probs, p_star,
        where=(probs > p_star),
        alpha=0.18,
        color="green",
        label="Challenge zone (p̂ > p*)",
    )
    ax.fill_between(
        deltas, probs, p_star,
        where=(probs <= p_star),
        alpha=0.08,
        color="red",
        label="Save challenge (p̂ ≤ p*)",
    )

    if cutoff is not None:
        ax.axvline(cutoff, color="gray", linestyle=":", lw=1.5,
                   label=f"Cutoff = {cutoff:.2f} in.")
        ax.annotate(
            f"{cutoff:.2f}\"",
            xy=(cutoff, p_star),
            xytext=(cutoff + 0.2, p_star + 0.08),
            fontsize=9,
            color="gray",
        )

    ax.set_xlabel("Distance from ABS boundary (inches)\n← Favorable to challenger  |  Unfavorable →", fontsize=10)
    ax.set_ylabel("Overturn probability", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(*delta_range)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    default_title = (
        f"Challenge Decision: p̂ vs p* | {balls}-{strikes} count, "
        f"{called_type} challenged by {initiator_type}"
    )
    ax.set_title(title or default_title, fontsize=11)

    if standalone:
        fig.tight_layout()
    return fig


def plot_location_threshold_by_count(
    p_star_by_count: dict[tuple[int, int], float],
    overturn_model: OverturnModel,
    called_type: str = "strike",
    initiator_type: str = "batter",
    counts_to_show: Optional[list[tuple[int, int]]] = None,
    delta_range: tuple[float, float] = (-4.0, 4.0),
) -> plt.Figure:
    """
    Multi-panel figure showing how the optimal cutoff shifts by count.

    Parameters
    ----------
    p_star_by_count : dict
        {(balls, strikes): p_star} mapping.
    overturn_model : OverturnModel
    counts_to_show : list of (balls, strikes), optional
        Subset of counts to display. Defaults to all keys in p_star_by_count.
    """
    if counts_to_show is None:
        counts_to_show = sorted(p_star_by_count.keys())

    n = len(counts_to_show)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (balls, strikes) in zip(axes_flat, counts_to_show):
        p_star = p_star_by_count.get((balls, strikes), 0.5)
        plot_location_threshold(
            p_star=p_star,
            overturn_model=overturn_model,
            called_type=called_type,
            balls=balls,
            strikes=strikes,
            initiator_type=initiator_type,
            delta_range=delta_range,
            title=f"{balls}-{strikes} count\np*={p_star:.3f}",
            ax=ax,
        )

    # Hide unused axes.
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Optimal Challenge Cutoff by Count ({called_type} challenged)",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_ev_gain_curve(
    threshold_model,
    state: GameState,
    overturn_model: OverturnModel,
    delta_range: tuple[float, float] = (-4.0, 4.0),
    n_points: int = 200,
) -> plt.Figure:
    """
    Plot expected WP gain from challenging as a function of location_delta.

    Zero-crossing marks the boundary between optimal challenge / save.
    """
    called_type = "strike" if state.challenger_type == "batter" else "ball"
    deltas, probs = overturn_model.location_curve(
        called_type=called_type,
        balls=state.balls,
        strikes=state.strikes,
        delta_range=delta_range,
        n_points=n_points,
    )

    if not hasattr(threshold_model, "ev_gain"):
        raise AttributeError("threshold_model must have an ev_gain(state, p) method.")

    ev_gains = np.array([threshold_model.ev_gain(state, p) for p in probs])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(deltas, ev_gains, color="darkgreen", lw=2.5)
    ax.axhline(0, color="black", lw=1)
    ax.fill_between(deltas, ev_gains, 0, where=(ev_gains > 0),
                    alpha=0.2, color="green", label="Challenge is EV+")
    ax.fill_between(deltas, ev_gains, 0, where=(ev_gains <= 0),
                    alpha=0.1, color="red", label="Save challenge")
    ax.set_xlabel("Distance from ABS boundary (inches)", fontsize=10)
    ax.set_ylabel("Expected WP gain from challenging", fontsize=10)
    ax.set_title(
        f"EV Gain from Challenging | Inning {state.inning}, "
        f"{state.half}, score {state.score_diff:+}, {state.outs} outs, "
        f"{state.balls}-{state.strikes}, k={state.challenges_remaining}",
        fontsize=11,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
