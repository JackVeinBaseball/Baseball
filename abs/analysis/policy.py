"""
ChallengePolicy: links the threshold model (p*) and the overturn model (p̂).

Decision rule: challenge whenever p̂ > p*(s).

This module is the top-level inference interface; downstream consumers
(scripts, notebooks, calibration analysis) call ChallengePolicy rather
than the individual model objects.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from abs.models.overturn_model import OverturnModel
from abs.models.simplified import SimplifiedThresholdModel
from abs.state import GameState


class ChallengePolicy:
    """
    Combines a threshold model and an overturn model into an actionable policy.

    Parameters
    ----------
    threshold_model : SimplifiedThresholdModel or DPSolver
        Provides p*(s) for a given GameState.
    overturn_model : OverturnModel
        Provides p̂(pitch) for pitch-level features.
    """

    def __init__(
        self,
        threshold_model: Union["SimplifiedThresholdModel", "DPSolver"],  # noqa: F821
        overturn_model: Optional[OverturnModel] = None,
    ) -> None:
        self.threshold_model = threshold_model
        self.overturn_model = overturn_model if overturn_model is not None else OverturnModel()

    def p_star(self, state: GameState) -> Optional[float]:
        """Return the optimal challenge threshold for this state."""
        return self.threshold_model.threshold(state)

    def p_hat(
        self,
        location_delta: float,
        called_type: str,
        balls: int,
        strikes: int,
        initiator_type: str,
    ) -> float:
        """Return the estimated overturn probability for this pitch."""
        return self.overturn_model.predict_single(
            location_delta, called_type, balls, strikes, initiator_type
        )

    def should_challenge(
        self,
        state: GameState,
        location_delta: float,
        initiator_type: str,
    ) -> dict:
        """
        Evaluate whether to challenge a specific pitch in a specific game state.

        Parameters
        ----------
        state : GameState
            Current game state (includes challenger_type, challenges_remaining).
        location_delta : float
            Signed distance from ABS boundary in inches.
        initiator_type : str
            "batter", "catcher", or "pitcher".

        Returns
        -------
        dict with keys:
            p_hat : float       — estimated overturn probability
            p_star : float|None — optimal threshold (None if k=0)
            recommend : bool    — True = challenge
            ev_gain : float     — expected WP gain from challenging at p̂
            reason : str        — human-readable explanation
        """
        called_type = "strike" if state.challenger_type == "batter" else "ball"
        p_h = self.p_hat(location_delta, called_type, state.balls, state.strikes, initiator_type)
        p_s = self.p_star(state)

        if p_s is None:
            return {
                "p_hat": p_h,
                "p_star": None,
                "recommend": False,
                "ev_gain": 0.0,
                "reason": "No challenges remaining.",
            }

        recommend = p_h > p_s

        # EV gain calculation.
        if hasattr(self.threshold_model, "ev_gain"):
            ev = self.threshold_model.ev_gain(state, p_h)
        else:
            ev = float("nan")

        if recommend:
            gap = p_h - p_s
            reason = (
                f"Challenge recommended: p̂={p_h:.3f} > p*={p_s:.3f} "
                f"(margin={gap:.3f})"
            )
        else:
            gap = p_s - p_h
            reason = (
                f"Save challenge: p̂={p_h:.3f} < p*={p_s:.3f} "
                f"(margin={gap:.3f})"
            )

        return {
            "p_hat": p_h,
            "p_star": p_s,
            "recommend": recommend,
            "ev_gain": ev,
            "reason": reason,
        }

    def evaluate_decisions(self, decisions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch-evaluate historical or hypothetical challenge decisions.

        Parameters
        ----------
        decisions_df : pd.DataFrame
            Required columns:
              - inning, half, score_diff, outs, base_state, balls, strikes,
                challenges_remaining, challenger_type (for GameState)
              - location_delta, initiator_type (for p̂)
            Optional columns:
              - challenged (bool): whether a challenge was actually taken
              - overturned (bool): whether the challenge succeeded

        Returns
        -------
        pd.DataFrame
            Original DataFrame with added columns:
            p_hat, p_star, recommend, ev_gain, was_optimal (if challenged present).
        """
        results = []
        for row in decisions_df.itertuples(index=False):
            try:
                state = GameState(
                    inning=int(row.inning),
                    half=str(row.half),
                    score_diff=int(row.score_diff),
                    outs=int(row.outs),
                    base_state=int(row.base_state),
                    balls=int(row.balls),
                    strikes=int(row.strikes),
                    challenges_remaining=int(row.challenges_remaining),
                    challenger_type=str(row.challenger_type),
                )
            except Exception as e:
                results.append({"p_hat": np.nan, "p_star": np.nan,
                                 "recommend": None, "ev_gain": np.nan,
                                 "_error": str(e)})
                continue

            decision = self.should_challenge(
                state,
                float(row.location_delta),
                str(row.initiator_type),
            )
            results.append({
                "p_hat": decision["p_hat"],
                "p_star": decision["p_star"],
                "recommend": decision["recommend"],
                "ev_gain": decision["ev_gain"],
            })

        result_df = pd.DataFrame(results)
        out = pd.concat(
            [decisions_df.reset_index(drop=True), result_df.reset_index(drop=True)],
            axis=1,
        )

        # Classify decisions as optimal/suboptimal if actual decisions are known.
        if "challenged" in out.columns:
            out["was_optimal"] = (
                (out["challenged"].astype(bool) == out["recommend"])
                | out["recommend"].isna()
            )
            if "overturned" in out.columns:
                out["overchallenged"] = (
                    out["challenged"].astype(bool) & ~out["recommend"]
                )
                out["underchallenged"] = (
                    ~out["challenged"].astype(bool) & out["recommend"]
                )

        return out
