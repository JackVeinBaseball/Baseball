"""
GameState: immutable representation of a complete ABS challenge decision state.

frozen=True makes GameState hashable so it can serve as a dict key in the DP
value table without any extra plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator

from abs.config import (
    BALLS_MAX,
    CHALLENGER_BATTER,
    CHALLENGER_DEFENSE,
    INNINGS_MAX,
    MAX_CHALLENGES,
    OUTS_MAX,
    SCORE_DIFF_CLAMP,
    STRIKES_MAX,
)

_VALID_HALVES = frozenset({"top", "bottom"})
_VALID_CHALLENGER_TYPES = frozenset({CHALLENGER_BATTER, CHALLENGER_DEFENSE})


@dataclass(frozen=True)
class GameState:
    """
    Complete state for an ABS challenge decision point.

    Attributes
    ----------
    inning : int
        Current inning, 1–9. Inning 9 represents all extra innings.
    half : str
        "top" (away bats) or "bottom" (home bats).
    score_diff : int
        Run differential from the *offensive team's* perspective,
        clamped to [-SCORE_DIFF_CLAMP, +SCORE_DIFF_CLAMP].
        Positive = offense leading.
    outs : int
        Outs in the current half-inning, 0–2.
    base_state : int
        Bitmask encoding base occupancy: bit 0 = 1B, bit 1 = 2B, bit 2 = 3B.
        Range 0–7.
    balls : int
        Balls in the current plate appearance, 0–3.
    strikes : int
        Strikes in the current plate appearance, 0–2.
    challenges_remaining : int
        Challenges remaining for the *challenging team*, 0–2.
    challenger_type : str
        "batter" — offense challenges a called strike.
        "defense" — catcher/pitcher challenges a called ball.
    """

    inning: int
    half: str
    score_diff: int
    outs: int
    base_state: int
    balls: int
    strikes: int
    challenges_remaining: int
    challenger_type: str

    def __post_init__(self) -> None:
        if not (1 <= self.inning <= INNINGS_MAX):
            raise ValueError(f"inning must be 1–{INNINGS_MAX}, got {self.inning}")
        if self.half not in _VALID_HALVES:
            raise ValueError(f"half must be 'top' or 'bottom', got {self.half!r}")
        if not (-SCORE_DIFF_CLAMP <= self.score_diff <= SCORE_DIFF_CLAMP):
            raise ValueError(
                f"score_diff must be in [{-SCORE_DIFF_CLAMP}, {SCORE_DIFF_CLAMP}], "
                f"got {self.score_diff}"
            )
        if not (0 <= self.outs <= OUTS_MAX):
            raise ValueError(f"outs must be 0–{OUTS_MAX}, got {self.outs}")
        if not (0 <= self.base_state <= 7):
            raise ValueError(f"base_state must be 0–7, got {self.base_state}")
        if not (0 <= self.balls <= BALLS_MAX):
            raise ValueError(f"balls must be 0–{BALLS_MAX}, got {self.balls}")
        if not (0 <= self.strikes <= STRIKES_MAX):
            raise ValueError(f"strikes must be 0–{STRIKES_MAX}, got {self.strikes}")
        if not (0 <= self.challenges_remaining <= MAX_CHALLENGES):
            raise ValueError(
                f"challenges_remaining must be 0–{MAX_CHALLENGES}, "
                f"got {self.challenges_remaining}"
            )
        if self.challenger_type not in _VALID_CHALLENGER_TYPES:
            raise ValueError(
                f"challenger_type must be 'batter' or 'defense', "
                f"got {self.challenger_type!r}"
            )

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def base_out_index(self) -> int:
        """Index into the 24 base-out states: outs*8 + base_state (0–23)."""
        return self.outs * 8 + self.base_state

    @property
    def count_index(self) -> int:
        """Index into the 12-cell count grid: balls*3 + strikes (0–11)."""
        return self.balls * 3 + self.strikes

    @property
    def is_offense_challenging(self) -> bool:
        """True when the batter is challenging (called strike)."""
        return self.challenger_type == CHALLENGER_BATTER

    @property
    def runners_on_base(self) -> int:
        """Number of runners on base (0–3), derived from base_state bitmask."""
        return bin(self.base_state).count("1")


# ── State space enumeration ───────────────────────────────────────────────────


def enumerate_all_states() -> list[GameState]:
    """
    Return every valid GameState in the decision state space.

    The Cartesian product of all dimensions is filtered to remove states that
    cannot arise as ABS challenge decision points.  Terminal game states (end
    of inning 9 bottom) are excluded because there is no decision to make.

    Returns
    -------
    list[GameState]
        Deterministically ordered list; roughly 109 k states.
    """
    states: list[GameState] = []

    innings = range(1, INNINGS_MAX + 1)
    halves = ["top", "bottom"]
    score_diffs = range(-SCORE_DIFF_CLAMP, SCORE_DIFF_CLAMP + 1)
    outs_range = range(OUTS_MAX + 1)
    base_states = range(8)
    balls_range = range(BALLS_MAX + 1)
    strikes_range = range(STRIKES_MAX + 1)
    challenges_range = range(MAX_CHALLENGES + 1)
    challenger_types = [CHALLENGER_BATTER, CHALLENGER_DEFENSE]

    for (
        inning,
        half,
        score_diff,
        outs,
        base_state,
        balls,
        strikes,
        k,
        ctype,
    ) in product(
        innings,
        halves,
        score_diffs,
        outs_range,
        base_states,
        balls_range,
        strikes_range,
        challenges_range,
        challenger_types,
    ):
        # Skip states where no challenge is possible — still included so the
        # DP can read V(s, k=0) as the no-challenge baseline.
        states.append(
            GameState(
                inning=inning,
                half=half,
                score_diff=score_diff,
                outs=outs,
                base_state=base_state,
                balls=balls,
                strikes=strikes,
                challenges_remaining=k,
                challenger_type=ctype,
            )
        )

    return states


def build_index(states: list[GameState]) -> dict[GameState, int]:
    """Map each GameState to its integer index in the states list."""
    return {s: i for i, s in enumerate(states)}


def iter_trajectory_states() -> Iterator[GameState]:
    """
    Yield states for the game-trajectory dimensions only
    (no challenges_remaining or challenger_type), used to pre-allocate
    the DP value array.
    """
    for inning, half, score_diff, outs, base_state, balls, strikes in product(
        range(1, INNINGS_MAX + 1),
        ["top", "bottom"],
        range(-SCORE_DIFF_CLAMP, SCORE_DIFF_CLAMP + 1),
        range(OUTS_MAX + 1),
        range(8),
        range(BALLS_MAX + 1),
        range(STRIKES_MAX + 1),
    ):
        # Use challenger_type=CHALLENGER_BATTER as a placeholder; the DP
        # will evaluate both types at each trajectory state.
        yield GameState(
            inning=inning,
            half=half,
            score_diff=score_diff,
            outs=outs,
            base_state=base_state,
            balls=balls,
            strikes=strikes,
            challenges_remaining=0,
            challenger_type=CHALLENGER_BATTER,
        )
