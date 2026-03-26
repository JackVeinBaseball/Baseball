"""
Pure state-transition functions for the ABS challenge framework.

Every function here is a pure function: given a GameState, return a new
GameState.  No mutation, no I/O.  These are called millions of times during
the DP solve, so they are kept simple and fast.
"""

from __future__ import annotations

from dataclasses import replace

from abs.config import (
    BALLS_MAX,
    CHALLENGER_BATTER,
    INNINGS_MAX,
    SCORE_DIFF_CLAMP,
    STRIKES_MAX,
)
from abs.state import GameState


# ── Internal helpers ──────────────────────────────────────────────────────────


def _clamp_score(score_diff: int) -> int:
    return max(-SCORE_DIFF_CLAMP, min(SCORE_DIFF_CLAMP, score_diff))


def _replace(s: GameState, **kwargs) -> GameState:
    """Wrapper around dataclasses.replace that clamps score_diff."""
    if "score_diff" in kwargs:
        kwargs["score_diff"] = _clamp_score(kwargs["score_diff"])
    return replace(s, **kwargs)


# ── Walk / Strikeout logic ────────────────────────────────────────────────────


def apply_walk(s: GameState) -> GameState:
    """
    Force-advance runners due to a walk (balls == 3, another ball called).

    Rules:
    - Runner on 1B is pushed forward if forced.
    - If bases are loaded, a run scores → offense score_diff +1.
    - Reset count to 0-0; put batter on 1B.
    """
    base = s.base_state
    score_delta = 0

    # Determine new base_state after walk.
    # Bit 0 = 1B, bit 1 = 2B, bit 2 = 3B.
    # Force advance: if 1B occupied, push to 2B (if 2B occupied, push to 3B,
    # and so on). Easiest to simulate by checking from the back.

    new_base = base | 0b001  # batter takes 1B

    if base & 0b001:          # runner on 1B was already there → force to 2B
        new_base = new_base | 0b010
        if base & 0b010:      # runner on 2B → force to 3B
            new_base = new_base | 0b100
            if base & 0b100:  # runner on 3B → scores
                new_base = new_base & ~0b100  # clear 3B (runner scored)
                score_delta = 1

    # Reset count and apply score change.
    return _replace(s, balls=0, strikes=0, base_state=new_base & 0b111,
                    score_diff=s.score_diff + score_delta)


def apply_strikeout(s: GameState) -> GameState:
    """
    Record an out due to strikeout (strikes == 2, another strike called).
    Does not advance runners (no dropped-third-strike logic for simplicity).
    Returns state after incrementing outs; may trigger advance_half_inning.
    """
    new_outs = s.outs + 1
    if new_outs >= 3:
        return advance_half_inning(s)
    return _replace(s, outs=new_outs, balls=0, strikes=0)


# ── Count advancement ─────────────────────────────────────────────────────────


def advance_count_ball(s: GameState) -> GameState:
    """
    A ball is called (or a called strike is overturned to ball).
    If balls == BALLS_MAX (3), result is a walk; otherwise increment balls.
    """
    if s.balls == BALLS_MAX:
        return apply_walk(s)
    return _replace(s, balls=s.balls + 1)


def advance_count_strike(s: GameState) -> GameState:
    """
    A strike is called (or a called ball is overturned to strike).
    If strikes == STRIKES_MAX (2), result is a strikeout; otherwise increment.
    """
    if s.strikes == STRIKES_MAX:
        return apply_strikeout(s)
    return _replace(s, strikes=s.strikes + 1)


# ── Half-inning advancement ───────────────────────────────────────────────────


def advance_half_inning(s: GameState) -> GameState:
    """
    When 3 outs are recorded: flip half-inning, reset bases/count/outs.
    If bottom of inning 9+ completes, the game is over — return a terminal
    state (inning=9, outs=3 is signalled by outs=0, half unchanged but
    inning stays at 9 so the DP can detect it as terminal via score_diff).

    Convention for terminal state:
      - inning = INNINGS_MAX
      - half = "bottom"
      - outs = 0  (reset)
      - base_state = 0
      - balls = 0, strikes = 0
      - score_diff unchanged (used to compute final WP)
    The DP's terminal-value assignment looks for this combination.
    """
    reset_kwargs = dict(outs=0, base_state=0, balls=0, strikes=0)

    if s.half == "top":
        # Top half done → bottom half of same inning begins.
        # From offense POV: away team just batted; now home team bats.
        # score_diff flips sign because the offensive team switches.
        return _replace(s, half="bottom", score_diff=-s.score_diff, **reset_kwargs)
    else:
        # Bottom half done → top of next inning.
        next_inning = min(s.inning + 1, INNINGS_MAX)
        # score_diff flips again (home finished batting, away bats next).
        return _replace(s, inning=next_inning, half="top",
                        score_diff=-s.score_diff, **reset_kwargs)


def is_terminal(s: GameState) -> bool:
    """
    Return True when the game has ended.

    A game ends after the bottom of inning 9 (or extra innings) if the home
    team is not tied or trailing.  We model extra innings by clamping inning
    to 9 and continuing; the game terminates when outs=0 and half="top" at
    inning=9 after the bottom half completes *and* the score is decided.

    Simplified terminal condition used by the DP:
    - We treat the *start* of a new top-of-inning-9+ as a terminal boundary
      and assign WP from the WP table (which handles end-of-game correctly).
    - Additionally, a walk-off in the bottom half (score_diff > 0 for the
      home team, i.e., score_diff < 0 from the *offensive* perspective when
      home is batting) ends the game immediately.

    In practice, the DP assigns boundary values at inning=INNINGS_MAX from
    the WP table and does not recurse further; this function marks those.
    """
    return s.inning == INNINGS_MAX and s.half == "top" and s.outs == 0 and s.balls == 0 and s.strikes == 0


# ── Challenge outcome transitions ─────────────────────────────────────────────


def state_after_overturn(s: GameState) -> GameState:
    """
    The challenged call is overturned.

    - Batter challenged a called strike → overturned to ball.
    - Defense challenged a called ball → overturned to strike.
    Challenges remaining is *not* decremented (successful challenge retained).
    """
    if s.challenger_type == CHALLENGER_BATTER:
        # Called strike → ball.
        return advance_count_ball(
            _replace(s, challenges_remaining=s.challenges_remaining)
        )
    else:
        # Called ball → strike.
        return advance_count_strike(
            _replace(s, challenges_remaining=s.challenges_remaining)
        )


def state_after_failed_challenge(s: GameState) -> GameState:
    """
    The challenge fails: original call stands and one challenge is lost.
    """
    if s.challenges_remaining == 0:
        raise ValueError("Cannot challenge with 0 challenges remaining.")
    original_call_state = state_after_no_challenge(s)
    return _replace(original_call_state,
                    challenges_remaining=s.challenges_remaining - 1)


def state_after_no_challenge(s: GameState) -> GameState:
    """
    No challenge taken: original call stands, count advances as called.

    - Batter did not challenge a called strike → advance_count_strike.
    - Defense did not challenge a called ball → advance_count_ball.
    Challenges remaining is unchanged.
    """
    if s.challenger_type == CHALLENGER_BATTER:
        return advance_count_strike(s)
    else:
        return advance_count_ball(s)
