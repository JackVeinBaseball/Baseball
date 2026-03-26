"""
Global constants and tunables for the ABS challenge strategy framework.
All magic numbers live here — nothing hardcoded elsewhere.
"""

# ── Game rules ────────────────────────────────────────────────────────────────
INNINGS_MAX: int = 9          # Inning 9 represents all extra innings as well
SCORE_DIFF_CLAMP: int = 10    # Score differential clamped to [-10, +10]
MAX_CHALLENGES: int = 2       # Teams start with 2 challenges per game

BALLS_MAX: int = 3            # 0-3; ball 4 → walk
STRIKES_MAX: int = 2          # 0-2; strike 3 → strikeout
OUTS_MAX: int = 2             # 0-2; out 3 → half-inning ends

# ── Challenger types ──────────────────────────────────────────────────────────
CHALLENGER_BATTER: str = "batter"      # Offense; challenges a called strike
CHALLENGER_DEFENSE: str = "defense"   # Catcher or pitcher; challenges a called ball

# ── DP solver ─────────────────────────────────────────────────────────────────
DP_CONVERGENCE_TOL: float = 1e-9   # Threshold below which WP differences are ignored

# ── Logistic WP model default coefficients ────────────────────────────────────
# Fitted to approximate Tango/MGL WP tables; used when CSV is not loaded.
# Feature vector: [1, score_diff, score_diff^2, inning_urgency, is_bottom, outs, runners, score*inning]
LOGISTIC_WP_COEFFS: dict[str, float] = {
    "intercept": 0.0,
    "score_diff": 0.38,
    "score_diff_sq": -0.012,
    "inning_urgency": 0.0,      # late-game coefficient; calibrated to 0 for symmetry
    "is_bottom": 0.0,
    "outs": -0.05,
    "runners": 0.02,
    "score_x_inning": 0.015,
}

# ── Overturn model default parametric coefficients ───────────────────────────
# p̂ = sigmoid(β₀ + β₁·location_delta + β₂·is_called_strike + β₃·count_leverage)
# location_delta: signed distance from ABS boundary in inches;
#   negative = clearly wrong call (favorable to challenger), positive = borderline
OVERTURN_COEFFS: dict[str, float] = {
    "intercept": 0.0,
    "location_delta": -1.8,     # more negative delta → higher overturn prob
    "is_called_strike": 0.1,    # slight asymmetry: CS challenges slightly easier
    "count_leverage": 0.05,     # higher-leverage counts get marginal boost
}

# ── Visualization ─────────────────────────────────────────────────────────────
FIGURE_DPI: int = 150
FIGURE_FORMAT: str = "png"

# Base state labels for display (bitmask: bit0=1B, bit1=2B, bit2=3B)
BASE_STATE_LABELS: dict[int, str] = {
    0b000: "---",
    0b001: "1--",
    0b010: "-2-",
    0b011: "12-",
    0b100: "--3",
    0b101: "1-3",
    0b110: "-23",
    0b111: "123",
}

# Count labels: (balls, strikes) → display string
def count_label(balls: int, strikes: int) -> str:
    return f"{balls}-{strikes}"
