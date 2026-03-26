"""
ABS Challenge Strategy Framework.

Two linked models:
  - p*(s): optimal challenge threshold by game state (simplified + DP versions)
  - p̂(pitch): estimated overturn probability from pitch-level features

Decision rule: challenge whenever p̂ > p*(s).
"""

__version__ = "0.1.0"
