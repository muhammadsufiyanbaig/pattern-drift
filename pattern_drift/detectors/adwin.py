"""
ADWIN — Adaptive Windowing drift detector.

Reference: Bifet & Gavaldà (2007), "Learning from Time-Changing Data with
Adaptive Windowing".

Algorithm
---------
Maintain a variable-length window W of recent values.  On every new
observation, attempt to split W at every possible cut-point.  A cut is
significant when the absolute difference between the left and right sub-window
means exceeds epsilon_cut (derived from Hoeffding's bound).  When a
significant cut is found the older (left) portion of the window is discarded
and drift is signalled.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Tuple

from .base import BaseDetector


class ADWIN(BaseDetector):
    """
    Parameters
    ----------
    delta : float
        Significance level (maps to the library's ``sensitivity`` parameter).
        Lower values make the detector *less* sensitive (fewer false positives).
        Typical range 0.001–0.1.
    max_window : int
        Hard cap on the number of observations retained in the window.
    """

    def __init__(self, delta: float = 0.002, max_window: int = 10_000) -> None:
        self.delta = delta
        self.max_window = max_window
        self._window: deque[float] = deque()
        self._total: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, value: float) -> Tuple[bool, float]:
        self._window.append(value)
        self._total += value

        if len(self._window) > self.max_window:
            removed = self._window.popleft()
            self._total -= removed

        drift_detected, score = self._check_drift()
        return drift_detected, score

    def reset(self) -> None:
        self._window.clear()
        self._total = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_drift(self) -> Tuple[bool, float]:
        n = len(self._window)
        if n < 2:
            return False, 0.0

        window_list = list(self._window)
        total = self._total
        best_score = 0.0

        left_sum = 0.0
        for i in range(1, n):
            left_sum += window_list[i - 1]
            right_sum = total - left_sum

            n0 = i
            n1 = n - i
            m0 = left_sum / n0
            m1 = right_sum / n1

            epsilon_cut = self._epsilon(n0, n1, n)
            diff = abs(m0 - m1)
            score = diff / (epsilon_cut + 1e-12)

            if score > best_score:
                best_score = score

            if diff >= epsilon_cut:
                # Discard left (older) portion — keep only the right sub-window
                self._window = deque(window_list[i:])
                self._total = right_sum
                normalised = min(score / 10.0, 1.0)
                return True, normalised

        return False, min(best_score / 10.0, 1.0)

    def _epsilon(self, n0: int, n1: int, n: int) -> float:
        """Hoeffding-based significance threshold for the cut point."""
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        return math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 * n / self.delta))
