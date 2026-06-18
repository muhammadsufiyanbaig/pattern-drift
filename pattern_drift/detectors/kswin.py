"""
KSWIN — Kolmogorov-Smirnov Windowed drift detector.

Reference: Raab, Heusinger & Schleif (2020), "Reactive Soft Prototype
Computing for Concept Drift Streams", Neurocomputing 416:340-351.

Algorithm
---------
Maintain a sliding *recent* window and a fixed *reference* window.  On each
update, run the two-sample Kolmogorov-Smirnov test between the two windows.
If the p-value is below the significance threshold, drift is declared.

Unlike ADWIN and Page-Hinkley, KSWIN is sensitive to *any* distributional
difference — mean, variance, skewness, tail behaviour — making it the right
choice when the shape of the distribution changes rather than just its mean.
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Sequence, Tuple

from .base import BaseDetector


class KSWIN(BaseDetector):
    """
    Parameters
    ----------
    alpha : float
        Significance level for the KS test (maps to ``sensitivity``).
        Typical range 0.001–0.05.
    window_size : int
        Total sliding window length.
    stat_size : int
        Size of the *recent* sub-window used in the KS test.
        Must be < window_size.
    reference : list[float] | None
        Optional pre-seeded reference distribution.  Updated via
        ``set_reference()`` on the parent monitor.
    """

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        reference: Optional[Sequence[float]] = None,
    ) -> None:
        if stat_size >= window_size:
            raise ValueError("stat_size must be smaller than window_size")
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self._window: deque[float] = deque(maxlen=window_size)
        self._reference: Optional[list[float]] = list(reference) if reference else None

    # ------------------------------------------------------------------

    def update(self, value: float) -> Tuple[bool, float]:
        self._window.append(value)

        if len(self._window) < self.window_size:
            return False, 0.0

        window_list = list(self._window)
        reference = self._reference if self._reference else window_list[: self.window_size - self.stat_size]
        recent = window_list[-self.stat_size :]

        stat, p_value = self._ks_test(reference, recent)

        score = max(0.0, 1.0 - p_value)  # high score → low p-value → more drift

        if p_value < self.alpha:
            # Slide the reference forward to the beginning of the stable window
            self._reference = None  # reset; will recalculate next round
            return True, score

        return False, score

    def reset(self) -> None:
        self._window.clear()
        self._reference = None

    def set_reference(self, data: Sequence[float]) -> None:
        """Manually pin the reference distribution."""
        self._reference = list(data)

    # ------------------------------------------------------------------

    @staticmethod
    def _ks_test(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
        """
        Two-sample KS test implemented without scipy dependency.

        Returns (statistic, p_value).
        The p-value approximation uses the Kolmogorov distribution.
        """
        import math

        a_sorted = sorted(a)
        b_sorted = sorted(b)
        n1 = len(a_sorted)
        n2 = len(b_sorted)

        # Build merged sorted sequence
        combined = sorted(set(a_sorted + b_sorted))
        max_diff = 0.0
        i, j = 0, 0
        for x in combined:
            while i < n1 and a_sorted[i] <= x:
                i += 1
            while j < n2 and b_sorted[j] <= x:
                j += 1
            diff = abs(i / n1 - j / n2)
            if diff > max_diff:
                max_diff = diff

        # Kolmogorov approximation for p-value
        en = math.sqrt(n1 * n2 / (n1 + n2))
        z = (en + 0.12 + 0.11 / en) * max_diff
        p_value = KSWIN._kolmogorov_p(z)
        return max_diff, p_value

    @staticmethod
    def _kolmogorov_p(z: float) -> float:
        """Approximate p-value from the Kolmogorov distribution."""
        import math

        if z < 0.2:
            return 1.0
        if z > 5.0:
            return 0.0

        p = 0.0
        for k in range(1, 101):
            sign = (-1) ** (k + 1)
            p += sign * math.exp(-2.0 * k * k * z * z)
        return max(0.0, min(1.0, 2.0 * p))
