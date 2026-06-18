"""
Page-Hinkley drift detector.

Reference: E.S. Page (1954), "Continuous Inspection Schemes", Biometrika.

Algorithm
---------
Maintain a running mean and a cumulative sum (U_t) of deviations from that
mean.  U_t is defined as the difference between the cumulative sum of
deviations and the observed minimum of that cumulative sum up to time t.
Drift is declared when U_t exceeds the threshold λ.

This implementation is highly memory-efficient: it stores only three scalars
per feature regardless of stream length.
"""
from __future__ import annotations

from typing import Tuple

from .base import BaseDetector


class PageHinkley(BaseDetector):
    """
    Parameters
    ----------
    delta : float
        Minimum detectable change magnitude (maps to ``sensitivity``).
        Lower = more sensitive to small shifts.
    lambda_ : float
        Detection threshold.  Larger values reduce false positives at the
        cost of slower detection.
    alpha : float
        Forgetting factor for the running mean (1.0 = no forgetting).
    """

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50.0,
        alpha: float = 1.0,
    ) -> None:
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self._n: int = 0
        self._mean: float = 0.0
        self._cumsum: float = 0.0
        self._min_cumsum: float = float("inf")

    # ------------------------------------------------------------------

    def update(self, value: float) -> Tuple[bool, float]:
        self._n += 1
        # Update running mean (with optional forgetting)
        self._mean = self.alpha * self._mean + (1.0 - self.alpha) * value if self._n > 1 else value
        if self._n == 1:
            self._mean = value

        deviation = value - self._mean - self.delta
        self._cumsum += deviation

        if self._cumsum < self._min_cumsum:
            self._min_cumsum = self._cumsum

        U = self._cumsum - self._min_cumsum
        score = U / (self.lambda_ + 1e-12)

        if U > self.lambda_:
            self.reset()
            return True, min(score, 1.0)

        return False, min(score, 1.0)

    def reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._cumsum = 0.0
        self._min_cumsum = float("inf")
