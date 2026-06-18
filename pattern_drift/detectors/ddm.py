"""
DDM — Drift Detection Method.

Reference: Gama, Medas, Castillo & Rodrigues (2004), "Learning with Drift
Detection", SBIA 2004.

Algorithm
---------
Track the running error rate p_t and its standard deviation σ_t.  Record the
historical minimum of p_t + σ_t as the stable baseline (p_min + σ_min).
Drift is declared when p_t + σ_t significantly exceeds that baseline.

DDM monitors the *output* of a classifier (prediction correctness) rather
than raw input features — making it the most direct signal that a deployed
model is already underperforming.
"""
from __future__ import annotations

import math
from typing import Tuple

from .base import BaseDetector


class DDM(BaseDetector):
    """
    Parameters
    ----------
    warning_level : float
        p + σ must exceed p_min + σ_min * warning_level to enter warning mode.
    drift_level : float
        p + σ must exceed p_min + σ_min * drift_level to declare drift.
    min_num_instances : int
        Minimum samples before drift can be flagged.
    """

    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_num_instances: int = 30,
    ) -> None:
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_num_instances = min_num_instances
        self.reset()

    # ------------------------------------------------------------------

    def update(self, value: float) -> Tuple[bool, float]:
        """
        Parameters
        ----------
        value : float
            1.0 if the model predicted correctly, 0.0 if incorrectly.
            Any numeric value is accepted; it is treated as a binary error
            signal (value > 0.5 → no error; ≤ 0.5 → error).
        """
        error = 1.0 if value <= 0.5 else 0.0

        self._n += 1
        self._p += (error - self._p) / self._n  # running mean
        self._sigma = math.sqrt(self._p * (1.0 - self._p) / self._n)

        if self._n < self.min_num_instances:
            return False, 0.0

        current = self._p + self._sigma

        if current < self._p_min + self._sigma_min:
            self._p_min = self._p
            self._sigma_min = self._sigma

        # DDM canonical thresholds (Gama et al. 2004):
        #   drift   : p_t + σ_t > p_min + drift_level   * σ_min
        #   (warning: p_t + σ_t > p_min + warning_level * σ_min  — not used here)
        drift_threshold = self._p_min + self.drift_level * self._sigma_min

        # Normalised score relative to the warning threshold for reporting
        warning_threshold = self._p_min + self.warning_level * self._sigma_min
        denominator = max(warning_threshold, 1e-12)
        score = max(0.0, (current - warning_threshold) / denominator)

        if current > drift_threshold:
            self.reset()
            return True, min(score, 1.0)

        return False, min(score, 1.0)

    def reset(self) -> None:
        self._n: int = 0
        self._p: float = 0.0
        self._sigma: float = 0.0
        self._p_min: float = float("inf")
        self._sigma_min: float = float("inf")
