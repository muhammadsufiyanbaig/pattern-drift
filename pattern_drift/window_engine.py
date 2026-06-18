"""
Retraining Window Engine — Stage 4 of the internal pipeline.

When drift is detected, this engine scans backward through the monitor's
recorded drift-score history to find the most recent continuous segment where
*all* feature scores were below the sensitivity threshold.

A configurable buffer (default 10 %) is trimmed from each end of the
suggested window to avoid edge effects.  A confidence score is returned
alongside the window bounds.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .result import RetrainingWindowResult


class RetrainingWindowEngine:
    """
    Parameters
    ----------
    sensitivity : float
        Drift score threshold; scores below this are considered "stable".
    buffer_pct : float
        Fraction of the stable window to trim from each end (default 0.10).
    """

    def __init__(self, sensitivity: float = 0.002, buffer_pct: float = 0.10) -> None:
        self.sensitivity = sensitivity
        self.buffer_pct = buffer_pct

    # ------------------------------------------------------------------

    def find_window(
        self,
        score_history: List[Dict[str, float]],
        drift_index: int,
    ) -> Optional[RetrainingWindowResult]:
        """
        Parameters
        ----------
        score_history : list of {feature: score} dicts, one per observation.
        drift_index   : index in score_history where drift was detected.

        Returns
        -------
        RetrainingWindowResult or None if no stable window found.
        """
        if drift_index <= 0 or not score_history:
            return None

        # Walk backward from the drift point to find the last stable window
        stable_end = drift_index - 1
        stable_start = stable_end

        for i in range(drift_index - 1, -1, -1):
            scores = score_history[i]
            if not scores:
                continue
            max_score = max(scores.values())
            if max_score < self.sensitivity:
                stable_start = i
            else:
                # First unstable point — stop
                break

        n_stable = stable_end - stable_start + 1
        if n_stable < 2:
            return None

        # Apply buffer
        buf = max(1, int(n_stable * self.buffer_pct))
        window_start = stable_start + buf
        window_end = stable_end - buf

        if window_start >= window_end:
            return None

        n_samples = window_end - window_start + 1

        # Confidence = fraction of stable samples within the raw stable region
        stable_count = sum(
            1
            for i in range(stable_start, stable_end + 1)
            if score_history[i] and max(score_history[i].values()) < self.sensitivity
        )
        confidence = stable_count / n_stable if n_stable > 0 else 0.0

        return RetrainingWindowResult(
            start=window_start,
            end=window_end,
            n_samples=n_samples,
            confidence=round(confidence, 4),
        )
