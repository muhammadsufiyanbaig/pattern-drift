"""
Drift Classifier — Stage 3 of the internal pipeline.

Inspects the temporal shape of drift signals across recent history and labels
the event as one of:

    sudden      — sharp, instantaneous change (detected in 1–2 observations)
    gradual     — slow build-up over many observations
    incremental — monotonically increasing drift score
    recurring   — drift after a previous stable period (implies seen before)
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple


_WINDOW = 50  # observations to look back for shape analysis


class DriftClassifier:
    def __init__(self) -> None:
        # score_history[feature] = deque of (drift_detected, score) pairs
        self._history: Dict[str, Deque[Tuple[bool, float]]] = {}
        self._drift_count: Dict[str, int] = {}

    # ------------------------------------------------------------------

    def record(self, feature: str, drift_detected: bool, score: float) -> None:
        """Record the latest detector output for a feature."""
        if feature not in self._history:
            self._history[feature] = deque(maxlen=_WINDOW)
            self._drift_count[feature] = 0
        self._history[feature].append((drift_detected, score))
        if drift_detected:
            self._drift_count[feature] += 1

    def classify(self, drifted_features: List[str]) -> Optional[str]:
        """
        Given a list of features that just triggered drift, return a drift type.
        Returns ``None`` if the list is empty.
        """
        if not drifted_features:
            return None

        types = [self._classify_feature(f) for f in drifted_features]

        # Priority: sudden > recurring > incremental > gradual
        for priority in ("sudden", "recurring", "incremental", "gradual"):
            if priority in types:
                return priority
        return types[0]

    def reset(self) -> None:
        self._history.clear()
        self._drift_count.clear()

    # ------------------------------------------------------------------

    def _classify_feature(self, feature: str) -> str:
        history = list(self._history.get(feature, []))
        if not history:
            return "sudden"

        scores = [s for _, s in history]
        detected_flags = [d for d, _ in history]

        # Sudden: the score jumped from near-zero to high in the last step
        if len(scores) >= 2 and scores[-2] < 0.2 and scores[-1] > 0.6:
            return "sudden"

        # Recurring: feature has drifted before (more than 1 previous drift event)
        if self._drift_count.get(feature, 0) > 1:
            return "recurring"

        # Incremental: monotonically increasing scores over the last N steps
        if len(scores) >= 5:
            tail = scores[-5:]
            if all(tail[i] <= tail[i + 1] for i in range(len(tail) - 1)):
                return "incremental"

        return "gradual"
