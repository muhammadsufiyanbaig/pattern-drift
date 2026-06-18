from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple


class BaseDetector(ABC):
    """
    Common interface for all drift detectors.

    Each detector instance monitors a single numeric feature stream.
    Call update(value) on every new observation; it returns (drift_detected, drift_score).
    """

    @abstractmethod
    def update(self, value: float) -> Tuple[bool, float]:
        """
        Ingest one observation.

        Returns
        -------
        drift_detected : bool
        drift_score    : float  (higher = more drift; 0.0 means stable)
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal state."""
