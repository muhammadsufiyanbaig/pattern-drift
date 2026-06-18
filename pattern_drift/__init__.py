"""
pattern-drift
=============
Automatic concept drift detection for streaming datasets.

Quick start
-----------
    from pattern_drift import DriftMonitor

    monitor = DriftMonitor(method="ADWIN", sensitivity=0.002)

    for record in stream:
        result = monitor.update(record)
        if result.drift_detected:
            print(f"Drift! type={result.drift_type}, features={result.drifted_features}")
"""

from .monitor import DriftMonitor
from .result import DriftResult, RetrainingWindowResult
from .dispatcher import AlertDispatcher
from .sklearn_wrapper import DriftDetector

__all__ = [
    "DriftMonitor",
    "DriftResult",
    "RetrainingWindowResult",
    "AlertDispatcher",
    "DriftDetector",
]

__version__ = "0.1.0"
