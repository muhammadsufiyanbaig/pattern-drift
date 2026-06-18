"""
scikit-learn Pipeline Wrapper.

Wraps DriftMonitor as a scikit-learn transformer so it can be inserted into
an existing sklearn Pipeline with zero architectural changes.

Example
-------
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from pattern_drift.sklearn_wrapper import DriftDetector

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("drift",  DriftDetector(method="ADWIN", sensitivity=0.002)),
    ])

    pipe.fit(X_train)
    for batch in stream:
        X_out = pipe.transform(batch)   # data passes through unchanged
        # drift events are handled by callbacks registered on the monitor
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

from .monitor import DriftMonitor
from .result import DriftResult


class DriftDetector:
    """
    A pass-through sklearn-compatible transformer that monitors data for drift.

    fit()       — sets the reference distribution (stores X, does nothing else)
    transform() — feeds each row through DriftMonitor.update() and returns X
                  unchanged so the pipeline can continue.

    Parameters
    ----------
    method, sensitivity, min_window, max_window, features, callbacks :
        Forwarded verbatim to DriftMonitor.
    """

    def __init__(
        self,
        method: str = "ADWIN",
        sensitivity: float = 0.002,
        min_window: int = 30,
        max_window: int = 10_000,
        features: Optional[List[str]] = None,
        callbacks: Optional[List[Callable[[DriftResult], None]]] = None,
    ) -> None:
        self.method = method
        self.sensitivity = sensitivity
        self.min_window = min_window
        self.max_window = max_window
        self.features = features
        self.callbacks = callbacks
        self._monitor: Optional[DriftMonitor] = None

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any = None) -> "DriftDetector":
        self._monitor = DriftMonitor(
            method=self.method,
            sensitivity=self.sensitivity,
            min_window=self.min_window,
            max_window=self.max_window,
            features=self.features,
            callbacks=self.callbacks,
        )
        self._monitor.set_reference(X)
        return self

    def transform(self, X: Any, y: Any = None) -> Any:
        if self._monitor is None:
            raise RuntimeError("Call fit() before transform().")
        self._monitor.update(X)
        return X

    def get_params(self, deep: bool = True) -> dict:
        return {
            "method": self.method,
            "sensitivity": self.sensitivity,
            "min_window": self.min_window,
            "max_window": self.max_window,
            "features": self.features,
            "callbacks": self.callbacks,
        }

    def set_params(self, **params: Any) -> "DriftDetector":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    @property
    def monitor(self) -> Optional[DriftMonitor]:
        """Direct access to the underlying DriftMonitor instance."""
        return self._monitor
