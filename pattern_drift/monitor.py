"""
DriftMonitor — the main public class of pattern-drift.

Usage
-----
    from pattern_drift import DriftMonitor

    monitor = DriftMonitor(method="ADWIN", sensitivity=0.002)

    for record in stream:
        result = monitor.update(record)
        if result.drift_detected:
            print(result)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .classifier import DriftClassifier
from .detectors import ADWIN, DDM, KSWIN, PageHinkley
from .detectors.base import BaseDetector
from .dispatcher import AlertDispatcher
from .feature_extractor import FeatureExtractor
from .result import DriftResult
from .window_engine import RetrainingWindowEngine

_METHODS = {
    "ADWIN": ADWIN,
    "PageHinkley": PageHinkley,
    "KSWIN": KSWIN,
    "DDM": DDM,
}


class DriftMonitor:
    """
    Continuous, per-feature drift monitor for streaming datasets.

    Parameters
    ----------
    method : str
        Detection algorithm.  One of: ``ADWIN`` (default), ``PageHinkley``,
        ``KSWIN``, ``DDM``.
    sensitivity : float
        Drift detection threshold (default 0.002).  Lower = more sensitive.
    min_window : int
        Minimum history size before drift can be reported (default 30).
    max_window : int
        Maximum number of records retained in memory (default 10 000).
    features : list[str] | None
        Column names to monitor.  ``None`` = auto-detect all numeric columns.
    callbacks : list[callable] | None
        Callables fired on every drift event, receiving a ``DriftResult``.
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
        if method not in _METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {list(_METHODS)}")

        self.method = method
        self.sensitivity = sensitivity
        self.min_window = min_window
        self.max_window = max_window

        self._extractor = FeatureExtractor(features)
        self._detectors: Dict[str, BaseDetector] = {}
        self._classifier = DriftClassifier()
        self._window_engine = RetrainingWindowEngine(sensitivity=sensitivity)
        self._dispatcher = AlertDispatcher()

        if callbacks:
            for cb in callbacks:
                self._dispatcher.register(cb)

        # History: list of {feature: score} dicts, one per update() call
        self._score_history: List[Dict[str, float]] = []
        self._n_updates: int = 0

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def update(self, data: Any) -> DriftResult:
        """
        Feed a single record or a micro-batch.

        Parameters
        ----------
        data : dict | pd.Series | pd.DataFrame
            A single row as a dict or Series, or a DataFrame.
            DataFrames are processed row-by-row; the returned DriftResult
            reflects the *last* row processed (or the first drift event found).

        Returns
        -------
        DriftResult
        """
        # Handle DataFrame micro-batches
        try:
            import pandas as pd

            if isinstance(data, pd.DataFrame):
                result = DriftResult(drift_detected=False, drift_type=None)
                for _, row in data.iterrows():
                    result = self._update_single(row)
                    if result.drift_detected:
                        return result
                return result
        except ImportError:
            pass

        return self._update_single(data)

    def reset(self) -> None:
        """Reset all internal detector state and history."""
        for det in self._detectors.values():
            det.reset()
        self._detectors.clear()
        self._extractor.reset()
        self._classifier.reset()
        self._score_history.clear()
        self._n_updates = 0

    def plot_drift_timeline(self) -> None:
        """
        Render a drift score timeline for all monitored features.
        Requires ``matplotlib`` (``pip install pattern-drift[viz]``).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Install the viz extra to use visualisation: "
                "pip install pattern-drift[viz]"
            )

        if not self._score_history:
            print("No drift history to plot.")
            return

        features = sorted(
            {k for record in self._score_history for k in record}
        )
        x = list(range(len(self._score_history)))

        fig, ax = plt.subplots(figsize=(12, 5))
        for feat in features:
            y = [record.get(feat, 0.0) for record in self._score_history]
            ax.plot(x, y, label=feat, alpha=0.8)

        ax.axhline(self.sensitivity, color="red", linestyle="--", label="Sensitivity threshold")
        ax.set_xlabel("Observation index")
        ax.set_ylabel("Drift score")
        ax.set_title("pattern-drift — Drift Score Timeline")
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    def export_report(self, path: str) -> None:
        """
        Export the full drift score history to JSON or CSV.

        Parameters
        ----------
        path : str
            File path.  Extension determines format: ``.json`` or ``.csv``.
        """
        if path.endswith(".csv"):
            self._export_csv(path)
        else:
            self._export_json(path)

    def set_reference(self, data: Any) -> None:
        """
        Manually update the reference distribution used for comparison.
        Currently applies to KSWIN detectors; other detectors ignore it.

        Parameters
        ----------
        data : dict | pd.Series | pd.DataFrame
            New reference data.  A DataFrame is used column-by-column.
        """
        try:
            import pandas as pd

            if isinstance(data, pd.DataFrame):
                for col in data.select_dtypes("number").columns:
                    det = self._detectors.get(col)
                    if isinstance(det, KSWIN):
                        det.set_reference(data[col].tolist())
                return
        except ImportError:
            pass

        record = self._extractor.extract(data)
        for feat, value in record.items():
            det = self._detectors.get(feat)
            if isinstance(det, KSWIN):
                det.set_reference([value])

    @classmethod
    def from_config(cls, path: str) -> "DriftMonitor":
        """
        Instantiate a DriftMonitor from a YAML config file.

        Requires PyYAML (``pip install pyyaml``).

        Example YAML
        ------------
        .. code-block:: yaml

            method: ADWIN
            sensitivity: 0.002
            min_window: 30
            max_window: 10000
            features:
              - age
              - income
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for from_config: pip install pyyaml")

        with open(path, "r") as fh:
            config = yaml.safe_load(fh)

        return cls(
            method=config.get("method", "ADWIN"),
            sensitivity=config.get("sensitivity", 0.002),
            min_window=config.get("min_window", 30),
            max_window=config.get("max_window", 10_000),
            features=config.get("features"),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_single(self, record: Any) -> DriftResult:
        features = self._extractor.extract(record)
        self._n_updates += 1

        step_scores: Dict[str, float] = {}
        drifted: List[str] = []

        for feat, value in features.items():
            det = self._get_or_create_detector(feat)
            drift_detected, score = det.update(value)
            step_scores[feat] = score
            self._classifier.record(feat, drift_detected, score)

            if drift_detected and self._n_updates >= self.min_window:
                drifted.append(feat)

        # Trim history to max_window
        self._score_history.append(step_scores)
        if len(self._score_history) > self.max_window:
            self._score_history.pop(0)

        if not drifted:
            return DriftResult(drift_detected=False, drift_type=None)

        drift_type = self._classifier.classify(drifted)
        max_score = max(step_scores.get(f, 0.0) for f in drifted)

        rw = self._window_engine.find_window(
            self._score_history, len(self._score_history) - 1
        )

        result = DriftResult(
            drift_detected=True,
            drift_type=drift_type,
            drifted_features=drifted,
            drift_score=round(max_score, 6),
            retraining_window=rw,
            timestamp=datetime.now(timezone.utc),
        )

        self._dispatcher.dispatch(result)
        return result

    def _get_or_create_detector(self, feature: str) -> BaseDetector:
        if feature not in self._detectors:
            cls = _METHODS[self.method]
            if self.method == "ADWIN":
                self._detectors[feature] = cls(delta=self.sensitivity, max_window=self.max_window)
            elif self.method == "PageHinkley":
                self._detectors[feature] = cls(delta=self.sensitivity)
            elif self.method == "KSWIN":
                self._detectors[feature] = cls(alpha=self.sensitivity)
            elif self.method == "DDM":
                self._detectors[feature] = cls(min_num_instances=self.min_window)
            else:
                self._detectors[feature] = cls()
        return self._detectors[feature]

    def _export_json(self, path: str) -> None:
        data = [
            {"index": i, **{k: v for k, v in record.items()}}
            for i, record in enumerate(self._score_history)
        ]
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)

    def _export_csv(self, path: str) -> None:
        if not self._score_history:
            return
        features = sorted({k for rec in self._score_history for k in rec})
        header = ["index"] + features
        rows = [header]
        for i, record in enumerate(self._score_history):
            rows.append([str(i)] + [str(record.get(f, "")) for f in features])
        with open(path, "w") as fh:
            fh.write("\n".join(",".join(row) for row in rows))
