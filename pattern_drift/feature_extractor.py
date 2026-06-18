"""
Feature Extractor — Stage 1 of the internal pipeline.

Converts an incoming record (dict, pandas Series, or single-row DataFrame)
into a plain ``{feature_name: float}`` mapping that the Detector Pool can
consume.
"""
from __future__ import annotations

from typing import Dict, List, Optional


class FeatureExtractor:
    """
    Parameters
    ----------
    features : list[str] | None
        Explicit list of column names to monitor.  ``None`` means
        "auto-detect all numeric columns on first observation".
    """

    def __init__(self, features: Optional[List[str]] = None) -> None:
        self._requested_features = features
        self._resolved_features: Optional[List[str]] = features

    # ------------------------------------------------------------------

    def extract(self, record: object) -> Dict[str, float]:
        """Return a ``{name: float}`` dict for a single record."""
        raw = self._to_dict(record)
        numeric = self._filter_numeric(raw)

        if self._resolved_features is None:
            # Auto-discover on first call
            self._resolved_features = sorted(numeric.keys())

        return {k: float(numeric[k]) for k in self._resolved_features if k in numeric}

    @property
    def features(self) -> Optional[List[str]]:
        return self._resolved_features

    def reset(self) -> None:
        # Only reset auto-discovered features; keep explicitly requested ones.
        if self._requested_features is None:
            self._resolved_features = None

    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(record: object) -> dict:
        """Convert various input types to a plain dict."""
        # pandas DataFrame (single row) or Series
        try:
            import pandas as pd

            if isinstance(record, pd.DataFrame):
                if len(record) != 1:
                    raise ValueError(
                        "When passing a DataFrame to update(), it must have exactly one row "
                        "or be a micro-batch — use update() with a full DataFrame for batches."
                    )
                return record.iloc[0].to_dict()
            if isinstance(record, pd.Series):
                return record.to_dict()
        except ImportError:
            pass

        if isinstance(record, dict):
            return record

        raise TypeError(
            f"Unsupported record type: {type(record).__name__}. "
            "Pass a dict, pandas Series, or single-row DataFrame."
        )

    @staticmethod
    def _filter_numeric(d: dict) -> dict:
        result = {}
        for k, v in d.items():
            try:
                result[k] = float(v)
            except (TypeError, ValueError):
                pass
        return result
