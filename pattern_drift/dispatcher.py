"""
Alert Dispatcher — Stage 5 of the internal pipeline.

Fires all registered callbacks when drift is detected.  Built-in helpers are
provided for Slack webhooks and generic HTTP webhooks.  Custom callbacks are
any callable accepting a DriftResult.
"""
from __future__ import annotations

import logging
from typing import Callable, List

from .result import DriftResult

logger = logging.getLogger(__name__)


class AlertDispatcher:
    def __init__(self) -> None:
        self._callbacks: List[Callable[[DriftResult], None]] = []

    # ------------------------------------------------------------------

    def register(self, callback: Callable[[DriftResult], None]) -> None:
        """Register any callable as a drift callback."""
        self._callbacks.append(callback)

    def dispatch(self, result: DriftResult) -> None:
        """Fire all callbacks with the DriftResult payload."""
        for cb in self._callbacks:
            try:
                cb(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Drift callback %s raised an exception: %s", cb, exc)

    def reset(self) -> None:
        self._callbacks.clear()

    # ------------------------------------------------------------------
    # Built-in callback factories
    # ------------------------------------------------------------------

    @staticmethod
    def slack_callback(webhook_url: str) -> Callable[[DriftResult], None]:
        """
        Returns a callback that POSTs a Slack-compatible JSON message.
        Requires the ``requests`` package (``pip install pattern-drift[alerts]``).
        """

        def _callback(result: DriftResult) -> None:
            try:
                import requests
            except ImportError:
                raise ImportError(
                    "Install the alerts extra to use Slack callbacks: "
                    "pip install pattern-drift[alerts]"
                )

            payload = {
                "text": (
                    f":warning: *Drift detected* ({result.drift_type})\n"
                    f"Features: {', '.join(result.drifted_features)}\n"
                    f"Score: {result.drift_score:.4f}\n"
                    f"Time: {result.timestamp.isoformat()}"
                )
            }
            requests.post(webhook_url, json=payload, timeout=5)

        return _callback

    @staticmethod
    def webhook_callback(url: str) -> Callable[[DriftResult], None]:
        """
        Returns a callback that POSTs the full DriftResult as JSON.
        Requires the ``requests`` package (``pip install pattern-drift[alerts]``).
        """

        def _callback(result: DriftResult) -> None:
            try:
                import requests
            except ImportError:
                raise ImportError(
                    "Install the alerts extra to use webhook callbacks: "
                    "pip install pattern-drift[alerts]"
                )

            payload = {
                "drift_detected": result.drift_detected,
                "drift_type": result.drift_type,
                "drifted_features": result.drifted_features,
                "drift_score": result.drift_score,
                "timestamp": result.timestamp.isoformat(),
                "retraining_window": (
                    {
                        "start": result.retraining_window.start,
                        "end": result.retraining_window.end,
                        "n_samples": result.retraining_window.n_samples,
                        "confidence": result.retraining_window.confidence,
                    }
                    if result.retraining_window
                    else None
                ),
            }
            requests.post(url, json=payload, timeout=5)

        return _callback

    @staticmethod
    def log_callback(level: str = "warning") -> Callable[[DriftResult], None]:
        """Returns a callback that logs the DriftResult via Python's logging."""
        log_fn = getattr(logger, level.lower(), logger.warning)

        def _callback(result: DriftResult) -> None:
            log_fn(
                "Drift detected | type=%s | features=%s | score=%.4f | ts=%s",
                result.drift_type,
                result.drifted_features,
                result.drift_score,
                result.timestamp.isoformat(),
            )

        return _callback
