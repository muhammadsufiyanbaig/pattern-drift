"""
Test category 9: Alert callback tests.

Verify callbacks:
  - Fire exactly once per drift event
  - Never fire on stable data
  - Receive a correct, fully-populated DriftResult payload
  - Multiple callbacks all fire independently
  - Dispatcher.log_callback, slack_callback, webhook_callback factories work correctly
"""
import random
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest

from pattern_drift import DriftMonitor, DriftResult
from pattern_drift.dispatcher import AlertDispatcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable(n: int, seed: int = 0):
    rng = random.Random(seed)
    return [{"x": rng.gauss(0.0, 0.1)} for _ in range(n)]


def _trigger_drift(monitor: DriftMonitor, shift: float = 50.0, n: int = 30):
    """Feed values that reliably trigger a drift alarm and return the event."""
    for _ in range(n):
        result = monitor.update({"x": shift})
        if result.drift_detected:
            return result
    return None


def _warmup(monitor: DriftMonitor, n: int = 100, seed: int = 1):
    rng = random.Random(seed)
    for _ in range(n):
        monitor.update({"x": rng.gauss(0.0, 0.1)})


# ---------------------------------------------------------------------------
# Firing behaviour
# ---------------------------------------------------------------------------

class TestCallbackFiring:
    def test_callback_fires_exactly_on_drift_events(self):
        """Callback count must equal the number of drift events."""
        fired_results = []
        monitor = DriftMonitor(
            method="PageHinkley", sensitivity=0.005, min_window=10,
            callbacks=[fired_results.append],
        )
        _warmup(monitor, 100)

        drift_count = 0
        for _ in range(60):
            result = monitor.update({"x": 100.0})
            if result.drift_detected:
                drift_count += 1

        assert len(fired_results) == drift_count, (
            f"Callback fired {len(fired_results)} times but {drift_count} drift events occurred"
        )

    def test_callback_never_fires_on_stable_data(self):
        fired = []
        monitor = DriftMonitor(
            method="ADWIN", sensitivity=0.002, min_window=30,
            callbacks=[fired.append],
        )
        for rec in _stable(300):
            monitor.update(rec)
        assert len(fired) == 0, f"Callback fired {len(fired)} times on stable data"

    def test_multiple_callbacks_all_fire(self):
        counts = [0, 0, 0]

        def cb0(r): counts[0] += 1
        def cb1(r): counts[1] += 1
        def cb2(r): counts[2] += 1

        monitor = DriftMonitor(
            method="PageHinkley", sensitivity=0.005, min_window=10,
            callbacks=[cb0, cb1, cb2],
        )
        _warmup(monitor, 100)
        _trigger_drift(monitor)

        # All three should have fired the same number of times
        assert counts[0] == counts[1] == counts[2], (
            f"Callbacks fired unequal counts: {counts}"
        )
        assert counts[0] > 0, "No callbacks fired"

    def test_callback_added_after_construction_fires(self):
        """Register a callback via dispatcher after monitor creation."""
        fired = []
        monitor = DriftMonitor(method="PageHinkley", sensitivity=0.005, min_window=10)
        monitor._dispatcher.register(fired.append)

        _warmup(monitor, 100)
        _trigger_drift(monitor)

        assert len(fired) > 0, "Late-registered callback should still fire"


# ---------------------------------------------------------------------------
# Payload correctness
# ---------------------------------------------------------------------------

class TestCallbackPayload:
    def test_payload_is_drift_result(self):
        payloads = []
        monitor = DriftMonitor(
            method="PageHinkley", sensitivity=0.005, min_window=10,
            callbacks=[payloads.append],
        )
        _warmup(monitor, 100)
        _trigger_drift(monitor)

        assert len(payloads) > 0
        for p in payloads:
            assert isinstance(p, DriftResult)

    def test_payload_drift_detected_is_true(self):
        payloads = []
        monitor = DriftMonitor(
            method="PageHinkley", sensitivity=0.005, min_window=10,
            callbacks=[payloads.append],
        )
        _warmup(monitor, 100)
        _trigger_drift(monitor)

        for p in payloads:
            assert p.drift_detected is True

    def test_payload_has_drifted_features(self):
        payloads = []
        monitor = DriftMonitor(
            method="PageHinkley", sensitivity=0.005, min_window=10,
            callbacks=[payloads.append],
        )
        _warmup(monitor, 100)
        _trigger_drift(monitor)

        for p in payloads:
            assert isinstance(p.drifted_features, list)
            assert len(p.drifted_features) > 0

    def test_payload_drift_type_is_valid(self):
        valid_types = {"sudden", "gradual", "incremental", "recurring"}
        payloads = []
        monitor = DriftMonitor(
            method="PageHinkley", sensitivity=0.005, min_window=10,
            callbacks=[payloads.append],
        )
        _warmup(monitor, 100)
        _trigger_drift(monitor)

        for p in payloads:
            assert p.drift_type in valid_types, f"Invalid drift_type: {p.drift_type!r}"

    def test_payload_timestamp_is_utc(self):
        payloads = []
        monitor = DriftMonitor(
            method="PageHinkley", sensitivity=0.005, min_window=10,
            callbacks=[payloads.append],
        )
        _warmup(monitor, 100)
        _trigger_drift(monitor)

        for p in payloads:
            assert p.timestamp.tzinfo is not None, "Timestamp must be timezone-aware"

    def test_payload_drift_score_non_negative(self):
        payloads = []
        monitor = DriftMonitor(
            method="PageHinkley", sensitivity=0.005, min_window=10,
            callbacks=[payloads.append],
        )
        _warmup(monitor, 100)
        _trigger_drift(monitor)

        for p in payloads:
            assert p.drift_score >= 0.0


# ---------------------------------------------------------------------------
# Built-in callback factories
# ---------------------------------------------------------------------------

class TestBuiltinCallbackFactories:
    def test_log_callback_calls_logger(self, caplog):
        """log_callback must emit a WARNING-level log record."""
        import logging
        cb = AlertDispatcher.log_callback(level="warning")
        result = DriftResult(
            drift_detected=True,
            drift_type="sudden",
            drifted_features=["x"],
            drift_score=0.75,
            timestamp=datetime.now(timezone.utc),
        )
        with caplog.at_level(logging.WARNING, logger="pattern_drift.dispatcher"):
            cb(result)
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.WARNING
        assert "sudden" in caplog.records[0].message

    def test_log_callback_invalid_level_falls_back_to_warning(self, caplog):
        """An unrecognised level name should still emit a log record (falls back to warning)."""
        import logging
        cb = AlertDispatcher.log_callback(level="nonexistent_level")
        result = DriftResult(
            drift_detected=True, drift_type="gradual",
            drifted_features=["y"], drift_score=0.3,
            timestamp=datetime.now(timezone.utc),
        )
        with caplog.at_level(logging.WARNING, logger="pattern_drift.dispatcher"):
            cb(result)  # should not raise
        assert len(caplog.records) == 1

    def test_slack_callback_raises_without_requests(self):
        """Without the requests extra, slack_callback raises ImportError."""
        cb = AlertDispatcher.slack_callback("https://hooks.slack.com/fake")
        result = DriftResult(
            drift_detected=True, drift_type="sudden",
            drifted_features=["x"], drift_score=0.9,
            timestamp=datetime.now(timezone.utc),
        )
        with patch.dict("sys.modules", {"requests": None}):
            with pytest.raises(ImportError, match="alerts"):
                cb(result)

    def test_webhook_callback_raises_without_requests(self):
        cb = AlertDispatcher.webhook_callback("https://example.com/drift")
        result = DriftResult(
            drift_detected=True, drift_type="gradual",
            drifted_features=["z"], drift_score=0.4,
            timestamp=datetime.now(timezone.utc),
        )
        with patch.dict("sys.modules", {"requests": None}):
            with pytest.raises(ImportError, match="alerts"):
                cb(result)

    def test_slack_callback_posts_correct_payload(self):
        requests_mock = MagicMock()
        cb = AlertDispatcher.slack_callback("https://hooks.slack.com/fake")
        result = DriftResult(
            drift_detected=True, drift_type="sudden",
            drifted_features=["revenue"], drift_score=0.88,
            timestamp=datetime.now(timezone.utc),
        )
        with patch.dict("sys.modules", {"requests": requests_mock}):
            cb(result)
        call_kwargs = requests_mock.post.call_args
        payload = call_kwargs[1]["json"]
        assert "text" in payload
        assert "sudden" in payload["text"]
        assert "revenue" in payload["text"]

    def test_webhook_callback_posts_full_result(self):
        requests_mock = MagicMock()
        cb = AlertDispatcher.webhook_callback("https://example.com/hook")
        result = DriftResult(
            drift_detected=True, drift_type="gradual",
            drifted_features=["age", "income"], drift_score=0.55,
            timestamp=datetime.now(timezone.utc),
        )
        with patch.dict("sys.modules", {"requests": requests_mock}):
            cb(result)
        call_kwargs = requests_mock.post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["drift_detected"] is True
        assert payload["drift_type"] == "gradual"
        assert set(payload["drifted_features"]) == {"age", "income"}
        assert payload["drift_score"] == 0.55
