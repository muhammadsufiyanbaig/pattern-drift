"""
Test category 2: Algorithm sensitivity calibration.

Verify that changing the `sensitivity` parameter makes detection
earlier or later as expected, and that the relationship is monotonic.
"""
import random
import pytest
from pattern_drift import DriftMonitor
from pattern_drift.detectors import ADWIN, PageHinkley, KSWIN


def _abrupt_stream(n_pre: int, n_post: int, shift: float, seed: int = 42):
    rng = random.Random(seed)
    pre = [rng.gauss(0.0, 0.1) for _ in range(n_pre)]
    post = [rng.gauss(shift, 0.1) for _ in range(n_post)]
    return pre + post


def _first_alarm_index(detector, stream):
    for i, v in enumerate(stream):
        if detector.update(v)[0]:
            return i
    return None


# ---------------------------------------------------------------------------
# ADWIN: higher delta → less stringent → detects earlier
# ---------------------------------------------------------------------------

class TestADWINSensitivity:
    """For ADWIN, delta is a significance level: larger delta = earlier alarm."""

    def _run(self, delta: float, stream: list):
        det = ADWIN(delta=delta)
        return _first_alarm_index(det, stream)

    def test_lenient_detects_before_strict(self):
        stream = _abrupt_stream(150, 150, shift=4.0)
        idx_lenient = self._run(delta=0.5, stream=stream)
        idx_strict  = self._run(delta=0.0001, stream=stream)

        assert idx_lenient is not None, "Lenient ADWIN should detect drift"
        assert idx_strict is not None, "Strict ADWIN should also detect drift (large shift)"
        assert idx_lenient <= idx_strict, (
            f"Lenient ({idx_lenient}) should fire no later than strict ({idx_strict})"
        )

    @pytest.mark.parametrize("delta", [0.5, 0.1, 0.01, 0.001])
    def test_all_detect_large_shift(self, delta):
        stream = _abrupt_stream(100, 200, shift=10.0)
        idx = self._run(delta=delta, stream=stream)
        assert idx is not None, f"ADWIN (delta={delta}) should detect a shift of 10.0"
        assert idx >= 100, f"Alarm at {idx} fired before drift point 100"


# ---------------------------------------------------------------------------
# PageHinkley: smaller delta → more sensitive → fires earlier
# ---------------------------------------------------------------------------

class TestPageHinkleySensitivity:
    """For PageHinkley, delta is minimum detectable change: smaller → more sensitive."""

    def _run(self, delta: float, lambda_: float, stream: list):
        det = PageHinkley(delta=delta, lambda_=lambda_)
        return _first_alarm_index(det, stream)

    def test_lower_lambda_detects_earlier(self):
        stream = _abrupt_stream(100, 200, shift=6.0)
        idx_fast = self._run(delta=0.005, lambda_=20.0, stream=stream)
        idx_slow = self._run(delta=0.005, lambda_=200.0, stream=stream)

        assert idx_fast is not None, "Low-threshold PH should detect drift"
        if idx_slow is not None:
            assert idx_fast <= idx_slow, (
                f"Lower lambda ({idx_fast}) should fire no later than higher ({idx_slow})"
            )

    @pytest.mark.parametrize("lambda_", [10.0, 30.0, 50.0, 100.0])
    def test_all_detect_large_shift(self, lambda_):
        stream = _abrupt_stream(100, 200, shift=15.0)
        idx = self._run(delta=0.005, lambda_=lambda_, stream=stream)
        assert idx is not None, f"PageHinkley (lambda={lambda_}) should detect shift of 15.0"
        assert idx >= 100


# ---------------------------------------------------------------------------
# KSWIN: smaller alpha → more stringent → detects later
# ---------------------------------------------------------------------------

class TestKSWINSensitivity:
    """For KSWIN, alpha is the KS significance level: larger → detects earlier."""

    def _run(self, alpha: float, stream: list):
        det = KSWIN(alpha=alpha, window_size=100, stat_size=30)
        return _first_alarm_index(det, stream)

    def test_larger_alpha_detects_before_smaller(self):
        stream = _abrupt_stream(200, 200, shift=5.0)
        idx_lenient = self._run(alpha=0.2, stream=stream)
        idx_strict  = self._run(alpha=0.001, stream=stream)

        assert idx_lenient is not None, "Lenient KSWIN should detect drift"
        if idx_strict is not None:
            assert idx_lenient <= idx_strict, (
                f"Lenient alpha ({idx_lenient}) should fire no later than strict ({idx_strict})"
            )


# ---------------------------------------------------------------------------
# DriftMonitor: sensitivity parameter wires through correctly
# ---------------------------------------------------------------------------

class TestMonitorSensitivityWiring:
    def test_adwin_sensitivity_stored(self):
        monitor = DriftMonitor(method="ADWIN", sensitivity=0.05)
        # Trigger detector creation
        monitor.update({"x": 1.0})
        det = monitor._detectors["x"]
        assert isinstance(det, ADWIN)
        assert det.delta == 0.05

    def test_pagehinkley_sensitivity_stored(self):
        monitor = DriftMonitor(method="PageHinkley", sensitivity=0.01)
        monitor.update({"x": 1.0})
        det = monitor._detectors["x"]
        assert isinstance(det, PageHinkley)
        assert det.delta == 0.01

    def test_kswin_sensitivity_stored(self):
        monitor = DriftMonitor(method="KSWIN", sensitivity=0.03)
        monitor.update({"x": 1.0})
        det = monitor._detectors["x"]
        assert isinstance(det, KSWIN)
        assert det.alpha == 0.03
