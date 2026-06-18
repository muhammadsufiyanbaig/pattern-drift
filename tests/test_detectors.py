"""Unit tests for individual drift detectors."""
import math
import pytest

from pattern_drift.detectors import ADWIN, DDM, KSWIN, PageHinkley


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_stream(n: int = 200, mean: float = 0.0, std: float = 0.1):
    """Deterministic pseudo-random stable stream."""
    import random
    rng = random.Random(42)
    return [rng.gauss(mean, std) for _ in range(n)]


def _drifted_stream(n_stable: int = 150, n_drift: int = 100, shift: float = 5.0):
    """Stable segment followed by a sudden mean shift."""
    stable = _stable_stream(n_stable)
    drifted = _stable_stream(n_drift, mean=shift)
    return stable + drifted


# ---------------------------------------------------------------------------
# ADWIN
# ---------------------------------------------------------------------------

class TestADWIN:
    def test_no_false_alarm_on_stable(self):
        det = ADWIN(delta=0.002)
        alarms = sum(1 for v in _stable_stream(300) if det.update(v)[0])
        assert alarms == 0, f"Expected 0 false alarms, got {alarms}"

    def test_detects_sudden_drift(self):
        det = ADWIN(delta=0.002)
        stream = _drifted_stream(n_stable=150, n_drift=80, shift=10.0)
        alarms = [i for i, v in enumerate(stream) if det.update(v)[0]]
        assert len(alarms) > 0, "ADWIN should detect the mean shift"
        assert alarms[0] > 100, "First alarm should be after stable period"

    def test_reset_clears_state(self):
        det = ADWIN(delta=0.002)
        for v in _stable_stream(50):
            det.update(v)
        det.reset()
        assert len(det._window) == 0
        assert det._total == 0.0

    def test_score_between_0_and_1(self):
        det = ADWIN(delta=0.002)
        for v in _stable_stream(100):
            _, score = det.update(v)
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


# ---------------------------------------------------------------------------
# PageHinkley
# ---------------------------------------------------------------------------

class TestPageHinkley:
    def test_detects_sudden_shift(self):
        det = PageHinkley(delta=0.005, lambda_=50.0)
        stream = _drifted_stream(n_stable=100, n_drift=50, shift=8.0)
        alarms = [i for i, v in enumerate(stream) if det.update(v)[0]]
        assert len(alarms) > 0, "PageHinkley should detect the mean shift"

    def test_no_false_alarm_on_stable(self):
        det = PageHinkley(delta=0.005, lambda_=50.0)
        alarms = sum(1 for v in _stable_stream(500) if det.update(v)[0])
        assert alarms == 0

    def test_reset(self):
        det = PageHinkley()
        for v in _stable_stream(50):
            det.update(v)
        det.reset()
        assert det._n == 0
        assert det._cumsum == 0.0


# ---------------------------------------------------------------------------
# KSWIN
# ---------------------------------------------------------------------------

class TestKSWIN:
    def test_detects_distribution_change(self):
        det = KSWIN(alpha=0.05, window_size=100, stat_size=30)
        # Stable: N(0, 0.1)
        stable = _stable_stream(200)
        # Drift: N(5, 0.1)
        drifted = _stable_stream(100, mean=5.0)
        stream = stable + drifted

        alarms = [i for i, v in enumerate(stream) if det.update(v)[0]]
        assert len(alarms) > 0, "KSWIN should detect the distribution change"

    def test_set_reference(self):
        det = KSWIN(alpha=0.05, window_size=50, stat_size=15)
        ref = _stable_stream(50)
        det.set_reference(ref)
        assert det._reference == ref

    def test_window_size_validation(self):
        with pytest.raises(ValueError):
            KSWIN(window_size=30, stat_size=30)


# ---------------------------------------------------------------------------
# DDM
# ---------------------------------------------------------------------------

class TestDDM:
    def test_detects_increasing_error_rate(self):
        det = DDM(min_num_instances=30, drift_level=3.0)
        # Stable: mostly correct (1.0)
        for _ in range(100):
            det.update(1.0)
        # Drift: mostly incorrect (0.0)
        alarms = [i for i in range(50) if det.update(0.0)[0]]
        assert len(alarms) > 0, "DDM should detect the error rate increase"

    def test_no_alarm_on_good_predictions(self):
        det = DDM(min_num_instances=30)
        alarms = sum(1 for _ in range(200) if det.update(1.0)[0])
        assert alarms == 0

    def test_reset(self):
        det = DDM()
        for _ in range(50):
            det.update(1.0)
        det.reset()
        assert det._n == 0
        assert det._p == 0.0
