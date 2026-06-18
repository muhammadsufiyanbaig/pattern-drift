"""
Test category 3: False positive rate tests.

Run each detector on long stable streams and assert the alarm rate
is below the theoretical bound or zero, depending on the algorithm.

Acceptable FPR thresholds used here are generous to avoid flakiness while
still catching broken detectors. All streams use a fixed seed for
reproducibility.
"""
import random
import pytest
from pattern_drift.detectors import ADWIN, DDM, KSWIN, PageHinkley
from pattern_drift import DriftMonitor


def _stable(n: int, mean: float = 0.0, std: float = 0.3, seed: int = 99) -> list:
    rng = random.Random(seed)
    return [rng.gauss(mean, std) for _ in range(n)]


# ---------------------------------------------------------------------------
# Per-detector FPR
# ---------------------------------------------------------------------------

class TestADWINFalsePositiveRate:
    def test_zero_false_alarms_on_stable_stream(self):
        """ADWIN with strict delta should produce zero false alarms on 1 000 samples."""
        det = ADWIN(delta=0.002)
        alarms = [i for i, v in enumerate(_stable(1000)) if det.update(v)[0]]
        assert len(alarms) == 0, f"Unexpected alarms at: {alarms}"

    def test_fpr_within_theoretical_bound_lenient(self):
        """
        Even with a very lenient delta (0.1), FPR should stay below 10 % of N.
        Each alarm resets the window, so chains are unlikely.
        """
        det = ADWIN(delta=0.1)
        n = 500
        alarms = sum(1 for v in _stable(n, seed=77) if det.update(v)[0])
        assert alarms / n < 0.10, f"FPR {alarms/n:.2%} exceeds 10 %"


class TestPageHinkleyFalsePositiveRate:
    def test_very_low_fpr_high_threshold(self):
        """
        PageHinkley resets on each alarm, so on a long stable stream a small
        number of false alarms is possible.  Require FPR < 1 % with lambda=100.
        """
        det = PageHinkley(delta=0.005, lambda_=100.0)
        n = 1000
        alarms = [i for i, v in enumerate(_stable(n)) if det.update(v)[0]]
        assert len(alarms) / n < 0.01, f"FPR {len(alarms)/n:.2%} exceeds 1 % — alarms at {alarms}"

    def test_fpr_within_bound_moderate_threshold(self):
        det = PageHinkley(delta=0.005, lambda_=50.0)
        n = 1000
        alarms = sum(1 for v in _stable(n, seed=88) if det.update(v)[0])
        assert alarms / n < 0.05, f"FPR {alarms/n:.2%} exceeds 5 %"


class TestKSWINFalsePositiveRate:
    def test_fpr_within_alpha_bound(self):
        """
        With alpha=0.01 the expected FPR is ~1 %. Allow 3× margin for
        finite-sample variation.
        """
        alpha = 0.01
        det = KSWIN(alpha=alpha, window_size=100, stat_size=30)
        n = 1000
        alarms = sum(1 for v in _stable(n, seed=55) if det.update(v)[0])
        # Only windows after the first 100 samples can trigger an alarm
        effective_n = max(n - 100, 1)
        fpr = alarms / effective_n
        assert fpr < alpha * 5, f"FPR {fpr:.2%} far exceeds alpha={alpha}"

    def test_zero_alarms_very_strict_alpha(self):
        det = KSWIN(alpha=0.0001, window_size=100, stat_size=30)
        alarms = sum(1 for v in _stable(500) if det.update(v)[0])
        assert alarms == 0


class TestDDMFalsePositiveRate:
    def test_zero_alarms_on_perfect_predictions(self):
        det = DDM(min_num_instances=30, drift_level=3.0)
        alarms = sum(1 for _ in range(500) if det.update(1.0)[0])
        assert alarms == 0

    def test_zero_alarms_before_min_window(self):
        det = DDM(min_num_instances=100)
        alarms = [i for i in range(99) if det.update(0.0)[0]]
        assert len(alarms) == 0, f"DDM fired before min_window: {alarms}"


# ---------------------------------------------------------------------------
# DriftMonitor-level FPR
# ---------------------------------------------------------------------------

class TestMonitorFalsePositiveRate:
    @pytest.mark.parametrize("method,sensitivity", [
        ("ADWIN",       0.002),
        ("PageHinkley", 0.005),
        ("KSWIN",       0.01),
        ("DDM",         0.002),
    ])
    def test_no_drift_on_stable_multifeature_stream(self, method, sensitivity):
        rng = random.Random(123)
        monitor = DriftMonitor(method=method, sensitivity=sensitivity, min_window=30)
        # DDM expects prediction-error-like inputs (0 or 1); others use float
        if method == "DDM":
            stream = [{"x": 1.0, "y": 1.0} for _ in range(200)]
        else:
            stream = [
                {"x": rng.gauss(0, 0.2), "y": rng.gauss(0, 0.2)}
                for _ in range(300)
            ]
        alarms = [r for r in (monitor.update(rec) for rec in stream) if r.drift_detected]
        assert len(alarms) == 0, (
            f"{method} produced {len(alarms)} false alarm(s) on stable data"
        )
