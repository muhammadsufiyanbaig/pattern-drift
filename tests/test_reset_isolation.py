"""
Test category 7: reset() state isolation tests.

After calling reset(), the monitor must behave exactly like a brand-new
instance: same outputs for the same subsequent inputs, zero residual state.
"""
import random
import pytest
from pattern_drift import DriftMonitor
from pattern_drift.detectors import ADWIN, PageHinkley, KSWIN, DDM


def _stable(n: int, seed: int = 0) -> list:
    rng = random.Random(seed)
    return [{"x": rng.gauss(0.0, 0.1)} for _ in range(n)]


# ---------------------------------------------------------------------------
# DriftMonitor.reset() — state attributes
# ---------------------------------------------------------------------------

class TestMonitorResetStateAttributes:
    def test_n_updates_reset_to_zero(self):
        monitor = DriftMonitor()
        for rec in _stable(50):
            monitor.update(rec)
        monitor.reset()
        assert monitor._n_updates == 0

    def test_score_history_cleared(self):
        monitor = DriftMonitor()
        for rec in _stable(50):
            monitor.update(rec)
        monitor.reset()
        assert monitor._score_history == []

    def test_detectors_cleared(self):
        monitor = DriftMonitor()
        for rec in _stable(50):
            monitor.update(rec)
        assert len(monitor._detectors) > 0
        monitor.reset()
        assert len(monitor._detectors) == 0

    def test_classifier_history_cleared(self):
        monitor = DriftMonitor()
        for rec in _stable(50):
            monitor.update(rec)
        monitor.reset()
        assert monitor._classifier._history == {}
        assert monitor._classifier._drift_count == {}

    def test_feature_extractor_resolved_features_reset(self):
        monitor = DriftMonitor()   # features=None → auto-detect
        for rec in _stable(10):
            monitor.update(rec)
        assert monitor._extractor.features is not None
        monitor.reset()
        # After reset the extractor rediscovers on next call
        assert monitor._extractor._resolved_features is None


# ---------------------------------------------------------------------------
# Behavioural equivalence: reset monitor == fresh monitor
# ---------------------------------------------------------------------------

class TestMonitorResetBehaviouralEquivalence:
    @pytest.mark.parametrize("method", ["ADWIN", "PageHinkley", "KSWIN"])
    def test_same_outputs_after_reset_vs_fresh(self, method):
        """
        Feed N stable records, reset, then feed M more records.
        The outputs must match a fresh monitor fed only the M records.
        """
        n_pre = 80
        n_post = 40
        seed = 42

        pre_stream = _stable(n_pre, seed=seed)
        post_stream = _stable(n_post, seed=seed + 1)

        # Monitor with history then reset
        monitor_reset = DriftMonitor(method=method, sensitivity=0.002, min_window=10)
        for rec in pre_stream:
            monitor_reset.update(rec)
        monitor_reset.reset()
        results_reset = [monitor_reset.update(rec) for rec in post_stream]

        # Fresh monitor — no prior history
        monitor_fresh = DriftMonitor(method=method, sensitivity=0.002, min_window=10)
        results_fresh = [monitor_fresh.update(rec) for rec in post_stream]

        for i, (r_reset, r_fresh) in enumerate(zip(results_reset, results_fresh)):
            assert r_reset.drift_detected == r_fresh.drift_detected, (
                f"Step {i}: reset={r_reset.drift_detected}, fresh={r_fresh.drift_detected}"
            )
            assert abs(r_reset.drift_score - r_fresh.drift_score) < 1e-9, (
                f"Step {i}: score mismatch reset={r_reset.drift_score:.6f} "
                f"fresh={r_fresh.drift_score:.6f}"
            )

    def test_reset_after_drift_clears_alarm_state(self):
        """Drift fires → reset → stable data should not fire again."""
        monitor = DriftMonitor(method="PageHinkley", sensitivity=0.005, min_window=10)
        rng = random.Random(7)
        # Stable
        for _ in range(100):
            monitor.update({"x": rng.gauss(0, 0.1)})
        # Drift
        for _ in range(50):
            monitor.update({"x": 100.0})

        monitor.reset()

        # Post-reset stable data should produce no alarms
        rng2 = random.Random(99)
        results = [monitor.update({"x": rng2.gauss(0, 0.1)}) for _ in range(200)]
        assert not any(r.drift_detected for r in results), (
            "Monitor should not fire after reset on stable data"
        )

    def test_multiple_resets_are_safe(self):
        monitor = DriftMonitor()
        for _ in range(3):
            for rec in _stable(20):
                monitor.update(rec)
            monitor.reset()
        # Should be fully clean after multiple resets
        assert monitor._n_updates == 0
        assert monitor._score_history == []

    def test_reset_preserves_configuration(self):
        """reset() must NOT touch configuration parameters."""
        monitor = DriftMonitor(
            method="KSWIN",
            sensitivity=0.01,
            min_window=50,
            max_window=5000,
        )
        for rec in _stable(30):
            monitor.update(rec)
        monitor.reset()
        assert monitor.method == "KSWIN"
        assert monitor.sensitivity == 0.01
        assert monitor.min_window == 50
        assert monitor.max_window == 5000


# ---------------------------------------------------------------------------
# Per-detector reset()
# ---------------------------------------------------------------------------

class TestDetectorResetIsolation:
    def _run_then_reset(self, detector, values: list):
        for v in values:
            detector.update(v)
        detector.reset()
        return detector

    def test_adwin_reset_matches_fresh(self):
        det_reset = self._run_then_reset(ADWIN(delta=0.002), [float(i) for i in range(50)])
        det_fresh = ADWIN(delta=0.002)
        # Feed same value; should produce same result
        v = 1.23
        assert det_reset.update(v) == det_fresh.update(v)

    def test_pagehinkley_reset_matches_fresh(self):
        det_reset = self._run_then_reset(PageHinkley(), [float(i) for i in range(50)])
        det_fresh = PageHinkley()
        v = 0.5
        assert det_reset.update(v) == det_fresh.update(v)

    def test_ddm_reset_matches_fresh(self):
        det_reset = self._run_then_reset(DDM(), [1.0] * 80)
        det_fresh = DDM()
        assert det_reset.update(1.0) == det_fresh.update(1.0)
