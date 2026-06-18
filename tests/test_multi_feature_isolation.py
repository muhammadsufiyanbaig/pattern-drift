"""
Test category 5: Multi-feature isolation tests.

Inject drift into exactly one feature and confirm it does not bleed
into the results or scores of the unaffected features.
"""
import random
import pytest
from pattern_drift import DriftMonitor


def _rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


def _stable_val(rng: random.Random) -> float:
    return rng.gauss(0.0, 0.1)


def _drifted_val(rng: random.Random) -> float:
    return rng.gauss(10.0, 0.1)


# ---------------------------------------------------------------------------
# Isolation: only the drifting feature appears in drifted_features
# ---------------------------------------------------------------------------

class TestMultiFeatureIsolation:
    @pytest.mark.parametrize("method", ["ADWIN", "PageHinkley"])
    def test_only_drifting_feature_reported(self, method):
        """
        Three features: 'a' drifts, 'b' and 'c' remain stable.
        Verify 'a' is the only entry in drifted_features.
        """
        monitor = DriftMonitor(method=method, sensitivity=0.002, min_window=30)
        rng = _rng(11)

        # Warm-up: all stable
        for _ in range(150):
            monitor.update({
                "a": _stable_val(rng),
                "b": _stable_val(rng),
                "c": _stable_val(rng),
            })

        # Inject drift on 'a' only
        drift_events = []
        for _ in range(200):
            result = monitor.update({
                "a": _drifted_val(rng),
                "b": _stable_val(rng),
                "c": _stable_val(rng),
            })
            if result.drift_detected:
                drift_events.append(result)

        assert len(drift_events) > 0, f"{method} should detect drift on feature 'a'"

        for event in drift_events:
            assert "a" in event.drifted_features, "'a' must appear in every drift event"
            spurious = set(event.drifted_features) - {"a"}
            assert not spurious, (
                f"Spurious features {spurious} reported as drifted — expected isolation"
            )

    def test_stable_feature_scores_remain_low(self):
        """The drift scores for non-drifting features should stay below sensitivity."""
        monitor = DriftMonitor(method="PageHinkley", sensitivity=0.005, min_window=30)
        rng = _rng(22)
        sensitivity = 0.005

        # Warm-up
        for _ in range(150):
            monitor.update({"a": _stable_val(rng), "b": _stable_val(rng)})

        # Drift 'a', keep 'b' stable; collect scores for 'b'
        b_scores = []
        for _ in range(100):
            monitor.update({"a": _drifted_val(rng), "b": _stable_val(rng)})
            if monitor._score_history:
                last = monitor._score_history[-1]
                if "b" in last:
                    b_scores.append(last["b"])

        # Most scores for 'b' should be well below threshold
        high_b = [s for s in b_scores if s > 0.5]
        assert len(high_b) == 0, (
            f"Feature 'b' had {len(high_b)} high-score observations despite being stable"
        )

    def test_each_feature_has_independent_detector(self):
        """Verify one detector instance per feature, not shared."""
        monitor = DriftMonitor(method="ADWIN")
        monitor.update({"x": 1.0, "y": 2.0, "z": 3.0})
        assert len(monitor._detectors) == 3
        assert "x" in monitor._detectors
        assert "y" in monitor._detectors
        assert "z" in monitor._detectors
        # All different instances
        assert monitor._detectors["x"] is not monitor._detectors["y"]
        assert monitor._detectors["y"] is not monitor._detectors["z"]

    def test_drift_in_two_of_three_features(self):
        """Drift in 'a' and 'b' but not 'c'; 'c' must never appear in drifted_features."""
        monitor = DriftMonitor(method="PageHinkley", sensitivity=0.002, min_window=20)
        rng = _rng(33)

        for _ in range(120):
            monitor.update({
                "a": _stable_val(rng),
                "b": _stable_val(rng),
                "c": _stable_val(rng),
            })

        c_reported = False
        for _ in range(150):
            result = monitor.update({
                "a": _drifted_val(rng),
                "b": _drifted_val(rng),
                "c": _stable_val(rng),
            })
            if result.drift_detected and "c" in result.drifted_features:
                c_reported = True
                break

        assert not c_reported, "Feature 'c' should never be reported as drifted"

    def test_feature_filter_restricts_monitoring(self):
        """When features=['a'], monitor ignores 'b' entirely."""
        monitor = DriftMonitor(method="ADWIN", features=["a"], min_window=10)
        rng = _rng(44)

        for _ in range(50):
            monitor.update({"a": _stable_val(rng), "b": _stable_val(rng)})

        assert "b" not in monitor._detectors, "'b' should not have a detector"
        assert "a" in monitor._detectors, "'a' must have a detector"
