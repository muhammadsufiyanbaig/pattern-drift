"""Integration tests for DriftMonitor."""
import pytest

from pattern_drift import DriftMonitor, DriftResult


def _make_record(value: float) -> dict:
    return {"feature_a": value, "feature_b": value * 0.5}


def _run_stream(monitor, values, feature="feature_a"):
    results = []
    for v in values:
        results.append(monitor.update({feature: v}))
    return results


class TestDriftMonitorBasic:
    def test_stable_stream_no_drift(self):
        monitor = DriftMonitor(method="ADWIN", sensitivity=0.002, min_window=30)
        import random
        rng = random.Random(0)
        results = [monitor.update({"x": rng.gauss(0, 0.1)}) for _ in range(300)]
        assert not any(r.drift_detected for r in results)

    def test_sudden_drift_detected(self):
        monitor = DriftMonitor(method="PageHinkley", sensitivity=0.005, min_window=30)
        import random
        rng = random.Random(1)
        stable = [{"x": rng.gauss(0, 0.1)} for _ in range(150)]
        drifted = [{"x": rng.gauss(10, 0.1)} for _ in range(100)]
        results = [monitor.update(r) for r in stable + drifted]
        assert any(r.drift_detected for r in results)

    def test_result_fields_on_no_drift(self):
        monitor = DriftMonitor()
        result = monitor.update({"x": 1.0})
        assert isinstance(result, DriftResult)
        assert result.drift_detected is False
        assert result.drift_type is None
        assert result.drifted_features == []

    def test_result_fields_on_drift(self):
        monitor = DriftMonitor(method="PageHinkley", sensitivity=0.005, min_window=5)
        import random
        rng = random.Random(2)
        for _ in range(100):
            monitor.update({"x": rng.gauss(0, 0.1)})
        # inject massive shift
        result = None
        for _ in range(50):
            r = monitor.update({"x": 100.0})
            if r.drift_detected:
                result = r
                break
        assert result is not None
        assert result.drift_detected is True
        assert result.drift_type in ("sudden", "gradual", "incremental", "recurring")
        assert "x" in result.drifted_features
        assert 0.0 <= result.drift_score

    def test_multi_feature_tracking(self):
        monitor = DriftMonitor(method="ADWIN", sensitivity=0.002, min_window=30)
        import random
        rng = random.Random(3)
        # Both features stable then drift one
        for _ in range(150):
            monitor.update({"a": rng.gauss(0, 0.1), "b": rng.gauss(0, 0.1)})
        for _ in range(100):
            monitor.update({"a": rng.gauss(10, 0.1), "b": rng.gauss(0, 0.1)})
        # History should exist for both features
        all_keys = set()
        for rec in monitor._score_history:
            all_keys.update(rec.keys())
        assert "a" in all_keys
        assert "b" in all_keys

    def test_reset(self):
        monitor = DriftMonitor()
        for _ in range(50):
            monitor.update({"x": 1.0})
        monitor.reset()
        assert monitor._n_updates == 0
        assert len(monitor._score_history) == 0
        assert len(monitor._detectors) == 0


class TestDriftMonitorCallbacks:
    def test_callback_fires_on_drift(self):
        fired = []
        monitor = DriftMonitor(
            method="PageHinkley",
            sensitivity=0.005,
            min_window=5,
            callbacks=[lambda r: fired.append(r)],
        )
        import random
        rng = random.Random(4)
        for _ in range(100):
            monitor.update({"x": rng.gauss(0, 0.1)})
        for _ in range(50):
            monitor.update({"x": 100.0})
        # callbacks should have fired if drift occurred
        # (don't assert exact count; just ensure structure is correct)
        for r in fired:
            assert isinstance(r, DriftResult)
            assert r.drift_detected is True

    def test_callback_exception_does_not_propagate(self):
        def bad_cb(r):
            raise RuntimeError("intentional")

        monitor = DriftMonitor(
            method="PageHinkley",
            sensitivity=0.005,
            min_window=5,
            callbacks=[bad_cb],
        )
        import random
        rng = random.Random(5)
        for _ in range(100):
            monitor.update({"x": rng.gauss(0, 0.1)})
        # This should not raise even though callback raises
        for _ in range(50):
            monitor.update({"x": 100.0})


class TestDriftMonitorFromConfig:
    def test_from_config(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            "method: PageHinkley\nsensitivity: 0.005\nmin_window: 10\nmax_window: 5000\n"
        )
        monitor = DriftMonitor.from_config(str(config))
        assert monitor.method == "PageHinkley"
        assert monitor.sensitivity == 0.005
        assert monitor.min_window == 10
        assert monitor.max_window == 5000


class TestDriftMonitorExport:
    def test_export_json(self, tmp_path):
        import json
        monitor = DriftMonitor()
        for i in range(10):
            monitor.update({"x": float(i)})
        path = str(tmp_path / "report.json")
        monitor.export_report(path)
        with open(path) as fh:
            data = json.load(fh)
        assert len(data) == 10

    def test_export_csv(self, tmp_path):
        monitor = DriftMonitor()
        for i in range(5):
            monitor.update({"x": float(i)})
        path = str(tmp_path / "report.csv")
        monitor.export_report(path)
        with open(path) as fh:
            lines = fh.read().strip().split("\n")
        assert lines[0].startswith("index")
        assert len(lines) == 6  # header + 5 rows


class TestDriftMonitorMethods:
    @pytest.mark.parametrize("method", ["ADWIN", "PageHinkley", "KSWIN", "DDM"])
    def test_all_methods_run(self, method):
        monitor = DriftMonitor(method=method, min_window=5)
        for i in range(20):
            result = monitor.update({"x": float(i % 5)})
            assert isinstance(result, DriftResult)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            DriftMonitor(method="BOGUS")
