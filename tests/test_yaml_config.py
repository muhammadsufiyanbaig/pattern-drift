"""
Test category 8: YAML config round-trip tests.

Verify DriftMonitor.from_config() parses every supported field correctly,
raises helpful errors on invalid input, and uses sensible defaults for
missing optional fields.
"""
import pytest
from pattern_drift import DriftMonitor


def _write_yaml(tmp_path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return str(p)


# ---------------------------------------------------------------------------
# Full round-trip
# ---------------------------------------------------------------------------

class TestYAMLRoundTrip:
    def test_all_fields_parsed(self, tmp_path):
        path = _write_yaml(tmp_path, """\
method: KSWIN
sensitivity: 0.03
min_window: 45
max_window: 7500
features:
  - age
  - income
  - churn_score
""")
        monitor = DriftMonitor.from_config(path)
        assert monitor.method == "KSWIN"
        assert monitor.sensitivity == 0.03
        assert monitor.min_window == 45
        assert monitor.max_window == 7500
        assert monitor._extractor.features == ["age", "income", "churn_score"]

    @pytest.mark.parametrize("method", ["ADWIN", "PageHinkley", "KSWIN", "DDM"])
    def test_all_methods_load(self, tmp_path, method):
        path = _write_yaml(tmp_path, f"method: {method}\n")
        monitor = DriftMonitor.from_config(path)
        assert monitor.method == method

    def test_round_trip_produces_functional_monitor(self, tmp_path):
        path = _write_yaml(tmp_path, """\
method: PageHinkley
sensitivity: 0.005
min_window: 10
max_window: 500
""")
        monitor = DriftMonitor.from_config(path)
        result = monitor.update({"x": 1.0})
        from pattern_drift import DriftResult
        assert isinstance(result, DriftResult)


# ---------------------------------------------------------------------------
# Defaults for missing optional fields
# ---------------------------------------------------------------------------

class TestYAMLDefaults:
    def test_method_defaults_to_adwin(self, tmp_path):
        path = _write_yaml(tmp_path, "sensitivity: 0.002\n")
        monitor = DriftMonitor.from_config(path)
        assert monitor.method == "ADWIN"

    def test_sensitivity_defaults(self, tmp_path):
        path = _write_yaml(tmp_path, "method: ADWIN\n")
        monitor = DriftMonitor.from_config(path)
        assert monitor.sensitivity == 0.002

    def test_min_window_defaults(self, tmp_path):
        path = _write_yaml(tmp_path, "method: ADWIN\n")
        monitor = DriftMonitor.from_config(path)
        assert monitor.min_window == 30

    def test_max_window_defaults(self, tmp_path):
        path = _write_yaml(tmp_path, "method: ADWIN\n")
        monitor = DriftMonitor.from_config(path)
        assert monitor.max_window == 10_000

    def test_features_defaults_to_none(self, tmp_path):
        path = _write_yaml(tmp_path, "method: ADWIN\n")
        monitor = DriftMonitor.from_config(path)
        # features=None means auto-detect
        assert monitor._extractor.features is None

    def test_empty_yaml_uses_all_defaults(self, tmp_path):
        path = _write_yaml(tmp_path, "{}\n")
        monitor = DriftMonitor.from_config(path)
        assert monitor.method == "ADWIN"
        assert monitor.sensitivity == 0.002


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestYAMLErrorHandling:
    def test_invalid_method_raises_value_error(self, tmp_path):
        path = _write_yaml(tmp_path, "method: BOGUS\n")
        with pytest.raises(ValueError, match="Unknown method"):
            DriftMonitor.from_config(path)

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            DriftMonitor.from_config("/nonexistent/path/config.yaml")

    def test_numeric_sensitivity_preserved(self, tmp_path):
        path = _write_yaml(tmp_path, "sensitivity: 1.5e-3\n")
        monitor = DriftMonitor.from_config(path)
        assert abs(monitor.sensitivity - 0.0015) < 1e-10

    def test_feature_list_order_preserved(self, tmp_path):
        path = _write_yaml(tmp_path, """\
features:
  - z_score
  - alpha
  - beta
""")
        monitor = DriftMonitor.from_config(path)
        assert monitor._extractor.features == ["z_score", "alpha", "beta"]
