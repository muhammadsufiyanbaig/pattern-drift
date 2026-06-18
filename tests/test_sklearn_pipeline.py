"""
Test category 11: scikit-learn pipeline compatibility tests.

Verify the DriftDetector wrapper:
  - Honours sklearn's fit/transform contract
  - Passes data through unchanged (shape, dtype, values)
  - Works inside a Pipeline with other transformers
  - get_params / set_params are consistent
  - transform() before fit() raises RuntimeError
  - Exposes the underlying monitor via .monitor
"""
import random
import pytest

sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")
pandas = pytest.importorskip("pandas", reason="pandas not installed")

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pattern_drift.sklearn_wrapper import DriftDetector
from pattern_drift import DriftMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(n: int, n_features: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(n_features)]
    return pd.DataFrame(rng.standard_normal((n, n_features)), columns=cols)


# ---------------------------------------------------------------------------
# fit / transform contract
# ---------------------------------------------------------------------------

class TestFitTransformContract:
    def test_fit_returns_self(self):
        det = DriftDetector(method="ADWIN")
        X = _df(50)
        result = det.fit(X)
        assert result is det

    def test_transform_returns_input_unchanged(self):
        det = DriftDetector(method="ADWIN")
        X_train = _df(100)
        X_test = _df(20, seed=99)
        det.fit(X_train)
        X_out = det.transform(X_test)
        # Output must be identical object or equal in value
        assert X_out is X_test or X_out.equals(X_test), (
            "transform() must return the input data unchanged"
        )

    def test_transform_before_fit_raises(self):
        det = DriftDetector()
        with pytest.raises(RuntimeError, match="fit"):
            det.transform(_df(10))

    def test_fit_sets_monitor_attribute(self):
        det = DriftDetector(method="PageHinkley")
        assert det.monitor is None
        det.fit(_df(50))
        assert det.monitor is not None
        assert isinstance(det.monitor, DriftMonitor)


# ---------------------------------------------------------------------------
# Shape and dtype preservation
# ---------------------------------------------------------------------------

class TestPassThrough:
    def test_shape_preserved(self):
        det = DriftDetector()
        X = _df(30, n_features=5)
        det.fit(X)
        X_out = det.transform(_df(10, n_features=5))
        assert X_out.shape == (10, 5)

    def test_column_names_preserved(self):
        det = DriftDetector()
        X = _df(30)
        det.fit(X)
        X_out = det.transform(_df(5))
        assert list(X_out.columns) == list(X.columns)

    def test_values_unchanged(self):
        det = DriftDetector()
        X_train = _df(50)
        det.fit(X_train)
        X_test = _df(10, seed=7)
        expected = X_test.values.copy()
        X_out = det.transform(X_test)
        np.testing.assert_array_equal(X_out.values, expected)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_works_inside_sklearn_pipeline(self):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("drift",  DriftDetector(method="ADWIN")),
        ])
        X_train = _df(100)
        pipe.fit(X_train)
        X_test = _df(20, seed=5)
        X_out = pipe.transform(X_test)
        # Data passes through the pipeline; shape must be preserved
        assert X_out.shape == X_test.shape

    def test_pipeline_fit_transform(self):
        pipe = Pipeline([
            ("drift", DriftDetector(method="PageHinkley")),
        ])
        X = _df(50)
        X_out = pipe.fit_transform(X)
        assert X_out.shape == X.shape

    def test_multiple_detectors_in_pipeline(self):
        """Two DriftDetectors on independent sub-features can coexist."""
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("drift1", DriftDetector(method="ADWIN", features=["f0", "f1"])),
            ("drift2", DriftDetector(method="KSWIN", features=["f2"])),
        ])
        X = _df(80, n_features=3)
        pipe.fit(X)
        X_out = pipe.transform(_df(20, n_features=3))
        assert X_out.shape == (20, 3)

    def test_drift_detected_inside_pipeline(self):
        """Inject drift after fit and confirm the monitor inside the pipeline fires."""
        fired = []
        det = DriftDetector(
            method="PageHinkley",
            sensitivity=0.005,
            callbacks=[fired.append],
        )
        pipe = Pipeline([("drift", det)])
        X_train = _df(100)
        pipe.fit(X_train)

        # Inject massive drift
        rng = np.random.default_rng(88)
        for _ in range(30):
            batch = pd.DataFrame(rng.standard_normal((1, 3)) + 50,
                                 columns=["f0", "f1", "f2"])
            pipe.transform(batch)

        # Callbacks should have fired
        assert len(fired) > 0, "DriftDetector inside pipeline should fire callbacks on drift"


# ---------------------------------------------------------------------------
# get_params / set_params
# ---------------------------------------------------------------------------

class TestGetSetParams:
    def test_get_params_returns_all_init_params(self):
        det = DriftDetector(method="KSWIN", sensitivity=0.01, min_window=40)
        params = det.get_params()
        assert params["method"] == "KSWIN"
        assert params["sensitivity"] == 0.01
        assert params["min_window"] == 40

    def test_set_params_updates_attributes(self):
        det = DriftDetector(method="ADWIN", sensitivity=0.002)
        det.set_params(method="DDM", sensitivity=0.05)
        assert det.method == "DDM"
        assert det.sensitivity == 0.05

    def test_get_params_round_trips(self):
        det = DriftDetector(method="PageHinkley", sensitivity=0.007, max_window=2000)
        params = det.get_params()
        det2 = DriftDetector(**params)
        assert det2.method == det.method
        assert det2.sensitivity == det.sensitivity
        assert det2.max_window == det.max_window

    def test_sklearn_clone_compatible(self):
        """sklearn.base.clone() uses get_params/set_params; must not raise."""
        from sklearn.base import clone
        det = DriftDetector(method="ADWIN", sensitivity=0.002, min_window=30)
        cloned = clone(det)
        assert cloned.method == det.method
        assert cloned.sensitivity == det.sensitivity


# ---------------------------------------------------------------------------
# monitor property
# ---------------------------------------------------------------------------

class TestMonitorProperty:
    def test_monitor_is_none_before_fit(self):
        det = DriftDetector()
        assert det.monitor is None

    def test_monitor_method_matches_config(self):
        det = DriftDetector(method="KSWIN")
        det.fit(_df(50))
        assert det.monitor.method == "KSWIN"

    def test_monitor_sensitivity_matches_config(self):
        det = DriftDetector(sensitivity=0.03)
        det.fit(_df(50))
        assert det.monitor.sensitivity == 0.03
