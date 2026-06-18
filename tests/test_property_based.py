"""
Test category 10: Property-based tests (Hypothesis).

Auto-generate arbitrary inputs and verify core invariants always hold
regardless of input shape or value.

Requires: pip install hypothesis
Tests are skipped gracefully if Hypothesis is not installed.
"""
import pytest

hypothesis = pytest.importorskip("hypothesis", reason="hypothesis not installed")

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from pattern_drift import DriftMonitor, DriftResult
from pattern_drift.detectors import ADWIN, PageHinkley, KSWIN, DDM

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

finite_float = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)

feature_name = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=20,
)

record_strategy = st.fixed_dictionaries(
    {"x": finite_float, "y": finite_float, "z": finite_float}
)

small_record = st.dictionaries(
    keys=feature_name,
    values=finite_float,
    min_size=1,
    max_size=5,
)


# ---------------------------------------------------------------------------
# Invariant: update() always returns a DriftResult
# ---------------------------------------------------------------------------

class TestUpdateAlwaysReturnsDriftResult:
    @given(record=record_strategy)
    @settings(max_examples=200)
    def test_result_type(self, record):
        monitor = DriftMonitor(method="ADWIN", min_window=2)
        result = monitor.update(record)
        assert isinstance(result, DriftResult)

    @given(records=st.lists(record_strategy, min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_all_results_are_drift_results(self, records):
        monitor = DriftMonitor(method="PageHinkley", min_window=2)
        for rec in records:
            result = monitor.update(rec)
            assert isinstance(result, DriftResult)


# ---------------------------------------------------------------------------
# Invariant: drift_score is always non-negative
# ---------------------------------------------------------------------------

class TestDriftScoreNonNegative:
    @given(records=st.lists(record_strategy, min_size=5, max_size=100))
    @settings(max_examples=100)
    def test_adwin_score_non_negative(self, records):
        monitor = DriftMonitor(method="ADWIN", min_window=2)
        for rec in records:
            result = monitor.update(rec)
            assert result.drift_score >= 0.0, f"Negative score: {result.drift_score}"

    @given(records=st.lists(record_strategy, min_size=5, max_size=100))
    @settings(max_examples=100)
    def test_pagehinkley_score_non_negative(self, records):
        monitor = DriftMonitor(method="PageHinkley", min_window=2)
        for rec in records:
            result = monitor.update(rec)
            assert result.drift_score >= 0.0


# ---------------------------------------------------------------------------
# Invariant: drifted_features is a subset of monitored features
# ---------------------------------------------------------------------------

class TestDriftedFeaturesSubsetOfMonitored:
    @given(records=st.lists(record_strategy, min_size=10, max_size=80))
    @settings(max_examples=100)
    def test_drifted_features_subset(self, records):
        monitor = DriftMonitor(method="PageHinkley", min_window=2, sensitivity=0.001)
        for rec in records:
            result = monitor.update(rec)
            if result.drift_detected:
                spurious = set(result.drifted_features) - set(rec.keys())
                assert not spurious, (
                    f"drifted_features {result.drifted_features} contains "
                    f"features not in record {list(rec.keys())}"
                )


# ---------------------------------------------------------------------------
# Invariant: drift_type is one of the valid labels or None
# ---------------------------------------------------------------------------

VALID_TYPES = {None, "sudden", "gradual", "incremental", "recurring"}


class TestDriftTypeIsValid:
    @given(records=st.lists(record_strategy, min_size=5, max_size=80))
    @settings(max_examples=100)
    def test_drift_type_valid(self, records):
        monitor = DriftMonitor(method="ADWIN", min_window=2)
        for rec in records:
            result = monitor.update(rec)
            assert result.drift_type in VALID_TYPES, (
                f"Invalid drift_type: {result.drift_type!r}"
            )

    @given(records=st.lists(record_strategy, min_size=5, max_size=50))
    @settings(max_examples=50)
    def test_drift_type_none_iff_no_drift(self, records):
        monitor = DriftMonitor(method="ADWIN", min_window=2)
        for rec in records:
            result = monitor.update(rec)
            if not result.drift_detected:
                assert result.drift_type is None
            else:
                assert result.drift_type is not None


# ---------------------------------------------------------------------------
# Invariant: n_updates increments by 1 per update()
# ---------------------------------------------------------------------------

class TestNUpdatesMonotonicity:
    @given(n=st.integers(min_value=1, max_value=100))
    @settings(max_examples=50)
    def test_n_updates_increments_correctly(self, n):
        monitor = DriftMonitor(method="ADWIN", min_window=2)
        for i in range(n):
            monitor.update({"x": float(i)})
        assert monitor._n_updates == n


# ---------------------------------------------------------------------------
# Invariant: score_history length bounded by max_window
# ---------------------------------------------------------------------------

class TestScoreHistoryBounded:
    @given(
        records=st.lists(record_strategy, min_size=1, max_size=200),
        max_window=st.integers(min_value=10, max_value=150),
    )
    @settings(max_examples=50)
    def test_history_length_bounded(self, records, max_window):
        monitor = DriftMonitor(method="ADWIN", max_window=max_window, min_window=2)
        for rec in records:
            monitor.update(rec)
        assert len(monitor._score_history) <= max_window


# ---------------------------------------------------------------------------
# Invariant: per-detector update returns (bool, float≥0)
# ---------------------------------------------------------------------------

class TestDetectorOutputTypes:
    @given(value=finite_float)
    @settings(max_examples=200)
    def test_adwin_output_types(self, value):
        det = ADWIN(delta=0.01)
        detected, score = det.update(value)
        assert isinstance(detected, bool)
        assert isinstance(score, float)
        assert score >= 0.0

    @given(value=finite_float)
    @settings(max_examples=200)
    def test_pagehinkley_output_types(self, value):
        det = PageHinkley()
        detected, score = det.update(value)
        assert isinstance(detected, bool)
        assert score >= 0.0

    @given(value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    @settings(max_examples=200)
    def test_ddm_output_types(self, value):
        det = DDM(min_num_instances=1)
        detected, score = det.update(value)
        assert isinstance(detected, bool)
        assert score >= 0.0
