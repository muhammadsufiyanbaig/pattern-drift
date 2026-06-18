"""
Test category 6: Retraining window correctness tests.

Verify the backward scan returns a window that:
  - Falls entirely within the pre-drift stable period
  - Has a sensible (high) confidence score
  - Respects the buffer trim on both ends
  - Has correct n_samples accounting
"""
import pytest
from pattern_drift.window_engine import RetrainingWindowEngine
from pattern_drift.result import RetrainingWindowResult


# ---------------------------------------------------------------------------
# History construction helpers
# ---------------------------------------------------------------------------

def _history(n_stable: int, n_drifted: int, sensitivity: float = 0.002):
    """Build a synthetic score_history list."""
    stable_score = sensitivity * 0.1    # well below threshold
    drift_score  = sensitivity * 10.0   # well above threshold
    return (
        [{"x": stable_score} for _ in range(n_stable)]
        + [{"x": drift_score} for _ in range(n_drifted)]
    )


# ---------------------------------------------------------------------------
# Core correctness
# ---------------------------------------------------------------------------

class TestRetrainingWindowCorrectness:
    def test_window_falls_within_stable_period(self):
        n_stable, n_drifted = 200, 30
        history = _history(n_stable, n_drifted)
        engine = RetrainingWindowEngine(sensitivity=0.002, buffer_pct=0.10)
        rw = engine.find_window(history, drift_index=n_stable)

        assert rw is not None
        # Window must be entirely inside [0, n_stable)
        assert rw.start >= 0, f"Window start {rw.start} is negative"
        assert rw.end < n_stable, (
            f"Window end {rw.end} is at or past drift point {n_stable}"
        )

    def test_window_does_not_overlap_drifted_region(self):
        n_stable, n_drifted = 150, 50
        history = _history(n_stable, n_drifted)
        engine = RetrainingWindowEngine(sensitivity=0.002)
        rw = engine.find_window(history, drift_index=n_stable)

        assert rw is not None
        assert rw.end < n_stable, "Window must not include any drifted samples"

    def test_high_confidence_on_clean_stable_region(self):
        """When the stable segment is long and perfectly stable, confidence → 1.0."""
        n_stable = 300
        history = _history(n_stable, 10)
        engine = RetrainingWindowEngine(sensitivity=0.002, buffer_pct=0.05)
        rw = engine.find_window(history, drift_index=n_stable)

        assert rw is not None
        assert rw.confidence >= 0.8, (
            f"Expected high confidence on clean stable segment, got {rw.confidence}"
        )

    def test_n_samples_matches_window_span(self):
        history = _history(200, 20)
        engine = RetrainingWindowEngine(sensitivity=0.002, buffer_pct=0.10)
        rw = engine.find_window(history, drift_index=200)

        assert rw is not None
        expected_n = rw.end - rw.start + 1
        assert rw.n_samples == expected_n, (
            f"n_samples={rw.n_samples} does not match span {expected_n}"
        )

    def test_confidence_between_0_and_1(self):
        history = _history(100, 10)
        engine = RetrainingWindowEngine(sensitivity=0.002)
        rw = engine.find_window(history, drift_index=100)

        if rw is not None:
            assert 0.0 <= rw.confidence <= 1.0


# ---------------------------------------------------------------------------
# Buffer trimming
# ---------------------------------------------------------------------------

class TestBufferTrimming:
    @pytest.mark.parametrize("buffer_pct", [0.05, 0.10, 0.20])
    def test_buffer_applied_to_both_ends(self, buffer_pct):
        n_stable = 200
        history = _history(n_stable, 20)
        engine = RetrainingWindowEngine(sensitivity=0.002, buffer_pct=buffer_pct)
        rw = engine.find_window(history, drift_index=n_stable)

        assert rw is not None
        min_buf = max(1, int(n_stable * buffer_pct))
        assert rw.start >= min_buf, (
            f"start={rw.start} should be >= buffer {min_buf}"
        )
        assert rw.end <= n_stable - 1 - min_buf, (
            f"end={rw.end} should leave room for buffer"
        )

    def test_zero_buffer_returns_near_full_stable_window(self):
        """
        buffer_pct=0.0 still applies a minimum buffer of 1 sample (max(1, 0) = 1)
        to avoid edge effects.  Verify the window is as large as possible given
        that minimum.
        """
        n_stable = 100
        history = _history(n_stable, 10)
        engine = RetrainingWindowEngine(sensitivity=0.002, buffer_pct=0.0)
        rw = engine.find_window(history, drift_index=n_stable)

        assert rw is not None
        # Minimum buffer of 1 sample applied on each side
        assert rw.start <= 1, f"start={rw.start} should be at most 1 with 0 % buffer"
        assert rw.end >= n_stable - 2, (
            f"end={rw.end} should be at least {n_stable-2} with 0 % buffer"
        )


# ---------------------------------------------------------------------------
# Edge / boundary cases
# ---------------------------------------------------------------------------

class TestRetrainingWindowEdgeCases:
    def test_returns_none_on_empty_history(self):
        engine = RetrainingWindowEngine()
        assert engine.find_window([], 0) is None

    def test_returns_none_when_drift_at_index_zero(self):
        engine = RetrainingWindowEngine()
        history = _history(100, 10)
        assert engine.find_window(history, 0) is None

    def test_returns_none_when_no_stable_segment(self):
        """All scores above threshold — no stable window exists."""
        engine = RetrainingWindowEngine(sensitivity=0.002)
        history = [{"x": 1.0} for _ in range(100)]  # all above threshold
        rw = engine.find_window(history, 80)
        assert rw is None

    def test_very_short_stable_segment_returns_none_or_valid(self):
        """With only 2 stable samples, buffer may collapse the window."""
        engine = RetrainingWindowEngine(sensitivity=0.002, buffer_pct=0.50)
        history = _history(2, 50)
        rw = engine.find_window(history, 2)
        # Either None (collapsed by buffer) or a valid result
        if rw is not None:
            assert rw.start <= rw.end
            assert rw.n_samples >= 1

    def test_multiple_calls_are_idempotent(self):
        """Calling find_window twice with the same args returns the same result."""
        engine = RetrainingWindowEngine(sensitivity=0.002, buffer_pct=0.10)
        history = _history(150, 20)
        rw1 = engine.find_window(history, 150)
        rw2 = engine.find_window(history, 150)

        if rw1 is None:
            assert rw2 is None
        else:
            assert rw1.start == rw2.start
            assert rw1.end == rw2.end
            assert rw1.confidence == rw2.confidence
