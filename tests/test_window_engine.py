"""Tests for the Retraining Window Engine."""
from pattern_drift.window_engine import RetrainingWindowEngine


class TestRetrainingWindowEngine:
    def _make_history(self, n_stable: int, n_unstable: int, threshold: float = 0.002):
        """Create a score history with stable then unstable records."""
        stable = [{"x": threshold * 0.5} for _ in range(n_stable)]
        unstable = [{"x": threshold * 5.0} for _ in range(n_unstable)]
        return stable + unstable

    def test_returns_none_on_empty(self):
        engine = RetrainingWindowEngine()
        assert engine.find_window([], 0) is None

    def test_returns_none_on_zero_index(self):
        engine = RetrainingWindowEngine()
        history = self._make_history(50, 10)
        assert engine.find_window(history, 0) is None

    def test_finds_stable_window(self):
        engine = RetrainingWindowEngine(sensitivity=0.002)
        history = self._make_history(100, 10)
        drift_idx = 100
        result = engine.find_window(history, drift_idx)
        assert result is not None
        assert result.start < result.end
        assert result.n_samples > 0
        assert 0.0 <= result.confidence <= 1.0

    def test_buffer_applied(self):
        engine = RetrainingWindowEngine(sensitivity=0.002, buffer_pct=0.10)
        history = self._make_history(100, 10)
        result = engine.find_window(history, 100)
        assert result is not None
        # Buffer of 10% means start > 0 and end < 99
        assert result.start > 0
        assert result.end < 100

    def test_no_stable_window_returns_none(self):
        engine = RetrainingWindowEngine(sensitivity=0.002)
        # All records above threshold
        history = [{"x": 1.0} for _ in range(50)]
        result = engine.find_window(history, 40)
        assert result is None
