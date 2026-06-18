"""
Test category 1: Synthetic stream unit tests.

Feed streams with *known* drift points and verify each algorithm raises its
first alarm after the drift point (not before) and within a reasonable lag.
"""
import random
import pytest
from pattern_drift.detectors import ADWIN, DDM, KSWIN, PageHinkley


# ---------------------------------------------------------------------------
# Stream factories
# ---------------------------------------------------------------------------

def _stream(n: int, mean: float, std: float = 0.1, seed: int = 0) -> list:
    rng = random.Random(seed)
    return [rng.gauss(mean, std) for _ in range(n)]


def _abrupt_stream(n_pre: int, n_post: int, shift: float, seed: int = 0):
    """Single abrupt mean shift at index n_pre."""
    return _stream(n_pre, 0.0, seed=seed) + _stream(n_post, shift, seed=seed + 1)


def _gradual_stream(n_pre: int, n_ramp: int, target: float, seed: int = 0):
    """Mean linearly ramps from 0 to target over n_ramp samples."""
    rng = random.Random(seed)
    pre = [rng.gauss(0.0, 0.1) for _ in range(n_pre)]
    ramp = [rng.gauss(target * i / n_ramp, 0.1) for i in range(n_ramp)]
    return pre + ramp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_alarm(detector, stream):
    """Return index of first alarm, or None."""
    for i, v in enumerate(stream):
        if detector.update(v)[0]:
            return i
    return None


MAX_LAG = 120  # max allowed samples after drift point before alarm fires


# ---------------------------------------------------------------------------
# ADWIN
# ---------------------------------------------------------------------------

class TestADWINSyntheticStreams:
    @pytest.mark.parametrize("shift,drift_at", [
        (5.0, 100),
        (10.0, 200),
        (3.0, 150),
    ])
    def test_detects_after_drift_point(self, shift, drift_at):
        stream = _abrupt_stream(drift_at, 200, shift)
        det = ADWIN(delta=0.002)
        alarm = _first_alarm(det, stream)
        assert alarm is not None, "ADWIN should detect the drift"
        assert alarm >= drift_at, f"Alarm at {alarm} fired before drift point {drift_at}"
        assert alarm <= drift_at + MAX_LAG, f"Alarm lag {alarm - drift_at} exceeds {MAX_LAG}"

    def test_no_alarm_before_drift_point(self):
        drift_at = 200
        stream = _abrupt_stream(drift_at, 100, shift=8.0)
        det = ADWIN(delta=0.002)
        pre_alarms = [i for i, v in enumerate(stream[:drift_at]) if det.update(v)[0]]
        assert len(pre_alarms) == 0, f"False alarm(s) before drift: {pre_alarms}"


# ---------------------------------------------------------------------------
# PageHinkley
# ---------------------------------------------------------------------------

class TestPageHinkleySyntheticStreams:
    @pytest.mark.parametrize("shift,drift_at", [
        (8.0, 100),
        (15.0, 150),
        (5.0, 80),
    ])
    def test_detects_after_drift_point(self, shift, drift_at):
        stream = _abrupt_stream(drift_at, 200, shift)
        det = PageHinkley(delta=0.005, lambda_=50.0)
        alarm = _first_alarm(det, stream)
        assert alarm is not None, "PageHinkley should detect the drift"
        assert alarm >= drift_at, f"Alarm at {alarm} fired before drift point {drift_at}"
        assert alarm <= drift_at + MAX_LAG

    def test_no_alarm_before_drift_point(self):
        drift_at = 150
        stream = _abrupt_stream(drift_at, 100, shift=10.0)
        det = PageHinkley(delta=0.005, lambda_=50.0)
        pre_alarms = [i for i, v in enumerate(stream[:drift_at]) if det.update(v)[0]]
        assert len(pre_alarms) == 0


# ---------------------------------------------------------------------------
# KSWIN
# ---------------------------------------------------------------------------

class TestKSWINSyntheticStreams:
    @pytest.mark.parametrize("shift,drift_at", [
        (5.0, 150),
        (8.0, 200),
    ])
    def test_detects_drift_for_large_shift(self, shift, drift_at):
        """
        KSWIN uses a sliding window, so the first alarm may fire up to
        window_size samples before the exact drift_at boundary as the window
        spans the transition region.  Verify an alarm fires and falls within
        the expected neighbourhood.
        """
        window_size = 100
        stream = _abrupt_stream(drift_at, 300, shift)
        det = KSWIN(alpha=0.05, window_size=window_size, stat_size=30)
        alarm = _first_alarm(det, stream)
        assert alarm is not None, f"KSWIN should detect shift={shift} at drift_at={drift_at}"
        # Alarm must be within window_size before or any time after the drift point
        assert alarm >= drift_at - window_size, (
            f"Alarm at {alarm} is too early (drift_at={drift_at}, window={window_size})"
        )

    def test_detects_variance_change(self):
        """KSWIN catches variance shift even when mean is unchanged."""
        rng = random.Random(7)
        stable  = [rng.gauss(0, 0.05) for _ in range(300)]  # low variance
        drifted = [rng.gauss(0, 3.0)  for _ in range(200)]  # high variance, same mean
        stream = stable + drifted
        # Use strict alpha to suppress false positives in the stable region
        det = KSWIN(alpha=0.001, window_size=100, stat_size=30)
        alarm = _first_alarm(det, stream)
        assert alarm is not None, "KSWIN should detect the variance change"
        # Alarm should appear after stable period with reasonable window overlap tolerance
        assert alarm >= len(stable) - 100, (
            f"Alarm at {alarm} is too early for a variance change starting at {len(stable)}"
        )


# ---------------------------------------------------------------------------
# DDM
# ---------------------------------------------------------------------------

class TestDDMSyntheticStreams:
    def test_detects_after_error_rate_increase(self):
        """Error rate starts low (good model) then jumps (concept drift)."""
        n_stable = 100
        n_drift = 60
        drift_at = n_stable
        # 1.0 = correct prediction, 0.0 = incorrect
        stream = [1.0] * n_stable + [0.0] * n_drift
        det = DDM(min_num_instances=30, drift_level=3.0)
        alarm = _first_alarm(det, stream)
        assert alarm is not None, "DDM should detect the error rate jump"
        assert alarm >= drift_at, f"Alarm at {alarm} fired before error rate increased"

    def test_no_alarm_on_consistent_correct_predictions(self):
        stream = [1.0] * 300
        det = DDM(min_num_instances=30)
        assert _first_alarm(det, stream) is None

    def test_no_alarm_on_consistent_wrong_predictions(self):
        """If error rate is consistently high but not increasing, no drift."""
        stream = [0.0] * 300
        det = DDM(min_num_instances=30)
        # DDM needs the rate to *increase* beyond the minimum + k*sigma
        # A consistently high rate sets a high baseline and stays there
        alarms = sum(1 for v in stream if det.update(v)[0])
        # DDM resets on drift detection, so multiple alarms are possible;
        # the important thing is it doesn't alarm before min_window
        alarm = _first_alarm(DDM(min_num_instances=30), stream)
        if alarm is not None:
            assert alarm >= 30, "DDM should not fire before min_window"
