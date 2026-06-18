"""
Test category 4: Drift type classification tests.

Feed shaped streams to the DriftClassifier and verify it returns the
correct label for each drift archetype.
"""
import pytest
from pattern_drift.classifier import DriftClassifier


# ---------------------------------------------------------------------------
# Helpers — build histories that satisfy each classifier branch
# ---------------------------------------------------------------------------

def _feed(classifier: DriftClassifier, feature: str, entries: list):
    """Feed (drift_detected, score) pairs into the classifier."""
    for detected, score in entries:
        classifier.record(feature, detected, score)


# ---------------------------------------------------------------------------
# Sudden
# ---------------------------------------------------------------------------

class TestSuddenClassification:
    def test_sudden_when_score_jumps_from_low_to_high(self):
        """scores[-2] < 0.2 and scores[-1] > 0.6 → sudden."""
        clf = DriftClassifier()
        # Build history: low scores, then one high score
        _feed(clf, "x", [(False, 0.05)] * 10 + [(True, 0.85)])
        result = clf.classify(["x"])
        assert result == "sudden"

    def test_sudden_on_empty_history(self):
        """If no history, default to sudden."""
        clf = DriftClassifier()
        # classify without any prior record calls
        result = clf._classify_feature("unseen_feature")
        assert result == "sudden"

    def test_sudden_priority_over_recurring(self):
        """sudden has higher priority than recurring."""
        clf = DriftClassifier()
        # Force drift count > 1 (recurring condition) AND score jump (sudden condition)
        _feed(clf, "x", [(True, 0.9)] * 3)   # drift_count = 3
        # Now inject the sudden pattern
        clf._history["x"].append((False, 0.05))
        clf._history["x"].append((True, 0.9))
        result = clf.classify(["x"])
        assert result == "sudden"


# ---------------------------------------------------------------------------
# Recurring
# ---------------------------------------------------------------------------

class TestRecurringClassification:
    def test_recurring_when_drift_count_exceeds_one(self):
        clf = DriftClassifier()
        # First drift
        _feed(clf, "x", [(False, 0.1)] * 5 + [(True, 0.3)])
        # Stable interlude
        _feed(clf, "x", [(False, 0.05)] * 10)
        # Second drift — drift_count is now 2
        _feed(clf, "x", [(True, 0.4)])
        result = clf.classify(["x"])
        assert result == "recurring"

    def test_not_recurring_on_first_drift(self):
        clf = DriftClassifier()
        _feed(clf, "x", [(False, 0.1)] * 5 + [(True, 0.4)])
        # drift_count is 1, should not be recurring
        result = clf._classify_feature("x")
        assert result != "recurring"


# ---------------------------------------------------------------------------
# Incremental
# ---------------------------------------------------------------------------

class TestIncrementalClassification:
    def test_incremental_on_monotonically_rising_scores(self):
        """Last 5 scores strictly non-decreasing → incremental."""
        clf = DriftClassifier()
        rising = [(False, 0.1), (False, 0.2), (False, 0.3), (False, 0.4), (True, 0.5)]
        _feed(clf, "x", rising)
        result = clf.classify(["x"])
        assert result == "incremental"

    def test_not_incremental_when_score_dips(self):
        clf = DriftClassifier()
        non_monotone = [(False, 0.1), (False, 0.5), (False, 0.3), (False, 0.4), (True, 0.6)]
        _feed(clf, "x", non_monotone)
        result = clf._classify_feature("x")
        assert result != "incremental"


# ---------------------------------------------------------------------------
# Gradual (catch-all)
# ---------------------------------------------------------------------------

class TestGradualClassification:
    def test_gradual_when_no_other_pattern_matches(self):
        """Drift count == 1, no sudden jump, scores not monotone → gradual."""
        clf = DriftClassifier()
        # Non-monotone, first drift, no score jump
        _feed(clf, "x", [(False, 0.3), (False, 0.1), (False, 0.4), (False, 0.2), (True, 0.35)])
        result = clf.classify(["x"])
        assert result == "gradual"


# ---------------------------------------------------------------------------
# Multi-feature priority
# ---------------------------------------------------------------------------

class TestMultiFeaturePriority:
    def test_sudden_beats_gradual_across_features(self):
        clf = DriftClassifier()
        # Feature a → gradual
        _feed(clf, "a", [(False, 0.3), (False, 0.1), (False, 0.2), (False, 0.15), (True, 0.25)])
        # Feature b → sudden
        _feed(clf, "b", [(False, 0.05)] * 5 + [(True, 0.9)])
        result = clf.classify(["a", "b"])
        assert result == "sudden"

    def test_recurring_beats_incremental_across_features(self):
        clf = DriftClassifier()
        # Feature a → incremental
        _feed(clf, "a", [(False, 0.1), (False, 0.2), (False, 0.3), (False, 0.4), (True, 0.5)])
        # Feature b → recurring (drift count = 2, no sudden jump, not incremental)
        _feed(clf, "b", [(True, 0.4)] * 2)   # force count = 2
        _feed(clf, "b", [(False, 0.1), (False, 0.3), (True, 0.35)])
        result = clf.classify(["a", "b"])
        assert result == "recurring"

    def test_reset_clears_drift_counts(self):
        clf = DriftClassifier()
        _feed(clf, "x", [(True, 0.5)] * 3)  # drift_count = 3
        clf.reset()
        _feed(clf, "x", [(False, 0.1)] * 3 + [(True, 0.4)])
        # After reset drift_count = 1, not recurring
        result = clf._classify_feature("x")
        assert result != "recurring"
