# pattern-drift

**Automatic concept drift detection for streaming datasets.**

`pattern-drift` is a Python library for data scientists and ML engineers working with time-sensitive models. It continuously monitors incoming data distributions, detects statistical drift, and recommends optimal retraining windows — keeping models accurate without manual monitoring.

---

## Installation

```bash
pip install pattern-drift
```

Optional extras:

```bash
pip install pattern-drift[viz]      # adds matplotlib for drift timeline visualisation
pip install pattern-drift[alerts]   # adds requests for Slack and webhook callbacks
pip install pattern-drift[all]      # everything
```

---

## Quick Start

```python
from pattern_drift import DriftMonitor

monitor = DriftMonitor(method="ADWIN", sensitivity=0.002)

for record in stream:                    # dict, pandas Series, or single-row DataFrame
    result = monitor.update(record)
    if result.drift_detected:
        print(f"Drift! type={result.drift_type}")
        print(f"Features: {result.drifted_features}")
        print(f"Score: {result.drift_score:.4f}")
        if result.retraining_window:
            rw = result.retraining_window
            print(f"Retrain on records {rw.start}–{rw.end} (confidence {rw.confidence:.2%})")
```

---

## Detection Algorithms

| Algorithm | Mechanism | Best For |
|-----------|-----------|----------|
| `ADWIN` (default) | Variable-length window split testing on mean differences | Gradual drift — adapts window size dynamically |
| `PageHinkley` | Cumulative sum of deviations from the running mean | Sudden drift — extremely fast and memory-efficient |
| `KSWIN` | Kolmogorov-Smirnov test comparing recent vs. reference window | Distribution shape changes beyond just mean shifts |
| `DDM` | Monitors prediction error rate vs. historical minimum | Classifier performance monitoring post-deployment |

Switch algorithms with a single parameter — no other code changes required:

```python
monitor = DriftMonitor(method="PageHinkley")
monitor = DriftMonitor(method="KSWIN")
monitor = DriftMonitor(method="DDM")
```

---

## API Reference

### `DriftMonitor`

```python
DriftMonitor(
    method="ADWIN",        # Detection algorithm
    sensitivity=0.002,     # Drift threshold — lower = more sensitive
    min_window=30,         # Minimum history before drift can be reported
    max_window=10_000,     # Maximum records retained in memory
    features=None,         # List of columns to monitor (None = auto-detect all numeric)
    callbacks=None,        # List of callables fired on drift
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `monitor.update(data)` | Feed a single row (dict/Series) or micro-batch (DataFrame). Returns `DriftResult`. |
| `monitor.reset()` | Reset all internal detector state and history. |
| `monitor.plot_drift_timeline()` | Render an interactive drift score timeline chart. |
| `monitor.export_report(path)` | Export full drift history to JSON or CSV. |
| `monitor.set_reference(data)` | Manually set the reference distribution for comparison. |
| `DriftMonitor.from_config(path)` | Class method — instantiate from a YAML config file. |

### `DriftResult` Fields

| Field | Type | Description |
|-------|------|-------------|
| `drift_detected` | `bool` | `True` if drift was found in any monitored feature |
| `drift_type` | `str \| None` | `sudden` · `gradual` · `incremental` · `recurring` |
| `drifted_features` | `list[str]` | Names of all features where drift was detected |
| `drift_score` | `float` | Maximum drift score across all features (0.0–1.0+) |
| `retraining_window` | `RetrainingWindowResult \| None` | Suggested retraining window with `start`, `end`, `n_samples`, `confidence` |
| `timestamp` | `datetime` | UTC datetime when the drift event was recorded |

---

## Alerts & Callbacks

```python
from pattern_drift import DriftMonitor
from pattern_drift.dispatcher import AlertDispatcher

monitor = DriftMonitor(
    callbacks=[
        AlertDispatcher.slack_callback("https://hooks.slack.com/..."),
        AlertDispatcher.webhook_callback("https://my-service/drift"),
        AlertDispatcher.log_callback(level="warning"),
        lambda result: print(result),          # custom inline callback
    ]
)
```

---

## YAML Configuration

```yaml
# drift_config.yaml
method: ADWIN
sensitivity: 0.002
min_window: 30
max_window: 10000
features:
  - age
  - income
  - session_duration
```

```python
monitor = DriftMonitor.from_config("drift_config.yaml")
```

---

## scikit-learn Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pattern_drift.sklearn_wrapper import DriftDetector

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("drift",  DriftDetector(method="ADWIN", sensitivity=0.002)),
])

pipe.fit(X_train)

for batch in stream:
    X_out = pipe.transform(batch)   # data passes through unchanged
```

---

## Visualisation

```python
monitor.plot_drift_timeline()           # interactive chart (requires matplotlib)
monitor.export_report("report.json")    # or "report.csv"
```

---

## Architecture

Each incoming record flows through five sequential stages:

1. **Feature Extractor** — splits each row into per-column numeric signals
2. **Detector Pool** — maintains one statistical detector per feature; computes drift score on every update
3. **Drift Classifier** — labels drift as `sudden` / `gradual` / `incremental` / `recurring` based on signal shape
4. **Retraining Window Engine** — scans history to find the last stable data window; returns a confidence-scored recommendation
5. **Alert Dispatcher** — fires registered callbacks (Slack, webhook, log, email, or custom)

---

## License

MIT
