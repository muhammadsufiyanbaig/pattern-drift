# pattern-drift — Configuration Report

**Version:** 0.1.0
**Document Type:** Configuration & Deployment Reference
**Last Updated:** March 2026

---

## Table of Contents

1. [Installation](#1-installation)
2. [DriftMonitor Parameters](#2-driftmonitor-parameters)
3. [Detection Algorithm Configuration](#3-detection-algorithm-configuration)
4. [YAML Configuration File](#4-yaml-configuration-file)
5. [Callback & Alert Configuration](#5-callback--alert-configuration)
6. [Optional Extras](#6-optional-extras)
7. [scikit-learn Pipeline Configuration](#7-scikit-learn-pipeline-configuration)
8. [Export & Reporting Configuration](#8-export--reporting-configuration)
9. [Parameter Tuning Guide](#9-parameter-tuning-guide)
10. [Environment Compatibility](#10-environment-compatibility)

---

## 1. Installation

### Standard Installation

```bash
pip install pattern-drift
```

### With Optional Extras

```bash
# Drift timeline charts
pip install pattern-drift[viz]

# Slack and HTTP webhook alerts
pip install pattern-drift[alerts]

# YAML config file support
pip install pattern-drift[yaml]

# pandas DataFrame integration
pip install pattern-drift[pandas]

# scikit-learn Pipeline wrapper
pip install pattern-drift[sklearn]

# Everything
pip install pattern-drift[all]
```

### Development Installation (from source)

```bash
git clone https://github.com/pattern-drift/pattern-drift
cd pattern-drift
pip install -e ".[dev]"
```

> **Minimum Python version:** 3.9
> **Zero mandatory runtime dependencies** — the core library uses only the Python standard library.

---

## 2. DriftMonitor Parameters

`DriftMonitor` is the primary public class. All parameters are set at instantiation and persist for the lifetime of the monitor.

```python
from pattern_drift import DriftMonitor

monitor = DriftMonitor(
    method      = "ADWIN",    # str
    sensitivity = 0.002,      # float
    min_window  = 30,         # int
    max_window  = 10_000,     # int
    features    = None,       # list[str] | None
    callbacks   = None,       # list[callable] | None
)
```

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"ADWIN"` | Detection algorithm. Must be one of: `"ADWIN"`, `"PageHinkley"`, `"KSWIN"`, `"DDM"`. |
| `sensitivity` | `float` | `0.002` | Primary drift threshold. Passed as the significance parameter to the selected algorithm. Lower values are more conservative (fewer false positives). |
| `min_window` | `int` | `30` | Minimum number of records that must be observed before any drift alarm can be raised. Prevents spurious early alarms on insufficient data. |
| `max_window` | `int` | `10_000` | Maximum records retained in the internal score-history buffer. Older records are evicted when the limit is reached. Controls memory usage. |
| `features` | `list[str] \| None` | `None` | Column names to monitor. `None` means auto-detect all numeric columns from the first observation. Non-numeric columns are silently ignored. |
| `callbacks` | `list[callable] \| None` | `None` | List of callables that fire when drift is detected. Each receives a `DriftResult` object. Exceptions in callbacks are caught and logged — they do not interrupt processing. |

### Parameter Validation

| Condition | Behaviour |
|-----------|-----------|
| `method` not in valid set | Raises `ValueError` with a helpful message listing valid options |
| `min_window > max_window` | Technically allowed; no alarms will fire until `min_window` records accumulate |
| `sensitivity <= 0` | Passed through to detector; extremely small values may prevent any detection |
| `features = []` | Empty list — no features are monitored, all calls return `drift_detected=False` |

---

## 3. Detection Algorithm Configuration

Each algorithm accepts the `sensitivity` parameter from `DriftMonitor` as its primary tuning knob. The table below shows the exact mapping.

### 3.1 ADWIN — Adaptive Windowing

**Use when:** Gradual or slow drift; you need adaptive memory management.

| `DriftMonitor` parameter | Maps to ADWIN | Role |
|--------------------------|---------------|------|
| `sensitivity` | `delta` | Hoeffding-bound significance level. Smaller = more conservative detection. |
| `max_window` | `max_window` | Hard cap on window length. |

**Internal logic:** ADWIN maintains a variable-length deque of recent observations. On every `update()`, it scans all possible cut points and applies Hoeffding's bound:

```
ε_cut = sqrt( (1 / 2m) × ln(4n / δ) )
```

where *n* is the total window size, *m* is the harmonic mean of sub-window sizes, and *δ* is the significance level (`sensitivity`). If `|mean_left − mean_right| ≥ ε_cut`, the old portion is discarded and drift is signalled.

**Recommended `sensitivity` range:** `0.0001` (very strict) → `0.1` (lenient)
**Default:** `0.002`

---

### 3.2 PageHinkley — Cumulative Sum

**Use when:** Sudden, abrupt shifts; memory and latency are critical constraints.

| `DriftMonitor` parameter | Maps to PageHinkley | Role |
|--------------------------|---------------------|------|
| `sensitivity` | `delta` | Minimum detectable mean change. Smaller = more sensitive to small deviations. |
| *(fixed)* | `lambda_` = `50.0` | Detection threshold. Increase to reduce false positives. |
| *(fixed)* | `alpha` = `1.0` | Forgetting factor for running mean (`1.0` = no forgetting). |

**Internal logic:** Maintains a running mean and cumulative deviation sum:

```
U_t = (Σ x_i − x̄ − δ) − min(Σ x_i − x̄ − δ)
```

Drift is declared when `U_t > λ`. The detector resets after each alarm.

**Recommended `sensitivity` range:** `0.001` → `0.05`
**Default:** `0.002`

> **Note:** To expose `lambda_` directly, create a `PageHinkley` detector instance manually and register it in a custom detector pool. The current API surface maps `sensitivity` → `delta` only.

---

### 3.3 KSWIN — Kolmogorov-Smirnov Windowed

**Use when:** Distribution shape changes (variance, skewness, tail behaviour) rather than mean shifts.

| `DriftMonitor` parameter | Maps to KSWIN | Role |
|--------------------------|---------------|------|
| `sensitivity` | `alpha` | KS-test significance level. Smaller = fewer false positives, later detection. |
| *(fixed)* | `window_size` = `100` | Total sliding window length. |
| *(fixed)* | `stat_size` = `30` | Size of the "recent" sub-window used in the KS comparison. |

**Internal logic:** Two windows are compared using the two-sample KS statistic (implemented in pure Python — no scipy dependency):

```
D = sup_x |F_ref(x) − F_recent(x)|
```

The p-value is approximated using the Kolmogorov distribution. If `p < alpha`, drift is declared.

**Recommended `sensitivity` range:** `0.001` → `0.05`
**Default:** `0.002`

---

### 3.4 DDM — Drift Detection Method

**Use when:** Monitoring classifier prediction error rate post-deployment.

| `DriftMonitor` parameter | Maps to DDM | Role |
|--------------------------|-------------|------|
| `sensitivity` | *(not directly used)* | Passed but does not directly map to DDM parameters in current version. |
| `min_window` | `min_num_instances` | Minimum samples before drift can be flagged. |
| *(fixed)* | `warning_level` = `2.0` | Warning threshold multiplier. |
| *(fixed)* | `drift_level` = `3.0` | Drift threshold multiplier. |

**Internal logic:** Tracks running error rate *p_t* and its standard deviation *σ_t*:

```
drift if:  p_t + σ_t  >  p_min + drift_level × σ_min
```

Input values are interpreted as binary correctness signals: `> 0.5` = correct, `≤ 0.5` = incorrect.

**Important:** DDM should receive prediction correctness signals (0.0/1.0), not raw feature values. It is the only algorithm designed for output-space monitoring.

---

## 4. YAML Configuration File

The `DriftMonitor.from_config(path)` class method loads all parameters from a YAML file. Requires `pyyaml` (`pip install pattern-drift[yaml]`).

### Full YAML Schema

```yaml
# drift_config.yaml
# All fields are optional — unspecified fields use the defaults shown below.

method: ADWIN           # string — ADWIN | PageHinkley | KSWIN | DDM
sensitivity: 0.002      # float  — primary detection threshold
min_window: 30          # int    — minimum history before alarms can fire
max_window: 10000       # int    — maximum history retained in memory
features:               # list[str] | absent — absent means auto-detect
  - feature_a
  - feature_b
  - feature_c
```

### Field Defaults

| Field | Default if absent |
|-------|-------------------|
| `method` | `"ADWIN"` |
| `sensitivity` | `0.002` |
| `min_window` | `30` |
| `max_window` | `10000` |
| `features` | `null` (auto-detect) |

### Usage

```python
monitor = DriftMonitor.from_config("drift_config.yaml")
```

> **Note:** Callbacks cannot be specified in YAML. Register them programmatically after loading:
>
> ```python
> monitor = DriftMonitor.from_config("config.yaml")
> monitor._dispatcher.register(my_callback)
> ```

### Algorithm-Specific Config Examples

**Gradual drift, conservative:**
```yaml
method: ADWIN
sensitivity: 0.0005
min_window: 50
max_window: 20000
```

**Fast response to sudden shifts:**
```yaml
method: PageHinkley
sensitivity: 0.01
min_window: 10
max_window: 5000
```

**Distribution shape monitoring:**
```yaml
method: KSWIN
sensitivity: 0.01
min_window: 100
max_window: 10000
features:
  - amount
  - session_duration
  - click_rate
```

**Classifier performance monitoring:**
```yaml
method: DDM
min_window: 50
max_window: 10000
```

---

## 5. Callback & Alert Configuration

### 5.1 Registering Callbacks at Construction

```python
monitor = DriftMonitor(
    callbacks=[
        my_callback_function,
        lambda result: print(result.drift_type),
    ]
)
```

### 5.2 Registering Callbacks After Construction

```python
monitor._dispatcher.register(my_callback)
```

### 5.3 Built-in Callback Factories

All factories return a `Callable[[DriftResult], None]`.

#### Logging Callback

```python
from pattern_drift.dispatcher import AlertDispatcher

cb = AlertDispatcher.log_callback(level="warning")
# level options: "debug", "info", "warning", "error", "critical"
```

No extra dependencies required.

#### Slack Webhook Callback

```python
cb = AlertDispatcher.slack_callback(
    webhook_url="https://hooks.slack.com/services/T.../B.../..."
)
```

Requires `pip install pattern-drift[alerts]`. Posts a formatted message to the Slack channel.

**Slack message format:**
```
:warning: *Drift detected* (sudden)
Features: feature_a, feature_b
Score: 0.8750
Time: 2026-03-14T10:30:00+00:00
```

#### HTTP Webhook Callback

```python
cb = AlertDispatcher.webhook_callback(url="https://my-service.com/drift")
```

Requires `pip install pattern-drift[alerts]`. POSTs a JSON payload:

```json
{
  "drift_detected": true,
  "drift_type": "sudden",
  "drifted_features": ["revenue"],
  "drift_score": 0.875,
  "timestamp": "2026-03-14T10:30:00+00:00",
  "retraining_window": {
    "start": 142,
    "end": 389,
    "n_samples": 248,
    "confidence": 0.9312
  }
}
```

#### Multiple Callbacks

```python
monitor = DriftMonitor(
    callbacks=[
        AlertDispatcher.log_callback("warning"),
        AlertDispatcher.slack_callback("https://hooks.slack.com/..."),
        AlertDispatcher.webhook_callback("https://api.myapp.com/drift"),
        lambda r: my_custom_handler(r),
    ]
)
```

All callbacks fire sequentially. Exceptions in any callback are caught, logged, and execution continues with the next callback.

---

## 6. Optional Extras

| Extra | Packages Installed | Unlocks |
|-------|--------------------|---------|
| `viz` | `matplotlib>=3.5` | `monitor.plot_drift_timeline()` |
| `alerts` | `requests>=2.28` | `AlertDispatcher.slack_callback()`, `.webhook_callback()` |
| `yaml` | `pyyaml>=6.0` | `DriftMonitor.from_config()` |
| `pandas` | `pandas>=1.5` | Passing `pd.DataFrame` / `pd.Series` to `update()` |
| `sklearn` | `scikit-learn>=1.1` | `DriftDetector` inside `sklearn.pipeline.Pipeline` |
| `all` | All of the above | Full feature set |
| `dev` | pytest, pytest-cov, pyyaml, pandas, matplotlib | Development and testing |

### Runtime Detection

When an optional extra is used without installation, a clear `ImportError` is raised:

```
ImportError: Install the viz extra to use visualisation: pip install pattern-drift[viz]
```

---

## 7. scikit-learn Pipeline Configuration

```python
from pattern_drift.sklearn_wrapper import DriftDetector

detector = DriftDetector(
    method      = "ADWIN",   # All DriftMonitor parameters are forwarded
    sensitivity = 0.002,
    min_window  = 30,
    max_window  = 10_000,
    features    = None,
    callbacks   = None,
)
```

### Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("drift",  DriftDetector(method="ADWIN", callbacks=[my_alert])),
])

pipe.fit(X_train)          # sets reference distribution
X_out = pipe.transform(X) # data passes through unchanged; drift monitored silently
```

### Accessing the Underlying Monitor

```python
monitor = pipe.named_steps["drift"].monitor
monitor.plot_drift_timeline()
monitor.export_report("report.json")
```

### sklearn Compatibility

- `get_params()` / `set_params()` — fully implemented
- `sklearn.base.clone()` — supported
- `fit_transform()` — supported (fit + transform in one call)
- `Pipeline` chaining — fully supported
- `GridSearchCV` — `method` and `sensitivity` can be grid-searched

---

## 8. Export & Reporting Configuration

### JSON Export

```python
monitor.export_report("drift_report.json")
```

**Output format:**
```json
[
  {"index": 0, "feature_a": 0.0012, "feature_b": 0.0008},
  {"index": 1, "feature_a": 0.0015, "feature_b": 0.0009},
  ...
]
```

One record per `update()` call. Scores are normalised drift scores in `[0.0, 1.0+]`.

### CSV Export

```python
monitor.export_report("drift_report.csv")
```

**Output format:**
```
index,feature_a,feature_b
0,0.0012,0.0008
1,0.0015,0.0009
...
```

### Visualisation

```python
monitor.plot_drift_timeline()
```

Requires `pip install pattern-drift[viz]`. Renders an interactive matplotlib figure with:
- One line per monitored feature
- Drift score on the Y-axis, observation index on the X-axis
- A horizontal dashed red line at the sensitivity threshold

### Reference Distribution Reset

```python
# Reset the reference window for KSWIN detectors (e.g., after intentional data migration)
monitor.set_reference(new_reference_dataframe)
```

---

## 9. Parameter Tuning Guide

### Choosing an Algorithm

| Scenario | Recommended Algorithm |
|----------|-----------------------|
| You expect slow, gradual model degradation | `ADWIN` |
| You expect overnight regime changes (market events, migrations) | `PageHinkley` |
| You care about variance, skewness, or tail behaviour changes | `KSWIN` |
| You want to monitor model accuracy directly (not input features) | `DDM` |
| Uncertain — safe default | `ADWIN` |

### Tuning `sensitivity`

| Desired Behaviour | Direction |
|-------------------|-----------|
| Fewer false positives, later detection | Decrease `sensitivity` |
| Faster detection, accept more false positives | Increase `sensitivity` |
| Production system with stable cost of retraining | Lower (0.0005 – 0.002) |
| Experimental or research use | Higher (0.01 – 0.1) |

### Tuning `min_window`

| Scenario | Recommended `min_window` |
|----------|--------------------------|
| Fast-changing streaming data (IoT, real-time) | `10` – `20` |
| Daily batch pipeline | `30` – `100` |
| Monthly model review process | `200` – `500` |

### Tuning `max_window`

| Scenario | Recommended `max_window` |
|----------|--------------------------|
| Memory-constrained environment | `1000` – `2000` |
| Standard ML pipeline | `10000` (default) |
| Long-horizon trend analysis | `50000` – `100000` |

### False Positive Management

| Tool | Effect |
|------|--------|
| Decrease `sensitivity` | Reduces ADWIN and KSWIN false positives |
| Increase `min_window` | Ensures detector is warmed up before alarming |
| Use `ADWIN` with strict `delta` | Theoretical FPR bounded by `delta` |
| Use KSWIN with small `alpha` (e.g., 0.001) | Strict KS-test significance |

---

## 10. Environment Compatibility

### Python Version Support

| Python Version | Supported |
|----------------|-----------|
| 3.9 | ✅ |
| 3.10 | ✅ |
| 3.11 | ✅ |
| 3.12 | ✅ |
| 3.13 | ✅ (tested) |
| < 3.9 | ❌ |

### Operating System

| Platform | Status |
|----------|--------|
| Linux (all distributions) | ✅ |
| macOS (Intel & Apple Silicon) | ✅ |
| Windows 10/11 | ✅ (tested on Windows 11 Pro) |

### Dependency Matrix

| Component | Mandatory | Package | Min Version |
|-----------|-----------|---------|-------------|
| Core | Yes | Python stdlib only | — |
| Visualisation | No | matplotlib | 3.5 |
| Alerts | No | requests | 2.28 |
| Config files | No | pyyaml | 6.0 |
| DataFrame input | No | pandas | 1.5 |
| Pipeline wrapper | No | scikit-learn | 1.1 |

### Integration Compatibility

| Framework | Compatible |
|-----------|-----------|
| Apache Kafka consumers | ✅ (feed records from consumer loop) |
| Apache Flink / Spark Streaming | ✅ (one monitor per partition) |
| FastAPI / Flask inference APIs | ✅ (update monitor in prediction handler) |
| MLflow | ✅ (log `DriftResult` as run metrics) |
| Airflow | ✅ (wrap in PythonOperator) |
| AWS SageMaker | ✅ (deploy as custom monitoring script) |
| Jupyter Notebooks | ✅ (full support including `plot_drift_timeline()`) |

---

*End of Configuration Report*
