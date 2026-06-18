from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class RetrainingWindowResult:
    """Suggested historical window to retrain on after drift is detected."""
    start: int          # index into the monitor's internal history
    end: int
    n_samples: int
    confidence: float   # 0.0 – 1.0; fraction of samples that were stable


@dataclass
class DriftResult:
    """Return value of DriftMonitor.update()."""
    drift_detected: bool
    drift_type: Optional[str]                       # sudden | gradual | incremental | recurring
    drifted_features: List[str] = field(default_factory=list)
    drift_score: float = 0.0                        # max score across all features, 0.0–1.0+
    retraining_window: Optional[RetrainingWindowResult] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
