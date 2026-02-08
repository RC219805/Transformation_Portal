"""APEX workflow contracts for end-to-end performance observability.

This module defines the three-layer contract architecture:
1. RunSpec - what we intend to test (immutable intent)
2. Observation - what actually happened (raw measurements)
3. Judgement - what we do with it (aggregation + decision)

These contracts are versioned and stable. Changes require migration plan.

Design principles:
- Contracts are dataclasses (serializable, typed, immutable where feasible)
- Each layer has single responsibility and clear boundaries
- Support dual-run (V1 + V2 on same commit)
- Enable zone-aware aggregation and worst-zone detection
- Schema evolution via version field and migration logic

Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from transformation_portal.metrics.performance_capsule import PerformanceCapsule

__version__ = "1.0.0"


@dataclass(frozen=True)
class RunSpec:
    """Immutable specification of what we intend to test.

    This is the "intent" layer - declared before execution.

    Attributes:
        run_id: Unique identifier for this run (e.g., commit_sha + timestamp)
        commit_sha: Git commit SHA being tested
        workflow_version: Pipeline version ("v1" or "v2")
        zones: List of zones to execute across (empty = single zone)
        device: Target device ("mps", "cuda", "cpu")
        backend_id: Depth backend identifier ("da3", "depth_pro", etc.)
        scene_type: Optional scene constraint (None = all scenes)
        timestamp: When this run was initiated (ISO8601)
        config_hash: Hash of exact configuration used
        runner_metadata: Optional runner details (CI job ID, instance type, etc.)
    """

    run_id: str
    commit_sha: str
    workflow_version: Literal["v1", "v2"]
    zones: List[str]
    device: str
    backend_id: str
    scene_type: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config_hash: str = ""
    runner_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunSpec:
        """Reconstruct from dict."""
        return cls(**data)

    def to_hash(self) -> str:
        """Compute deterministic hash of this RunSpec."""
        json_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]


@dataclass
class Observation:
    """Raw measurements from execution (what actually happened).

    This is the "reality" layer - captured during/after execution.

    Attributes:
        run_spec: The RunSpec this observation corresponds to
        zone: Zone where this observation was captured (None = unknown)
        capsules: List of PerformanceCapsules captured during this run
        phase_timings: Aggregated phase timings (optional, for pipeline-level)
        resource_metadata: Resource usage metadata (peak memory, GPU util, etc.)
        errors: List of errors encountered (if any)
        captured_at: When this observation was captured (ISO8601)
    """

    run_spec: RunSpec
    zone: Optional[str]
    capsules: List[PerformanceCapsule]
    phase_timings: Dict[str, float] = field(default_factory=dict)
    resource_metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    captured_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "run_spec": self.run_spec.to_dict(),
            "zone": self.zone,
            "capsules": [c.to_dict() for c in self.capsules],
            "phase_timings": self.phase_timings,
            "resource_metadata": self.resource_metadata,
            "errors": self.errors,
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Observation:
        """Reconstruct from dict."""
        return cls(
            run_spec=RunSpec.from_dict(data["run_spec"]),
            zone=data["zone"],
            capsules=[PerformanceCapsule.from_dict(c) for c in data["capsules"]],
            phase_timings=data.get("phase_timings", {}),
            resource_metadata=data.get("resource_metadata", {}),
            errors=data.get("errors", []),
            captured_at=data.get("captured_at", datetime.now(timezone.utc).isoformat()),
        )

    @property
    def has_errors(self) -> bool:
        """Check if this observation has errors."""
        return len(self.errors) > 0

    @property
    def sample_count(self) -> int:
        """Number of capsules in this observation."""
        return len(self.capsules)


@dataclass
class BucketStats:
    """Statistics for a performance bucket.

    Attributes:
        bucket_name: Name of the bucket
        count: Number of samples
        p50: Median latency (seconds)
        p95: p95 latency (seconds)
        p99: Optional p99 latency (seconds)
        mean: Mean latency (seconds)
        min: Minimum latency (seconds)
        max: Maximum latency (seconds)
        threshold_p50: Configured p50 threshold
        threshold_p95: Configured p95 threshold
        pass_fail: Verdict ("pass", "warn", "fail")
        is_insufficient_data: Whether count < minimum sample size (never blocks)
    """

    bucket_name: str
    count: int
    p50: float
    p95: float
    p99: Optional[float]
    mean: float
    min: float
    max: float
    threshold_p50: float
    threshold_p95: float
    pass_fail: Literal["pass", "warn", "fail"]
    is_insufficient_data: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BucketStats:
        """Reconstruct from dict."""
        return cls(**data)


@dataclass
class RegressionReport:
    """Regression detection results comparing current to baseline.

    Attributes:
        baseline_run_id: Run ID of the baseline (e.g., previous commit or main)
        baseline_commit_sha: Commit SHA of the baseline
        current_run_id: Run ID being evaluated
        current_commit_sha: Commit SHA being evaluated
        bucket_regressions: Dict mapping bucket_name -> regression delta (fraction)
        max_regression: Maximum regression across all buckets
        max_regression_bucket: Bucket with maximum regression
        status: Overall verdict ("pass", "warn", "fail")
        explanation: Human-readable summary
    """

    baseline_run_id: str
    baseline_commit_sha: str
    current_run_id: str
    current_commit_sha: str
    bucket_regressions: Dict[str, float]
    max_regression: float
    max_regression_bucket: str
    status: Literal["pass", "warn", "fail"]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RegressionReport:
        """Reconstruct from dict."""
        return cls(**data)


@dataclass
class Judgement:
    """Final judgement after aggregation and comparison (what we do with it).

    This is the "decision" layer - computed from Observations.

    Attributes:
        run_id: Run ID being judged
        workflow_version: Workflow version ("v1" or "v2")
        zone: Zone this judgement applies to (None = global/all zones)
        bucket_stats: Per-bucket statistics (bucket_name -> BucketStats)
        regression_report: Optional regression comparison to baseline
        pass_fail: Overall verdict ("pass", "warn", "fail")
        explanation: Human-readable explanation of verdict
        worst_zone_p95: Worst-zone p95 across all zones (for global judgement)
        worst_zone_name: Name of the zone with worst p95
        judged_at: When this judgement was made (ISO8601)
    """

    run_id: str
    workflow_version: Literal["v1", "v2"]
    zone: Optional[str]
    bucket_stats: Dict[str, BucketStats]
    regression_report: Optional[RegressionReport]
    pass_fail: Literal["pass", "warn", "fail"]
    explanation: str
    worst_zone_p95: Optional[float] = None
    worst_zone_name: Optional[str] = None
    judged_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "run_id": self.run_id,
            "workflow_version": self.workflow_version,
            "zone": self.zone,
            "bucket_stats": {k: v.to_dict() for k, v in self.bucket_stats.items()},
            "regression_report": self.regression_report.to_dict() if self.regression_report else None,
            "pass_fail": self.pass_fail,
            "explanation": self.explanation,
            "worst_zone_p95": self.worst_zone_p95,
            "worst_zone_name": self.worst_zone_name,
            "judged_at": self.judged_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Judgement:
        """Reconstruct from dict."""
        return cls(
            run_id=data["run_id"],
            workflow_version=data["workflow_version"],
            zone=data["zone"],
            bucket_stats={k: BucketStats.from_dict(v) for k, v in data["bucket_stats"].items()},
            regression_report=(
                RegressionReport.from_dict(data["regression_report"]) if data.get("regression_report") else None
            ),
            pass_fail=data["pass_fail"],
            explanation=data["explanation"],
            worst_zone_p95=data.get("worst_zone_p95"),
            worst_zone_name=data.get("worst_zone_name"),
            judged_at=data.get("judged_at", datetime.now(timezone.utc).isoformat()),
        )

    @property
    def is_blocking(self) -> bool:
        """Check if this judgement should block a release."""
        return self.pass_fail == "fail"
