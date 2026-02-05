#!/usr/bin/env python3
"""Performance ledger tool for pipeline regression detection.

Parses manifests from batch runs, computes statistics, and compares against
baselines to detect performance regressions.

v1.6 "Boss Mode++" Enhancements:
- Multi-backend aware latency stats:
  - Latency statistics are computed on the *comparison backend* only (baseline backend).
  - Always reports per-backend latency summary to expose fallback/mixed runs.
- Forensic stats:
  - Bootstrap confidence intervals for mean + p95 deltas (diff = current - baseline).
  - Outlier attribution: top slowest and p95 tail contributors (with identifiers).
- Failure taxonomy:
  - Hierarchical error classification (Buckets / Signatures / Exception Types) with examples.
- Latency visualization:
  - ASCII histogram for the comparison backend.
- Robust environment capture:
  - Works even if torch is missing, stubbed, or mocked as None.

Usage:
    # Capture baseline
    python tools/performance_ledger.py \
        --manifests-dir ./output/prod_run/manifests \
        --output ./docs/performance/baselines/baseline_v2.0.0.json \
        --baseline-version v2.0.0 \
        --quality-tier apex

    # Compare against baseline
    python tools/performance_ledger.py \
        --baseline ./docs/performance/baselines/baseline_v2.0.0.json \
        --compare ./output/experimental_run/manifests \
        --output ./output/perf_report.md \
        --emit-json ./output/perf_full.json

Exit codes (compare mode):
    0 = OK (or only noise/potential regressions and --strict not set)
    1 = Significant regressions detected (or any regression with --strict)
    2 = Backend mismatch (comparison invalid)
    3 = Insufficient latency data (cannot compute latency significance; failure-rate regressions still take precedence)
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TOOL_VERSION = "1.6.0"

# Regression thresholds (per ADR-023) - defaults
DEFAULT_REGRESSION_THRESHOLDS = {
    "p95_worsening_pct": 10.0,
    "mean_worsening_pct": 15.0,
    "failure_rate_increase": 0.0,  # fraction (not percent)
}

# Bootstrap defaults
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_MIN_SAMPLES = 5

# Forensics defaults
DEFAULT_TOP_SLOWEST = 5
DEFAULT_TOP_P95_CONTRIB = 10
DEFAULT_HIST_BINS = 10
DEFAULT_HIST_WIDTH = 24


# ----------------------------
# Dataclasses
# ----------------------------

@dataclass
class EnvironmentMetadata:
    """Environment metadata for baseline reproducibility."""
    python: str
    torch: Optional[str]
    device: str
    os: str
    cpu: Optional[str] = None
    memory_gb: Optional[int] = None


@dataclass
class BackendComplianceMetadata:
    """Backend selection compliance metrics."""
    requested_backend: str
    dominant_resolved_backend: str
    fallback_count: int
    total_count: int
    fallback_rate_pct: float
    distribution: Dict[str, int]
    consistency_status: str = "unknown"

    @property
    def is_consistent(self) -> bool:
        return self.fallback_count == 0 and len(self.distribution) == 1


@dataclass
class TimingSample:
    """A single timing measurement linked to an artifact and resolved backend."""
    duration: float
    identifier: str
    resolved_backend: str


@dataclass
class Outlier:
    """Details about a performance outlier."""
    identifier: str
    duration: float
    z_score: float


@dataclass
class Statistics:
    """Runtime statistics for a batch run (timings computed on a chosen subset)."""
    count: int
    mean_sec: float
    median_sec: float
    p90_sec: float
    p95_sec: float
    min_sec: float
    max_sec: float
    std_dev: float
    success_rate: float
    total_sec: Optional[float] = None

    # Forensic: Outliers (computed on timing sample set)
    slowest_artifacts: List[Outlier] = field(default_factory=list)
    p95_contributors: List[Outlier] = field(default_factory=list)

    # Failure taxonomy (computed on failures across whole run)
    error_distribution: Optional[Dict[str, int]] = None          # signature -> count
    error_examples: Optional[Dict[str, str]] = None              # signature -> example
    error_buckets: Optional[Dict[str, int]] = None               # bucket -> count
    error_bucket_examples: Optional[Dict[str, str]] = None       # bucket -> example
    error_exc_types: Optional[Dict[str, int]] = None             # type -> count


@dataclass
class Baseline:
    """Performance baseline schema."""
    version: str
    backend: str  # baseline comparison backend
    quality_tier: str
    environment: EnvironmentMetadata
    statistics: Statistics
    compliance: Optional[BackendComplianceMetadata] = None

    # Raw samples for bootstrap significance (comparison backend only)
    raw_samples: Optional[List[float]] = None

    # Optional richer baseline context (not required for compare)
    raw_samples_by_backend: Optional[Dict[str, List[float]]] = None
    latency_summary_by_backend: Optional[Dict[str, Dict[str, float]]] = None

    captured_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    captured_by: str = f"tools/performance_ledger.py v{TOOL_VERSION}"
    notes: Optional[str] = None


@dataclass
class Regression:
    """Detected regression."""
    metric: str
    baseline: float
    current: float
    change_pct: float
    threshold_pct: float
    status: str
    significance: str = "unknown"  # "significant", "noise", "potential", "unknown"
    confidence_interval: Optional[Tuple[float, float]] = None    # diff (current - baseline)


# ----------------------------
# Schema-safe dataclass loading
# ----------------------------

def _dataclass_from_dict(cls, data: Optional[Dict[str, Any]]):
    """Create dataclass instance from dict, ignoring unknown keys."""
    if data is None:
        return None
    allowed = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in allowed}
    return cls(**filtered)


def _load_baseline(path: Path) -> Baseline:
    """Load baseline JSON with schema tolerance."""
    try:
        with path.open("r", encoding="utf-8") as f:
            base = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Baseline not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Baseline JSON invalid: {e}")

    env = _dataclass_from_dict(EnvironmentMetadata, base.get("environment"))
    stats = _dataclass_from_dict(Statistics, base.get("statistics"))
    comp = _dataclass_from_dict(BackendComplianceMetadata, base.get("compliance"))

    if env is None or stats is None:
        raise ValueError("Baseline missing required 'environment' or 'statistics' blocks.")

    return Baseline(
        version=base.get("version", "unknown"),
        backend=base.get("backend", "unknown"),
        quality_tier=base.get("quality_tier", "unknown"),
        environment=env,
        statistics=stats,
        compliance=comp,
        raw_samples=base.get("raw_samples") or [],
        raw_samples_by_backend=base.get("raw_samples_by_backend"),
        latency_summary_by_backend=base.get("latency_summary_by_backend"),
        captured_at=base.get("captured_at", datetime.now(timezone.utc).isoformat()),
        captured_by=base.get("captured_by", f"tools/performance_ledger.py v{TOOL_VERSION}"),
        notes=base.get("notes"),
    )


# ----------------------------
# Manifest I/O
# ----------------------------

def parse_manifests(manifests_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load all manifest JSONs from directory."""
    if not manifests_dir.exists():
        raise FileNotFoundError(f"Manifests directory not found: {manifests_dir}")

    manifest_files = sorted(manifests_dir.glob("*.json"))
    if not manifest_files:
        raise ValueError(f"No JSON manifests found in {manifests_dir}")

    valid_manifests: List[Dict[str, Any]] = []
    failed_manifests: List[Dict[str, Any]] = []

    for manifest_file in manifest_files:
        try:
            with manifest_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Inject filename as fallback identifier (used by outlier attribution)
            data.setdefault("identifier", manifest_file.name)

            if any(k in data for k in ("timing", "status", "v2", "backend_selection", "depth")):
                valid_manifests.append(data)
            else:
                failed_manifests.append({"file": str(manifest_file), "error": "Missing key blocks"})
        except json.JSONDecodeError as e:
            logger.warning(f"Skipping invalid JSON {manifest_file}: {e}")
            failed_manifests.append({"file": str(manifest_file), "error": str(e)})

    logger.info(f"Loaded {len(valid_manifests)} manifests from {manifests_dir}")
    return valid_manifests, failed_manifests


# ----------------------------
# Backend resolution helpers
# ----------------------------

def _infer_backend_from_legacy_manifest(m: Dict[str, Any]) -> str:
    """Infer resolved backend for pre-ADR-023 manifests."""
    depth_meta = m.get("depth") if isinstance(m.get("depth"), dict) else {}
    stats = depth_meta.get("stats") if isinstance(depth_meta.get("stats"), dict) else {}

    if "backend" in stats and isinstance(stats["backend"], str):
        return stats["backend"]

    model = depth_meta.get("model")
    if isinstance(model, str):
        ml = model.lower()
        if "depth-pro" in ml or "depth_pro" in ml:
            return "depth_pro"
    return "da3"


def _resolve_backend_selection(m: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return (requested_backend, resolved_backend, resolution_status)."""
    bs = m.get("backend_selection")
    if isinstance(bs, dict) and bs:
        requested = (bs.get("requested_backend") or "auto").strip()
        resolved = (bs.get("resolved_backend") or "unknown").strip()
        status = (bs.get("resolution_status") or "unknown").strip().lower()
        return requested, resolved, status

    # Legacy inference
    return "unknown", _infer_backend_from_legacy_manifest(m), "inferred"


def analyze_backend_compliance(manifests: List[Dict[str, Any]]) -> BackendComplianceMetadata:
    """Analyze backend selection truth and fallback rates."""
    requested_counts: Counter = Counter()
    resolved_counts: Counter = Counter()
    fallback_count = 0
    total = 0

    for m in manifests:
        requested, resolved, status = _resolve_backend_selection(m)
        if "fallback" in status:
            fallback_count += 1

        requested_counts[requested] += 1
        resolved_counts[resolved] += 1
        total += 1

    dominant_resolved = resolved_counts.most_common(1)[0][0] if resolved_counts else "unknown"
    dominant_requested = requested_counts.most_common(1)[0][0] if requested_counts else "auto"
    fallback_rate = (fallback_count / total * 100.0) if total > 0 else 0.0

    consistency = "clean"
    if total == 0:
        consistency = "empty"
    elif fallback_rate > 50.0:
        consistency = "fallback_heavy"
    elif len(resolved_counts) > 1:
        consistency = "mixed"

    return BackendComplianceMetadata(
        requested_backend=dominant_requested,
        dominant_resolved_backend=dominant_resolved,
        fallback_count=fallback_count,
        total_count=total,
        fallback_rate_pct=fallback_rate,
        distribution=dict(resolved_counts),
        consistency_status=consistency,
    )


# ----------------------------
# Failure Taxonomy Logic
# ----------------------------

_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
_HEX_RE = re.compile(r"\b0x[0-9a-f]+\b", re.IGNORECASE)
_LINE_RE = re.compile(r"\bline\s+\d+\b", re.IGNORECASE)
_FILELINE_RE = re.compile(r"\bFile\s+\"[^\"]+\",?\s+line\s+\d+\b", re.IGNORECASE)
_POSIX_PATH_RE = re.compile(r"(/[^ \n\t\"']+)+")
_WIN_PATH_RE = re.compile(r"\b[A-Za-z]:\\(?:[^ \n\t\"']+\\)*[^ \n\t\"']+\b")
_FLOAT_RE = re.compile(r"\b\d+\.\d+\b")
_INT_RE = re.compile(r"\b\d+\b")
_DIM_RE = re.compile(r"\b\d+\s*[x×]\s*\d+\b", re.IGNORECASE)
_SHAPE_RE = re.compile(r"\b(?:shape|sizes?)=\(?[\d,\s]+\)?\b", re.IGNORECASE)
_MEM_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:kb|mb|gb|tb|kib|mib|gib|tib)\b", re.IGNORECASE)
_TENSOR_RE = re.compile(r"\btensor\([^)]*\)\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_EXC_PREFIX_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\s*:\s*(.*)$")


def normalize_error_signature(msg: str) -> Tuple[str, str]:
    """Normalize error message to stable signature. Returns (signature, raw_example)."""
    raw = (msg or "").strip()
    if not raw:
        return ("unknown error", "unknown error")

    s = raw
    if "\n" in s:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if lines:
            s = lines[-1] if (":" in lines[-1] or "error" in lines[-1].lower()) else lines[0]

    s_l = s.strip().lower()
    s_l = _FILELINE_RE.sub("file <path> line <n>", s_l)
    s_l = _LINE_RE.sub("line <n>", s_l)
    s_l = _UUID_RE.sub("<uuid>", s_l)
    s_l = _HEX_RE.sub("<hex>", s_l)
    s_l = _WIN_PATH_RE.sub("<path>", s_l)
    s_l = _POSIX_PATH_RE.sub("<path>", s_l)
    s_l = _TENSOR_RE.sub("tensor(<...>)", s_l)
    s_l = _SHAPE_RE.sub("shape=<shape>", s_l)
    s_l = _DIM_RE.sub("<dim>", s_l)
    s_l = _MEM_RE.sub("<mem>", s_l)
    s_l = _FLOAT_RE.sub("<f>", s_l)
    s_l = _INT_RE.sub("<n>", s_l)
    s_l = _WHITESPACE_RE.sub(" ", s_l).strip()
    return (s_l or "unknown error", raw)


def classify_error(msg: str) -> Tuple[str, str, List[str], str]:
    """Classify error into exception type, normalized signature, and buckets."""
    sig, raw = normalize_error_signature(msg)

    exc_type = "unknown"
    raw_last = ""
    if raw.strip():
        raw_lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        raw_last = raw_lines[-1] if raw_lines else raw.strip()

    m = _EXC_PREFIX_RE.match(raw_last)
    if m:
        exc_type = m.group(1)
        body = m.group(2) or ""
        sig2, _ = normalize_error_signature(f"{exc_type}: {body}")
        sig = sig2 or sig
    elif ":" in sig:
        prefix = sig.split(":", 1)[0].strip()
        if prefix.endswith("error") or prefix.endswith("exception"):
            exc_type = prefix[:1].upper() + prefix[1:]

    s, r = sig.lower(), raw.lower()

    def has(*terms: str) -> bool:
        return any(t in s or t in r for t in terms)

    buckets: List[str] = []

    if has("out of memory", "oom", "cuda out of memory", "mps out of memory", "tried to allocate", "allocation", "<mem>"):
        buckets.append("oom")
    if has("no module named", "importerror", "modulenotfounderror", "cannot import", "undefined symbol"):
        buckets.append("import")
    if has("has no attribute", "attributeerror"):
        buckets.append("api")
    if has("backend mismatch", "invalid comparison", "baseline mismatch", "resolved backend", "requested backend"):
        buckets.append("backend")
    if has("no such file", "filenotfounderror", "enoent", "ioerror"):
        buckets.append("io")
    if has("permission denied", "eacces", "operation not permitted"):
        buckets.append("permission")
    if has("timeout", "timed out", "deadline exceeded"):
        buckets.append("timeout")
    if has("shape", "sizes", "dimension", "broadcast", "size mismatch", "matmul"):
        buckets.append("shape")
    if has("expected all tensors to be on the same device", "device", "cuda", "mps", "cpu"):
        buckets.append("device")
    if has("connection", "dns", "ssl", "handshake", "http", "429", "503", "network"):
        buckets.append("network")
    if has("nan", "inf", "overflow", "underflow", "invalid value"):
        buckets.append("numerics")

    if not buckets:
        buckets = ["unknown"]

    return exc_type, sig, buckets, raw


# ----------------------------
# Extraction + statistics
# ----------------------------

def extract_data(
    manifests: List[Dict[str, Any]],
) -> Tuple[
    List[TimingSample],
    int,
    int,
    List[str],
    Dict[str, str],
    Counter,
    Dict[str, str],
    Counter,
]:
    """Extract timings (with identifiers + resolved backend) and failure taxonomy."""
    samples: List[TimingSample] = []
    success_count = 0
    failure_count = 0

    signatures: List[str] = []
    sig_examples: Dict[str, str] = {}
    bucket_counts: Counter = Counter()
    bucket_examples: Dict[str, str] = {}
    exc_type_counts: Counter = Counter()

    ok_statuses = {"ok", "success", "passed"}
    fail_statuses = {"fail", "failed", "error"}

    for m in manifests:
        # Identifier (prefer input.image_path basename if present)
        identifier = m.get("identifier", "unknown")
        if isinstance(m.get("input"), dict):
            p = m["input"].get("image_path")
            if p:
                identifier = Path(p).name

        _requested, resolved_backend, _status = _resolve_backend_selection(m)

        # Timing
        timing = m.get("timing") if isinstance(m.get("timing"), dict) else {}
        total_sec = timing.get("total_seconds")

        # Status
        v2 = m.get("v2") if isinstance(m.get("v2"), dict) else {}
        status = str(v2.get("status") or m.get("status") or "").lower().strip()

        is_success = False
        if status in ok_statuses:
            is_success = True
        elif status in fail_statuses:
            is_success = False
        elif m.get("depth") is not None:
            is_success = True  # legacy/PBR assumption

        if is_success:
            success_count += 1
            if total_sec is not None:
                try:
                    samples.append(
                        TimingSample(duration=float(total_sec), identifier=str(identifier), resolved_backend=str(resolved_backend))
                    )
                except (TypeError, ValueError):
                    pass
        else:
            failure_count += 1
            err_msg = (
                v2.get("error_message")
                or m.get("error_message")
                or m.get("error")
                or status
                or "unknown error"
            )
            exc_type, sig, buckets, ex = classify_error(str(err_msg))

            signatures.append(sig)
            sig_examples.setdefault(sig, ex)
            exc_type_counts[exc_type] += 1
            for b in buckets:
                bucket_counts[b] += 1
                bucket_examples.setdefault(b, ex)

    return samples, success_count, failure_count, signatures, sig_examples, bucket_counts, bucket_examples, exc_type_counts


def compute_statistics(
    samples: List[TimingSample],
    success_count: int,
    failure_count: int,
    signatures: List[str],
    sig_examples: Dict[str, str],
    bucket_counts: Counter,
    bucket_examples: Dict[str, str],
    exc_type_counts: Counter,
    *,
    top_slowest: int = DEFAULT_TOP_SLOWEST,
    top_p95: int = DEFAULT_TOP_P95_CONTRIB,
) -> Statistics:
    """Compute stats (over provided timing samples) plus failure taxonomy (over whole run)."""
    total_runs = success_count + failure_count
    success_rate = success_count / total_runs if total_runs > 0 else 0.0

    # Failure Taxonomy (Top N)
    top_sigs = dict(Counter(signatures).most_common(5)) if signatures else {}
    top_sig_ex = {k: sig_examples.get(k, "") for k in top_sigs}

    top_buckets = dict(bucket_counts.most_common(8)) if bucket_counts else {}
    top_bucket_ex = {k: bucket_examples.get(k, "") for k in top_buckets}

    top_exc = dict(exc_type_counts.most_common(8)) if exc_type_counts else {}

    if not samples:
        return Statistics(
            count=0,
            mean_sec=0.0,
            median_sec=0.0,
            p90_sec=0.0,
            p95_sec=0.0,
            min_sec=0.0,
            max_sec=0.0,
            std_dev=0.0,
            success_rate=success_rate,
            total_sec=0.0,
            slowest_artifacts=[],
            p95_contributors=[],
            error_distribution=top_sigs,
            error_examples=top_sig_ex,
            error_buckets=top_buckets,
            error_bucket_examples=top_bucket_ex,
            error_exc_types=top_exc,
        )

    durations = np.array([s.duration for s in samples], dtype=float)
    mean_val = float(np.mean(durations))
    std_dev = float(np.std(durations))

    # Top slowest
    sorted_samples = sorted(samples, key=lambda x: x.duration, reverse=True)
    slowest: List[Outlier] = []
    for s in sorted_samples[: max(0, int(top_slowest))]:
        z = (s.duration - mean_val) / std_dev if std_dev > 0 else 0.0
        slowest.append(Outlier(identifier=s.identifier, duration=s.duration, z_score=z))

    # p95 contributors (tail drivers)
    p95_cutoff = float(np.percentile(durations, 95))
    p95_samples = [s for s in sorted_samples if s.duration >= p95_cutoff][: max(0, int(top_p95))]
    p95_contrib: List[Outlier] = []
    for s in p95_samples:
        z = (s.duration - mean_val) / std_dev if std_dev > 0 else 0.0
        p95_contrib.append(Outlier(identifier=s.identifier, duration=s.duration, z_score=z))

    return Statistics(
        count=int(durations.size),
        mean_sec=mean_val,
        median_sec=float(np.median(durations)),
        p90_sec=float(np.percentile(durations, 90)),
        p95_sec=float(np.percentile(durations, 95)),
        min_sec=float(np.min(durations)),
        max_sec=float(np.max(durations)),
        std_dev=std_dev,
        success_rate=success_rate,
        total_sec=float(np.sum(durations)),
        slowest_artifacts=slowest,
        p95_contributors=p95_contrib,
        error_distribution=top_sigs,
        error_examples=top_sig_ex,
        error_buckets=top_buckets,
        error_bucket_examples=top_bucket_ex,
        error_exc_types=top_exc,
    )


def compute_latency_summary_by_backend(samples: List[TimingSample]) -> Dict[str, Dict[str, float]]:
    """Compute a compact per-backend latency summary (success timings only)."""
    by_backend: Dict[str, List[float]] = {}
    for s in samples:
        by_backend.setdefault(s.resolved_backend or "unknown", []).append(float(s.duration))

    summary: Dict[str, Dict[str, float]] = {}
    for backend, vals in by_backend.items():
        arr = np.array(vals, dtype=float)
        if arr.size == 0:
            continue
        summary[backend] = {
            "count": int(arr.size),
            "mean_sec": float(np.mean(arr)),
            "median_sec": float(np.median(arr)),
            "p95_sec": float(np.percentile(arr, 95)),
            "min_sec": float(np.min(arr)),
            "max_sec": float(np.max(arr)),
        }
    return summary


def raw_samples_by_backend(samples: List[TimingSample]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for s in samples:
        out.setdefault(s.resolved_backend or "unknown", []).append(float(s.duration))
    return out


# ----------------------------
# Bootstrap / significance
# ----------------------------

def bootstrap_ci_for_stat(
    baseline_samples: np.ndarray,
    current_samples: np.ndarray,
    stat_fn,
    *,
    iterations: int,
    confidence_level: float,
    seed: int = 42,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> Optional[Tuple[float, float]]:
    """Bootstrap CI for (stat(current) - stat(baseline)). Returns None if too few samples."""
    if baseline_samples.size < min_samples or current_samples.size < min_samples:
        return None

    rng = np.random.default_rng(seed)
    diffs = np.empty(int(iterations), dtype=float)

    n_b = baseline_samples.size
    n_c = current_samples.size

    for i in range(int(iterations)):
        b = rng.choice(baseline_samples, size=n_b, replace=True)
        c = rng.choice(current_samples, size=n_c, replace=True)
        diffs[i] = float(stat_fn(c) - stat_fn(b))

    alpha = (1.0 - confidence_level) / 2.0
    lo = float(np.percentile(diffs, alpha * 100.0))
    hi = float(np.percentile(diffs, (1.0 - alpha) * 100.0))
    return (lo, hi)


def detect_regressions(
    baseline: Baseline,
    current_stats: Statistics,
    current_samples: List[TimingSample],
    thresholds: Dict[str, float],
    *,
    bootstrap_iterations: int,
    confidence_level: float,
    min_samples: int,
) -> List[Regression]:
    """Compare current stats against baseline with significance testing."""
    regressions: List[Regression] = []
    bs = baseline.statistics

    b_raw = np.array(baseline.raw_samples or [], dtype=float)
    c_raw = np.array([s.duration for s in current_samples], dtype=float)

    def pct_change(base: float, curr: float) -> float:
        return ((curr - base) / base) * 100.0 if base > 0 else 0.0

    def check_latency_metric(metric_name: str, base_val: float, curr_val: float, thresh_pct: float):
        if base_val <= 0 or curr_val <= 0:
            return

        chg_pct = pct_change(base_val, curr_val)
        if chg_pct <= thresh_pct:
            return

        ci = bootstrap_ci_for_stat(
            b_raw,
            c_raw,
            (np.mean if metric_name == "mean_sec" else (lambda x: np.percentile(x, 95))),
            iterations=bootstrap_iterations,
            confidence_level=confidence_level,
            min_samples=min_samples,
        )

        if ci is None:
            significance = "potential"
        else:
            significance = "significant" if ci[0] > 0 else "noise"

        regressions.append(
            Regression(
                metric=metric_name,
                baseline=base_val,
                current=curr_val,
                change_pct=chg_pct,
                threshold_pct=thresh_pct,
                status="regression",
                significance=significance,
                confidence_interval=ci,
            )
        )

    check_latency_metric("p95_sec", bs.p95_sec, current_stats.p95_sec, thresholds["p95_worsening_pct"])
    check_latency_metric("mean_sec", bs.mean_sec, current_stats.mean_sec, thresholds["mean_worsening_pct"])

    # Failure-rate regression (does not require timings)
    fail_rate_base = 1.0 - bs.success_rate
    fail_rate_curr = 1.0 - current_stats.success_rate
    fail_increase = fail_rate_curr - fail_rate_base

    if fail_increase > thresholds["failure_rate_increase"]:
        regressions.append(
            Regression(
                metric="failure_rate",
                baseline=fail_rate_base * 100.0,
                current=fail_rate_curr * 100.0,
                change_pct=fail_increase * 100.0,
                threshold_pct=thresholds["failure_rate_increase"] * 100.0,
                status="regression",
                significance="significant",
                confidence_interval=None,
            )
        )

    return regressions


# ----------------------------
# Histogram
# ----------------------------

def render_histogram(data: List[float], *, bins: int = DEFAULT_HIST_BINS, width: int = DEFAULT_HIST_WIDTH) -> str:
    """Render ASCII histogram."""
    if not data:
        return ""
    try:
        counts, edges = np.histogram(data, bins=int(bins))
        max_count = int(max(counts)) if len(counts) else 0
        if max_count == 0:
            return "No Data"
        lines: List[str] = []
        for i, count in enumerate(counts):
            bar = "█" * int((count / max_count) * int(width))
            lines.append(f"{edges[i]:6.2f}s ┤ {bar:<{int(width)}} ({int(count)})")
        return "\n".join(lines)
    except Exception:
        return "Histogram error"


# ----------------------------
# Reporting
# ----------------------------

def format_markdown(
    baseline: Baseline,
    current_stats: Statistics,
    regressions: List[Regression],
    env: EnvironmentMetadata,
    compliance: BackendComplianceMetadata,
    latency_samples: List[TimingSample],
    latency_summary: Dict[str, Dict[str, float]],
    *,
    backend_mismatch: bool,
    insufficient_latency_data: bool,
    current_success: int,
    current_failure: int,
    failed_manifests: int,
    hist_bins: int,
    hist_width: int,
) -> str:
    """Generate comprehensive forensic report."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    significant_regs = [r for r in regressions if r.significance == "significant"]

    status_icon = "✅"
    status_text = "PERFORMANCE OK"
    if backend_mismatch:
        status_icon, status_text = "❌", "INVALID (Backend mismatch)"
    elif significant_regs:
        status_icon, status_text = "❌", f"REGRESSION ({len(significant_regs)} significant)"
    elif insufficient_latency_data:
        status_icon, status_text = "⚠️", "INSUFFICIENT LATENCY DATA (significance unavailable)"
    elif regressions:
        status_icon, status_text = "⚠️", "POTENTIAL REGRESSION (noise/potential)"

    # Executive delta safety
    mean_delta_pct = "n/a"
    if baseline.statistics.mean_sec > 0 and current_stats.mean_sec > 0:
        mean_delta_pct = f"{((current_stats.mean_sec - baseline.statistics.mean_sec) / baseline.statistics.mean_sec * 100.0):+.1f}%"

    lines: List[str] = [
        "# Performance & Compliance Ledger",
        "",
        f"**Date:** {now_utc} | **Tool:** v{TOOL_VERSION}",
        "",
        "## 1. Executive Summary",
        "",
        f"### {status_icon} {status_text}",
        "",
        "| Category | Baseline | Current | Delta |",
        "|----------|----------|---------|-------|",
        f"| **Backend** | `{baseline.backend}` | `{compliance.dominant_resolved_backend}` | {'✅ Match' if not backend_mismatch else '❌ Mismatch'} |",
        f"| **Timing Samples (comp backend)** | {baseline.statistics.count} | {current_stats.count} | {current_stats.count - baseline.statistics.count:+} |",
        f"| **Mean Latency (comp backend)** | {baseline.statistics.mean_sec:.3f}s | {current_stats.mean_sec:.3f}s | {mean_delta_pct} |",
        f"| **Success Rate (overall)** | {baseline.statistics.success_rate*100:.1f}% | {current_stats.success_rate*100:.1f}% | {(current_stats.success_rate-baseline.statistics.success_rate)*100:+.1f}% |",
        f"| **Current Manifests** | — | {compliance.total_count} | ok={current_success}, fail={current_failure}, parse_failed={failed_manifests} |",
        "",
    ]

    # Compliance
    dist_str = ", ".join([f"{k}:{v}" for k, v in compliance.distribution.items()])
    lines.extend([
        "## 2. Compliance & Consistency",
        "",
        f"- **Backend Requested:** `{compliance.requested_backend}`",
        f"- **Backend Resolved (dominant):** `{compliance.dominant_resolved_backend}`",
        f"- **Fallbacks:** {compliance.fallback_count}/{compliance.total_count} ({compliance.fallback_rate_pct:.1f}%)",
        f"- **Consistency:** `{compliance.consistency_status}` | Distribution: {dist_str}",
        "",
    ])

    # Per-backend latency summary (success timings)
    if latency_summary:
        lines.extend([
            "### Per-Backend Latency Summary (success timings)",
            "",
            "| Resolved Backend | Samples | Mean | Median | p95 | Min | Max |",
            "|------------------|---------|------|--------|-----|-----|-----|",
        ])
        for backend, s in sorted(latency_summary.items(), key=lambda kv: (-kv[1].get("count", 0), kv[0])):
            lines.append(
                f"| `{backend}` | {int(s.get('count', 0))} | {s.get('mean_sec', 0.0):.3f}s | {s.get('median_sec', 0.0):.3f}s | "
                f"{s.get('p95_sec', 0.0):.3f}s | {s.get('min_sec', 0.0):.3f}s | {s.get('max_sec', 0.0):.3f}s |"
            )
        lines.append("")

    # Regressions
    if regressions and not backend_mismatch:
        lines.extend([
            "## 3. Regression Forensics",
            "",
            "| Metric | Baseline | Current | Change | Significance | CI (diff, 95%) |",
            "|--------|----------|---------|--------|--------------|----------------|",
        ])
        for r in regressions:
            sig_icon = "🔴" if r.significance == "significant" else ("🟡" if r.significance in {"potential", "noise"} else "⚪")
            ci_str = "n/a"
            if r.confidence_interval is not None:
                ci_str = f"[{r.confidence_interval[0]:+.3f}, {r.confidence_interval[1]:+.3f}]"
            lines.append(
                f"| {r.metric} | {r.baseline:.3f} | {r.current:.3f} | {r.change_pct:+.1f}% | {sig_icon} {r.significance} | {ci_str} |"
            )
        lines.append("")

    # Histogram (comparison backend only)
    latency_values = [s.duration for s in latency_samples]
    if latency_values and not backend_mismatch:
        lines.extend([
            "### Latency Distribution (comparison backend)",
            "```text",
            render_histogram(latency_values, bins=hist_bins, width=hist_width),
            "```",
            "",
        ])

    # Outliers
    if current_stats.slowest_artifacts:
        lines.extend([
            "## 4. Top Slowest Artifacts (comparison backend)",
            "",
            "| Rank | Identifier | Duration | Z-Score |",
            "|------|------------|----------|---------|",
        ])
        for i, out in enumerate(current_stats.slowest_artifacts, 1):
            lines.append(f"| #{i} | `{out.identifier}` | **{out.duration:.2f}s** | {out.z_score:.1f}σ |")
        lines.append("")

    # p95 contributors
    if current_stats.p95_contributors:
        lines.extend([
            "## 5. p95 Contributors (tail drivers)",
            "Artifacts at/above the p95 cutoff (comparison backend):",
            "",
            "| Rank | Identifier | Duration | Z-Score |",
            "|------|------------|----------|---------|",
        ])
        for i, out in enumerate(current_stats.p95_contributors, 1):
            lines.append(f"| #{i} | `{out.identifier}` | **{out.duration:.2f}s** | {out.z_score:.1f}σ |")
        lines.append("")

    # Failure taxonomy
    if current_failure > 0:
        lines.extend(["## 6. Failure Taxonomy", ""])
        if current_stats.error_buckets:
            lines.extend([
                "### Top Buckets",
                "| Count | Bucket | Example |",
                "|-------|--------|---------|",
            ])
            for b, cnt in current_stats.error_buckets.items():
                ex = (current_stats.error_bucket_examples or {}).get(b, "")
                ex_short = (ex[:120] + "...") if len(ex) > 120 else ex
                lines.append(f"| {cnt} | `{b}` | `{ex_short}` |")
            lines.append("")

        if current_stats.error_distribution:
            lines.extend([
                "### Top Signatures",
                "| Count | Normalized Signature | Example |",
                "|-------|----------------------|---------|",
            ])
            for sig, cnt in current_stats.error_distribution.items():
                ex = (current_stats.error_examples or {}).get(sig, "")
                ex_short = (ex[:120] + "...") if len(ex) > 120 else ex
                sig_short = (sig[:120] + "...") if len(sig) > 120 else sig
                lines.append(f"| {cnt} | `{sig_short}` | `{ex_short}` |")
            lines.append("")

        if current_stats.error_exc_types:
            lines.extend([
                "### Exception Type Mix",
                "| Count | Exception Type |",
                "|-------|----------------|",
            ])
            for et, cnt in current_stats.error_exc_types.items():
                lines.append(f"| {cnt} | `{et}` |")
            lines.append("")

    # Environment
    lines.extend([
        "## 7. Environment",
        f"- **OS:** {env.os} | **Device:** {env.device} | **Torch:** {env.torch or 'N/A'}",
        f"- **Python:** {env.python} | **CPU:** {env.cpu or 'N/A'} | **Memory:** {str(env.memory_gb)+' GB' if env.memory_gb else 'N/A'}",
        "",
    ])

    return "\n".join(lines)


# ----------------------------
# Environment capture (robust torch)
# ----------------------------

def capture_environment() -> EnvironmentMetadata:
    """Capture environment metadata safely (torch may be absent, stubbed, or mocked)."""
    python_ver = sys.version.split()[0]
    os_str = f"{platform.system()}-{platform.release()}-{platform.machine()}"
    cpu = platform.processor() or None

    torch_ver: Optional[str] = None
    device = "cpu"

    try:
        import torch  # type: ignore
        torch_ver = getattr(torch, "__version__", None)

        # CUDA first (most specific)
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and getattr(cuda, "is_available", None) and cuda.is_available():
            device = "cuda"
        else:
            backends = getattr(torch, "backends", None)
            mps = getattr(backends, "mps", None) if backends is not None else None
            if mps is not None and getattr(mps, "is_available", None) and mps.is_available():
                device = "mps"
    except Exception:
        torch_ver = None
        device = "cpu"

    mem_gb: Optional[int] = None
    try:
        import psutil  # type: ignore
        mem_gb = int(psutil.virtual_memory().total / (1024**3))
    except Exception:
        mem_gb = None

    return EnvironmentMetadata(
        python=python_ver,
        torch=torch_ver,
        device=device,
        os=os_str,
        cpu=cpu,
        memory_gb=mem_gb,
    )


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Performance Ledger v" + TOOL_VERSION)

    # Mode args
    parser.add_argument("--manifests-dir", type=Path, help="Directory containing manifest JSONs (capture mode).")
    parser.add_argument("--compare", type=Path, help="Directory containing manifest JSONs (compare mode).")
    parser.add_argument("--baseline", type=Path, help="Baseline JSON path (compare mode).")

    # Outputs
    parser.add_argument("--output", type=Path, required=True, help="Output file: baseline JSON (capture) or report MD (compare).")
    parser.add_argument("--emit-json", type=Path, help="Emit rich JSON context (compare mode).")

    # Baseline metadata
    parser.add_argument("--baseline-version", default="auto")
    parser.add_argument("--quality-tier", default="unknown")
    parser.add_argument("--notes")

    # Thresholds (CLI overrides)
    parser.add_argument("--p95-threshold", type=float, default=DEFAULT_REGRESSION_THRESHOLDS["p95_worsening_pct"])
    parser.add_argument("--mean-threshold", type=float, default=DEFAULT_REGRESSION_THRESHOLDS["mean_worsening_pct"])
    parser.add_argument(
        "--failure-rate-threshold",
        type=float,
        default=DEFAULT_REGRESSION_THRESHOLDS["failure_rate_increase"] * 100.0,
        help="Allowed increase in failure rate in percentage points (default: 0.0). Example: 0.5 means allow +0.5% failures.",
    )

    # Significance knobs
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--confidence-level", type=float, default=DEFAULT_CONFIDENCE_LEVEL)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES, help="Minimum samples required for latency significance (bootstrap).")

    # Forensics knobs
    parser.add_argument("--top-slowest", type=int, default=DEFAULT_TOP_SLOWEST)
    parser.add_argument("--top-p95", type=int, default=DEFAULT_TOP_P95_CONTRIB)
    parser.add_argument("--hist-bins", type=int, default=DEFAULT_HIST_BINS)
    parser.add_argument("--hist-width", type=int, default=DEFAULT_HIST_WIDTH)

    # Flags
    parser.add_argument("--strict", action="store_true", help="Fail (exit 1) on any regression, even if noise/potential.")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    mode = "compare" if args.compare else "capture"
    target_dir = args.compare if mode == "compare" else args.manifests_dir

    if not target_dir:
        logger.error("Must specify --manifests-dir (capture) or --compare (compare).")
        return 1

    # Load manifests
    try:
        manifests, failed = parse_manifests(target_dir)
    except Exception as e:
        logger.error(str(e))
        return 1
    if not manifests:
        logger.error("No valid manifests found.")
        return 1

    # Extract data
    samples_all, succ_cnt, fail_cnt, sigs, sig_ex, buckets, bucket_ex, exc_types = extract_data(manifests)
    compliance = analyze_backend_compliance(manifests)
    env = capture_environment()

    # Compute per-backend latency summary over all success samples
    latency_summary = compute_latency_summary_by_backend(samples_all)

    logger.info(
        f"Processed {compliance.total_count} manifests "
        f"(ok={succ_cnt}, fail={fail_cnt}, parse_failed={len(failed)}, timing_samples={len(samples_all)})"
    )

    # Capture mode: baseline statistics computed on dominant backend only
    if mode == "capture":
        baseline_backend = compliance.dominant_resolved_backend
        samples_backend = [s for s in samples_all if s.resolved_backend == baseline_backend]

        stats = compute_statistics(
            samples_backend,
            succ_cnt,
            fail_cnt,
            sigs,
            sig_ex,
            buckets,
            bucket_ex,
            exc_types,
            top_slowest=args.top_slowest,
            top_p95=args.top_p95,
        )

        baseline = Baseline(
            version=args.baseline_version,
            backend=baseline_backend,
            quality_tier=args.quality_tier,
            environment=env,
            statistics=stats,
            compliance=compliance,
            raw_samples=[s.duration for s in samples_backend],
            raw_samples_by_backend=raw_samples_by_backend(samples_all),
            latency_summary_by_backend=latency_summary,
            notes=args.notes,
        )
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(asdict(baseline), f, indent=2)
        logger.info(f"Baseline captured: {args.output}")
        return 0

    # Compare mode
    if not args.baseline:
        logger.error("--baseline required for comparison.")
        return 1

    try:
        baseline = _load_baseline(args.baseline)
    except Exception as e:
        logger.error(f"Failed to load baseline: {e}")
        return 1

    backend_mismatch = baseline.backend != compliance.dominant_resolved_backend

    # Comparison backend = baseline.backend (by definition)
    comp_backend = baseline.backend
    current_samples_backend = [s for s in samples_all if s.resolved_backend == comp_backend]

    # Compute current stats on comparison backend samples, but success/failure/taxonomy on whole run
    current_stats = compute_statistics(
        current_samples_backend,
        succ_cnt,
        fail_cnt,
        sigs,
        sig_ex,
        buckets,
        bucket_ex,
        exc_types,
        top_slowest=args.top_slowest,
        top_p95=args.top_p95,
    )

    # Insufficient latency data: not enough samples for bootstrap CI (baseline OR current)
    b_raw = np.array(baseline.raw_samples or [], dtype=float)
    c_raw = np.array([s.duration for s in current_samples_backend], dtype=float)
    insufficient_latency_data = (b_raw.size < int(args.min_samples)) or (c_raw.size < int(args.min_samples))

    thresholds = {
        "p95_worsening_pct": float(args.p95_threshold),
        "mean_worsening_pct": float(args.mean_threshold),
        "failure_rate_increase": float(args.failure_rate_threshold) / 100.0,  # pct points -> fraction
    }

    regressions: List[Regression] = []
    if not backend_mismatch:
        regressions = detect_regressions(
            baseline,
            current_stats,
            current_samples_backend,
            thresholds,
            bootstrap_iterations=int(args.bootstrap_iterations),
            confidence_level=float(args.confidence_level),
            min_samples=int(args.min_samples),
        )

    # Render report (always, even if mismatch, to surface compliance/failures)
    report = format_markdown(
        baseline=baseline,
        current_stats=current_stats,
        regressions=regressions,
        env=env,
        compliance=compliance,
        latency_samples=current_samples_backend,
        latency_summary=latency_summary,
        backend_mismatch=backend_mismatch,
        insufficient_latency_data=insufficient_latency_data,
        current_success=succ_cnt,
        current_failure=fail_cnt,
        failed_manifests=len(failed),
        hist_bins=int(args.hist_bins),
        hist_width=int(args.hist_width),
    )
    with args.output.open("w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved: {args.output}")

    # Exit code precedence:
    # 2 backend mismatch > 1 significant regression > 3 insufficient latency data > 0 OK (or strict handling)
    significant_regs = [r for r in regressions if r.significance == "significant"]

    exit_code = 0
    status = "ok"
    if backend_mismatch:
        exit_code, status = 2, "backend_mismatch"
    elif significant_regs:
        exit_code, status = 1, "regression"
    elif insufficient_latency_data:
        exit_code, status = 3, "insufficient_latency_data"
    elif regressions:
        exit_code, status = (1 if args.strict else 0), ("regression_strict" if args.strict else "potential_regression")

    # Emit JSON
    if args.emit_json:
        out_json = {
            "tool_version": TOOL_VERSION,
            "status": status,
            "exit_code": exit_code,
            "thresholds": {
                "p95_worsening_pct": thresholds["p95_worsening_pct"],
                "mean_worsening_pct": thresholds["mean_worsening_pct"],
                "failure_rate_increase_pct_points": thresholds["failure_rate_increase"] * 100.0,
            },
            "bootstrap": {
                "iterations": int(args.bootstrap_iterations),
                "confidence_level": float(args.confidence_level),
                "min_samples": int(args.min_samples),
            },
            "comparison": {
                "backend": comp_backend,
                "baseline_backend": baseline.backend,
                "current_dominant_backend": compliance.dominant_resolved_backend,
                "backend_mismatch": backend_mismatch,
            },
            "regressions": [asdict(r) for r in regressions],
            "significant_regressions": [asdict(r) for r in significant_regs],
            "outliers": {
                "slowest": [asdict(o) for o in current_stats.slowest_artifacts],
                "p95_contributors": [asdict(o) for o in current_stats.p95_contributors],
            },
            "baseline": {
                "version": baseline.version,
                "backend": baseline.backend,
                "quality_tier": baseline.quality_tier,
                "captured_at": baseline.captured_at,
            },
            "current": {
                "manifest_total": compliance.total_count,
                "success": succ_cnt,
                "failure": fail_cnt,
                "timing_samples_total": len(samples_all),
                "timing_samples_comparison_backend": current_stats.count,
                "parse_failed": len(failed),
            },
            "latency_summary_by_backend": latency_summary,
            "stats": asdict(current_stats),
            "failure_analysis": {
                "buckets": current_stats.error_buckets or {},
                "bucket_examples": current_stats.error_bucket_examples or {},
                "top_signatures": current_stats.error_distribution or {},
                "signature_examples": current_stats.error_examples or {},
                "exception_types": current_stats.error_exc_types or {},
            },
            "compliance": asdict(compliance),
            "insufficient_latency_data": insufficient_latency_data,
            "failed_manifest_details": failed,
        }
        with args.emit_json.open("w", encoding="utf-8") as f:
            json.dump(out_json, f, indent=2)
        logger.info(f"Rich JSON saved: {args.emit_json}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
