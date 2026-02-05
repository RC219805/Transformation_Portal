#!/usr/bin/env python3
"""Performance ledger tool for pipeline regression detection.

Parses manifests from batch runs, computes statistics, and compares against
baselines to detect performance regressions.

v1.5 "Forensic" Enhancements:
- Outlier Attribution: Identifies specific images causing p95 spikes.
- Statistical Significance: Uses Bootstrap Resampling (NumPy) for 95% CIs.
- Raw Data Persistence: Baselines store raw samples for rigorous comparison.
- Forensic Reporting: Separates "Noise" from "Signal" in regressions.

Usage:
    # Capture baseline (now stores raw samples for future bootstrapping)
    python tools/performance_ledger.py \
        --manifests-dir ./output/prod_run/manifests \
        --output ./docs/performance/baselines/baseline_v2.0.0.json

    # Compare against baseline (Forensic Mode)
    python tools/performance_ledger.py \
        --baseline ./docs/performance/baselines/baseline_v2.0.0.json \
        --compare ./output/experimental_run/manifests \
        --output ./output/perf_report.md \
        --emit-json ./output/perf_full.json \
        --strict

Exit codes:
    0 = OK
    1 = Regressions detected (Statistically Significant)
    2 = Backend mismatch
    3 = Insufficient data
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

TOOL_VERSION = "1.5.0"

# Regression thresholds (per ADR-023)
DEFAULT_REGRESSION_THRESHOLDS = {
    "p95_worsening_pct": 10.0,
    "mean_worsening_pct": 15.0,
    "failure_rate_increase": 0.0,
}

# Bootstrap settings
BOOTSTRAP_ITERATIONS = 2000
CONFIDENCE_LEVEL = 0.95


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
    """A single timing measurement linked to an artifact."""
    duration: float
    identifier: str  # filename or manifest ID


@dataclass
class Outlier:
    """Details about a performance outlier."""
    identifier: str
    duration: float
    z_score: float


@dataclass
class Statistics:
    """Runtime statistics for a batch run."""
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
    
    # Forensic: Outliers
    slowest_artifacts: List[Outlier] = field(default_factory=list)

    # Forensic: Failure Taxonomy
    error_distribution: Optional[Dict[str, int]] = None          # signature -> count
    error_examples: Optional[Dict[str, str]] = None              # signature -> example
    error_buckets: Optional[Dict[str, int]] = None               # bucket -> count
    error_bucket_examples: Optional[Dict[str, str]] = None       # bucket -> example
    error_exc_types: Optional[Dict[str, int]] = None             # type -> count


@dataclass
class Baseline:
    """Performance baseline schema."""
    version: str
    backend: str
    quality_tier: str
    environment: EnvironmentMetadata
    statistics: Statistics
    compliance: Optional[BackendComplianceMetadata] = None
    # Raw samples needed for bootstrap comparison
    raw_samples: Optional[List[float]] = None
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
    significance: str = "unknown"  # "significant", "noise", "unknown"
    confidence_interval: Optional[Tuple[float, float]] = None


def _dataclass_from_dict(cls, data: Optional[Dict[str, Any]]):
    """Create dataclass instance from dict, ignoring unknown keys."""
    if data is None:
        return None
    allowed = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in allowed}
    return cls(**filtered)


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
            
            # Inject filename as fallback identifier
            if "identifier" not in data:
                data["identifier"] = manifest_file.name

            if any(k in data for k in ("timing", "status", "v2", "backend_selection", "depth")):
                valid_manifests.append(data)
            else:
                failed_manifests.append({"file": str(manifest_file), "error": "Missing key blocks"})
        except json.JSONDecodeError as e:
            logger.warning(f"Skipping invalid JSON {manifest_file}: {e}")
            failed_manifests.append({"file": str(manifest_file), "error": str(e)})

    logger.info(f"Loaded {len(valid_manifests)} manifests from {manifests_dir}")
    return valid_manifests, failed_manifests


def analyze_backend_compliance(manifests: List[Dict[str, Any]]) -> BackendComplianceMetadata:
    """Analyze backend selection truth and fallback rates."""
    requested_counts = Counter()
    resolved_counts = Counter()
    fallback_count = 0
    total = 0

    for m in manifests:
        bs = m.get("backend_selection")
        if isinstance(bs, dict) and bs:
            requested = (bs.get("requested_backend") or "auto").strip()
            resolved = (bs.get("resolved_backend") or "unknown").strip()
            status = (bs.get("resolution_status") or "unknown").strip().lower()
            if "fallback" in status:
                fallback_count += 1
        else:
            # Legacy inference
            depth_meta = m.get("depth") if isinstance(m.get("depth"), dict) else {}
            stats = depth_meta.get("stats") if isinstance(depth_meta.get("stats"), dict) else {}
            if "backend" in stats and isinstance(stats["backend"], str):
                resolved = stats["backend"]
            elif "model" in depth_meta and isinstance(depth_meta.get("model"), str):
                resolved = "depth_pro" if "depth-pro" in depth_meta["model"].lower() else "da3"
            else:
                resolved = "da3"
            requested = "unknown"

        requested_counts[requested] += 1
        resolved_counts[resolved] += 1
        total += 1

    dominant_resolved = resolved_counts.most_common(1)[0][0] if resolved_counts else "unknown"
    dominant_requested = requested_counts.most_common(1)[0][0] if requested_counts else "auto"
    fallback_rate = (fallback_count / total * 100.0) if total > 0 else 0.0

    consistency = "clean"
    if total == 0: consistency = "empty"
    elif fallback_rate > 50.0: consistency = "fallback_heavy"
    elif len(resolved_counts) > 1: consistency = "mixed"

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
    """Normalize error message to stable signature."""
    raw = (msg or "").strip()
    if not raw: return ("unknown error", "unknown error")
    
    s = raw
    if "\n" in s:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if lines: s = lines[-1] if (":" in lines[-1] or "error" in lines[-1].lower()) else lines[0]

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
    """Classify error into type, signature, and buckets."""
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
    def has(*terms: str) -> bool: return any(t in s or t in r for t in terms)
    
    buckets = []
    if has("out of memory", "oom", "allocation", "<mem>"): buckets.append("oom")
    if has("no module named", "importerror", "modulenotfounderror"): buckets.append("import")
    if has("has no attribute", "attributeerror"): buckets.append("api")
    if has("backend mismatch", "invalid comparison"): buckets.append("backend")
    if has("no such file", "filenotfounderror", "ioerror"): buckets.append("io")
    if has("permission denied", "eacces"): buckets.append("permission")
    if has("timeout", "timed out"): buckets.append("timeout")
    if has("shape", "sizes", "dimension", "broadcast"): buckets.append("shape")
    if has("device", "cuda", "mps"): buckets.append("device")
    if has("connection", "dns", "ssl", "http"): buckets.append("network")
    if has("nan", "inf", "overflow"): buckets.append("numerics")
    
    if not buckets: buckets = ["unknown"]
    return exc_type, sig, buckets, raw


def extract_data(
    manifests: List[Dict[str, Any]]
) -> Tuple[
    List[TimingSample], int, int,
    List[str], Dict[str, str], Counter, Dict[str, str], Counter
]:
    """Extract timings (with identifiers) and failure taxonomy."""
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
        # Resolve Identifier
        identifier = m.get("identifier", "unknown")
        if "input" in m and isinstance(m["input"], dict):
            p = m["input"].get("image_path")
            if p: identifier = Path(p).name

        # Timing
        timing = m.get("timing") if isinstance(m.get("timing"), dict) else {}
        total_sec = timing.get("total_seconds")

        # Status
        v2 = m.get("v2") if isinstance(m.get("v2"), dict) else {}
        status = str(v2.get("status") or m.get("status") or "").lower().strip()

        is_success = False
        if status in ok_statuses: is_success = True
        elif status in fail_statuses: is_success = False
        elif m.get("depth") is not None: is_success = True # Legacy/PBR assumption

        if is_success:
            success_count += 1
            if total_sec is not None:
                try:
                    val = float(total_sec)
                    samples.append(TimingSample(duration=val, identifier=identifier))
                except (TypeError, ValueError): pass
        else:
            failure_count += 1
            err_msg = v2.get("error_message") or m.get("error") or status or "unknown error"
            exc_type, sig, buckets, ex = classify_error(str(err_msg))
            
            signatures.append(sig)
            if sig not in sig_examples: sig_examples[sig] = ex
            exc_type_counts[exc_type] += 1
            for b in buckets:
                bucket_counts[b] += 1
                if b not in bucket_examples: bucket_examples[b] = ex

    return (
        samples, success_count, failure_count,
        signatures, sig_examples, bucket_counts, bucket_examples, exc_type_counts
    )


def compute_statistics(
    samples: List[TimingSample],
    success_count: int,
    failure_count: int,
    signatures: List[str],
    sig_examples: Dict[str, str],
    bucket_counts: Counter,
    bucket_examples: Dict[str, str],
    exc_type_counts: Counter,
) -> Statistics:
    """Compute statistics including outliers and failure taxonomy."""
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
            count=0, mean_sec=0.0, median_sec=0.0, p90_sec=0.0, p95_sec=0.0,
            min_sec=0.0, max_sec=0.0, std_dev=0.0, success_rate=success_rate,
            total_sec=0.0,
            error_distribution=top_sigs, error_examples=top_sig_ex,
            error_buckets=top_buckets, error_bucket_examples=top_bucket_ex,
            error_exc_types=top_exc
        )

    durations = np.array([s.duration for s in samples], dtype=float)
    mean_val = float(np.mean(durations))
    std_dev = float(np.std(durations))

    # Identify Outliers (Top 5 slowest)
    sorted_samples = sorted(samples, key=lambda x: x.duration, reverse=True)
    outliers = []
    for s in sorted_samples[:5]:
        z_score = (s.duration - mean_val) / std_dev if std_dev > 0 else 0.0
        outliers.append(Outlier(identifier=s.identifier, duration=s.duration, z_score=z_score))

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
        slowest_artifacts=outliers,
        error_distribution=top_sigs, error_examples=top_sig_ex,
        error_buckets=top_buckets, error_bucket_examples=top_bucket_ex,
        error_exc_types=top_exc
    )


def bootstrap_confidence_interval(
    baseline_samples: np.ndarray, 
    current_samples: np.ndarray, 
    iterations: int = BOOTSTRAP_ITERATIONS
) -> Tuple[float, float]:
    """Calculate 95% CI for the difference in means (Current - Baseline)."""
    if len(baseline_samples) < 5 or len(current_samples) < 5:
        return (0.0, 0.0)

    diffs = []
    rng = np.random.default_rng(42)

    for _ in range(iterations):
        b_resample = rng.choice(baseline_samples, size=len(baseline_samples), replace=True)
        c_resample = rng.choice(current_samples, size=len(current_samples), replace=True)
        diffs.append(np.mean(c_resample) - np.mean(b_resample))

    diffs = np.array(diffs)
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    lower = np.percentile(diffs, alpha * 100)
    upper = np.percentile(diffs, (1.0 - alpha) * 100)
    
    return (float(lower), float(upper))


def detect_regressions(
    baseline: Baseline, 
    current_stats: Statistics, 
    current_samples: List[TimingSample],
    thresholds: Dict[str, float]
) -> List[Regression]:
    """Compare current stats against baseline with significance testing."""
    regressions: List[Regression] = []
    bs = baseline.statistics

    b_raw = np.array(baseline.raw_samples) if baseline.raw_samples else np.array([])
    c_raw = np.array([s.duration for s in current_samples])

    def check_metric(metric_name: str, base_val: float, curr_val: float, thresh_pct: float):
        if base_val <= 0 or curr_val <= 0: return
        
        pct_change = ((curr_val - base_val) / base_val) * 100.0
        
        if pct_change > thresh_pct:
            significance = "unknown"
            ci = None
            
            # Bootstrap for mean significance
            if metric_name == "mean_sec" and len(b_raw) > 0 and len(c_raw) > 0:
                ci = bootstrap_confidence_interval(b_raw, c_raw)
                significance = "significant" if ci[0] > 0 else "noise"
            elif metric_name == "p95_sec":
                significance = "significant" if pct_change > (thresh_pct * 2) else "potential"

            regressions.append(Regression(
                metric=metric_name,
                baseline=base_val,
                current=curr_val,
                change_pct=pct_change,
                threshold_pct=thresh_pct,
                status="regression",
                significance=significance,
                confidence_interval=ci
            ))

    check_metric("p95_sec", bs.p95_sec, current_stats.p95_sec, thresholds["p95_worsening_pct"])
    check_metric("mean_sec", bs.mean_sec, current_stats.mean_sec, thresholds["mean_worsening_pct"])

    fail_rate_base = 1.0 - bs.success_rate
    fail_rate_curr = 1.0 - current_stats.success_rate
    fail_increase = fail_rate_curr - fail_rate_base

    if fail_increase > thresholds["failure_rate_increase"]:
        regressions.append(Regression(
            "failure_rate", fail_rate_base * 100.0, fail_rate_curr * 100.0,
            fail_increase * 100.0, thresholds["failure_rate_increase"] * 100.0,
            "regression", "significant"
        ))

    return regressions


def _render_histogram(data: List[float], bins: int = 10, width: int = 24) -> str:
    """Render ASCII histogram."""
    if not data: return ""
    try:
        counts, edges = np.histogram(data, bins=bins)
        max_count = int(max(counts)) if len(counts) else 0
        if max_count == 0: return "No Data"
        lines = []
        for i, count in enumerate(counts):
            bar = "█" * int((count / max_count) * width)
            lines.append(f"{edges[i]:6.2f}s ┤ {bar:<{width}} ({int(count)})")
        return "\n".join(lines)
    except Exception: return "Histogram error"


def format_markdown(
    baseline: Baseline,
    current_stats: Statistics,
    regressions: List[Regression],
    env: EnvironmentMetadata,
    compliance: BackendComplianceMetadata,
    timings: List[float],
    *,
    backend_mismatch: bool = False,
    insufficient_data: bool = False,
    current_success: int = 0,
    current_failure: int = 0,
    failed_manifests: int = 0,
) -> str:
    """Generate comprehensive forensic report."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    status_icon = "✅"
    status_text = "PERFORMANCE OK"
    if backend_mismatch: status_icon, status_text = "❌", "INVALID (Mismatch)"
    elif insufficient_data: status_icon, status_text = "⚠️", "INSUFFICIENT DATA"
    elif regressions:
        sig_count = sum(1 for r in regressions if r.significance == "significant")
        if sig_count: status_icon, status_text = "❌", f"REGRESSION ({sig_count} significant)"
        else: status_icon, status_text = "⚠️", "POTENTIAL REGRESSION (Noise?)"

    lines = [
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
        f"| **Backend** | `{baseline.backend}` | `{compliance.dominant_resolved_backend}` | { '✅ Match' if not backend_mismatch else '❌ Mismatch' } |",
        f"| **Samples** | {baseline.statistics.count} | {current_stats.count} | {current_stats.count - baseline.statistics.count:+} |",
        f"| **Mean Latency** | {baseline.statistics.mean_sec:.3f}s | {current_stats.mean_sec:.3f}s | {((current_stats.mean_sec - baseline.statistics.mean_sec)/baseline.statistics.mean_sec*100):+.1f}% |",
        "",
    ]

    # Compliance
    lines.extend([
        "## 2. Compliance & Consistency",
        "",
        f"- **Backend Requested:** `{compliance.requested_backend}`",
        f"- **Backend Resolved:** `{compliance.dominant_resolved_backend}`",
        f"- **Fallbacks:** {compliance.fallback_count} / {compliance.total_count} ({compliance.fallback_rate_pct:.1f}%)",
        ""
    ])

    # Regressions
    if regressions and not backend_mismatch:
        lines.extend([
            "## 3. Regression Forensics",
            "",
            "| Metric | Baseline | Current | Change | Significance | CI (95%) |",
            "|--------|----------|---------|--------|--------------|----------|"
        ])
        for r in regressions:
            sig_icon = "🔴" if r.significance == "significant" else "🟡"
            ci_str = f"[{r.confidence_interval[0]:+.2f}, {r.confidence_interval[1]:+.2f}]" if r.confidence_interval else "n/a"
            lines.append(f"| {r.metric} | {r.baseline:.2f} | {r.current:.2f} | {r.change_pct:+.1f}% | {sig_icon} {r.significance} | {ci_str} |")
        lines.append("")

    # Latency Histogram
    if timings and not insufficient_data:
        lines.extend([
            "### Latency Distribution",
            "```text",
            _render_histogram(timings),
            "```",
            ""
        ])

    # Outliers
    if current_stats.slowest_artifacts:
        lines.extend([
            "## 4. Top 5 Slowest Artifacts",
            "",
            "| Rank | Identifier | Duration | Z-Score |",
            "|------|------------|----------|---------|"
        ])
        for i, out in enumerate(current_stats.slowest_artifacts, 1):
            lines.append(f"| #{i} | `{out.identifier}` | **{out.duration:.2f}s** | {out.z_score:.1f}σ |")
        lines.append("")

    # Failure Taxonomy
    if current_failure > 0:
        lines.extend(["## 5. Failure Taxonomy", ""])
        if current_stats.error_buckets:
            lines.extend([
                "### Top Buckets",
                "| Count | Bucket | Example |",
                "|-------|--------|---------|"
            ])
            for b, cnt in current_stats.error_buckets.items():
                ex = current_stats.error_bucket_examples.get(b, "")[:80] + "..."
                lines.append(f"| {cnt} | `{b}` | `{ex}` |")
            lines.append("")
        
        if current_stats.error_distribution:
            lines.extend([
                "### Top Signatures",
                "| Count | Normalized Signature |",
                "|-------|----------------------|"
            ])
            for sig, cnt in current_stats.error_distribution.items():
                lines.append(f"| {cnt} | `{sig[:100]}...` |")
            lines.append("")

    # Environment
    lines.extend([
        "## 6. Environment",
        f"- **OS:** {env.os} | **Device:** {env.device} | **Torch:** {env.torch or 'N/A'}",
        ""
    ])

    return "\n".join(lines)


def capture_environment() -> EnvironmentMetadata:
    """Capture environment metadata safely."""
    python_ver = sys.version.split()[0]
    os_str = f"{platform.system()}-{platform.release()}-{platform.machine()}"
    torch_ver, device, mem_gb = None, "cpu", None
    
    try:
        import torch
        torch_ver = torch.__version__
        if torch.backends.mps.is_available(): device = "mps"
        elif torch.cuda.is_available(): device = "cuda"
    except ImportError: pass

    try:
        import psutil
        mem_gb = int(psutil.virtual_memory().total / (1024**3))
    except ImportError: pass

    return EnvironmentMetadata(python=python_ver, torch=torch_ver, device=device, os=os_str, memory_gb=mem_gb)


def main() -> int:
    parser = argparse.ArgumentParser(description="Performance Ledger v" + TOOL_VERSION)
    parser.add_argument("--manifests-dir", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-json", type=Path)
    parser.add_argument("--baseline-version", default="auto")
    parser.add_argument("--quality-tier", default="unknown")
    parser.add_argument("--notes")
    
    # Flags
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    mode = "compare" if args.compare else "capture"
    target_dir = args.compare if mode == "compare" else args.manifests_dir
    
    if not target_dir:
        logger.error("Must specify --manifests-dir or --compare")
        return 1

    # Load & Analyze
    manifests, failed = parse_manifests(target_dir)
    if not manifests: return 1
    
    # Extract data with new Tuple signature
    (
        samples, succ_cnt, fail_cnt,
        sigs, sig_ex, buckets, bucket_ex, exc_types
    ) = extract_data(manifests)

    stats = compute_statistics(
        samples, succ_cnt, fail_cnt,
        sigs, sig_ex, buckets, bucket_ex, exc_types
    )
    compliance = analyze_backend_compliance(manifests)
    env = capture_environment()

    # Capture Mode
    if mode == "capture":
        raw_samples = [s.duration for s in samples]
        baseline = Baseline(
            version=args.baseline_version,
            backend=compliance.dominant_resolved_backend,
            quality_tier=args.quality_tier,
            environment=env,
            statistics=stats,
            compliance=compliance,
            raw_samples=raw_samples,
            notes=args.notes
        )
        with args.output.open("w") as f: json.dump(asdict(baseline), f, indent=2)
        logger.info(f"Baseline captured: {args.output}")
        return 0

    # Compare Mode
    if not args.baseline: return 1
    try:
        with args.baseline.open() as f:
            base_data = json.load(f)
            baseline = Baseline(
                version=base_data.get("version", "unknown"),
                backend=base_data.get("backend", "unknown"),
                quality_tier=base_data.get("quality_tier", "unknown"),
                environment=_dataclass_from_dict(EnvironmentMetadata, base_data.get("environment")),
                statistics=_dataclass_from_dict(Statistics, base_data.get("statistics")),
                compliance=_dataclass_from_dict(BackendComplianceMetadata, base_data.get("compliance")),
                raw_samples=base_data.get("raw_samples"),
                notes=base_data.get("notes")
            )
    except Exception as e:
        logger.error(f"Failed to load baseline: {e}")
        return 1

    backend_mismatch = baseline.backend != compliance.dominant_resolved_backend
    insufficient_data = stats.count == 0 or baseline.statistics.count == 0

    regressions = []
    if not backend_mismatch and not insufficient_data:
        regressions = detect_regressions(baseline, stats, samples, DEFAULT_REGRESSION_THRESHOLDS)

    report = format_markdown(
        baseline, stats, regressions, env, compliance, [s.duration for s in samples],
        backend_mismatch=backend_mismatch,
        insufficient_data=insufficient_data,
        current_success=succ_cnt,
        current_failure=fail_cnt,
        failed_manifests=len(failed)
    )

    with args.output.open("w") as f: f.write(report)
    
    if args.emit_json:
        out_json = {
            "status": "fail" if regressions else "ok",
            "regressions": [asdict(r) for r in regressions],
            "outliers": [asdict(o) for o in stats.slowest_artifacts],
            "stats": asdict(stats),
            "compliance": asdict(compliance)
        }
        with args.emit_json.open("w") as f: json.dump(out_json, f, indent=2)

    if args.strict and (regressions or backend_mismatch): return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
