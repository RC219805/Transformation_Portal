"""APEX visual quality evalset and benchmark helpers.

The helpers in this module are intentionally offline-safe. They validate the
quality corpus, emit deterministic readiness reports, and provide a seam for
benchmark execution without forcing model downloads in unit tests.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from transformation_portal.evals.metrics import lpips_score, ssim
from transformation_portal.ingest.canonical_json import dump_json
from transformation_portal.lux_depth_v3.model_registry import MODEL_REGISTRY, UsageClass

APEX_EVALSET_SCHEMA_VERSION = "apex_evalset.v1"
APEX_EVAL_REPORT_VERSION = "apex_eval_report.v1"
DEPTH_BACKEND_BENCHMARK_REPORT_VERSION = "depth_backend_benchmark_report.v1"
DEPTH_PRO_BACKENDS = {"depth_pro"}


def _lpips_available() -> bool:
    """Return whether LPIPS can produce authoritative visible-delta evidence."""
    try:
        importlib.import_module("lpips")
        importlib.import_module("torch")
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class ApexEvalAsset:
    """Single source image entry in an APEX visual evalset."""

    asset_id: str
    asset_ref: str
    sha256: str
    scene_type: str
    expected_materials: tuple[str, ...]
    risk_zones: tuple[str, ...]
    reject_if: tuple[str, ...]
    manual_quality_score: float | None = None
    notes: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ApexEvalAsset":
        required = ("asset_id", "asset_ref", "sha256", "scene_type")
        missing = [key for key in required if not str(payload.get(key, "")).strip()]
        if missing:
            raise ValueError(f"APEX eval asset missing required fields: {', '.join(missing)}")

        manual_score = payload.get("manual_quality_score")
        if manual_score is not None:
            manual_score = float(manual_score)
            if not 0.0 <= manual_score <= 1.0:
                raise ValueError("manual_quality_score must be null or in [0, 1]")

        return cls(
            asset_id=str(payload["asset_id"]),
            asset_ref=str(payload["asset_ref"]),
            sha256=str(payload["sha256"]).lower(),
            scene_type=str(payload["scene_type"]),
            expected_materials=tuple(str(item) for item in payload.get("expected_materials", [])),
            risk_zones=tuple(str(item) for item in payload.get("risk_zones", [])),
            reject_if=tuple(str(item) for item in payload.get("reject_if", [])),
            manual_quality_score=manual_score,
            notes=(str(payload["notes"]) if payload.get("notes") is not None else None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "asset_ref": self.asset_ref,
            "sha256": self.sha256,
            "scene_type": self.scene_type,
            "expected_materials": list(self.expected_materials),
            "risk_zones": list(self.risk_zones),
            "reject_if": list(self.reject_if),
            "manual_quality_score": self.manual_quality_score,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ApexEvalSet:
    """Loaded APEX visual evalset."""

    evalset_id: str
    version: str
    description: str
    assets: tuple[ApexEvalAsset, ...]
    source_path: Path
    repo_root: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_asset_path(self, asset: ApexEvalAsset) -> Path:
        path = Path(asset.asset_ref)
        if path.is_absolute():
            return path
        return self.repo_root / path


@dataclass(frozen=True)
class DepthBackendRunResult:
    """One backend execution result consumed by benchmark reporting."""

    backend: str
    asset_id: str
    status: str
    runtime_ms: float | None = None
    depth_map: np.ndarray | None = None
    depth_path: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


DepthBackendRunner = Callable[[str, ApexEvalAsset, Path, str], DepthBackendRunResult]


def repository_root(start: Path | None = None) -> Path:
    """Resolve the repository root from this package location or a caller path."""
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    return Path.cwd().resolve()


def load_apex_evalset(evalset_path: Path | str, *, repo_root: Path | None = None) -> ApexEvalSet:
    """Load an APEX evalset JSON file or directory containing ``evalset.json``."""
    root = (repo_root or repository_root()).resolve()
    path = Path(evalset_path)
    if not path.is_absolute():
        path = root / path
    if path.is_dir():
        path = path / "evalset.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != APEX_EVALSET_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported APEX evalset schema: {payload.get('schema_version')!r}; " f"expected {APEX_EVALSET_SCHEMA_VERSION!r}"
        )
    assets_raw = payload.get("assets")
    if not isinstance(assets_raw, list) or not assets_raw:
        raise ValueError("APEX evalset must contain a non-empty assets list")
    assets = tuple(ApexEvalAsset.from_mapping(item) for item in assets_raw)
    return ApexEvalSet(
        evalset_id=str(payload.get("evalset_id") or path.parent.name),
        version=str(payload.get("version") or "v1"),
        description=str(payload.get("description") or ""),
        assets=assets,
        source_path=path,
        repo_root=root,
        metadata=dict(payload.get("metadata") or {}),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_status(evalset: ApexEvalSet, asset: ApexEvalAsset) -> dict[str, Any]:
    path = evalset.resolve_asset_path(asset)
    if not path.is_file():
        return {
            "asset_id": asset.asset_id,
            "status": "missing_asset",
            "asset_ref": asset.asset_ref,
            "resolved_path": str(path),
            "expected_sha256": asset.sha256,
            "actual_sha256": None,
        }

    actual_sha = sha256_file(path)
    status = "ready" if actual_sha == asset.sha256 else "checksum_mismatch"
    return {
        "asset_id": asset.asset_id,
        "status": status,
        "asset_ref": asset.asset_ref,
        "resolved_path": str(path),
        "expected_sha256": asset.sha256,
        "actual_sha256": actual_sha,
        "size_bytes": int(path.stat().st_size),
    }


def visible_delta_metrics(reference: Path, candidate: Path) -> dict[str, Any]:
    """Compute deterministic visible-delta metrics for two rendered images."""
    from PIL import Image

    ref = np.asarray(Image.open(reference).convert("RGB"), dtype=np.float32) / 255.0
    cand = np.asarray(Image.open(candidate).convert("RGB"), dtype=np.float32) / 255.0
    if ref.shape != cand.shape:
        return {
            "status": "shape_mismatch",
            "reference_shape": list(ref.shape),
            "candidate_shape": list(cand.shape),
            "ssim": None,
            "lpips": None,
            "delta_e_proxy_mean_abs": None,
            "delta_e_proxy_max_abs": None,
            "metric_warnings": [],
        }

    abs_delta = np.abs(cand - ref)
    metric_warnings: list[str] = []
    try:
        ssim_value: float | None = float(ssim(cand, ref))
    except Exception:
        ssim_value = None
        metric_warnings.append("ssim_unavailable")
    if _lpips_available():
        try:
            lpips_value = float(lpips_score(cand, ref))
            if not math.isfinite(lpips_value):
                lpips_value = None
                metric_warnings.append("lpips_nonfinite")
        except Exception:
            lpips_value = None
            metric_warnings.append("lpips_unavailable")
    else:
        lpips_value = None
        metric_warnings.append("lpips_unavailable")
    return {
        "status": "partial_metrics" if metric_warnings else "ok",
        "ssim": ssim_value,
        "lpips": lpips_value,
        "delta_e_proxy_mean_abs": float(abs_delta.mean()),
        "delta_e_proxy_max_abs": float(abs_delta.max()),
        "metric_warnings": metric_warnings,
    }


def build_apex_eval_report(
    evalset_path: Path | str,
    *,
    output_dir: Path | str,
    candidate_outputs: Mapping[str, Mapping[str, Path | str]] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build and persist an APEX eval report.

    ``candidate_outputs`` maps candidate name to ``asset_id -> output path``.
    When no candidate output is provided, the report still validates corpus
    readiness and leaves visual-delta metrics unset.
    """
    evalset = load_apex_evalset(evalset_path, repo_root=repo_root)
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = evalset.repo_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    candidate_outputs = candidate_outputs or {}
    assets_report = []
    for asset in evalset.assets:
        status = asset_status(evalset, asset)
        candidates = []
        for candidate_name, outputs in sorted(candidate_outputs.items()):
            output_value = outputs.get(asset.asset_id)
            if output_value is None:
                candidates.append(
                    {
                        "candidate": candidate_name,
                        "status": "missing_candidate_output",
                        "metrics": {},
                    }
                )
                continue
            candidate_path = Path(output_value)
            if not candidate_path.is_absolute():
                candidate_path = evalset.repo_root / candidate_path
            if status["status"] != "ready":
                candidates.append(
                    {
                        "candidate": candidate_name,
                        "status": "source_not_ready",
                        "metrics": {},
                    }
                )
                continue
            if not candidate_path.is_file():
                candidates.append(
                    {
                        "candidate": candidate_name,
                        "status": "missing_candidate_output",
                        "output_path": str(candidate_path),
                        "metrics": {},
                    }
                )
                continue
            metrics = visible_delta_metrics(Path(status["resolved_path"]), candidate_path)
            candidates.append(
                {
                    "candidate": candidate_name,
                    "status": metrics.pop("status"),
                    "output_path": str(candidate_path),
                    "metrics": {
                        "depth_edge_fidelity": None,
                        "material_precision": None,
                        "pixel_op_false_positive_risk": None,
                        **metrics,
                    },
                }
            )

        assets_report.append(
            {
                **asset.to_dict(),
                "asset_status": status,
                "candidates": candidates,
            }
        )

    ready_count = sum(1 for item in assets_report if item["asset_status"]["status"] == "ready")
    report = {
        "schema_version": APEX_EVAL_REPORT_VERSION,
        "evalset": {
            "evalset_id": evalset.evalset_id,
            "version": evalset.version,
            "description": evalset.description,
            "source_path": str(evalset.source_path),
            "asset_count": len(evalset.assets),
            "ready_asset_count": ready_count,
        },
        "quality_trajectory": {
            "depth_pro_role": "research_quality_yardstick",
            "da3_metric_role": "commercial_safe_baseline",
            "pixel_op_authority": "calibrated_material_confidence_required",
            "apex_noop_policy": "fail_closed_when_masks_and_implemented_ops_apply_zero_operations",
        },
        "assets": assets_report,
    }
    report_path = output_root / "apex_eval_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        dump_json(report, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return report


def _backend_license_tier(backend: str) -> str:
    normalized = backend.strip().lower().replace("-", "_")
    if normalized in DEPTH_PRO_BACKENDS:
        return "research_only"
    spec = MODEL_REGISTRY.get(normalized)
    if spec and spec.usage_class == UsageClass.COMMERCIAL_OK:
        return "commercial_ok"
    if spec and spec.usage_class == UsageClass.NON_COMMERCIAL_ONLY:
        return "research_only"
    return "unknown"


def _depth_edge_score(depth_map: np.ndarray | None) -> float | None:
    if depth_map is None:
        return None
    arr = np.asarray(depth_map, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return None
    finite = np.isfinite(arr)
    if not finite.any():
        return 0.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    gy, gx = np.gradient(arr)
    grad = np.sqrt(gx * gx + gy * gy)
    p95 = float(np.percentile(grad, 95))
    return float(np.clip(p95, 0.0, 1.0))


def _architectural_plausibility(depth_map: np.ndarray | None) -> float | None:
    if depth_map is None:
        return None
    arr = np.asarray(depth_map, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    spread = float(np.percentile(finite, 95) - np.percentile(finite, 5))
    saturation = float(np.mean((finite <= 1e-4) | (finite >= 1.0 - 1e-4)))
    return float(np.clip(spread * (1.0 - saturation), 0.0, 1.0))


def build_depth_backend_benchmark_report(
    evalset_path: Path | str,
    *,
    backends: Sequence[str],
    quality_tier: str,
    output_dir: Path | str,
    non_commercial_ok: bool = False,
    accept_depth_pro_license: bool = False,
    runner: DepthBackendRunner | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build a governed depth backend comparison report."""
    evalset = load_apex_evalset(evalset_path, repo_root=repo_root)
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = evalset.repo_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    backend_reports: list[dict[str, Any]] = []
    for backend in backends:
        normalized_backend = backend.strip().lower().replace("-", "_")
        license_tier = _backend_license_tier(normalized_backend)
        backend_report = {
            "backend": normalized_backend,
            "license_tier": license_tier,
            "quality_tier": quality_tier,
            "status": "ready",
            "assets": [],
        }
        if normalized_backend in DEPTH_PRO_BACKENDS and not (non_commercial_ok and accept_depth_pro_license):
            backend_report["status"] = "license_blocked"
            backend_report["license_requirements"] = {
                "non_commercial_ok": bool(non_commercial_ok),
                "accept_apple_depth_pro_research_license": bool(accept_depth_pro_license),
            }
            backend_reports.append(backend_report)
            continue

        asset_scores: list[float] = []
        runtimes: list[float] = []
        for asset in evalset.assets:
            source_status = asset_status(evalset, asset)
            if source_status["status"] != "ready":
                backend_report["assets"].append(
                    {
                        "asset_id": asset.asset_id,
                        "status": source_status["status"],
                        "source": source_status,
                    }
                )
                continue
            if runner is None:
                backend_report["assets"].append(
                    {
                        "asset_id": asset.asset_id,
                        "status": "not_executed",
                        "source": source_status,
                        "metrics": {
                            "depth_edge_score": None,
                            "boundary_halo_risk": None,
                            "architectural_plausibility": None,
                            "runtime_ms": None,
                        },
                    }
                )
                continue
            started = time.perf_counter()
            run_result = runner(normalized_backend, asset, output_root, quality_tier)
            runtime_ms = (
                float(run_result.runtime_ms)
                if run_result.runtime_ms is not None
                else round((time.perf_counter() - started) * 1000.0, 3)
            )
            edge_score = _depth_edge_score(run_result.depth_map)
            plausibility = _architectural_plausibility(run_result.depth_map)
            if edge_score is not None:
                asset_scores.append(edge_score)
            runtimes.append(runtime_ms)
            backend_report["assets"].append(
                {
                    "asset_id": asset.asset_id,
                    "status": run_result.status,
                    "source": source_status,
                    "depth_path": run_result.depth_path,
                    "provenance": run_result.provenance,
                    "error": run_result.error,
                    "metrics": {
                        "depth_edge_score": edge_score,
                        "boundary_halo_risk": None if edge_score is None else float(np.clip(1.0 - edge_score, 0.0, 1.0)),
                        "architectural_plausibility": plausibility,
                        "runtime_ms": runtime_ms,
                    },
                }
            )
        if asset_scores:
            backend_report["depth_edge_score"] = float(np.mean(asset_scores))
            backend_report["boundary_halo_risk"] = float(np.clip(1.0 - backend_report["depth_edge_score"], 0.0, 1.0))
        else:
            backend_report["depth_edge_score"] = None
            backend_report["boundary_halo_risk"] = None
        backend_report["runtime_ms"] = float(np.mean(runtimes)) if runtimes else None
        backend_reports.append(backend_report)

    report = {
        "schema_version": DEPTH_BACKEND_BENCHMARK_REPORT_VERSION,
        "evalset": {
            "evalset_id": evalset.evalset_id,
            "version": evalset.version,
            "asset_count": len(evalset.assets),
            "source_path": str(evalset.source_path),
        },
        "quality_tier": quality_tier,
        "backends": backend_reports,
    }
    report_path = output_root / "depth_backend_comparison_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        dump_json(report, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return report


def parse_candidate_outputs(values: Iterable[str]) -> dict[str, dict[str, Path]]:
    """Parse ``candidate:asset_id=path`` CLI values."""
    parsed: dict[str, dict[str, Path]] = {}
    for value in values:
        candidate_part, sep, output_path = value.partition("=")
        if sep != "=":
            raise ValueError(f"Invalid candidate output {value!r}; expected candidate:asset_id=path")
        candidate, sep, asset_id = candidate_part.partition(":")
        if sep != ":" or not candidate or not asset_id:
            raise ValueError(f"Invalid candidate output {value!r}; expected candidate:asset_id=path")
        parsed.setdefault(candidate, {})[asset_id] = Path(output_path)
    return parsed
