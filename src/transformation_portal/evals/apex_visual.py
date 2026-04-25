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
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from transformation_portal.evals.apex_metrics import compute_apex_metrics
from transformation_portal.evals.asset_resolver import ResolvedAsset, resolve_manifest_path
from transformation_portal.evals.image_io import load_16bit_tiff
from transformation_portal.evals.image_metadata import inspect_reference_image, normalize_image_format
from transformation_portal.evals.metrics import lpips_score, ssim
from transformation_portal.ingest.canonical_json import dump_json
from transformation_portal.lux_depth_v3.model_registry import MODEL_REGISTRY, UsageClass

APEX_EVALSET_SCHEMA_VERSION = "apex_evalset.v1"
APEX_EVAL_REPORT_VERSION = "apex_eval_report.v1"
APEX_METRIC_CONTRACT_VERSION = "apex_metrics.v1"
DEPTH_BACKEND_BENCHMARK_REPORT_VERSION = "depth_backend_benchmark_report.v1"
DEPTH_PRO_BACKENDS = {"depth_pro"}
CANONICAL_APEX_DATASET_TIER = "canonical_apex"
DEFAULT_DATASET_TIER = "smoke_or_readiness"
CANONICAL_APEX_ASSET_ROLE = "canonical_apex_reference"
DEFAULT_ASSET_ROLE = "compatibility_fixture"
DATASET_TIERS = frozenset(
    {
        CANONICAL_APEX_DATASET_TIER,
        DEFAULT_DATASET_TIER,
        "delivery_preview",
        "synthetic_smoke",
        "compatibility_fixture",
    }
)
ASSET_ROLES = frozenset(
    {
        CANONICAL_APEX_ASSET_ROLE,
        "delivery_preview",
        "synthetic_smoke",
        "compatibility_fixture",
    }
)
CANONICAL_REFERENCE_FORMATS = frozenset({"tif", "tiff"})
PROVENANCE_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _lpips_available() -> bool:
    """Return whether LPIPS can produce authoritative visible-delta evidence."""
    try:
        importlib.import_module("lpips")
        importlib.import_module("torch")
    except ImportError:
        return False
    return True


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return _bool_with_default(value, False)


def _bool_with_default(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _normalize_image_format(value: Any) -> str | None:
    return normalize_image_format(value)


def _normalize_source_raw_format(value: Any) -> str | None:
    normalized = _normalize_optional_str(value)
    if normalized is None:
        return None
    return normalized.lower().lstrip(".")


def _optional_provenance_sha(value: Any, *, field_name: str) -> str | None:
    normalized = _normalize_optional_str(value)
    if normalized is None:
        return None
    if not PROVENANCE_SHA_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a lowercase 64-character hexadecimal SHA-256 digest")
    return normalized


def _normalized_format(asset: "ApexEvalAsset") -> str | None:
    explicit = _normalize_image_format(asset.canonical_format)
    if explicit:
        return explicit
    path_hint = asset.reference_path or asset.asset_ref
    return _normalize_image_format(Path(path_hint).suffix)


def _reference_bit_depth(asset: "ApexEvalAsset") -> int | None:
    if asset.canonical_bit_depth is not None:
        return int(asset.canonical_bit_depth)
    if _normalized_format(asset) == "jpeg":
        return 8
    return None


def _bit_depth_from_dtype(dtype: Any) -> int | None:
    try:
        np_dtype = np.dtype(dtype)
    except TypeError:
        return None
    if np_dtype.kind == "b":
        return 1
    if np_dtype.kind in {"u", "i", "f"}:
        return int(np_dtype.itemsize * 8)
    return None


def _bit_depth_from_pil_mode(mode: str | None) -> int | None:
    if mode in {"1"}:
        return 1
    if mode in {"L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr"}:
        return 8
    if mode in {"I;16", "I;16B", "I;16L"}:
        return 16
    if mode in {"I", "F"}:
        return 32
    return None


def _tiff_bit_depth(path: Path) -> int | None:
    try:
        tifffile = importlib.import_module("tifffile")
        with tifffile.TiffFile(path) as tiff:
            if not tiff.pages:
                return None
            page = tiff.pages[0]
            dtype_bits = _bit_depth_from_dtype(getattr(page, "dtype", None))
            if dtype_bits is not None:
                return dtype_bits
            bits_per_sample = getattr(page, "bitspersample", None)
            if isinstance(bits_per_sample, (tuple, list)):
                return int(max(bits_per_sample)) if bits_per_sample else None
            if bits_per_sample is not None:
                return int(bits_per_sample)
    except (ImportError, OSError, ValueError, TypeError, AttributeError, IndexError):
        return None
    return None


def _reference_image_metadata(path: Path) -> dict[str, Any]:
    return inspect_reference_image(path)


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
    asset_role: str = DEFAULT_ASSET_ROLE
    reference_path: str | None = None
    delivery_path: str | None = None
    canonical_bit_depth: int | None = None
    canonical_format: str | None = None
    canonical_color_space: str | None = None
    evaluate_at_native_resolution: bool = False
    allow_downsampled_model_inference: bool = True
    preserve_16bit_intermediates: bool = False
    canonical_scoring_eligible: bool | None = None
    canonical_scoring_blocked_reason: str | None = None
    manual_quality_score: float | None = None
    source_raw_path: str | None = None
    source_raw_format: str | None = None
    source_raw_sha256: str | None = None
    raw_development_profile: str | None = None
    raw_development_settings_sha256: str | None = None
    canonical_icc_profile_name: str | None = None
    canonical_icc_profile_sha256: str | None = None
    working_color_space: str | None = None
    working_transfer_function: str | None = None
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

        asset_role = str(payload.get("asset_role") or DEFAULT_ASSET_ROLE)
        if asset_role not in ASSET_ROLES:
            raise ValueError(f"Unsupported APEX asset_role {asset_role!r}")

        canonical_bit_depth = _optional_int(payload.get("canonical_bit_depth"))
        canonical_format = _normalize_optional_str(payload.get("canonical_format"))
        reference_path = _normalize_optional_str(payload.get("reference_path")) or str(payload["asset_ref"])
        delivery_path = _normalize_optional_str(payload.get("delivery_path"))
        canonical_eligible = _optional_bool(payload.get("canonical_scoring_eligible"))

        return cls(
            asset_id=str(payload["asset_id"]),
            asset_ref=str(payload["asset_ref"]),
            sha256=str(payload["sha256"]).lower(),
            scene_type=str(payload["scene_type"]),
            expected_materials=tuple(str(item) for item in payload.get("expected_materials", [])),
            risk_zones=tuple(str(item) for item in payload.get("risk_zones", [])),
            reject_if=tuple(str(item) for item in payload.get("reject_if", [])),
            asset_role=asset_role,
            reference_path=reference_path,
            delivery_path=delivery_path,
            canonical_bit_depth=canonical_bit_depth,
            canonical_format=canonical_format,
            canonical_color_space=_normalize_optional_str(payload.get("canonical_color_space")),
            evaluate_at_native_resolution=_bool_with_default(payload.get("evaluate_at_native_resolution"), False),
            allow_downsampled_model_inference=_bool_with_default(payload.get("allow_downsampled_model_inference"), True),
            preserve_16bit_intermediates=_bool_with_default(payload.get("preserve_16bit_intermediates"), False),
            canonical_scoring_eligible=canonical_eligible,
            canonical_scoring_blocked_reason=_normalize_optional_str(payload.get("canonical_scoring_blocked_reason")),
            manual_quality_score=manual_score,
            source_raw_path=_normalize_optional_str(payload.get("source_raw_path")),
            source_raw_format=_normalize_source_raw_format(payload.get("source_raw_format")),
            source_raw_sha256=_optional_provenance_sha(
                payload.get("source_raw_sha256"),
                field_name="source_raw_sha256",
            ),
            raw_development_profile=_normalize_optional_str(payload.get("raw_development_profile")),
            raw_development_settings_sha256=_optional_provenance_sha(
                payload.get("raw_development_settings_sha256"),
                field_name="raw_development_settings_sha256",
            ),
            canonical_icc_profile_name=_normalize_optional_str(payload.get("canonical_icc_profile_name")),
            canonical_icc_profile_sha256=_optional_provenance_sha(
                payload.get("canonical_icc_profile_sha256"),
                field_name="canonical_icc_profile_sha256",
            ),
            working_color_space=_normalize_optional_str(payload.get("working_color_space")),
            working_transfer_function=_normalize_optional_str(payload.get("working_transfer_function")),
            notes=(str(payload["notes"]) if payload.get("notes") is not None else None),
        )

    def provenance_dict(self) -> dict[str, Any]:
        values = {
            "source_raw_path": self.source_raw_path,
            "source_raw_format": self.source_raw_format,
            "source_raw_sha256": self.source_raw_sha256,
            "raw_development_profile": self.raw_development_profile,
            "raw_development_settings_sha256": self.raw_development_settings_sha256,
            "canonical_icc_profile_name": self.canonical_icc_profile_name,
            "canonical_icc_profile_sha256": self.canonical_icc_profile_sha256,
            "working_color_space": self.working_color_space,
            "working_transfer_function": self.working_transfer_function,
        }
        return {key: value for key, value in values.items() if value is not None}

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "asset_id": self.asset_id,
            "asset_ref": self.asset_ref,
            "sha256": self.sha256,
            "scene_type": self.scene_type,
            "expected_materials": list(self.expected_materials),
            "risk_zones": list(self.risk_zones),
            "reject_if": list(self.reject_if),
            "asset_role": self.asset_role,
            "reference_path": self.reference_path,
            "delivery_path": self.delivery_path,
            "canonical_bit_depth": self.canonical_bit_depth,
            "canonical_format": self.canonical_format,
            "canonical_color_space": self.canonical_color_space,
            "evaluate_at_native_resolution": self.evaluate_at_native_resolution,
            "allow_downsampled_model_inference": self.allow_downsampled_model_inference,
            "preserve_16bit_intermediates": self.preserve_16bit_intermediates,
            "canonical_scoring_eligible": self.canonical_scoring_eligible,
            "canonical_scoring_blocked_reason": self.canonical_scoring_blocked_reason,
            "manual_quality_score": self.manual_quality_score,
            "notes": self.notes,
        }
        payload.update(self.provenance_dict())
        return payload


@dataclass(frozen=True)
class ApexEvalSet:
    """Loaded APEX visual evalset."""

    evalset_id: str
    version: str
    description: str
    assets: tuple[ApexEvalAsset, ...]
    source_path: Path
    repo_root: Path
    asset_root: Path | None = None
    dataset_tier: str = DEFAULT_DATASET_TIER
    canonical_bit_depth: int | None = None
    canonical_format: str | None = None
    canonical_color_space: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_asset(self, asset: ApexEvalAsset) -> ResolvedAsset:
        return resolve_manifest_path(asset.asset_ref, repo_root=self.repo_root, asset_root=self.asset_root)

    def resolve_reference(self, asset: ApexEvalAsset) -> ResolvedAsset:
        return resolve_manifest_path(
            asset.reference_path or asset.asset_ref, repo_root=self.repo_root, asset_root=self.asset_root
        )

    def resolve_asset_path(self, asset: ApexEvalAsset) -> Path:
        resolution = self.resolve_asset(asset)
        if resolution.escaped_asset_root:
            raise ValueError(f"APEX asset_ref escapes asset root: {asset.asset_ref}")
        if resolution.resolved_path is None:
            raise ValueError(f"APEX asset_ref did not resolve to a filesystem path: {asset.asset_ref}")
        return resolution.resolved_path

    def resolve_reference_path(self, asset: ApexEvalAsset) -> Path:
        reported = asset.reference_path or asset.asset_ref
        resolution = self.resolve_reference(asset)
        if resolution.escaped_asset_root:
            raise ValueError(f"APEX reference_path escapes asset root: {reported}")
        if resolution.resolved_path is None:
            raise ValueError(f"APEX reference_path did not resolve to a filesystem path: {reported}")
        return resolution.resolved_path


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


def load_apex_evalset(
    evalset_path: Path | str,
    *,
    repo_root: Path | None = None,
    asset_root: Path | str | None = None,
) -> ApexEvalSet:
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
    dataset_tier = str(payload.get("dataset_tier") or DEFAULT_DATASET_TIER)
    if dataset_tier not in DATASET_TIERS:
        raise ValueError(f"Unsupported APEX dataset_tier {dataset_tier!r}")
    return ApexEvalSet(
        evalset_id=str(payload.get("evalset_id") or path.parent.name),
        version=str(payload.get("version") or "v1"),
        description=str(payload.get("description") or ""),
        assets=assets,
        source_path=path,
        repo_root=root,
        asset_root=Path(asset_root).expanduser() if asset_root is not None and str(asset_root).strip() else None,
        dataset_tier=dataset_tier,
        canonical_bit_depth=_optional_int(payload.get("canonical_bit_depth")),
        canonical_format=_normalize_optional_str(payload.get("canonical_format")),
        canonical_color_space=_normalize_optional_str(payload.get("canonical_color_space")),
        metadata=dict(payload.get("metadata") or {}),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_scoring_status(
    evalset: ApexEvalSet,
    asset: ApexEvalAsset,
    *,
    asset_ready: bool,
    reference_metadata: Mapping[str, Any] | None = None,
    path_blocked_reason: str | None = None,
    path_field: str | None = None,
) -> dict[str, Any]:
    reference_metadata = reference_metadata or {}
    declared_format = _normalized_format(asset)
    declared_bit_depth = _reference_bit_depth(asset)
    detected_format = _normalize_image_format(reference_metadata.get("detected_reference_format"))
    detected_bit_depth = _optional_int(reference_metadata.get("detected_reference_bit_depth"))
    detected_dimensions = reference_metadata.get("detected_reference_dimensions")
    detected_mode = reference_metadata.get("detected_reference_mode")
    detected_channel_count = reference_metadata.get("detected_reference_channel_count")
    detected_icc_profile_name = reference_metadata.get("detected_reference_icc_profile_name")
    detected_icc_profile_sha256 = reference_metadata.get("detected_reference_icc_profile_sha256")
    metadata_status = str(reference_metadata.get("observable_reference_metadata_status") or "not_checked")
    metadata_error = _normalize_optional_str(reference_metadata.get("observable_reference_metadata_error"))
    reference_format = detected_format or declared_format
    reference_bit_depth = detected_bit_depth if detected_bit_depth is not None else declared_bit_depth
    eligible = True
    blocked_reason: str | None = None

    if path_blocked_reason:
        eligible = False
        blocked_reason = path_blocked_reason
    elif not asset_ready:
        eligible = False
        blocked_reason = "asset_not_ready"
    elif evalset.dataset_tier != CANONICAL_APEX_DATASET_TIER:
        eligible = False
        blocked_reason = "noncanonical_dataset_tier"
    elif asset.asset_role != CANONICAL_APEX_ASSET_ROLE:
        eligible = False
        blocked_reason = "asset_role_not_canonical"
    elif metadata_status == "missing_reference_asset":
        eligible = False
        blocked_reason = "missing_reference_asset"
    elif metadata_status != "ok":
        eligible = False
        blocked_reason = "missing_observable_reference_metadata"
    elif declared_format is not None and reference_format != declared_format:
        eligible = False
        blocked_reason = "reference_format_mismatch"
    elif reference_bit_depth is None:
        eligible = False
        blocked_reason = "missing_16bit_reference"
    elif reference_bit_depth < 16:
        eligible = False
        blocked_reason = "reference_bit_depth_below_16"
    elif reference_format not in CANONICAL_REFERENCE_FORMATS:
        eligible = False
        blocked_reason = "non_tiff_reference"
    elif not asset.evaluate_at_native_resolution:
        eligible = False
        blocked_reason = "native_resolution_evaluation_required"
    elif not asset.preserve_16bit_intermediates:
        eligible = False
        blocked_reason = "preserve_16bit_intermediates_required"
    elif asset.canonical_scoring_eligible is False:
        eligible = False
        blocked_reason = asset.canonical_scoring_blocked_reason or "canonical_scoring_disabled"

    if not eligible and asset.canonical_scoring_blocked_reason and not path_blocked_reason:
        blocked_reason = asset.canonical_scoring_blocked_reason

    return {
        "asset_role": asset.asset_role,
        "reference_path": asset.reference_path or asset.asset_ref,
        "delivery_path": asset.delivery_path,
        "reference_bit_depth": reference_bit_depth,
        "reference_format": reference_format,
        "declared_reference_bit_depth": declared_bit_depth,
        "declared_reference_format": declared_format,
        "detected_reference_bit_depth": detected_bit_depth,
        "detected_reference_format": detected_format,
        "detected_reference_dimensions": detected_dimensions,
        "detected_reference_mode": detected_mode,
        "detected_reference_channel_count": detected_channel_count,
        "detected_reference_icc_profile_name": detected_icc_profile_name,
        "detected_reference_icc_profile_sha256": detected_icc_profile_sha256,
        "observable_reference_metadata_status": metadata_status,
        "observable_reference_metadata_error": metadata_error,
        "reference_color_space": asset.canonical_color_space,
        "evaluate_at_native_resolution": asset.evaluate_at_native_resolution,
        "allow_downsampled_model_inference": asset.allow_downsampled_model_inference,
        "preserve_16bit_intermediates": asset.preserve_16bit_intermediates,
        "canonical_scoring_eligible": eligible,
        "canonical_scoring_blocked_reason": None if eligible else blocked_reason,
        "path_field": path_field if path_blocked_reason else None,
    }


def canonical_scoring_status(evalset: ApexEvalSet, asset: ApexEvalAsset) -> dict[str, Any]:
    """Return canonical APEX quality scoring eligibility for an eval asset."""
    asset_resolution = evalset.resolve_asset(asset)
    reference_resolution = evalset.resolve_reference(asset)
    path = asset_resolution.resolved_path
    reference_path = reference_resolution.resolved_path
    path_blocked_reason = None
    path_field = None
    if asset_resolution.escaped_asset_root:
        path_blocked_reason = "path_escapes_asset_root"
        path_field = "asset_ref"
    elif reference_resolution.escaped_asset_root:
        path_blocked_reason = "path_escapes_asset_root"
        path_field = "reference_path"
    reference_metadata = _reference_image_metadata(reference_path) if path is not None and path.is_file() else None
    return _canonical_scoring_status(
        evalset,
        asset,
        asset_ready=path is not None and path.is_file() and sha256_file(path) == asset.sha256,
        reference_metadata=reference_metadata,
        path_blocked_reason=path_blocked_reason,
        path_field=path_field,
    )


def asset_status(evalset: ApexEvalSet, asset: ApexEvalAsset) -> dict[str, Any]:
    asset_resolution = evalset.resolve_asset(asset)
    reference_resolution = evalset.resolve_reference(asset)
    path = asset_resolution.resolved_path
    reference_path = reference_resolution.resolved_path
    path_blocked_reason = None
    path_field = None
    if asset_resolution.escaped_asset_root:
        path_blocked_reason = "path_escapes_asset_root"
        path_field = "asset_ref"
    elif reference_resolution.escaped_asset_root:
        path_blocked_reason = "path_escapes_asset_root"
        path_field = "reference_path"

    if path is None or not path.is_file() or path_blocked_reason:
        return {
            "asset_id": asset.asset_id,
            "status": "missing_asset",
            "asset_ref": asset.asset_ref,
            "asset_resolution": asset_resolution.to_report_dict(),
            "reference_resolution": reference_resolution.to_report_dict(),
            "expected_sha256": asset.sha256,
            "actual_sha256": None,
            **_canonical_scoring_status(
                evalset,
                asset,
                asset_ready=False,
                path_blocked_reason=path_blocked_reason,
                path_field=path_field,
            ),
        }

    actual_sha = sha256_file(path)
    status = "ready" if actual_sha == asset.sha256 else "checksum_mismatch"
    reference_metadata = _reference_image_metadata(reference_path)
    return {
        "asset_id": asset.asset_id,
        "status": status,
        "asset_ref": asset.asset_ref,
        "asset_resolution": asset_resolution.to_report_dict(),
        "reference_resolution": reference_resolution.to_report_dict(),
        "expected_sha256": asset.sha256,
        "actual_sha256": actual_sha,
        "size_bytes": int(path.stat().st_size),
        **_canonical_scoring_status(
            evalset,
            asset,
            asset_ready=status == "ready",
            reference_metadata=reference_metadata,
        ),
    }


def _visible_delta_unavailable(status: str, **extra: Any) -> dict[str, Any]:
    return {
        "status": status,
        "ssim": None,
        "lpips": None,
        "delta_e_proxy_mean_abs": None,
        "delta_e_proxy_max_abs": None,
        "metric_warnings": [],
        **extra,
    }


def _load_rgb_image(path: Path, *, role: str) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return np.asarray(rgb, dtype=np.float32) / 255.0, None
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        return None, _visible_delta_unavailable(
            "unreadable_image",
            unreadable_role=role,
            unreadable_path=str(path),
            error=str(exc),
        )


def visible_delta_metrics(reference: Path, candidate: Path) -> dict[str, Any]:
    """Compute deterministic visible-delta metrics for two rendered images."""
    ref, ref_error = _load_rgb_image(reference, role="reference")
    if ref_error is not None:
        return ref_error
    cand, cand_error = _load_rgb_image(candidate, role="candidate")
    if cand_error is not None:
        return cand_error

    assert ref is not None
    assert cand is not None
    if ref.shape != cand.shape:
        return _visible_delta_unavailable(
            "shape_mismatch",
            reference_shape=list(ref.shape),
            candidate_shape=list(cand.shape),
            status_aliases=["invalid_candidate_dimensions"],
        )

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


def _candidate_report_base(
    *,
    candidate_name: str,
    status: str,
    candidate_path: Path | None,
    reference_resolution: ResolvedAsset,
    metrics: Mapping[str, Any] | None = None,
    metrics_authoritative: bool = False,
    reason: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidate": candidate_name,
        "status": status,
        "metric_contract": APEX_METRIC_CONTRACT_VERSION,
        "metrics_authoritative": bool(metrics_authoritative),
        "evaluation_target_path": reference_resolution.reported_path,
        "reference_resolution": reference_resolution.to_report_dict(),
        "candidate_output": {
            "path": str(candidate_path) if candidate_path is not None else None,
        },
        "metrics": dict(metrics or {}),
    }
    if candidate_path is not None:
        payload["output_path"] = str(candidate_path)
    if reason:
        payload["reason"] = reason
    if extra:
        payload.update(dict(extra))
    return payload


def _read_image_array(path: Path, *, require_16bit_tiff: bool, role: str) -> tuple[np.ndarray | None, str | None, str | None]:
    if require_16bit_tiff:
        array, metadata = load_16bit_tiff(path)
        if array is not None:
            return array, None, None
        reason = str(metadata.get("reason") or metadata.get("observable_reference_metadata_status") or "invalid_input")
        if reason in {"non_tiff_reference", "reference_bit_depth_below_16", "loaded_bit_depth_below_16"}:
            return None, f"unsupported_{role}_bit_depth", reason
        return None, f"unreadable_{role}", reason

    from PIL import Image, UnidentifiedImageError

    try:
        if _normalize_image_format(path.suffix) == "tiff":
            tifffile = importlib.import_module("tifffile")
            return np.asarray(tifffile.imread(path)), None, None
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB")), None, None
    except (ImportError, OSError, ValueError, TypeError, UnidentifiedImageError) as exc:
        return None, f"unreadable_{role}", str(exc)


def _candidate_metrics_report(
    *,
    evalset: ApexEvalSet,
    asset: ApexEvalAsset,
    asset_report_status: Mapping[str, Any],
    candidate_name: str,
    output_value: Path | str | None,
) -> dict[str, Any]:
    reference_resolution = evalset.resolve_reference(asset)
    candidate_path = Path(output_value) if output_value is not None else None
    if candidate_path is not None and not candidate_path.is_absolute():
        candidate_path = evalset.repo_root / candidate_path

    if candidate_path is None or not candidate_path.is_file():
        return _candidate_report_base(
            candidate_name=candidate_name,
            status="missing_candidate",
            candidate_path=candidate_path,
            reference_resolution=reference_resolution,
            reason="candidate_output_missing",
        )

    if asset_report_status.get("status") != "ready":
        return _candidate_report_base(
            candidate_name=candidate_name,
            status="metrics_not_computed",
            candidate_path=candidate_path,
            reference_resolution=reference_resolution,
            reason=f"asset_status_{asset_report_status.get('status')}",
        )

    reference_path = reference_resolution.resolved_path
    if reference_path is None or reference_resolution.escaped_asset_root or not reference_path.is_file():
        return _candidate_report_base(
            candidate_name=candidate_name,
            status="missing_reference",
            candidate_path=candidate_path,
            reference_resolution=reference_resolution,
            reason="reference_path_missing_or_unresolved",
        )

    canonical_comparison = (
        bool(asset_report_status.get("canonical_scoring_eligible"))
        or evalset.dataset_tier == CANONICAL_APEX_DATASET_TIER
        or asset.asset_role == CANONICAL_APEX_ASSET_ROLE
    )
    reference, reference_status, reference_reason = _read_image_array(
        reference_path,
        require_16bit_tiff=canonical_comparison,
        role="reference",
    )
    if reference_status:
        return _candidate_report_base(
            candidate_name=candidate_name,
            status=reference_status,
            candidate_path=candidate_path,
            reference_resolution=reference_resolution,
            reason=reference_reason,
        )

    candidate, candidate_status, candidate_reason = _read_image_array(
        candidate_path,
        require_16bit_tiff=canonical_comparison,
        role="candidate",
    )
    if candidate_status:
        return _candidate_report_base(
            candidate_name=candidate_name,
            status=candidate_status,
            candidate_path=candidate_path,
            reference_resolution=reference_resolution,
            reason=candidate_reason,
        )

    assert reference is not None
    assert candidate is not None
    metrics = compute_apex_metrics(
        reference,
        candidate,
        working_color_space=asset.working_color_space,
        working_transfer_function=asset.working_transfer_function,
    )
    visible_status = str(metrics.get("visible_delta", {}).get("status") or "")
    if visible_status == "dimension_mismatch":
        return _candidate_report_base(
            candidate_name=candidate_name,
            status="dimension_mismatch",
            candidate_path=candidate_path,
            reference_resolution=reference_resolution,
            metrics=metrics,
            reason="dimension_mismatch",
        )
    if visible_status != "ok":
        return _candidate_report_base(
            candidate_name=candidate_name,
            status="metrics_not_computed",
            candidate_path=candidate_path,
            reference_resolution=reference_resolution,
            metrics=metrics,
            reason=visible_status or "metrics_not_computed",
        )
    return _candidate_report_base(
        candidate_name=candidate_name,
        status="ok",
        candidate_path=candidate_path,
        reference_resolution=reference_resolution,
        metrics=metrics,
        metrics_authoritative=True,
    )


def build_apex_eval_report(
    evalset_path: Path | str,
    *,
    output_dir: Path | str,
    candidate_outputs: Mapping[str, Mapping[str, Path | str]] | None = None,
    repo_root: Path | None = None,
    asset_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build and persist an APEX eval report.

    ``candidate_outputs`` maps candidate name to ``asset_id -> output path``.
    When no candidate output is provided, the report still validates corpus
    readiness and leaves candidate metrics unset.
    """
    evalset = load_apex_evalset(evalset_path, repo_root=repo_root, asset_root=asset_root)
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = evalset.repo_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    candidate_outputs = candidate_outputs or {}
    assets_report = []
    for asset in evalset.assets:
        status = asset_status(evalset, asset)
        canonical_fields = {
            "asset_role": status["asset_role"],
            "reference_path": status["reference_path"],
            "delivery_path": status["delivery_path"],
            "reference_bit_depth": status["reference_bit_depth"],
            "reference_format": status["reference_format"],
            "declared_reference_bit_depth": status["declared_reference_bit_depth"],
            "declared_reference_format": status["declared_reference_format"],
            "detected_reference_bit_depth": status["detected_reference_bit_depth"],
            "detected_reference_format": status["detected_reference_format"],
            "detected_reference_dimensions": status["detected_reference_dimensions"],
            "detected_reference_mode": status["detected_reference_mode"],
            "detected_reference_channel_count": status["detected_reference_channel_count"],
            "detected_reference_icc_profile_name": status["detected_reference_icc_profile_name"],
            "detected_reference_icc_profile_sha256": status["detected_reference_icc_profile_sha256"],
            "observable_reference_metadata_status": status["observable_reference_metadata_status"],
            "observable_reference_metadata_error": status["observable_reference_metadata_error"],
            "reference_color_space": status["reference_color_space"],
            "evaluate_at_native_resolution": status["evaluate_at_native_resolution"],
            "allow_downsampled_model_inference": status["allow_downsampled_model_inference"],
            "preserve_16bit_intermediates": status["preserve_16bit_intermediates"],
            "canonical_scoring_eligible": status["canonical_scoring_eligible"],
            "canonical_scoring_blocked_reason": status["canonical_scoring_blocked_reason"],
            "path_field": status["path_field"],
            "asset_resolution": status["asset_resolution"],
            "reference_resolution": status["reference_resolution"],
        }
        candidates = []
        for candidate_name, outputs in sorted(candidate_outputs.items()):
            candidates.append(
                _candidate_metrics_report(
                    evalset=evalset,
                    asset=asset,
                    asset_report_status=status,
                    candidate_name=candidate_name,
                    output_value=outputs.get(asset.asset_id),
                )
            )

        assets_report.append(
            {
                **asset.to_dict(),
                **canonical_fields,
                "asset_status": status,
                "candidates": candidates,
            }
        )

    ready_count = sum(1 for item in assets_report if item["asset_status"]["status"] == "ready")
    canonical_count = sum(
        1 for item in assets_report if item["asset_status"]["status"] == "ready" and item["canonical_scoring_eligible"]
    )
    missing_count = sum(1 for item in assets_report if item["asset_status"]["status"] == "missing_asset")
    noncanonical_count = sum(
        1 for item in assets_report if item["asset_status"]["status"] == "ready" and not item["canonical_scoring_eligible"]
    )
    blocked_reason_counts: dict[str, int] = {}
    for item in assets_report:
        reason = item["canonical_scoring_blocked_reason"]
        if item["asset_status"]["status"] == "ready" and reason:
            blocked_reason_counts[reason] = blocked_reason_counts.get(reason, 0) + 1
    report_path = output_root / "apex_eval_report.json"
    report = {
        "schema_version": APEX_EVAL_REPORT_VERSION,
        "report_path": str(report_path),
        "evalset": {
            "evalset_id": evalset.evalset_id,
            "version": evalset.version,
            "description": evalset.description,
            "source_path": str(evalset.source_path),
            "repo_root": str(evalset.repo_root),
            "dataset_tier": evalset.dataset_tier,
            "canonical_bit_depth": evalset.canonical_bit_depth,
            "canonical_format": evalset.canonical_format,
            "canonical_color_space": evalset.canonical_color_space,
            "asset_count": len(evalset.assets),
            "ready_asset_count": ready_count,
            "canonical_scoring_eligible_count": canonical_count,
            "missing_asset_count": missing_count,
            "noncanonical_asset_count": noncanonical_count,
            "canonical_scoring_blocked_reason_counts": blocked_reason_counts,
        },
        "quality_trajectory": {
            "depth_pro_role": "research_quality_yardstick",
            "da3_metric_role": "commercial_safe_baseline",
            "pixel_op_authority": "calibrated_material_confidence_required",
            "apex_noop_policy": "fail_closed_when_masks_and_implemented_ops_apply_zero_operations",
        },
        "assets": assets_report,
    }
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


def _depth_case_io_metadata(
    evalset: ApexEvalSet,
    asset: ApexEvalAsset,
    source_status: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_path = asset.reference_path or asset.asset_ref
    reference_resolution = evalset.resolve_reference(asset)
    reference_bit_depth = (
        _optional_int(source_status.get("reference_bit_depth")) if source_status is not None else _reference_bit_depth(asset)
    )
    reference_format = (
        _normalize_image_format(source_status.get("reference_format"))
        if source_status is not None
        else _normalized_format(asset)
    )
    reference_dimensions = source_status.get("detected_reference_dimensions") if source_status is not None else None
    downsampled = bool(asset.allow_downsampled_model_inference)
    model_input = {
        "derived_from": reference_path,
        "input_bit_depth": 8 if downsampled else reference_bit_depth,
        "input_color_space": "srgb" if downsampled else asset.canonical_color_space,
        "input_resolution": None,
        "reference_resolution_dimensions": reference_dimensions,
        "downsampled_for_inference": downsampled,
        "reference_resolution": reference_resolution.to_report_dict(),
    }
    evaluation_target = {
        "path": reference_path,
        "bit_depth": reference_bit_depth,
        "format": reference_format,
        "dimensions": reference_dimensions,
        "color_space": asset.canonical_color_space,
        "evaluate_at_native_resolution": asset.evaluate_at_native_resolution,
        "reference_resolution": reference_resolution.to_report_dict(),
    }
    evaluation_target.update(asset.provenance_dict())
    return model_input, evaluation_target


def _merge_mapping(base: dict[str, Any], override: Any) -> dict[str, Any]:
    if isinstance(override, Mapping):
        return {**base, **dict(override)}
    return base


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
    asset_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build a governed depth backend comparison report."""
    evalset = load_apex_evalset(evalset_path, repo_root=repo_root, asset_root=asset_root)
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
            model_input, evaluation_target = _depth_case_io_metadata(evalset, asset, source_status)
            if source_status["status"] != "ready":
                backend_report["assets"].append(
                    {
                        "asset_id": asset.asset_id,
                        "status": source_status["status"],
                        "source": source_status,
                        "model_input": model_input,
                        "evaluation_target": evaluation_target,
                    }
                )
                continue
            if runner is None:
                backend_report["assets"].append(
                    {
                        "asset_id": asset.asset_id,
                        "status": "not_executed",
                        "source": source_status,
                        "model_input": model_input,
                        "evaluation_target": evaluation_target,
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
            provenance = dict(run_result.provenance)
            backend_report["assets"].append(
                {
                    "asset_id": asset.asset_id,
                    "status": run_result.status,
                    "source": source_status,
                    "depth_path": run_result.depth_path,
                    "model_input": _merge_mapping(model_input, provenance.get("model_input")),
                    "evaluation_target": _merge_mapping(evaluation_target, provenance.get("evaluation_target")),
                    "provenance": provenance,
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

    report_path = output_root / "depth_backend_comparison_report.json"
    report = {
        "schema_version": DEPTH_BACKEND_BENCHMARK_REPORT_VERSION,
        "report_path": str(report_path),
        "evalset": {
            "evalset_id": evalset.evalset_id,
            "version": evalset.version,
            "asset_count": len(evalset.assets),
            "source_path": str(evalset.source_path),
        },
        "quality_tier": quality_tier,
        "backends": backend_reports,
    }
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
