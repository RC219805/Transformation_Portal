"""Portal job artifact cataloging and response metadata helpers.

This app-independent module validates, indexes, and serializes job artifacts
while preserving the legacy app.py payload contract through injected runtime
limits and compatibility wrappers.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple
from urllib.parse import quote

from transformation_portal.lux_depth_v3.run_card_contract import infer_run_card_version

MAX_INDEXED_ARTIFACTS = 200
ARTIFACT_FINGERPRINT_MAX_BYTES = 8 * 1024 * 1024
_ARTIFACT_FINGERPRINT_CHUNK_BYTES = 1024 * 1024
JOB_RUN_SUMMARY_MAX_BYTES = 1024 * 1024


@dataclass(frozen=True)
class JobRunMetadata:
    output_dir: Path
    run_card_path: Optional[Path] = None
    run_card_payload: Optional[Dict[str, Any]] = None
    batch_manifest_path: Optional[Path] = None
    batch_manifest_payload: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class JobArtifactIndexResult:
    items: List[Dict[str, Any]]
    artifact_lookup: Dict[str, Path]
    artifacts: Dict[str, Any]


class ArtifactPathValidationError(ValueError):
    """Base class for bounded artifact-path validation failures."""


class InvalidArtifactPathError(ArtifactPathValidationError):
    """Artifact path is empty or malformed."""


class AbsoluteArtifactPathError(ArtifactPathValidationError):
    """Artifact path attempted to use an absolute path."""


class ArtifactPathOutsideJobOutputDirError(ArtifactPathValidationError):
    """Artifact path attempted to escape the job output directory."""


def _infer_artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {
        ".json",
        ".yaml",
        ".yml",
        ".txt",
        ".md",
        ".log",
        ".csv",
    }:
        return "metadata"
    if suffix in {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".webp",
        ".gif",
        ".avif",
        ".exr",
    }:
        return "image"
    if suffix in {".zip", ".tar", ".gz", ".tgz", ".bag"}:
        return "archive"
    return "file"


def _artifact_content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _artifact_media_kind(path: Path) -> str:
    artifact_type = _infer_artifact_type(path)
    if artifact_type == "image":
        return "image"
    if artifact_type == "metadata":
        return "metadata"
    if artifact_type == "archive":
        return "archive"
    return "file"


def _artifact_is_previewable(path: Path) -> bool:
    return _artifact_media_kind(path) == "image" and _artifact_content_type(path).startswith("image/")


# MIME types browsers reliably render via <img> / <picture>. TIFF, EXR, and
# similar formats are excluded so the portal never asks the browser to decode
# them; previewing those goes through a sibling PNG proxy when available.
_BROWSER_PREVIEWABLE_MIME_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "image/avif",
    }
)


def _artifact_is_browser_previewable(path: Path) -> bool:
    return _artifact_content_type(path).lower() in _BROWSER_PREVIEWABLE_MIME_TYPES


def _relative_artifact_path(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _artifact_preview_proxy_path(path: Path) -> Optional[Path]:
    """Return a sibling PNG proxy for browser-unfriendly image artifacts."""

    if _artifact_is_browser_previewable(path):
        return None
    if _artifact_media_kind(path) != "image":
        return None
    candidate = path.with_name(path.name + ".preview.png")
    try:
        if candidate.is_file():
            return candidate
    except OSError:
        return None
    return None


def _add_artifact_preview_proxy_lookup(
    lookup: Dict[str, Path],
    *,
    output_dir: Path,
    artifact_path: Path,
) -> None:
    proxy_path = _artifact_preview_proxy_path(artifact_path)
    if proxy_path is None:
        return
    try:
        resolved_output_dir = Path(os.path.realpath(output_dir.expanduser()))
        resolved_proxy_path = Path(os.path.realpath(proxy_path))
        proxy_relative_path = _relative_artifact_path(resolved_proxy_path, resolved_output_dir)
    except (OSError, ValueError):
        return
    lookup.setdefault(proxy_relative_path, resolved_proxy_path)


def _safe_artifact_attachment_filename(path: Path) -> str:
    """Return an ASCII-safe filename for Content-Disposition attachments."""

    candidate = re.sub(r"[^A-Za-z0-9._-]", "_", path.name or "")
    return candidate or "download"


def _artifact_response_headers(path: Path) -> Dict[str, str]:
    """Build response headers for a job artifact download."""

    headers: Dict[str, str] = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }
    if not _artifact_is_previewable(path):
        filename = _safe_artifact_attachment_filename(path)
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return headers


def _artifact_display_label(role: str) -> str:
    return {
        "primary_preview": "Primary Preview",
        "review_preview": "Review Preview",
        "supporting_preview": "Supporting Preview",
        "run_card": "Run Card",
        "report": "Report",
        "manifest": "Manifest",
        "vlm_caption": "Advisory Caption",
        "archive": "Archive",
        "log": "Log",
        "metadata": "Metadata",
    }.get(role, "File")


_STEM_NOISE_RE = re.compile(
    r"(master16|upscaled16|final|result|render|beauty|marketing|depth|preview"
    r"|thumb|debug|segmentation|overlay|mask|albedo|normal|roughness|metallic|ao)"
)


def _artifact_compare_group(relative_path: str, path: Path) -> str:
    if not _artifact_is_previewable(path):
        return ""
    artifact_path = PurePosixPath(relative_path)
    parent = artifact_path.parent.as_posix()
    if parent == ".":
        parent = ""
    raw_stem = artifact_path.stem.lower()
    simplified_stem = _STEM_NOISE_RE.sub(" ", raw_stem)
    normalized_stem = re.sub(r"[^a-z0-9]+", "-", simplified_stem).strip("-")
    if not normalized_stem:
        normalized_stem = re.sub(r"[^a-z0-9]+", "-", raw_stem).strip("-")
    batch_hint = _artifact_batch_hint(relative_path)
    return "|".join(part for part in (batch_hint, parent, normalized_stem) if part)


def _artifact_display_hint(relative_path: str, path: Path) -> Dict[str, Any]:
    lower_name = relative_path.lower()
    stem_lower = PurePosixPath(relative_path).stem.lower()
    artifact_type = _infer_artifact_type(path)
    if _artifact_is_previewable(path):
        if re.search(
            r"(mask|matte|thumb|preview|debug|overlay|segmentation|albedo|normal|roughness|metallic|ao)",
            lower_name,
        ):
            role = "supporting_preview"
            priority = 700
        elif re.search(r"(master16|upscaled16|final|result|render|beauty|marketing|depth)", lower_name):
            role = "primary_preview"
            priority = 1000
        else:
            role = "review_preview"
            priority = 850
    elif lower_name.endswith(".vlm_captioning.sidecar.json"):
        role = "vlm_caption"
        priority = 300
    elif lower_name.endswith(".vlm_captioning.raw.txt"):
        role = "log"
        priority = 160
    elif "/captioning/" in f"/{lower_name}":
        role = "metadata"
        priority = 120
    elif "run_card" in lower_name:
        role = "run_card"
        priority = 320
    elif "report" in lower_name:
        role = "report"
        priority = 280
    elif "manifest" in lower_name:
        role = "manifest"
        priority = 240
    elif artifact_type == "archive":
        role = "archive"
        priority = 180
    elif lower_name.endswith(".log") or "/logs/" in lower_name or re.search(r"(^|[._\-\s])log($|[._\-\s])", stem_lower):
        role = "log"
        priority = 160
    elif artifact_type == "metadata":
        role = "metadata"
        priority = 120
    else:
        role = "file"
        priority = 100

    hint: Dict[str, Any] = {
        "role": role,
        "priority": priority,
        "label": _artifact_display_label(role),
    }
    compare_group = _artifact_compare_group(relative_path, path)
    if compare_group:
        hint["compare_group"] = compare_group
    return hint


def _artifact_url(job_id: str, relative_path: str) -> str:
    return f"/v1/jobs/{quote(str(job_id), safe='')}" f"/artifacts/{quote(relative_path, safe='/')}"


def _artifact_fingerprint(
    path: Path,
    size_bytes: Optional[int],
    *,
    max_bytes: int = ARTIFACT_FINGERPRINT_MAX_BYTES,
    chunk_bytes: int = _ARTIFACT_FINGERPRINT_CHUNK_BYTES,
) -> Tuple[Optional[str], str]:
    """Return ``(sha256_hex, status)`` for an artifact."""

    if size_bytes is None:
        return None, "unavailable"
    if size_bytes > max_bytes:
        return None, "skipped_size"
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return None, "unavailable"
    return digest.hexdigest(), "ok"


def _serialize_indexed_artifact(
    *,
    job_id: str,
    relative_path: str,
    path: Path,
    fingerprint_max_bytes: int = ARTIFACT_FINGERPRINT_MAX_BYTES,
    fingerprint_chunk_bytes: int = _ARTIFACT_FINGERPRINT_CHUNK_BYTES,
) -> Dict[str, Any]:
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = None

    content_type = _artifact_content_type(path)
    sha256_hex, fingerprint_status = _artifact_fingerprint(
        path,
        size_bytes,
        max_bytes=fingerprint_max_bytes,
        chunk_bytes=fingerprint_chunk_bytes,
    )
    download_url = _artifact_url(job_id, relative_path)
    proxy_path = _artifact_preview_proxy_path(path)
    proxy_relative = f"{relative_path}.preview.png" if proxy_path is not None else None
    browser_previewable = _artifact_is_browser_previewable(path) or proxy_path is not None
    payload: Dict[str, Any] = {
        "artifact_type": _infer_artifact_type(path),
        "media_kind": _artifact_media_kind(path),
        "previewable": _artifact_is_previewable(path),
        "browser_previewable": browser_previewable,
        "content_type": content_type,
        "mime_type": content_type,
        "display_hint": _artifact_display_hint(relative_path, path),
        "url": download_url,
        "download_url": download_url,
        "path": relative_path,
        "relative_path": relative_path,
        "size_bytes": size_bytes,
        "fingerprint_status": fingerprint_status,
    }
    if proxy_relative is not None:
        payload["preview_url"] = _artifact_url(job_id, proxy_relative)
        payload["preview_mime_type"] = "image/png"
    if sha256_hex is not None:
        payload["sha256"] = sha256_hex
    return payload


def _coerce_nonnegative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _captioning_artifact_counts_from_paths(paths: Iterable[str]) -> Dict[str, int]:
    counts = {"sidecar_count": 0, "raw_count": 0, "proxy_count": 0}
    for raw_path in paths:
        lower_path = str(raw_path or "").replace("\\", "/").strip().lower()
        if not lower_path:
            continue
        if lower_path.endswith(".vlm_captioning.sidecar.json"):
            counts["sidecar_count"] += 1
        elif lower_path.endswith(".vlm_captioning.raw.txt"):
            counts["raw_count"] += 1
        elif "/captioning/" in f"/{lower_path}" and re.search(r"_proxy\.(?:png|jpe?g)$", lower_path):
            counts["proxy_count"] += 1
    return counts


def _captioning_artifact_counts_from_run_card(payload: Mapping[str, Any]) -> Dict[str, int]:
    artifact_index = payload.get("artifact_index")
    if not isinstance(artifact_index, list):
        return {"sidecar_count": 0, "raw_count": 0, "proxy_count": 0}
    return _captioning_artifact_counts_from_paths(
        str(artifact.get("relative_path") or artifact.get("path") or "")
        for artifact in artifact_index
        if isinstance(artifact, Mapping)
    )


def _captioning_artifact_counts_from_job_artifacts(artifacts: Mapping[str, Any]) -> Dict[str, int]:
    items = artifacts.get("items") if isinstance(artifacts, Mapping) else None
    if not isinstance(items, list):
        return {"sidecar_count": 0, "raw_count": 0, "proxy_count": 0}
    return _captioning_artifact_counts_from_paths(
        str(item.get("relative_path") or item.get("path") or "") for item in items if isinstance(item, Mapping)
    )


def _load_bounded_json_object(
    path: Path,
    *,
    max_bytes: int = JOB_RUN_SUMMARY_MAX_BYTES,
) -> Optional[Dict[str, Any]]:
    try:
        size_bytes = path.stat().st_size
    except OSError:
        return None
    if size_bytes <= 0 or size_bytes > max_bytes:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_artifact_relative_path(artifact_path: str) -> str:
    raw = str(artifact_path or "").strip()
    if not raw or raw.startswith("~") or "\x00" in raw or "\\" in raw:
        raise InvalidArtifactPathError

    candidate = PurePosixPath(raw)
    if candidate.is_absolute():
        raise AbsoluteArtifactPathError

    normalized = candidate.as_posix()
    if normalized in {"", "."}:
        raise InvalidArtifactPathError
    if any(part == ".." for part in candidate.parts):
        raise ArtifactPathOutsideJobOutputDirError

    return normalized


def _load_bounded_run_card_payload(
    path: Optional[Path],
    *,
    max_bytes: int = JOB_RUN_SUMMARY_MAX_BYTES,
) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    payload = _load_bounded_json_object(path, max_bytes=max_bytes)
    if payload is None:
        return None

    batch_id = str(payload.get("batch_id") or "").strip()
    artifact_index = payload.get("artifact_index")
    if not batch_id:
        return None
    try:
        infer_run_card_version(payload)
    except ValueError:
        return None
    if artifact_index is None:
        total_images = _coerce_nonnegative_int(payload.get("total_images"))
        success_count = _coerce_nonnegative_int(payload.get("success_count"))
        error_count = _coerce_nonnegative_int(payload.get("error_count"))
        if total_images is None and (success_count is None or error_count is None):
            return None
        return payload
    if not isinstance(artifact_index, list) or not artifact_index:
        return None
    for artifact in artifact_index:
        if not isinstance(artifact, Mapping):
            return None
        candidate_path = artifact.get("relative_path") or artifact.get("path")
        if not isinstance(candidate_path, str) or not candidate_path.strip():
            return None
        try:
            _normalize_artifact_relative_path(candidate_path)
        except ArtifactPathValidationError:
            return None
    return payload


def _artifact_batch_hint(relative_path: str) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}_\d{6}", PurePosixPath(relative_path).stem)
    return match.group(0) if match else ""


def _artifact_recency_key(relative_path: str, artifact_path: Path) -> Tuple[str, float, str]:
    batch_hint = _artifact_batch_hint(relative_path)
    try:
        modified_time = artifact_path.stat().st_mtime
    except OSError:
        modified_time = -1.0
    return (batch_hint, modified_time, relative_path)


def _find_newest_artifact_path(output_dir: Path, candidates: List[Path]) -> Optional[Path]:
    normalized_candidates: List[Tuple[str, Path]] = []
    for candidate in candidates:
        try:
            resolved = Path(os.path.realpath(candidate))
        except OSError:
            continue
        if not resolved.exists() or not resolved.is_file():
            continue
        try:
            relative_path = _relative_artifact_path(resolved, output_dir)
        except ValueError:
            continue
        normalized_candidates.append((relative_path, resolved))
    if not normalized_candidates:
        return None
    _, artifact_path = max(
        normalized_candidates,
        key=lambda item: _artifact_recency_key(item[0], item[1]),
    )
    return artifact_path


def _resolve_artifact_path_within_output_dir(
    output_dir: Path,
    relative_path: str,
) -> Optional[Tuple[str, Path]]:
    try:
        normalized_relative_path = _normalize_artifact_relative_path(relative_path)
    except ArtifactPathValidationError:
        return None
    resolved_candidate = Path(
        os.path.realpath(
            output_dir / Path(*PurePosixPath(normalized_relative_path).parts),
        )
    )
    try:
        canonical_relative_path = _relative_artifact_path(resolved_candidate, output_dir)
    except ValueError:
        return None
    if not resolved_candidate.exists() or not resolved_candidate.is_file():
        return None
    return canonical_relative_path, resolved_candidate


def _resolve_job_run_metadata(
    output_dir: Optional[Path],
    *,
    max_bytes: int = JOB_RUN_SUMMARY_MAX_BYTES,
) -> Optional[JobRunMetadata]:
    if output_dir is None:
        return None
    output_dir = Path(os.path.realpath(output_dir.expanduser()))
    if not output_dir.exists() or not output_dir.is_dir():
        return None

    batch_manifest_dir = output_dir / "manifests"
    run_card_candidates: List[Tuple[int, Tuple[str, float, str], Path, Dict[str, Any], Optional[Path]]] = []
    for candidate in output_dir.glob("run_card_*.json"):
        try:
            resolved_candidate = Path(os.path.realpath(candidate))
            relative_path = _relative_artifact_path(resolved_candidate, output_dir)
        except (OSError, ValueError):
            continue
        run_card_payload = _load_bounded_run_card_payload(resolved_candidate, max_bytes=max_bytes)
        if run_card_payload is None:
            continue
        batch_id = str(run_card_payload.get("batch_id") or "").strip()
        matching_manifest_path: Optional[Path] = None
        if batch_id:
            manifest_candidate = batch_manifest_dir / f"batch_{batch_id}.json"
            if manifest_candidate.exists() and manifest_candidate.is_file():
                matching_manifest_path = Path(os.path.realpath(manifest_candidate))
        run_card_candidates.append(
            (
                1 if matching_manifest_path is not None else 0,
                _artifact_recency_key(relative_path, resolved_candidate),
                resolved_candidate,
                run_card_payload,
                matching_manifest_path,
            )
        )

    run_card_path: Optional[Path] = None
    run_card_payload: Optional[Dict[str, Any]] = None
    batch_manifest_path: Optional[Path] = None
    batch_manifest_payload: Optional[Dict[str, Any]] = None
    if run_card_candidates:
        _, _, run_card_path, run_card_payload, batch_manifest_path = max(
            run_card_candidates,
            key=lambda item: (item[0], item[1]),
        )
        if batch_manifest_path is not None:
            batch_manifest_payload = _load_bounded_json_object(batch_manifest_path, max_bytes=max_bytes)
    elif batch_manifest_dir.exists() and batch_manifest_dir.is_dir():
        batch_manifest_path = _find_newest_artifact_path(
            output_dir,
            list(batch_manifest_dir.glob("batch_*.json")),
        )
        if batch_manifest_path is not None:
            batch_manifest_payload = _load_bounded_json_object(batch_manifest_path, max_bytes=max_bytes)

    return JobRunMetadata(
        output_dir=output_dir,
        run_card_path=run_card_path,
        run_card_payload=run_card_payload,
        batch_manifest_path=batch_manifest_path,
        batch_manifest_payload=batch_manifest_payload,
    )


def _build_scoped_job_artifacts(
    *,
    job_id: str,
    output_dir: Path,
    candidate_paths: List[Path],
    max_indexed_artifacts: int = MAX_INDEXED_ARTIFACTS,
    fingerprint_max_bytes: int = ARTIFACT_FINGERPRINT_MAX_BYTES,
    fingerprint_chunk_bytes: int = _ARTIFACT_FINGERPRINT_CHUNK_BYTES,
) -> Tuple[List[Dict[str, Any]], Dict[str, Path], bool]:
    discovered: Dict[str, Path] = {}
    for candidate_path in candidate_paths:
        try:
            resolved_path = Path(os.path.realpath(candidate_path))
        except OSError:
            continue
        if not resolved_path.exists() or not resolved_path.is_file():
            continue
        try:
            relative_path = _relative_artifact_path(resolved_path, output_dir)
        except ValueError:
            continue
        discovered[relative_path] = resolved_path

    ordered_candidates = sorted(
        discovered.items(),
        key=lambda item: (item[0].casefold(), item[0]),
    )
    truncated = len(ordered_candidates) > max_indexed_artifacts
    selected_candidates = ordered_candidates[:max_indexed_artifacts]

    items = [
        _serialize_indexed_artifact(
            job_id=job_id,
            relative_path=relative_path,
            path=path,
            fingerprint_max_bytes=fingerprint_max_bytes,
            fingerprint_chunk_bytes=fingerprint_chunk_bytes,
        )
        for relative_path, path in selected_candidates
    ]
    selected_lookup = {relative_path: path for relative_path, path in selected_candidates}
    for relative_path, path in selected_candidates:
        _add_artifact_preview_proxy_lookup(
            selected_lookup,
            output_dir=output_dir,
            artifact_path=path,
        )
    return items, selected_lookup, truncated


def _build_scoped_job_artifacts_from_run_metadata(
    *,
    job_id: str,
    metadata: JobRunMetadata,
    max_indexed_artifacts: int = MAX_INDEXED_ARTIFACTS,
    fingerprint_max_bytes: int = ARTIFACT_FINGERPRINT_MAX_BYTES,
    fingerprint_chunk_bytes: int = _ARTIFACT_FINGERPRINT_CHUNK_BYTES,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Path], bool]]:
    artifact_index = None
    if metadata.run_card_path is not None and metadata.run_card_payload is not None:
        artifact_index = metadata.run_card_payload.get("artifact_index")
        if isinstance(artifact_index, list):
            candidate_paths: List[Path] = [metadata.run_card_path]
            for artifact_entry in artifact_index:
                if not isinstance(artifact_entry, dict):
                    continue
                artifact_relative_path = artifact_entry.get("relative_path") or artifact_entry.get("path")
                if not isinstance(artifact_relative_path, str) or not artifact_relative_path.strip():
                    continue
                resolved = _resolve_artifact_path_within_output_dir(
                    metadata.output_dir,
                    artifact_relative_path,
                )
                if resolved is None:
                    continue
                _, resolved_path = resolved
                candidate_paths.append(resolved_path)
            if len(candidate_paths) > 1:
                return _build_scoped_job_artifacts(
                    job_id=job_id,
                    output_dir=metadata.output_dir,
                    candidate_paths=candidate_paths,
                    max_indexed_artifacts=max_indexed_artifacts,
                    fingerprint_max_bytes=fingerprint_max_bytes,
                    fingerprint_chunk_bytes=fingerprint_chunk_bytes,
                )

    if metadata.batch_manifest_path is not None:
        candidate_paths = [metadata.batch_manifest_path]
        if metadata.run_card_path is not None and isinstance(artifact_index, list) and artifact_index:
            candidate_paths.insert(0, metadata.run_card_path)
        return _build_scoped_job_artifacts(
            job_id=job_id,
            output_dir=metadata.output_dir,
            candidate_paths=candidate_paths,
            max_indexed_artifacts=max_indexed_artifacts,
            fingerprint_max_bytes=fingerprint_max_bytes,
            fingerprint_chunk_bytes=fingerprint_chunk_bytes,
        )

    return None


def _validate_resolved_job_artifact_path(
    output_dir: Optional[Path],
    resolved_artifact: Path,
) -> tuple[Path, Path, str]:
    if output_dir is None:
        raise FileNotFoundError("job_output_dir_missing")

    output_dir = Path(os.path.realpath(output_dir.expanduser()))
    if not output_dir.exists() or not output_dir.is_dir():
        raise FileNotFoundError("job_output_dir_missing")

    resolved = Path(os.path.realpath(resolved_artifact))
    try:
        relative_path = _relative_artifact_path(resolved, output_dir)
    except ValueError as exc:
        raise ArtifactPathOutsideJobOutputDirError from exc

    return output_dir, resolved, relative_path


def _hydrate_artifact_lookup_from_items(
    *,
    items: Any,
    output_dir: Optional[Path],
) -> Dict[str, Path]:
    if not isinstance(items, list) or not items:
        return {}
    if output_dir is None:
        return {}

    lookup: Dict[str, Path] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        candidate_path = item.get("relative_path") or item.get("path")
        try:
            normalized = _normalize_artifact_relative_path(str(candidate_path or ""))
        except ValueError:
            continue
        resolved_candidate = Path(output_dir) / Path(*PurePosixPath(normalized).parts)
        try:
            _, resolved, canonical_relative_path = _validate_resolved_job_artifact_path(output_dir, resolved_candidate)
        except (ValueError, FileNotFoundError):
            continue
        if not resolved.exists() or not resolved.is_file():
            continue
        lookup[canonical_relative_path] = resolved
        _add_artifact_preview_proxy_lookup(
            lookup,
            output_dir=output_dir,
            artifact_path=resolved,
        )
    return lookup


def _iter_job_artifact_files(output_dir: Path) -> Iterator[Path]:
    for root, dirnames, filenames in os.walk(output_dir, followlinks=False):
        root_path = Path(root)
        dirnames[:] = sorted(dirname for dirname in dirnames if not (root_path / dirname).is_symlink())
        for filename in sorted(filenames):
            yield root_path / filename


def _index_job_artifacts(
    *,
    job_id: str,
    output_dir: Optional[Path],
    max_indexed_artifacts: int = MAX_INDEXED_ARTIFACTS,
    fingerprint_max_bytes: int = ARTIFACT_FINGERPRINT_MAX_BYTES,
    fingerprint_chunk_bytes: int = _ARTIFACT_FINGERPRINT_CHUNK_BYTES,
    run_summary_max_bytes: int = JOB_RUN_SUMMARY_MAX_BYTES,
) -> JobArtifactIndexResult:
    if output_dir is None:
        artifacts = {
            "output_dir": None,
            "items": [],
            "indexed_count": 0,
            "truncated": False,
        }
        return JobArtifactIndexResult(items=[], artifact_lookup={}, artifacts=artifacts)
    if not output_dir.exists() or not output_dir.is_dir():
        artifacts = {
            "output_dir": str(output_dir),
            "items": [],
            "indexed_count": 0,
            "truncated": False,
        }
        return JobArtifactIndexResult(items=[], artifact_lookup={}, artifacts=artifacts)

    output_dir = Path(os.path.realpath(output_dir.expanduser()))
    metadata = _resolve_job_run_metadata(output_dir, max_bytes=run_summary_max_bytes)
    if metadata is not None:
        scoped_artifacts = _build_scoped_job_artifacts_from_run_metadata(
            job_id=job_id,
            metadata=metadata,
            max_indexed_artifacts=max_indexed_artifacts,
            fingerprint_max_bytes=fingerprint_max_bytes,
            fingerprint_chunk_bytes=fingerprint_chunk_bytes,
        )
        if scoped_artifacts is not None:
            items, artifact_lookup, truncated = scoped_artifacts
            artifacts = {
                "output_dir": str(output_dir),
                "items": items,
                "indexed_count": len(items),
                "truncated": truncated,
            }
            return JobArtifactIndexResult(items=items, artifact_lookup=artifact_lookup, artifacts=artifacts)

    selected: List[tuple[tuple[str, str], str, Path]] = []
    selected_keys: List[tuple[str, str]] = []
    total_files = 0
    for path in _iter_job_artifact_files(output_dir):
        if not path.is_file():
            continue

        resolved_path = Path(os.path.realpath(path))
        try:
            resolved_path.relative_to(output_dir)
        except ValueError:
            continue
        total_files += 1
        try:
            relative_path = _relative_artifact_path(path, output_dir)
        except ValueError:
            relative_path = path.name

        key = (relative_path.casefold(), relative_path)

        if len(selected) < max_indexed_artifacts:
            insert_at = bisect_left(selected_keys, key)
            selected_keys.insert(insert_at, key)
            selected.insert(insert_at, (key, relative_path, resolved_path))
            continue

        if key >= selected_keys[-1]:
            continue

        insert_at = bisect_left(selected_keys, key)
        selected_keys.insert(insert_at, key)
        selected.insert(insert_at, (key, relative_path, resolved_path))
        selected_keys.pop()
        selected.pop()

    truncated = total_files > max_indexed_artifacts

    items: List[Dict[str, Any]] = []
    selected_lookup: Dict[str, Path] = {}
    for _, relative_path, path in selected:
        items.append(
            _serialize_indexed_artifact(
                job_id=job_id,
                relative_path=relative_path,
                path=path,
                fingerprint_max_bytes=fingerprint_max_bytes,
                fingerprint_chunk_bytes=fingerprint_chunk_bytes,
            )
        )
        selected_lookup[relative_path] = path
        _add_artifact_preview_proxy_lookup(
            selected_lookup,
            output_dir=output_dir,
            artifact_path=path,
        )

    artifacts = {
        "output_dir": str(output_dir),
        "items": items,
        "indexed_count": len(items),
        "truncated": truncated,
    }
    return JobArtifactIndexResult(items=items, artifact_lookup=selected_lookup, artifacts=artifacts)
