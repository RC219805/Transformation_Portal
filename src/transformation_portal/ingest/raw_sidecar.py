"""RAW metadata sidecar generation helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional

# Re-export the lux_depth_v3 RAW extension whitelist as the single source of
# truth so the sidecar generator and the depth pipeline never disagree on
# which files count as RAW.
from ..lux_depth_v3.raw_loader import RAW_EXTENSIONS as RAW_EXTENSIONS
from .canonical_json import dumps_json

RAW_SIDECAR_SCHEMA = "raw-image-sidecar/v2"
EXIFTOOL_VERSION_TIMEOUT_SECONDS = 5
EXIFTOOL_METADATA_TIMEOUT_SECONDS = 30
_VOLATILE_EXIFTOOL_KEYS = {
    "FileAccessDate",
    "FileInodeChangeDate",
}


@dataclass(frozen=True)
class RawSidecarResult:
    input_path: Path
    output_path: Path
    rawpy_available: bool
    rawpy_ok: bool
    rawpy_error: Optional[str] = None


def is_raw_image_path(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTENSIONS


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Dict[str, Any]) -> str:
    return dumps_json(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ": "),
    )


def _write_json_atomic(payload: Dict[str, Any], output_path: Path, *, fsync: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(_canonical_json(payload))
            if fsync:
                handle.flush()
                import os

                os.fsync(handle.fileno())
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _sanitize_exiftool_payload(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in raw_payload.items():
        terminal_key = key.split(":")[-1]
        if terminal_key in _VOLATILE_EXIFTOOL_KEYS:
            continue
        cleaned[key] = value
    return cleaned


def _get_exiftool_version(exiftool_path: str) -> str:
    completed = subprocess.run(
        [exiftool_path, "-ver"],
        check=True,
        capture_output=True,
        text=True,
        timeout=EXIFTOOL_VERSION_TIMEOUT_SECONDS,
    )
    return completed.stdout.strip()


def _run_exiftool_json(input_path: Path, exiftool_path: str) -> Dict[str, Any]:
    completed = subprocess.run(
        [exiftool_path, "-json", "-G1", "-a", "-u", "-n", str(input_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=EXIFTOOL_METADATA_TIMEOUT_SECONDS,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise ValueError("Unexpected exiftool JSON payload shape")
    return payload[0]


def _rawpy_status_payload() -> tuple[Optional[ModuleType], Dict[str, Any]]:
    try:
        import rawpy  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return None, {
            "available": False,
            "ok": False,
            "error": str(exc),
            "version": None,
            "libraw_version": None,
        }

    rawpy_version = getattr(rawpy, "__version__", None)
    libraw_version = getattr(rawpy, "libraw_version", None)
    if isinstance(libraw_version, (tuple, list)):
        libraw_version = ".".join(str(part) for part in libraw_version)
    elif libraw_version is not None:
        libraw_version = str(libraw_version)

    return rawpy, {
        "available": True,
        "ok": True,
        "error": None,
        "version": str(rawpy_version) if rawpy_version is not None else None,
        "libraw_version": libraw_version,
    }


def _read_rawpy_metadata(input_path: Path) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    rawpy_module, status = _rawpy_status_payload()
    if rawpy_module is None:
        return None, status

    try:
        with rawpy_module.imread(str(input_path)) as raw:
            payload = {
                "raw_type": str(raw.raw_type),
                "sizes": {
                    "raw_height": raw.sizes.raw_height,
                    "raw_width": raw.sizes.raw_width,
                    "height": raw.sizes.height,
                    "width": raw.sizes.width,
                    "top_margin": raw.sizes.top_margin,
                    "left_margin": raw.sizes.left_margin,
                    "iheight": raw.sizes.iheight,
                    "iwidth": raw.sizes.iwidth,
                    "pixel_aspect": raw.sizes.pixel_aspect,
                    "flip": raw.sizes.flip,
                    "crop_left_margin": raw.sizes.crop_left_margin,
                    "crop_top_margin": raw.sizes.crop_top_margin,
                    "crop_width": raw.sizes.crop_width,
                    "crop_height": raw.sizes.crop_height,
                },
                "color_desc": (raw.color_desc.decode("ascii", errors="replace") if raw.color_desc is not None else None),
                "num_colors": raw.num_colors,
                "black_level_per_channel": (
                    list(raw.black_level_per_channel) if raw.black_level_per_channel is not None else None
                ),
                "white_level": raw.white_level,
                "camera_whitebalance": (list(raw.camera_whitebalance) if raw.camera_whitebalance is not None else None),
                "daylight_whitebalance": (list(raw.daylight_whitebalance) if raw.daylight_whitebalance is not None else None),
                "raw_pattern": raw.raw_pattern.tolist() if raw.raw_pattern is not None else None,
            }
            return payload, status
    except Exception as exc:  # noqa: BLE001
        failed_status = dict(status)
        failed_status["ok"] = False
        failed_status["error"] = str(exc)
        return None, failed_status


def _build_file_metadata(
    input_path: Path,
    *,
    size_bytes: Optional[int] = None,
    sha256: Optional[str] = None,
) -> Dict[str, Any]:
    stat = input_path.stat()
    return {
        "source_file": str(input_path),
        "file_name": input_path.name,
        "suffix": input_path.suffix.lower(),
        "size_bytes": stat.st_size if size_bytes is None else size_bytes,
        "sha256": _sha256_file(input_path) if sha256 is None else sha256,
    }


def build_raw_sidecar_payload(
    input_path: Path,
    *,
    exiftool_path: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
    file_sha256: Optional[str] = None,
    precomputed_exiftool_payload: Optional[Dict[str, Any]] = None,
    precomputed_exiftool_version: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_exiftool = exiftool_path or shutil.which("exiftool")
    if precomputed_exiftool_version is None or precomputed_exiftool_payload is None:
        if not resolved_exiftool:
            raise FileNotFoundError("exiftool not found on PATH")

    exiftool_version = (
        precomputed_exiftool_version if precomputed_exiftool_version is not None else _get_exiftool_version(resolved_exiftool)
    )
    exiftool_payload = _sanitize_exiftool_payload(
        precomputed_exiftool_payload
        if precomputed_exiftool_payload is not None
        else _run_exiftool_json(input_path, resolved_exiftool)
    )
    rawpy_payload, rawpy_status = _read_rawpy_metadata(input_path)

    return {
        "sidecar_schema": RAW_SIDECAR_SCHEMA,
        "source_file": str(input_path),
        "file": _build_file_metadata(
            input_path,
            size_bytes=file_size_bytes,
            sha256=file_sha256,
        ),
        "capture_status": {
            "exiftool": {
                "available": True,
                "ok": True,
                "error": None,
                "version": exiftool_version,
            },
            "rawpy": rawpy_status,
        },
        "metadata_exiftool": exiftool_payload,
        "metadata_rawpy": rawpy_payload,
    }


def generate_raw_sidecar(
    input_path: Path,
    *,
    output_path: Optional[Path] = None,
    exiftool_path: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
    file_sha256: Optional[str] = None,
    precomputed_exiftool_payload: Optional[Dict[str, Any]] = None,
    precomputed_exiftool_version: Optional[str] = None,
    fsync: bool = False,
) -> RawSidecarResult:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    resolved_output = output_path or input_path.with_name(f"{input_path.stem}.raw.sidecar.json")
    payload = build_raw_sidecar_payload(
        input_path,
        exiftool_path=exiftool_path,
        file_size_bytes=file_size_bytes,
        file_sha256=file_sha256,
        precomputed_exiftool_payload=precomputed_exiftool_payload,
        precomputed_exiftool_version=precomputed_exiftool_version,
    )
    _write_json_atomic(payload, resolved_output, fsync=fsync)

    rawpy_status = payload["capture_status"]["rawpy"]
    return RawSidecarResult(
        input_path=input_path,
        output_path=resolved_output,
        rawpy_available=bool(rawpy_status.get("available", False)),
        rawpy_ok=bool(rawpy_status.get("ok", False)),
        rawpy_error=rawpy_status.get("error"),
    )
