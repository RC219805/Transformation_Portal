"""Deterministic canonicalization and extraction for Phase 4 capture metadata."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import unicodedata
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

from .exceptions import (
    ConfigValidationError,
    ExtractionFailure,
    PathNormalizationError,
    SchemaValidationError,
    StrictWarningsError,
)

SUPPORTED_EXTENSIONS = frozenset(
    {
        ".dng",
        ".tif",
        ".tiff",
        ".cr2",
        ".cr3",
        ".nef",
        ".arw",
        ".raf",
        ".rw2",
        ".orf",
    }
)

_GPS_DATE_RE = re.compile(r"^(\d{4})[:\-](\d{2})[:\-](\d{2})$")
_GPS_TIME_RE = re.compile(r"^(\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")
_EXIF_DATETIME_RE = re.compile(r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})$")
_OFFSET_RE = re.compile(r"^([+-])(\d{2}):(\d{2})$")
_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:[\\/]")
_PARENT_SEGMENT_RE = re.compile(r"(^|/)\.\.(/|$)")
_WARNING_CODE_RE = re.compile(r"^WARN_[A-Z0-9_]+$")

_REQUIRED_TOP_LEVEL_KEYS = {
    "metadata_contract_version",
    "tag_whitelist",
    "datetime_precedence",
    "rounding_rules",
    "orientation_mapping",
    "warning_codes",
    "path_normalization",
}
_SUPPORTED_DATETIME_PRECEDENCE = {
    "GPSDateStamp+GPSTimeStamp",
    "DateTimeOriginal+OffsetTimeOriginal",
}
_ROUNDING_KEYS = {
    "gps_decimal_places",
    "focal_length_decimal_places",
    "aperture_decimal_places",
    "shutter_speed_decimal_places",
    "exposure_ev_decimal_places",
    "rounding_mode",
}
_PATH_POLICY_KEYS = {
    "forbid_absolute",
    "forbid_backslash",
    "forbid_dotdot",
    "forbid_leading_dot_slash",
}
_ORIENTATION_KEYS = {"1", "2", "3", "4", "5", "6", "7", "8"}
_ORIENTATION_VALUES = {
    "Horizontal",
    "MirrorHorizontal",
    "Rotate180",
    "MirrorVertical",
    "MirrorHorizontalRotate270CW",
    "Rotate90CW",
    "MirrorHorizontalRotate90CW",
    "Rotate270CW",
}
_IMPLEMENTATION_REQUIRED_WARNING_CODES = {
    "WARN_DATETIME_NO_TZ",
    "WARN_GPS_PARSE_FAIL",
    "WARN_INVALID_ORIENTATION",
}
EXIFTOOL_TIMEOUT_SECONDS = 120


def load_capture_metadata_config(config_path: Path) -> dict[str, Any]:
    """Load and minimally validate canonicalization config."""
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigValidationError(f"unable to load config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigValidationError("config root must be an object")

    missing = sorted(_REQUIRED_TOP_LEVEL_KEYS - set(payload))
    if missing:
        raise ConfigValidationError(f"config missing required key(s): {', '.join(missing)}")
    unknown = sorted(set(payload) - _REQUIRED_TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigValidationError(f"config contains unknown key(s): {', '.join(unknown)}")

    if payload["metadata_contract_version"] != "tp.meta.capture.v1":
        raise ConfigValidationError("metadata_contract_version must be tp.meta.capture.v1")

    if not isinstance(payload["tag_whitelist"], list) or not payload["tag_whitelist"]:
        raise ConfigValidationError("tag_whitelist must be a non-empty list")
    if any((not isinstance(tag, str) or not tag.strip()) for tag in payload["tag_whitelist"]):
        raise ConfigValidationError("tag_whitelist entries must be non-empty strings")
    if len(set(payload["tag_whitelist"])) != len(payload["tag_whitelist"]):
        raise ConfigValidationError("tag_whitelist must not contain duplicates")

    if not isinstance(payload["datetime_precedence"], list) or not payload["datetime_precedence"]:
        raise ConfigValidationError("datetime_precedence must be a non-empty list")
    precedence_values = payload["datetime_precedence"]
    if any((not isinstance(source, str) or source not in _SUPPORTED_DATETIME_PRECEDENCE) for source in precedence_values):
        raise ConfigValidationError(
            "datetime_precedence contains unsupported source; allowed values are "
            f"{', '.join(sorted(_SUPPORTED_DATETIME_PRECEDENCE))}"
        )
    if len(set(precedence_values)) != len(precedence_values):
        raise ConfigValidationError("datetime_precedence must not contain duplicates")

    rounding_rules = payload["rounding_rules"]
    if not isinstance(rounding_rules, dict):
        raise ConfigValidationError("rounding_rules must be an object")
    missing_rounding = sorted(_ROUNDING_KEYS - set(rounding_rules))
    if missing_rounding:
        raise ConfigValidationError(f"rounding_rules missing key(s): {', '.join(missing_rounding)}")
    unknown_rounding = sorted(set(rounding_rules) - _ROUNDING_KEYS)
    if unknown_rounding:
        raise ConfigValidationError(f"rounding_rules contains unknown key(s): {', '.join(unknown_rounding)}")
    for key in sorted(_ROUNDING_KEYS - {"rounding_mode"}):
        value = rounding_rules[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ConfigValidationError(f"rounding_rules.{key} must be a non-negative integer")
    if rounding_rules["rounding_mode"] != "bankers":
        raise ConfigValidationError("rounding_rules.rounding_mode must be bankers")

    orientation_mapping = payload["orientation_mapping"]
    if not isinstance(orientation_mapping, dict):
        raise ConfigValidationError("orientation_mapping must be an object")
    mapping_keys = set(orientation_mapping)
    if mapping_keys != _ORIENTATION_KEYS:
        missing_orientation = sorted(_ORIENTATION_KEYS - mapping_keys)
        unknown_orientation = sorted(mapping_keys - _ORIENTATION_KEYS)
        details: list[str] = []
        if missing_orientation:
            details.append(f"missing={','.join(missing_orientation)}")
        if unknown_orientation:
            details.append(f"unknown={','.join(unknown_orientation)}")
        raise ConfigValidationError(f"orientation_mapping keys must be 1..8 ({'; '.join(details)})")
    if any((not isinstance(value, str) or value not in _ORIENTATION_VALUES) for value in orientation_mapping.values()):
        raise ConfigValidationError("orientation_mapping values must match metadata schema orientation enum")

    if not isinstance(payload["warning_codes"], list) or not payload["warning_codes"]:
        raise ConfigValidationError("warning_codes must be a non-empty list")
    if any((not isinstance(code, str) or not _WARNING_CODE_RE.match(code)) for code in payload["warning_codes"]):
        raise ConfigValidationError("warning_codes entries must match ^WARN_[A-Z0-9_]+$")
    if len(set(payload["warning_codes"])) != len(payload["warning_codes"]):
        raise ConfigValidationError("warning_codes must not contain duplicates")
    missing_required_warning_codes = sorted(_IMPLEMENTATION_REQUIRED_WARNING_CODES - set(payload["warning_codes"]))
    if missing_required_warning_codes:
        raise ConfigValidationError(
            "warning_codes missing implementation-required code(s): " f"{', '.join(missing_required_warning_codes)}"
        )

    path_policy = payload["path_normalization"]
    if not isinstance(path_policy, dict):
        raise ConfigValidationError("path_normalization must be an object")
    missing_policy = sorted(_PATH_POLICY_KEYS - set(path_policy))
    if missing_policy:
        raise ConfigValidationError(f"path_normalization missing key(s): {', '.join(missing_policy)}")
    unknown_policy = sorted(set(path_policy) - _PATH_POLICY_KEYS)
    if unknown_policy:
        raise ConfigValidationError(f"path_normalization contains unknown key(s): {', '.join(unknown_policy)}")
    for key in sorted(_PATH_POLICY_KEYS):
        if not isinstance(path_policy[key], bool):
            raise ConfigValidationError(f"path_normalization.{key} must be boolean")

    return payload


def compute_config_fingerprint_sha256(config: dict[str, Any]) -> str:
    """Compute deterministic fingerprint over canonical JSON config bytes."""
    canonical = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def normalize_relative_path(raw_relative_path: str, path_policy: dict[str, Any] | None = None) -> str:
    """Normalize and validate deterministic relative-path constraints."""
    policy = path_policy or {
        "forbid_absolute": True,
        "forbid_backslash": True,
        "forbid_dotdot": True,
        "forbid_leading_dot_slash": True,
    }

    if not raw_relative_path:
        raise PathNormalizationError("relative path is empty")
    if policy.get("forbid_absolute", True) and raw_relative_path.startswith("/"):
        raise PathNormalizationError(f"absolute path forbidden: {raw_relative_path}")
    if policy.get("forbid_absolute", True) and _DRIVE_PREFIX_RE.match(raw_relative_path):
        raise PathNormalizationError(f"drive-qualified path forbidden: {raw_relative_path}")
    if policy.get("forbid_backslash", True) and "\\" in raw_relative_path:
        raise PathNormalizationError(f"backslash forbidden: {raw_relative_path}")
    if policy.get("forbid_leading_dot_slash", True) and raw_relative_path.startswith("./"):
        raise PathNormalizationError(f"leading ./ forbidden: {raw_relative_path}")
    if policy.get("forbid_dotdot", True) and _PARENT_SEGMENT_RE.search(raw_relative_path):
        raise PathNormalizationError(f".. segment forbidden: {raw_relative_path}")
    if "//" in raw_relative_path:
        raise PathNormalizationError(f"empty path segment forbidden: {raw_relative_path}")
    if raw_relative_path.endswith("/"):
        raise PathNormalizationError(f"trailing slash forbidden: {raw_relative_path}")
    return raw_relative_path


def normalize_string(value: Any) -> str | None:
    """NFC-normalize and trim ASCII whitespace for string fields."""
    if value is None:
        return None
    normalized = unicodedata.normalize("NFC", str(value)).strip(" \t\r\n\f\v")
    return normalized or None


def _tag_value(raw_tags: dict[str, Any], tag: str) -> Any:
    """Lookup tag with group and ungrouped fallbacks."""
    candidates = [tag]
    if ":" in tag:
        candidates.append(tag.split(":", 1)[1])
    else:
        candidates.extend(
            [
                f"EXIF:{tag}",
                f"XMP:{tag}",
                f"Composite:{tag}",
                f"File:{tag}",
            ]
        )
    for candidate in candidates:
        if candidate in raw_tags:
            return raw_tags[candidate]
    return None


def _parse_decimal(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise InvalidOperation("bool is not a numeric tag value")
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise InvalidOperation("non-finite float")
        return Decimal(str(value))
    if isinstance(value, str):
        stripped = value.strip()
        if "/" in stripped and stripped.count("/") == 1:
            numerator, denominator = stripped.split("/", 1)
            return Decimal(numerator) / Decimal(denominator)
        return Decimal(stripped)
    raise InvalidOperation(f"unsupported numeric type: {type(value).__name__}")


def _round_half_even(value: Any, places: int) -> float:
    decimal_value = _parse_decimal(value)
    quant = Decimal("1").scaleb(-places)
    rounded = decimal_value.quantize(quant, rounding=ROUND_HALF_EVEN)
    return float(rounded)


def _parse_gps_date(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, str):
        return None
    match = _GPS_DATE_RE.match(value.strip())
    if not match:
        return None
    year, month, day = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return year, month, day


def _parse_gps_time(value: Any) -> tuple[int, int, int] | None:
    if isinstance(value, str):
        match = _GPS_TIME_RE.match(value.strip())
        if not match:
            return None
        hour = int(match.group(1))
        minute = int(match.group(2))
        second = int(Decimal(match.group(3)).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))
        return hour, minute, min(second, 59)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            hour = int(_parse_decimal(value[0]))
            minute = int(_parse_decimal(value[1]))
            second = int(_parse_decimal(value[2]).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))
        except (InvalidOperation, ValueError):
            return None
        return hour, minute, min(second, 59)
    return None


def _parse_datetime_original(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    match = _EXIF_DATETIME_RE.match(value.strip())
    if not match:
        return None
    year, month, day, hour, minute, second = [int(match.group(i)) for i in range(1, 7)]
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


def _parse_offset(value: Any) -> timezone | None:
    if not isinstance(value, str):
        return None
    match = _OFFSET_RE.match(value.strip())
    if not match:
        return None
    sign = 1 if match.group(1) == "+" else -1
    hours = int(match.group(2))
    minutes = int(match.group(3))
    total_minutes = sign * (hours * 60 + minutes)
    return timezone(timedelta(minutes=total_minutes))


def _derive_capture_datetime_utc(
    raw_tags: dict[str, Any],
    datetime_precedence: list[str],
    warnings: set[str],
) -> str | None:
    for source in datetime_precedence:
        if source == "GPSDateStamp+GPSTimeStamp":
            date_value = _tag_value(raw_tags, "GPSDateStamp")
            time_value = _tag_value(raw_tags, "GPSTimeStamp")
            date_parts = _parse_gps_date(date_value)
            time_parts = _parse_gps_time(time_value)
            if date_parts and time_parts:
                try:
                    dt = datetime(
                        date_parts[0],
                        date_parts[1],
                        date_parts[2],
                        time_parts[0],
                        time_parts[1],
                        time_parts[2],
                        tzinfo=timezone.utc,
                    )
                except ValueError:
                    warnings.add("WARN_GPS_PARSE_FAIL")
                    continue
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        if source == "DateTimeOriginal+OffsetTimeOriginal":
            datetime_original = _parse_datetime_original(_tag_value(raw_tags, "DateTimeOriginal"))
            offset = _parse_offset(_tag_value(raw_tags, "OffsetTimeOriginal"))
            if datetime_original and offset:
                localized = datetime_original.replace(tzinfo=offset)
                utc_dt = localized.astimezone(timezone.utc)
                return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    warnings.add("WARN_DATETIME_NO_TZ")
    return None


def _parse_rounded_optional(
    raw_tags: dict[str, Any],
    tag: str,
    places: int,
    warnings: set[str],
    warning_code: str | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> float | None:
    value = _tag_value(raw_tags, tag)
    if value is None:
        return None
    try:
        rounded = _round_half_even(value, places)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        if warning_code:
            warnings.add(warning_code)
        return None
    if lower is not None and rounded < lower:
        if warning_code:
            warnings.add(warning_code)
        return None
    if upper is not None and rounded > upper:
        if warning_code:
            warnings.add(warning_code)
        return None
    return rounded


def _map_orientation(raw_tags: dict[str, Any], mapping: dict[str, str], warnings: set[str]) -> str | None:
    value = _tag_value(raw_tags, "Orientation")
    if value is None:
        return None
    try:
        numeric = int(_parse_decimal(value))
    except (InvalidOperation, ValueError):
        warnings.add("WARN_INVALID_ORIENTATION")
        return None
    key = str(numeric)
    mapped = mapping.get(key)
    if mapped is None:
        warnings.add("WARN_INVALID_ORIENTATION")
    return mapped


def _canonical_warning_codes(warnings: set[str], allowed_codes: list[str]) -> list[str]:
    allowed = set(allowed_codes)
    return sorted({code for code in warnings if code in allowed})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_capture_files(input_root: Path, path_policy: dict[str, Any]) -> list[tuple[str, Path]]:
    if not input_root.exists() or not input_root.is_dir():
        raise PathNormalizationError(f"input root must be an existing directory: {input_root}")

    discovered: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for file_path in sorted(input_root.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        raw_relative = file_path.relative_to(input_root).as_posix()
        relative_path = normalize_relative_path(raw_relative, path_policy=path_policy)
        if relative_path in seen:
            raise PathNormalizationError(f"duplicate relative path after normalization: {relative_path}")
        seen.add(relative_path)
        discovered.append((relative_path, file_path))
    discovered.sort(key=lambda item: item[0])
    return discovered


def _run_exiftool(file_paths: list[Path], tag_whitelist: list[str]) -> dict[str, dict[str, Any]]:
    if not file_paths:
        return {}

    command = ["exiftool", "-json", "-n"]
    command.extend(f"-{tag}" for tag in tag_whitelist)
    command.extend(str(path) for path in file_paths)

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=EXIFTOOL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExtractionFailure(f"exiftool timed out after {EXIFTOOL_TIMEOUT_SECONDS}s") from exc
    except OSError as exc:
        raise ExtractionFailure(f"failed to execute exiftool: {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip() or "(no stderr)"
        raise ExtractionFailure(f"exiftool failed with code {result.returncode}: {stderr}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ExtractionFailure(f"invalid exiftool JSON output: {exc}") from exc

    if not isinstance(payload, list):
        raise ExtractionFailure("exiftool payload must be a JSON array")

    by_path: dict[str, dict[str, Any]] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            raise ExtractionFailure("exiftool entry must be an object")
        source_file = entry.get("SourceFile")
        if not isinstance(source_file, str):
            raise ExtractionFailure("exiftool entry missing SourceFile")
        resolved = str(Path(source_file).resolve())
        by_path[resolved] = entry
    return by_path


def _build_metadata_object(
    *,
    relative_path: str,
    file_path: Path,
    raw_tags: dict[str, Any],
    config: dict[str, Any],
    config_fingerprint_sha256: str,
    extractor_name: str,
    extractor_version: str,
) -> dict[str, Any]:
    warnings: set[str] = set()
    rounding_rules = config["rounding_rules"]
    orientation_mapping = config["orientation_mapping"]

    record: dict[str, Any] = {
        "metadata_contract_version": config["metadata_contract_version"],
        "relative_path": relative_path,
        "file_sha256": _sha256_file(file_path),
        "capture_datetime_utc": _derive_capture_datetime_utc(
            raw_tags=raw_tags,
            datetime_precedence=list(config["datetime_precedence"]),
            warnings=warnings,
        ),
        "camera_make": normalize_string(_tag_value(raw_tags, "Make")),
        "camera_model": normalize_string(_tag_value(raw_tags, "Model")),
        "lens_model": normalize_string(_tag_value(raw_tags, "LensModel")),
        "gps_latitude": _parse_rounded_optional(
            raw_tags,
            "GPSLatitude",
            places=int(rounding_rules["gps_decimal_places"]),
            warnings=warnings,
            warning_code="WARN_GPS_PARSE_FAIL",
            lower=-90,
            upper=90,
        ),
        "gps_longitude": _parse_rounded_optional(
            raw_tags,
            "GPSLongitude",
            places=int(rounding_rules["gps_decimal_places"]),
            warnings=warnings,
            warning_code="WARN_GPS_PARSE_FAIL",
            lower=-180,
            upper=180,
        ),
        "focal_length_mm": _parse_rounded_optional(
            raw_tags,
            "FocalLength",
            places=int(rounding_rules["focal_length_decimal_places"]),
            warnings=warnings,
        ),
        "aperture_fnumber": _parse_rounded_optional(
            raw_tags,
            "FNumber",
            places=int(rounding_rules["aperture_decimal_places"]),
            warnings=warnings,
        ),
        "shutter_speed_seconds": _parse_rounded_optional(
            raw_tags,
            "ExposureTime",
            places=int(rounding_rules["shutter_speed_decimal_places"]),
            warnings=warnings,
        ),
        "exposure_compensation_ev": _parse_rounded_optional(
            raw_tags,
            "ExposureCompensation",
            places=int(rounding_rules["exposure_ev_decimal_places"]),
            warnings=warnings,
        ),
        "orientation": _map_orientation(raw_tags, dict(orientation_mapping), warnings),
        "extractor": {
            "name": extractor_name,
            "version": extractor_version,
            "config_fingerprint_sha256": config_fingerprint_sha256,
        },
    }
    record["extraction_warnings"] = _canonical_warning_codes(warnings, list(config["warning_codes"]))
    return record


def _validate_records(records: list[dict[str, Any]], schema_path: Path) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise SchemaValidationError("jsonschema dependency is required for metadata schema validation") from exc

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(f"unable to load metadata schema {schema_path}: {exc}") from exc

    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.exceptions.SchemaError as exc:
        raise SchemaValidationError(f"invalid schema {schema_path}: {exc.message}") from exc

    validator = jsonschema.Draft202012Validator(schema)
    for index, record in enumerate(records):
        errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        if errors:
            first = errors[0]
            path = ".".join(str(part) for part in first.path) or "<root>"
            raise SchemaValidationError(f"record[{index}] schema validation failed at {path}: {first.message}")


def extract_capture_metadata_records(
    *,
    input_root: Path,
    config: dict[str, Any],
    strict: bool,
    schema_path: Path,
    extractor_name: str = "extract_capture_metadata.py",
    extractor_version: str = "phase4c-v1",
    exif_runner: Callable[[list[Path], list[str]], dict[str, dict[str, Any]]] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Extract deterministic capture metadata records for files under input_root.

    Args:
        input_root: Root directory to discover image files from.
        config: Canonicalization configuration dictionary.
        strict: If True, fail on any extraction warnings.
        schema_path: Path to the metadata JSON schema for validation.
        extractor_name: Name to embed in extractor metadata.
        extractor_version: Version to embed in extractor metadata.
        exif_runner: Optional custom ExifTool runner function.
        progress_callback: Optional callback for progress reporting.
            Called with (current, total, message) during processing.

    Returns:
        List of validated capture metadata record dictionaries.
    """
    path_policy = config.get("path_normalization")
    if not isinstance(path_policy, dict):
        raise ConfigValidationError("path_normalization policy must be an object")

    if progress_callback:
        progress_callback(0, 0, "Discovering capture files...")

    discovered = _discover_capture_files(input_root, path_policy=path_policy)
    file_paths = [path for _, path in discovered]
    total_files = len(discovered)

    if progress_callback:
        progress_callback(0, total_files, f"Found {total_files} capture files, extracting EXIF metadata...")

    runner = exif_runner or _run_exiftool
    raw_by_path = runner(file_paths, list(config["tag_whitelist"]))
    fingerprint = compute_config_fingerprint_sha256(config)

    if progress_callback:
        progress_callback(0, total_files, "Building metadata records...")

    records: list[dict[str, Any]] = []
    for index, (relative_path, file_path) in enumerate(discovered):
        raw_tags = raw_by_path.get(str(file_path.resolve()), {})
        record = _build_metadata_object(
            relative_path=relative_path,
            file_path=file_path,
            raw_tags=raw_tags,
            config=config,
            config_fingerprint_sha256=fingerprint,
            extractor_name=extractor_name,
            extractor_version=extractor_version,
        )
        if strict and record["extraction_warnings"]:
            joined = ", ".join(record["extraction_warnings"])
            raise StrictWarningsError(f"strict mode failed for {relative_path}: {joined}")
        records.append(record)

        if progress_callback:
            progress_callback(index + 1, total_files, f"Processed {relative_path}")

    if progress_callback:
        progress_callback(total_files, total_files, "Validating records against schema...")

    _validate_records(records, schema_path=schema_path)

    if progress_callback:
        progress_callback(total_files, total_files, f"Extraction complete: {total_files} records")

    return records


def write_capture_metadata_artifact(records: list[dict[str, Any]], out_path: Path) -> None:
    """Write canonical JSON artifact for capture metadata records."""
    payload = (
        json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload)
