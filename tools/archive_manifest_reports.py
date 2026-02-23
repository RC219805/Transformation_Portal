#!/usr/bin/env python3
"""
archive_manifest_reports.py

Deterministic, manifest-first reporting tool for the Patrick W. Price "Archive_Manifest"
(ExifTool CSV) to support Phase 2: Ground Truth Vault + Spatial AI substrate.

INTEGRITY CONTRACT ANCHORS:
  1. Path Canonicalization:
     - Backslash → forward slash normalization
     - Repeated slash collapse (//+ → /)
     - Trailing slash removal
     - root_marker split with fallback to full path
     - Leading slash removal (lstrip "/") to prevent empty origin_drive from absolute paths
     - Empty relpath placeholder ("." for edge cases like SourceFile="/")
     - CRITICAL: origin_drive and partition derived from dir_rel (not relpath)
       to prevent drive-root file misclassification

  2. Grouping Determinism:
     - Explicit sort_values(["origin_drive", "basekey", "relpath"]) before groupby
     - Grouping by (origin_drive, basekey) to prevent cross-drive collisions
     - Stable pandas groupby(sort=True) for reproducibility

  3. Output Stability:
     - Explicit column ordering locked in code
     - encoding="utf-8", lineterminator="\n" for cross-platform byte-level reproducibility
     - Floats rounded to 6 decimals in JSON summary
     - Helper columns (is_*) dropped before CSV write

  4. Semantic Integrity Guards:
     - Required column validation (fail-fast on missing columns)
     - root_marker coverage threshold (warn by default, fail closed in strict mode)
     - filesize_approx_bytes_decimal: DIAGNOSTIC ONLY (decimal units, not forensic anchor)

  5. Chunk Splitting:
     - split_manifest.json emitted with part order + original_size for recombination verification
     - Atomic file write pattern (temp → rename)

  6. Risk Scoring:
     - risk_model_version and risk_weights documented in summary.json
     - Future weight changes must bump version

Inputs:
  - ExifTool CSV with at least these columns:
      SourceFile, FileName, FileSize, Model, DateTimeOriginal, ImageSize, Quality,
      FocalLength, ShutterSpeed, Aperture, ISO, WhiteBalance, Flash

Outputs (in --outdir):
  - archive_index_normalized.csv.gz
      Per-file normalized index (path normalization, role/category tagging, approx bytes).
  - asset_grouping_report.csv.gz
      One row per (origin_drive + directory + basename) group with counts & anomaly flags.
  - anomaly_hotspots.csv
      Partition-level aggregation for triage (by origin_drive + partition).
  - summary.json
      High-signal rollups + risk model version + weights.

Optional:
  - --by-drive : emit per-origin-drive partitions under outdir/by_origin_drive/
  - --chunk-mb : split large outputs into .partNNN files for transport (recombine with cat)

Dependencies:
  - Requires numpy and pandas (pip install -r requirements/tools-archive.txt for pinned versions)

Determinism properties:
  - Stable path canonicalization
  - Stable groupby ordering (sort=True)
  - Stable column ordering for outputs
  - Cross-platform byte-level reproducibility (UTF-8 + LF line endings)

NOTE: This tool is intentionally "no hashes" because it operates on the ExifTool manifest only.
      Integrity hashing (SHA-256) should be implemented as a separate vault ingest step.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "archive_manifest_reports.py requires optional dependencies 'numpy' and 'pandas'. "
        "Install pinned tool dependencies with: pip install -r requirements/tools-archive.txt "
        "(or, if you explicitly prefer unpinned versions, run: pip install numpy pandas)."
    ) from exc

_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]+)\s*$")

DETERMINISTIC_GZIP_COMPRESSION = {"method": "gzip", "compresslevel": 9, "mtime": 0}
DEFAULT_MIN_ROOT_MARKER_COVERAGE = 0.95

ENUM_DOMAIN_DEFAULTS: Dict[str, Dict[str, set[str]]] = {
    "archive_index_normalized.csv.gz": {
        "role": {"primary_raw", "primary_raster", "primary_video", "sidecar", "other"},
        "category": {"RAW", "JPEG", "TIFF_non_RAW", "Video", "XMP", "THM", "Other"},
    },
    "asset_grouping_report.csv.gz": {
        "asset_type": {"raw", "raster", "video", "raw+raster", "sidecar_only", "other", "hybrid"},
    },
}


def atomic_write(path: Path, writer_func: Callable[[Path], None]) -> None:
    """Write path atomically by writing to a temp file then replacing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        writer_func(tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_csv_atomic(df: pd.DataFrame, path: Path, *, compression: Optional[str | Dict[str, Any]] = None) -> None:
    """Write CSV deterministically and atomically."""

    def _write(tmp_path: Path) -> None:
        csv_compression: Optional[str | Dict[str, Any]] = compression
        if compression == "gzip":
            csv_compression = dict(DETERMINISTIC_GZIP_COMPRESSION)
        elif isinstance(compression, dict) and compression.get("method") == "gzip":
            csv_compression = dict(DETERMINISTIC_GZIP_COMPRESSION)
            csv_compression.update(compression)
            csv_compression["mtime"] = 0

        df.to_csv(
            tmp_path,
            index=False,
            compression=csv_compression,
            encoding="utf-8",
            lineterminator="\n",
        )

    atomic_write(path, _write)


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2, sort_keys: bool = True) -> None:
    """Write JSON deterministically and atomically."""

    def _write(tmp_path: Path) -> None:
        with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=indent, sort_keys=sort_keys)

    atomic_write(path, _write)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write raw bytes deterministically and atomically."""

    def _write(tmp_path: Path) -> None:
        with tmp_path.open("wb") as handle:
            handle.write(data)

    atomic_write(path, _write)


def _enum_domains_for_output(out_file: str, schema: Dict[str, Any]) -> Dict[str, set[str]]:
    """Extract enum domain requirements from schema plus governance defaults."""
    enum_domains = {col: set(vals) for col, vals in ENUM_DOMAIN_DEFAULTS.get(out_file, {}).items()}
    for col_spec in schema.get("columns", []):
        if not isinstance(col_spec, dict):
            continue
        col_name = col_spec.get("name")
        enum_values = col_spec.get("enum")
        if isinstance(col_name, str) and isinstance(enum_values, list):
            enum_domains[col_name] = {str(v) for v in enum_values}
    return enum_domains


def _read_csv_header_and_invalid_enums(
    out_file: str,
    out_path: Path,
    schema: Dict[str, Any],
) -> tuple[List[str], Dict[str, set[str]]]:
    """Read CSV header and collect invalid enum values for schema-governed columns."""
    enum_domains = _enum_domains_for_output(out_file, schema)
    invalid_values: Dict[str, set[str]] = {}

    if out_file.endswith(".gz"):
        with gzip.open(out_path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            actual_cols_ordered = list(reader.fieldnames or [])
            cols_to_check = {k: v for k, v in enum_domains.items() if k in actual_cols_ordered}
            if cols_to_check:
                for row in reader:
                    for col_name, allowed in cols_to_check.items():
                        value = row.get(col_name, "")
                        if value not in allowed:
                            invalid_values.setdefault(col_name, set()).add(value)
    else:
        with out_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            actual_cols_ordered = list(reader.fieldnames or [])
            cols_to_check = {k: v for k, v in enum_domains.items() if k in actual_cols_ordered}
            if cols_to_check:
                for row in reader:
                    for col_name, allowed in cols_to_check.items():
                        value = row.get(col_name, "")
                        if value not in allowed:
                            invalid_values.setdefault(col_name, set()).add(value)

    return actual_cols_ordered, invalid_values


def validate_outputs_against_schemas(outdir: Path) -> None:
    """Validate CSV outputs against documented column schemas."""
    schema_dir = Path(__file__).parent.parent / "docs" / "archive" / "schemas"

    validations = {
        "archive_index_normalized.csv.gz": "archive_index_normalized.schema.json",
        "asset_grouping_report.csv.gz": "asset_grouping_report.schema.json",
        "anomaly_hotspots.csv": "anomaly_hotspots.schema.json",
    }

    for out_file, schema_file in validations.items():
        out_path = outdir / out_file
        schema_path = schema_dir / schema_file

        if not out_path.exists():
            raise SystemExit(f"Missing output file: {out_file}")
        if not schema_path.exists():
            raise SystemExit(f"Missing schema file: {schema_file}")

        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        expected_cols_ordered = [c["name"] for c in schema["columns"]]
        expected_cols = set(expected_cols_ordered)
        actual_cols_ordered, invalid_enum_values = _read_csv_header_and_invalid_enums(out_file, out_path, schema)

        actual_cols = set(actual_cols_ordered)

        if actual_cols != expected_cols:
            raise SystemExit(
                f"Schema mismatch in {out_file}\n"
                f"Expected columns (set): {sorted(expected_cols)}\n"
                f"Actual columns (set):   {sorted(actual_cols)}"
            )

        if actual_cols_ordered != expected_cols_ordered:
            raise SystemExit(
                f"Column order mismatch in {out_file}\n"
                f"Expected order: {expected_cols_ordered}\n"
                f"Actual order:   {actual_cols_ordered}"
            )

        if invalid_enum_values:
            pretty: Dict[str, List[str]] = {}
            for col_name, vals in invalid_enum_values.items():
                pretty[col_name] = sorted("<empty>" if v == "" else str(v) for v in vals)
            raise SystemExit(f"Enum domain mismatch in {out_file}\n" f"Invalid values: {json.dumps(pretty, sort_keys=True)}")

        print(f"✅ Schema validation passed: {out_file}")


def parse_filesize_to_bytes(s: object) -> Optional[int]:
    """Parse ExifTool human-readable FileSize to approximate bytes (decimal units).

    INTEGRITY NOTE: This uses DECIMAL multipliers (MB = 10^6, not 2^20).
    Results are approximate and MUST NOT be used as forensic integrity anchors.
    For audit-grade integrity, use hash-first vault layer with exact byte counts.

    Args:
        s: FileSize string from ExifTool (e.g., "10 MB", "2.5 GB", "100 bytes")

    Returns:
        Approximate byte count (int) or None if parsing fails
    """
    if s is None:
        return None
    if isinstance(s, float) and np.isnan(s):
        return None
    s = str(s).strip()
    if not s:
        return None

    m = _SIZE_RE.match(s)
    if not m:
        if s.lower().endswith("bytes"):
            try:
                return int(float(s.lower().replace("bytes", "").strip()))
            except (TypeError, ValueError):
                return None
        return None

    num = float(m.group(1))
    unit = m.group(2).lower()

    if unit in {"byte", "bytes", "b"}:
        mult = 1
    elif unit in {"kb", "k"}:
        mult = 10**3
    elif unit == "kib":
        mult = 2**10
    elif unit in {"mb", "m"}:
        mult = 10**6
    elif unit == "mib":
        mult = 2**20
    elif unit in {"gb", "g"}:
        mult = 10**9
    elif unit == "gib":
        mult = 2**30
    elif unit in {"tb", "t"}:
        mult = 10**12
    elif unit == "tib":
        mult = 2**40
    else:
        return None

    return int(round(num * mult))


def split_file(path: Path, chunk_bytes: int) -> tuple[list[str], int]:
    """Split a file into .partNNN chunks.

    Returns:
        (parts, original_size) for integrity verification during recombination.
    """
    parts: List[str] = []
    original_size = path.stat().st_size
    with path.open("rb") as f:
        i = 1
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            part_path = path.with_name(f"{path.name}.part{i:03d}")
            write_bytes_atomic(part_path, chunk)
            parts.append(str(part_path))
            i += 1
    return parts, original_size


def build_reports(
    input_csv: str,
    outdir: str,
    root_marker: str = "All Archive/",
    by_drive: bool = True,
    chunk_mb: int = 0,
    validate_schemas: bool = False,
    strict_root_marker: bool = False,
    min_root_marker_coverage: float = DEFAULT_MIN_ROOT_MARKER_COVERAGE,
) -> Dict[str, object]:
    """Build deterministic archive index, asset grouping, and hotspot reports.

    INTEGRITY CONTRACT:
    - Deterministic path canonicalization (backslash→slash, collapse //, strip trailing /)
    - Fail-fast validation: required columns must be present
    - origin_drive and partition derived from dir_rel (not relpath) to prevent misclassification
    - Explicit sort before groupby for reproducible ordering
    - Group by (origin_drive, basekey) to prevent cross-drive collisions
    - Explicit column ordering + UTF-8 + LF line endings for byte-level reproducibility
    - Floats rounded to 6 decimals in JSON for cross-version stability

    Args:
        input_csv: Path to ExifTool CSV manifest
        outdir: Output directory for reports (created if missing)
        root_marker: Substring used to strip prefix from SourceFile (default: "All Archive/")
        by_drive: If True, emit per-drive partitions in outdir/by_origin_drive/
        chunk_mb: If >0, split large outputs into N MB chunks
        validate_schemas: If True, validate output CSVs against documented column schemas
        strict_root_marker: If True, fail when root_marker coverage is below threshold
        min_root_marker_coverage: Minimum required fraction of rows containing root_marker

    Returns:
        Dict mapping output keys to file paths (+ split_manifest if chunking enabled)

    Raises:
        SystemExit: If required columns are missing or root_marker validation fails
    """
    input_csv_path = Path(input_csv)
    outdir_path = Path(outdir)
    df = pd.read_csv(input_csv_path, dtype=str, keep_default_na=False)

    if not 0.0 <= min_root_marker_coverage <= 1.0:
        raise SystemExit("--min-root-marker-coverage must be between 0.0 and 1.0")

    normalized_root_marker = re.sub(r"/+", "/", root_marker.replace("\\", "/")).strip("/")
    if not normalized_root_marker:
        raise SystemExit("--root-marker must include at least one non-slash character")
    root_marker = f"{normalized_root_marker}/"

    # Validate required columns (fail-fast)
    required_cols = {
        "SourceFile",
        "FileName",
        "FileSize",
        "Model",
        "DateTimeOriginal",
        "ImageSize",
        "Quality",
        "FocalLength",
        "ShutterSpeed",
        "Aperture",
        "ISO",
        "WhiteBalance",
        "Flash",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    # Normalize path with deterministic canonicalization
    sf = df["SourceFile"].astype(str).str.replace("\\", "/", regex=False)
    # Collapse repeated slashes and strip trailing slashes
    sf = sf.str.replace(r"/+", "/", regex=True).str.rstrip("/")

    marker_found = sf.str.contains(root_marker, regex=False)
    rel = sf.str.split(root_marker, n=1, expand=True)
    if marker_found.any() and rel.shape[1] > 1:
        relpath = np.where(marker_found, rel[1].fillna("").values, sf.values)
    else:
        relpath = sf.values
    marker_coverage = float(marker_found.mean()) if len(df) > 0 else 0.0

    relpath = pd.Series(relpath, name="relpath").astype(str)
    # CRITICAL: Strip leading slashes to prevent empty origin_drive when root_marker is missing
    # and fallback produces absolute paths (e.g., /vault/All Archive/DriveA/...)
    # This stabilizes absolute-path fallback and prevents cross-environment drift.
    relpath = relpath.str.lstrip("/")
    empty_relpath = relpath.eq("") | relpath.isna()
    if empty_relpath.any():
        if strict_root_marker:
            raise SystemExit(f"Found {int(empty_relpath.sum())} rows with empty relpath after canonicalization.")
        # Handle edge case: if relpath becomes empty after lstrip (e.g., SourceFile was just "/"),
        # replace with a placeholder to prevent downstream empty-string issues
        relpath = relpath.mask(empty_relpath, ".")

    # Validate root_marker coverage. Strict mode fails closed; default mode warns.
    if marker_coverage < min_root_marker_coverage:
        unmatched_examples = sf[~marker_found].head(5).tolist()
        coverage_msg = (
            f"root_marker coverage {marker_coverage:.3f} < {min_root_marker_coverage:.3f} " f"for marker '{root_marker}'"
        )
        if strict_root_marker:
            raise SystemExit(
                f"{coverage_msg}. Non-comparable run context. " f"Examples (unmatched SourceFile): {unmatched_examples}"
            )
        print(
            f"Warning: {coverage_msg}. Outputs may be system-context-dependent. "
            f"Examples (unmatched SourceFile): {unmatched_examples}",
            file=sys.stderr,
        )

    # Split relpath into (dir_rel, filename) while handling no-separator rows.
    split_path = relpath.str.rsplit("/", n=1, expand=True)
    if split_path.shape[1] == 1:
        dir_rel = pd.Series("", index=relpath.index, dtype=str)
        filename = split_path[0].fillna("")
    else:
        has_sep = relpath.str.contains("/", regex=False)
        dir_rel = split_path[0].where(has_sep, "").fillna("")
        filename = split_path[1].where(has_sep, split_path[0]).fillna("")

    # CRITICAL: Derive origin_drive and partition from dir_rel (not relpath)
    # to avoid misclassifying drive-root files as partitions
    parts = dir_rel.str.split("/", n=2, expand=True)
    if parts.shape[1] >= 1:
        origin_drive = parts[0].fillna("")
    else:
        origin_drive = pd.Series("", index=dir_rel.index)

    if parts.shape[1] >= 2:
        partition = parts[1].fillna("")
    else:
        partition = pd.Series("", index=dir_rel.index)

    ext = filename.str.extract(r"\.([^\.]+)$", expand=False).fillna("").str.lower()
    basename = filename.str.replace(r"\.[^\.]+$", "", regex=True)
    basekey = (dir_rel + "/" + basename).str.strip("/")

    filesize_bytes = df["FileSize"].apply(parse_filesize_to_bytes).astype("Int64")

    dto = df["DateTimeOriginal"].astype(str)
    dt_missing = (dto == "") | dto.str.lower().isin({"nan", "none", "null", "-"})

    quality = df["Quality"].astype(str)
    quality_is_raw = quality.str.upper().eq("RAW")

    raw_exts = {"cr2", "crw", "dng", "nef", "arw"}
    video_exts = {"mov", "mp4", "m4v", "dv", "mts"}
    raster_exts = {"jpg", "jpeg", "tif", "tiff", "png", "heic", "psd"}
    sidecar_exts = {"xmp", "thm"}

    is_raw = ext.isin(raw_exts) | ((ext.isin({"tif", "tiff"})) & quality_is_raw)
    is_video = ext.isin(video_exts)
    is_sidecar = ext.isin(sidecar_exts)
    is_raster = ext.isin(raster_exts) & (~is_raw) & (~is_video) & (~is_sidecar)

    role = np.select(
        [is_sidecar, is_video, is_raw, is_raster],
        ["sidecar", "primary_video", "primary_raw", "primary_raster"],
        default="other",
    )

    category = np.select(
        [
            ext.isin({"xmp"}),
            ext.isin({"thm"}),
            is_video,
            is_raw,
            ext.isin({"jpg", "jpeg"}),
            ext.isin({"tif", "tiff"}) & (~quality_is_raw),
        ],
        ["XMP", "THM", "Video", "RAW", "JPEG", "TIFF_non_RAW"],
        default="Other",
    )

    dir_split = dir_rel.str.split("/", n=1, expand=True)
    if dir_split.shape[1] > 1:
        dir_within_drive = dir_split[1].fillna("")
    else:
        dir_within_drive = pd.Series("", index=dir_rel.index, dtype=str)

    archive_index = pd.DataFrame(
        {
            "SourceFile": df["SourceFile"],
            "relpath": relpath,
            "origin_drive": origin_drive,
            "partition": partition,
            "dir_within_drive": dir_within_drive,
            "FileName": df["FileName"],
            "ext": ext,
            "basename": basename,
            "basekey": basekey,
            "role": role,
            "category": category,
            "filesize_approx_bytes_decimal": filesize_bytes,
            "Model": df["Model"],
            "DateTimeOriginal": df["DateTimeOriginal"],
            "dt_missing": dt_missing.astype(bool),
            "ImageSize": df["ImageSize"],
            "Quality": df["Quality"],
            "FocalLength": df["FocalLength"],
            "ShutterSpeed": df["ShutterSpeed"],
            "Aperture": df["Aperture"],
            "ISO": df["ISO"],
            "WhiteBalance": df["WhiteBalance"],
            "Flash": df["Flash"],
        }
    )

    # Flags for grouping
    archive_index["is_primary_raw"] = archive_index["role"].eq("primary_raw")
    archive_index["is_primary_raster"] = archive_index["role"].eq("primary_raster")
    archive_index["is_primary_video"] = archive_index["role"].eq("primary_video")
    archive_index["is_xmp"] = archive_index["ext"].eq("xmp")
    archive_index["is_thm"] = archive_index["ext"].eq("thm")
    archive_index["is_sidecar"] = archive_index["role"].eq("sidecar")
    archive_index["is_other"] = archive_index["role"].eq("other")
    archive_index["is_jpeg"] = archive_index["ext"].isin(["jpg", "jpeg"])

    # Lock determinism: explicit sort before groupby
    archive_index = archive_index.sort_values(["origin_drive", "basekey", "relpath"]).reset_index(drop=True)

    # CRITICAL: Group by (origin_drive, basekey) to avoid cross-drive collisions
    g = archive_index.groupby(["origin_drive", "basekey"], sort=True)
    g_size = g.size()

    asset = pd.DataFrame(
        {
            "origin_drive": [key[0] for key in g_size.index],
            "basekey": [key[1] for key in g_size.index],
            "n_files": g_size.values,
            "partition": g["partition"].first().values,
            "dir_rel": g["dir_within_drive"].first().values,
            "basename": g["basename"].first().values,
            "n_raw_files": g["is_primary_raw"].sum().astype(int).values,
            "n_raster_files": g["is_primary_raster"].sum().astype(int).values,
            "n_video_files": g["is_primary_video"].sum().astype(int).values,
            "n_jpeg_files": g["is_jpeg"].sum().astype(int).values,
            "n_xmp": g["is_xmp"].sum().astype(int).values,
            "n_thm": g["is_thm"].sum().astype(int).values,
            "n_sidecar": g["is_sidecar"].sum().astype(int).values,
            "n_other_files": g["is_other"].sum().astype(int).values,
            "any_dt_missing": g["dt_missing"].any().values,
        }
    )

    has_raw = asset["n_raw_files"] > 0
    has_raster = asset["n_raster_files"] > 0
    has_video = asset["n_video_files"] > 0
    has_primary = has_raw | has_raster | has_video
    has_sidecar = asset["n_sidecar"] > 0
    has_other = asset["n_other_files"] > 0

    asset["asset_type"] = np.select(
        [
            has_raw & ~has_raster & ~has_video & ~has_other,
            has_raster & ~has_raw & ~has_video & ~has_other,
            has_video & ~has_raw & ~has_raster & ~has_other,
            has_raw & has_raster & ~has_video & ~has_other,
            ~has_primary & has_sidecar & ~has_other,
            has_other & ~has_raw & ~has_raster & ~has_video,
        ],
        [
            "raw",
            "raster",
            "video",
            "raw+raster",
            "sidecar_only",
            "other",
        ],
        default="hybrid",
    )

    # Anomaly flags
    asset["flag_xmp_orphan_no_raw_jpeg"] = (asset["n_xmp"] > 0) & ((asset["n_raw_files"] + asset["n_jpeg_files"]) == 0)
    asset["flag_xmp_orphan_no_image_any_raster"] = (asset["n_xmp"] > 0) & (
        (asset["n_raw_files"] + asset["n_raster_files"]) == 0
    )
    asset["flag_sidecar_only"] = asset["asset_type"].eq("sidecar_only")
    asset["flag_video_still_collision"] = (asset["n_video_files"] > 0) & ((asset["n_raw_files"] + asset["n_raster_files"]) > 0)
    asset["flag_container_imovie"] = asset["dir_rel"].str.contains(r"\.imovielibrary", case=False, regex=True)
    asset["flag_still_missing_datetime"] = ((asset["n_raw_files"] + asset["n_raster_files"]) > 0) & (asset["any_dt_missing"])

    # Hotspots (by origin_drive + partition)
    hot = (
        asset.groupby(["origin_drive", "partition"], sort=True)
        .agg(
            asset_groups=("basekey", "size"),
            xmp_orphan_raw_jpeg=("flag_xmp_orphan_no_raw_jpeg", "sum"),
            sidecar_only=("flag_sidecar_only", "sum"),
            video_still_collision=("flag_video_still_collision", "sum"),
            imovie_container=("flag_container_imovie", "sum"),
            still_missing_datetime=("flag_still_missing_datetime", "sum"),
            other_only=("asset_type", lambda s: int((s == "other").sum())),
            hybrid=("asset_type", lambda s: int((s == "hybrid").sum())),
        )
        .reset_index()
    )

    file_agg = (
        archive_index.groupby(["origin_drive", "partition"], sort=True)
        .agg(
            files=("relpath", "size"),
            approx_bytes_decimal=(
                "filesize_approx_bytes_decimal",
                lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum(),
            ),
            raw_files=("category", lambda s: int((s == "RAW").sum())),
            jpeg_files=("category", lambda s: int((s == "JPEG").sum())),
            video_files=("category", lambda s: int((s == "Video").sum())),
            xmp_files=("category", lambda s: int((s == "XMP").sum())),
            thm_files=("category", lambda s: int((s == "THM").sum())),
            other_files=("category", lambda s: int((s == "Other").sum())),
            dt_missing_files=("dt_missing", lambda s: int(s.astype(bool).sum())),
        )
        .reset_index()
    )

    hotspots = hot.merge(file_agg, on=["origin_drive", "partition"], how="left")
    hotspots["approx_TB"] = hotspots["approx_bytes_decimal"] / 1e12
    hotspots["risk_score"] = (
        hotspots["xmp_orphan_raw_jpeg"] * 5
        + hotspots["sidecar_only"] * 4
        + hotspots["imovie_container"] * 3
        + hotspots["still_missing_datetime"] * 1
        + hotspots["video_still_collision"] * 10
        + hotspots["hybrid"] * 2
    )
    hotspots = hotspots.sort_values(["risk_score", "xmp_orphan_raw_jpeg", "sidecar_only", "files"], ascending=False)

    # Output paths
    outdir_path.mkdir(parents=True, exist_ok=True)
    out_archive_index = outdir_path / "archive_index_normalized.csv.gz"
    out_asset_groups = outdir_path / "asset_grouping_report.csv.gz"
    out_hotspots = outdir_path / "anomaly_hotspots.csv"
    out_summary = outdir_path / "summary.json"

    helper_cols = [
        "is_primary_raw",
        "is_primary_raster",
        "is_primary_video",
        "is_xmp",
        "is_thm",
        "is_sidecar",
        "is_other",
        "is_jpeg",
    ]
    archive_index_out = archive_index.drop(columns=helper_cols)

    # Lock explicit column ordering for CSV stability
    archive_index_col_order = [
        "SourceFile",
        "relpath",
        "origin_drive",
        "partition",
        "dir_within_drive",
        "FileName",
        "ext",
        "basename",
        "basekey",
        "role",
        "category",
        "filesize_approx_bytes_decimal",
        "Model",
        "DateTimeOriginal",
        "dt_missing",
        "ImageSize",
        "Quality",
        "FocalLength",
        "ShutterSpeed",
        "Aperture",
        "ISO",
        "WhiteBalance",
        "Flash",
    ]
    archive_index_out = archive_index_out[archive_index_col_order]

    asset_col_order = [
        "origin_drive",
        "basekey",
        "n_files",
        "partition",
        "dir_rel",
        "basename",
        "n_raw_files",
        "n_raster_files",
        "n_video_files",
        "n_jpeg_files",
        "n_xmp",
        "n_thm",
        "n_sidecar",
        "n_other_files",
        "any_dt_missing",
        "asset_type",
        "flag_xmp_orphan_no_raw_jpeg",
        "flag_xmp_orphan_no_image_any_raster",
        "flag_sidecar_only",
        "flag_video_still_collision",
        "flag_container_imovie",
        "flag_still_missing_datetime",
    ]
    asset = asset[asset_col_order]

    write_csv_atomic(archive_index_out, out_archive_index, compression=DETERMINISTIC_GZIP_COMPRESSION)
    write_csv_atomic(asset, out_asset_groups, compression=DETERMINISTIC_GZIP_COMPRESSION)
    write_csv_atomic(hotspots, out_hotspots)

    # Summary metrics
    video_rows = archive_index_out[archive_index_out["category"] == "Video"]
    by_ext = (
        video_rows.groupby("ext")
        .agg(total=("ext", "size"), missing=("dt_missing", "sum"))
        .assign(missing_rate=lambda d: (d["missing"] / d["total"]).round(6))
        .sort_values("total", ascending=False)
        .reset_index()
        .to_dict(orient="records")
    )

    summary = {
        "rows_total_files": int(len(archive_index_out)),
        "rows_total_asset_groups": int(len(asset)),
        "orphan_xmp_groups_raw_jpeg": int(asset["flag_xmp_orphan_no_raw_jpeg"].sum()),
        "orphan_xmp_groups_no_image_any_raster": int(asset["flag_xmp_orphan_no_image_any_raster"].sum()),
        "sidecar_only_groups": int(asset["flag_sidecar_only"].sum()),
        "asset_type_counts": asset["asset_type"].value_counts().to_dict(),
        "video_datetimeoriginal_missing_rate": round(float(video_rows["dt_missing"].mean()), 6) if len(video_rows) else None,
        "video_datetimeoriginal_missing_by_ext": by_ext,
        "risk_model_version": "1.0",
        "risk_weights": {
            "xmp_orphan_raw_jpeg": 5,
            "sidecar_only": 4,
            "imovie_container": 3,
            "still_missing_datetime": 1,
            "video_still_collision": 10,
            "hybrid": 2,
        },
    }
    write_json_atomic(out_summary, summary, indent=2, sort_keys=True)

    # Validate schema if requested
    if validate_schemas:
        validate_outputs_against_schemas(outdir_path)

    outputs = {
        "archive_index_gz": str(out_archive_index),
        "asset_groups_gz": str(out_asset_groups),
        "hotspots_csv": str(out_hotspots),
        "summary_json": str(out_summary),
    }

    # Optional: partitions by drive
    if by_drive:
        out_by_drive = outdir_path / "by_origin_drive"
        out_by_drive.mkdir(parents=True, exist_ok=True)

        drive_values = sorted(archive_index_out["origin_drive"].unique())
        used_drive_labels: set[str] = set()
        for drv in drive_values:
            raw_label = drv if drv else "__ROOT__"
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_label)[:80]
            drive_hash = hashlib.sha1(raw_label.encode("utf-8")).hexdigest()[:8]

            # Avoid ambiguous/empty labels and prevent collisions after sanitize+truncate.
            if not safe or not safe.strip("_"):
                safe = f"drive__{drive_hash}"

            if safe in used_drive_labels:
                suffix = f"__{drive_hash}"
                safe = f"{safe[: max(1, 80 - len(suffix))]}{suffix}"

            used_drive_labels.add(safe)

            ai_path = out_by_drive / f"archive_index__{safe}.csv.gz"
            ag_path = out_by_drive / f"asset_groups__{safe}.csv.gz"
            write_csv_atomic(
                archive_index_out.loc[archive_index_out["origin_drive"] == drv],
                ai_path,
                compression=DETERMINISTIC_GZIP_COMPRESSION,
            )
            write_csv_atomic(
                asset.loc[asset["origin_drive"] == drv],
                ag_path,
                compression=DETERMINISTIC_GZIP_COMPRESSION,
            )

        outputs["by_drive_dir"] = str(out_by_drive)

    # Optional: split outputs into chunks
    if chunk_mb and chunk_mb > 0:
        chunk_bytes = int(chunk_mb) * 1024 * 1024
        split_manifest: Dict[str, Dict[str, object]] = {}
        for k, p in outputs.items():
            if not isinstance(p, str):
                continue
            output_path = Path(p)
            if not output_path.is_file():
                continue
            if output_path.stat().st_size > chunk_bytes:
                parts, orig_size = split_file(output_path, chunk_bytes)
                split_manifest[str(output_path)] = {
                    "parts": parts,
                    "original_size": orig_size,
                    "part_count": len(parts),
                }
        if split_manifest:
            outputs["split_manifest"] = split_manifest
            # Write split manifest to JSON for recombination verification
            split_manifest_path = outdir_path / "split_manifest.json"
            write_json_atomic(split_manifest_path, split_manifest, indent=2, sort_keys=True)
            outputs["split_manifest_json"] = str(split_manifest_path)

    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to archive_manifest.csv (ExifTool output)")
    ap.add_argument("--outdir", required=True, help="Output directory for reports")
    ap.add_argument("--root-marker", default="All Archive/", help="Substring used to strip prefix from SourceFile")
    ap.add_argument("--no-by-drive", action="store_true", help="Disable per-drive partition outputs")
    ap.add_argument("--chunk-mb", type=int, default=0, help="If >0, split large outputs into N MB parts")
    ap.add_argument(
        "--strict-root-marker",
        action="store_true",
        default=False,
        help=(
            "Fail closed when root marker coverage is below --min-root-marker-coverage "
            "or canonicalization yields empty relpath rows"
        ),
    )
    ap.add_argument(
        "--min-root-marker-coverage",
        type=float,
        default=DEFAULT_MIN_ROOT_MARKER_COVERAGE,
        help="Minimum required fraction [0.0-1.0] of rows containing root_marker (default: 0.95)",
    )
    ap.add_argument(
        "--validate-schemas",
        action="store_true",
        default=False,
        help="Validate all generated CSV outputs against column schemas " "(fails fast on schema drift)",
    )
    args = ap.parse_args()

    outputs = build_reports(
        input_csv=args.input,
        outdir=args.outdir,
        root_marker=args.root_marker,
        by_drive=not args.no_by_drive,
        chunk_mb=args.chunk_mb,
        validate_schemas=args.validate_schemas,
        strict_root_marker=args.strict_root_marker,
        min_root_marker_coverage=args.min_root_marker_coverage,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
