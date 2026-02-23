#!/usr/bin/env python3
"""
archive_manifest_reports.py

Deterministic, manifest-first reporting tool for the Patrick W. Price "Archive_Manifest"
(ExifTool CSV) to support Phase 2: Ground Truth Vault + Spatial AI substrate.

Inputs:
  - ExifTool CSV with at least these columns:
      SourceFile, FileName, FileSize, Model, DateTimeOriginal, ImageSize, Quality,
      FocalLength, ShutterSpeed, Aperture, ISO, WhiteBalance, Flash

Outputs (in --outdir):
  - archive_index_normalized.csv.gz
      Per-file normalized index (path normalization, role/category tagging, approx bytes).
  - asset_grouping_report.csv.gz
      One row per (directory + basename) group ("basekey") with counts & anomaly flags.
  - anomaly_hotspots.csv
      Partition-level aggregation for triage (by origin_drive + partition).
  - summary.json
      High-signal rollups.

Optional:
  - --by-drive : emit per-origin-drive partitions under outdir/by_origin_drive/
  - --chunk-mb : split large outputs into .partNNN files for transport (recombine with cat)
  - Requires optional dependencies: numpy, pandas

Determinism properties:
  - Stable path canonicalization
  - Stable groupby ordering (sort=True)
  - Stable column ordering for outputs

NOTE: This tool is intentionally "no hashes" because it operates on the ExifTool manifest only.
      Integrity hashing (SHA-256) should be implemented as a separate vault ingest step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "archive_manifest_reports.py requires optional dependencies 'numpy' and 'pandas'. "
        "Install them with: pip install numpy pandas"
    ) from exc

_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]+)\s*$")


def parse_filesize_to_bytes(s: object) -> Optional[int]:
    """Parse ExifTool human-readable FileSize to approximate bytes (decimal units)."""
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
            except Exception:
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


def split_file(path: str, chunk_bytes: int) -> List[str]:
    """Split a file into .partNNN chunks."""
    parts: List[str] = []
    with open(path, "rb") as f:
        i = 1
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            part_path = f"{path}.part{i:03d}"
            with open(part_path, "wb") as pf:
                pf.write(chunk)
            parts.append(part_path)
            i += 1
    return parts


def build_reports(
    input_csv: str,
    outdir: str,
    root_marker: str = "All Archive/",
    by_drive: bool = True,
    chunk_mb: int = 0,
) -> Dict[str, object]:
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)

    # Normalize path
    sf = df["SourceFile"].astype(str).str.replace("\\", "/", regex=False)
    rel = sf.str.split(root_marker, n=1, expand=True)
    relpath = np.where(rel.shape[1] > 1, rel[1], sf.values)
    relpath = pd.Series(relpath, name="relpath").astype(str)

    parts = relpath.str.split("/", n=2, expand=True)
    origin_drive = parts[0].fillna("")
    partition = parts[1].fillna("")

    # Split relpath into (dir_rel, filename) while handling no-separator rows.
    split_path = relpath.str.rsplit("/", n=1, expand=True)
    if split_path.shape[1] == 1:
        dir_rel = pd.Series("", index=relpath.index, dtype=str)
        filename = split_path[0].fillna("")
    else:
        has_sep = relpath.str.contains("/", regex=False)
        dir_rel = split_path[0].where(has_sep, "").fillna("")
        filename = split_path[1].where(has_sep, split_path[0]).fillna("")

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

    archive_index = pd.DataFrame(
        {
            "SourceFile": df["SourceFile"],
            "relpath": relpath,
            "origin_drive": origin_drive,
            "partition": partition,
            "dir_within_drive": dir_rel.str.split("/", n=1, expand=True)[1].fillna(""),
            "FileName": df["FileName"],
            "ext": ext,
            "basename": basename,
            "basekey": basekey,
            "role": role,
            "category": category,
            "filesize_approx_bytes": filesize_bytes,
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

    g = archive_index.groupby("basekey", sort=True)

    asset = pd.DataFrame(
        {
            "basekey": g.size().index,
            "n_files": g.size().values,
            "origin_drive": g["origin_drive"].first().values,
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
            approx_bytes=("filesize_approx_bytes", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
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
    hotspots["approx_TB"] = hotspots["approx_bytes"] / 1e12
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
    os.makedirs(outdir, exist_ok=True)
    out_archive_index = os.path.join(outdir, "archive_index_normalized.csv.gz")
    out_asset_groups = os.path.join(outdir, "asset_grouping_report.csv.gz")
    out_hotspots = os.path.join(outdir, "anomaly_hotspots.csv")
    out_summary = os.path.join(outdir, "summary.json")

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

    archive_index_out.to_csv(out_archive_index, index=False, compression="gzip")
    asset.to_csv(out_asset_groups, index=False, compression="gzip")
    hotspots.to_csv(out_hotspots, index=False)

    # Summary metrics
    video_rows = archive_index_out[archive_index_out["category"] == "Video"]
    by_ext = (
        video_rows.groupby("ext")
        .agg(total=("ext", "size"), missing=("dt_missing", "sum"))
        .assign(missing_rate=lambda d: d["missing"] / d["total"])
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
        "video_datetimeoriginal_missing_rate": float(video_rows["dt_missing"].mean()) if len(video_rows) else None,
        "video_datetimeoriginal_missing_by_ext": by_ext,
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    outputs = {
        "archive_index_gz": out_archive_index,
        "asset_groups_gz": out_asset_groups,
        "hotspots_csv": out_hotspots,
        "summary_json": out_summary,
    }

    # Optional: partitions by drive
    if by_drive:
        out_by_drive = os.path.join(outdir, "by_origin_drive")
        os.makedirs(out_by_drive, exist_ok=True)

        drive_values = sorted(archive_index_out["origin_drive"].unique())
        for drv in drive_values:
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", drv)[:80]
            ai_path = os.path.join(out_by_drive, f"archive_index__{safe}.csv.gz")
            ag_path = os.path.join(out_by_drive, f"asset_groups__{safe}.csv.gz")
            archive_index_out.loc[archive_index_out["origin_drive"] == drv].to_csv(ai_path, index=False, compression="gzip")
            asset.loc[asset["origin_drive"] == drv].to_csv(ag_path, index=False, compression="gzip")

        outputs["by_drive_dir"] = out_by_drive

    # Optional: split outputs into chunks
    if chunk_mb and chunk_mb > 0:
        chunk_bytes = int(chunk_mb) * 1024 * 1024
        split_manifest: Dict[str, List[str]] = {}
        for k, p in outputs.items():
            if not isinstance(p, str) or not os.path.isfile(p):
                continue
            if os.path.getsize(p) > chunk_bytes:
                split_manifest[p] = split_file(p, chunk_bytes)
        outputs["split_manifest"] = split_manifest

    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to archive_manifest.csv (ExifTool output)")
    ap.add_argument("--outdir", required=True, help="Output directory for reports")
    ap.add_argument("--root-marker", default="All Archive/", help="Substring used to strip prefix from SourceFile")
    ap.add_argument("--no-by-drive", action="store_true", help="Disable per-drive partition outputs")
    ap.add_argument("--chunk-mb", type=int, default=0, help="If >0, split large outputs into N MB parts")
    args = ap.parse_args()

    outputs = build_reports(
        input_csv=args.input,
        outdir=args.outdir,
        root_marker=args.root_marker,
        by_drive=not args.no_by_drive,
        chunk_mb=args.chunk_mb,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
