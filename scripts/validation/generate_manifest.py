#!/usr/bin/env python3
"""Generate/update baseline manifest for data/benchmark_datasets/validation_v1.

What it does:
- Reads existing manifest header (tool versions, presets, etc.)
- Scans inputs + baseline output folders for image files
- Computes SHA256, size, and mtime for each file
- Stamps host metadata (hostname/user/os/python)
- Writes back a deterministic, sorted manifest.json

Usage:
  python scripts/validation/generate_manifest.py
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

DATASET_ROOT = Path("data/benchmark_datasets/validation_v1")
INPUT_DIR = DATASET_ROOT / "input"
BASELINES_DIR = DATASET_ROOT / "baselines"
MANIFEST_PATH = BASELINES_DIR / "manifest.json"

BASELINE_IDS = [
    "topaz_photo",
    "topaz_gigapixel",
    "topaz_video",
    "adobe_sr",
    "adobe_neutral",
]

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iter_images(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def build_file_record(path: Path, tool: str | None) -> dict:
    stat = path.stat()
    return {
        "tool": tool,
        "relpath": str(path.as_posix()),
        "file": path.name,
        "bytes": stat.st_size,
        "mtime_utc": iso_utc(stat.st_mtime),
        "sha256": sha256_file(path),
    }


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"manifest.json not found at {MANIFEST_PATH}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    # Host stamping (non-destructive: keeps user-provided labels)
    host = manifest.get("host", {})
    host.update(
        {
            "hostname": platform.node(),
            "user": getpass.getuser(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    manifest["host"] = host

    # Inputs
    inputs = []
    for p in iter_images(INPUT_DIR):
        inputs.append(build_file_record(p, tool=None))
    manifest["inputs"] = inputs

    # Baselines
    files = []
    for baseline_id in BASELINE_IDS:
        folder = BASELINES_DIR / baseline_id
        for p in iter_images(folder):
            files.append(build_file_record(p, tool=baseline_id))

    # Deterministic sort
    files.sort(key=lambda r: (r.get("tool") or "", r.get("file") or ""))
    manifest["files"] = files

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Updated manifest: {len(inputs)} inputs, {len(files)} baseline outputs.")


if __name__ == "__main__":
    main()
