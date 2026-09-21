#!/usr/bin/env python3
"""Verify the editorial dependency closure and exercise real image/PDF encoders."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    """Load repository tooling explicitly, without enabling an ambient import root."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load editorial tooling: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def smoke(lock: Path, expected_prefix: Path, raw_file: Path | None = None) -> dict:
    """Check installed versions, uint16 TIFF, encoded JPEG, PDF images, and optional RAW."""
    runtime = load_module("editorial_runtime_check_helpers", REPO_ROOT / "scripts/setup/editorial_runtime.py")
    runtime.require_target()
    if Path(sys.prefix).resolve() != expected_prefix.resolve() or sys.prefix == sys.base_prefix:
        raise ValueError("Editorial smoke must use the selected isolated runtime")
    expected = {name: value[0] for name, value in runtime.parse_lock(lock.read_bytes()).items()}
    installed = {}
    for distribution in importlib.metadata.distributions():
        name = runtime.normalize(distribution.metadata["Name"])
        if name in installed:
            raise ValueError(f"Duplicate installed editorial distribution: {name}")
        installed[name] = distribution.version
    if installed != expected:
        raise ValueError("Installed editorial distribution closure differs from the exact lock")
    for name in (
        "numpy",
        "PIL",
        "scipy",
        "cv2",
        "tifffile",
        "imagecodecs",
        "yaml",
        "tqdm",
        "rawpy",
        "reportlab",
        "exifread",
        "piexif",
    ):
        importlib.import_module(name)

    import cv2
    import numpy as np
    import rawpy
    import tifffile
    from PIL import Image

    editorial = load_module("editorial_runtime_smoke_tool", REPO_ROOT / "tools/ad_editorial_post_pipeline.py")
    with tempfile.TemporaryDirectory(prefix="tp-editorial-smoke-") as temporary:
        root = Path(temporary)
        master = np.linspace(0.05, 0.75, 32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3)
        tiff = root / "master.tif"
        editorial.save_tiff16_prophoto(master, tiff, None)
        encoded = tifffile.imread(tiff)
        if encoded.dtype != np.uint16 or encoded.shape != master.shape or np.unique(encoded).size <= 256:
            raise ValueError("Editorial TIFF failed real RGB uint16 precision smoke")
        independent = cv2.imread(str(tiff), cv2.IMREAD_UNCHANGED)
        if independent is None or not np.array_equal(independent[..., ::-1], encoded):
            raise ValueError("Editorial TIFF failed independent OpenCV decoding")
        jpeg = root / "preview.jpg"
        editorial.save_jpeg_srgb(master, jpeg, None)
        with Image.open(jpeg) as preview:
            preview.load()
            if preview.mode != "RGB" or not preview.info.get("icc_profile"):
                raise ValueError("Editorial JPEG failed RGB/ICC encoder smoke")
        pdf = root / "contact.pdf"
        editorial.build_contact_sheet([jpeg], pdf, caption="Editorial runtime encoding smoke")
        pdf_bytes = pdf.read_bytes()
        if not pdf_bytes.startswith(b"%PDF-") or b"/Subtype /Image" not in pdf_bytes:
            raise ValueError("Editorial contact sheet did not contain an encoded image")
    result = {
        "schema": "tp.editorial.runtime.smoke.v1",
        "python": sys.version.split()[0],
        "packages": installed,
        "libraw": list(rawpy.libraw_version),
        "tiff_rgb16": True,
        "jpeg_icc": True,
        "pdf_image": True,
        "raw_decode": "not_exercised",
    }
    if raw_file is not None:
        decoded = editorial.raw_to_prophoto_tiff(raw_file)
        if decoded.ndim != 3 or decoded.shape[2] != 3 or not np.isfinite(decoded).all():
            raise ValueError("Editorial RAW decode failed shape/finiteness validation")
        result["raw_decode"] = {"shape": list(decoded.shape), "dtype": str(decoded.dtype)}
    return result


def main(argv: list[str] | None = None) -> int:
    """Emit machine-readable successful smoke evidence; fail on any missing capability."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--expected-prefix", type=Path, required=True)
    parser.add_argument("--raw-file", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(smoke(args.lock, args.expected_prefix, args.raw_file), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
