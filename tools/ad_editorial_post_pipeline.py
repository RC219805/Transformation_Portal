#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""File: tools/ad_editorial_post_pipeline.py

Editorial RAW workflow with explicit linear-ProPhoto processing.

Maintain the existing CLI and project layout. TIFFs carry genuine uint16 RGB
and a matching matrix/shaper ICC transfer; web delivery converts to encoded
sRGB. HDR requires measured, constant-aperture/ISO brackets. Panorama features
use disposable proxies, while warping and feather blending use float masters.

See docs/guides/AD_EDITORIAL_POST_PIPELINE.md for supported dependencies,
acceptance limits, and operator commands. Local synthetic checks do not certify
photographic output or an end-to-end installed editorial runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rawpy
import tifffile
import yaml  # FIXED: Added missing import
from PIL import Image, ImageCms, ImageOps
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader  # FIXED: Added missing import
from reportlab.pdfgen import canvas
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Optional deps
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import exifread
except Exception:  # pragma: no cover
    exifread = None

try:
    import piexif  # JPEG fallback
except Exception:  # pragma: no cover
    piexif = None


# ----------------------------- logging ------------------------------------- #

LOG = logging.getLogger("ad_post")
TIFF_SUFFIX = ".tif"
PDF_SUFFIX = ".pdf"


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ----------------------------- helpers ------------------------------------- #


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def human_sort_key(p: Path) -> Tuple:
    s = p.name
    return tuple(int(t) if t.isdigit() else t.lower() for t in split_tokens(s))


def split_tokens(s: str) -> List[str]:
    out, token = [], ""
    for ch in s:
        if ch.isdigit() and (not token or token[-1].isdigit()):
            token += ch
        else:
            if token:
                out.append(token)
            token = ch
    if token:
        out.append(token)
    return out


def has_exiftool() -> bool:
    try:
        subprocess.run(
            ["exiftool", "-ver"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=10,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def load_icc_bytes(icc_path: Optional[Path]) -> Optional[bytes]:
    if not icc_path:
        return None
    if icc_path.exists():
        return icc_path.read_bytes()
    raise FileNotFoundError(f"Configured ICC profile not found: {icc_path}")


def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


def tiff_filename(stem: str, marker: str = "") -> str:
    return f"{stem}{marker}{TIFF_SUFFIX}"


def contact_sheet_filename(style: str) -> str:
    return f"contact_{style}{PDF_SUFFIX}"


def project_relative_path(project_root: Path, configured_path: object, default: Path) -> Path:
    """Resolve tool config paths relative to the configured project root."""
    raw_path = default if configured_path in (None, "") else Path(configured_path)
    expanded = raw_path.expanduser()
    if expanded.is_absolute():
        return expanded
    return project_root / expanded


# FIXED: Proper hash verification logic
def copy_and_verify(src: Path, dst: Path) -> None:
    """Copy file with hash verification. Skip if destination already matches."""
    if dst.exists():
        src_hash = sha256sum(src)
        dst_hash = sha256sum(dst)
        if src_hash == dst_hash:
            LOG.debug("File already exists with matching hash: %s", dst)
            return
        LOG.warning("Hash mismatch for existing %s, re-copying", dst)

    # Perform copy
    shutil.copy2(src, dst)

    # Verify copy succeeded
    src_hash = sha256sum(src)
    dst_hash = sha256sum(dst)
    if src_hash != dst_hash:
        # Clean up corrupted file
        try:
            dst.unlink()
        except Exception:
            pass
        raise RuntimeError(f"Hash mismatch copying {src} -> {dst}")


# FIXED: Atomic file write wrapper
def atomic_write(path: Path, writer_func, *args, **kwargs) -> None:
    """
    Atomically write a file by writing to a temp file first, then renaming.

    Args:
        path: Destination path
        writer_func: Function that takes a path and writes to it
        *args, **kwargs: Additional arguments for writer_func
    """
    # Unique, exclusively created siblings avoid competing writers or following a
    # pre-existing predictable .tmp symlink. Replacement remains on one filesystem.
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        writer_func(temp_path, *args, **kwargs)
        temp_path.replace(path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            LOG.warning("Unable to remove editorial temporary file %s", temp_path)


# ----------------------------- config -------------------------------------- #


@dataclass
class PipelineConfig:
    project_name: str
    project_root: Path
    input_raw_dir: Path
    backup_raw_dir: Optional[Path]
    rename: Dict
    selects: Dict
    icc: Dict
    processing: Dict
    styles: Dict
    consistency: Dict
    retouch: Dict
    export: Dict
    metadata: Dict
    deliver: Dict

    @staticmethod
    def from_yaml(path: Path) -> "PipelineConfig":
        data = json.loads(json.dumps(_read_yaml(path)))  # ensure plain types
        root = Path(data["project_root"]).expanduser().resolve()

        cfg = PipelineConfig(
            project_name=data["project_name"],
            project_root=root,
            input_raw_dir=Path(data["input_raw_dir"]).expanduser().resolve(),
            backup_raw_dir=(Path(data["backup_raw_dir"]).expanduser().resolve() if data.get("backup_raw_dir") else None),
            rename=data.get("rename", {"enabled": False}),
            selects=data.get("selects", {"use_csv": False}),
            icc=data.get("icc", {}),
            processing=data.get(
                "processing",
                {
                    "workers": 4,
                    "enable_hdr": False,
                    "enable_pano": False,
                    "auto_upright": True,
                    "upright_max_deg": 3.0,
                },
            ),
            styles=data.get("styles", {}),
            consistency=data.get("consistency", {"target_median": 0.42, "wb_neutralize": True}),
            retouch=data.get("retouch", {"dust_remove": False, "hotspot_reduce": False}),
            export=data.get(
                "export",
                {
                    "web_long_edge_px": 2500,
                    "jpeg_quality": 96,
                    "sharpen_web_amount": 0.35,
                    "sharpen_print_amount": 0.1,
                },
            ),
            metadata=data.get("metadata", {}),
            deliver=data.get("deliver", {"zip": True}),
        )

        # FIXED: Validate configuration
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """FIXED: Comprehensive config validation to prevent runtime errors."""
        errors = []

        # Validate project_name
        if not self.project_name or not self.project_name.strip():
            errors.append("project_name cannot be empty")

        # Validate paths
        if not self.input_raw_dir.exists():
            errors.append(f"input_raw_dir does not exist: {self.input_raw_dir}")

        if self.metadata.get("csv_path"):
            metadata_path = project_relative_path(
                self.project_root, self.metadata["csv_path"], self.project_root / "DOCS" / "metadata.csv"
            )
            if not metadata_path.is_file():
                errors.append(f"Configured metadata CSV does not exist: {metadata_path}")

        # Validate processing params
        workers = self.processing.get("workers", 4)
        if not isinstance(workers, int) or not (1 <= workers <= 64):
            errors.append(f"processing.workers must be 1-64, got {workers}")

        upright_max = self.processing.get("upright_max_deg", 3.0)
        if not isinstance(upright_max, (int, float)) or not (0 <= upright_max <= 15):
            errors.append(f"processing.upright_max_deg must be 0-15, got {upright_max}")

        hdr_gap = self.processing.get("hdr_group_gap_sec", 2.0)
        if not isinstance(hdr_gap, (int, float)) or hdr_gap < 0:
            errors.append(f"processing.hdr_group_gap_sec must be >= 0, got {hdr_gap}")

        # Validate export params
        web_edge = self.export.get("web_long_edge_px", 2500)
        if not isinstance(web_edge, int) or not (100 <= web_edge <= 10000):
            errors.append(f"export.web_long_edge_px must be 100-10000, got {web_edge}")

        jpeg_quality = self.export.get("jpeg_quality", 96)
        if not isinstance(jpeg_quality, int) or not (1 <= jpeg_quality <= 100):
            errors.append(f"export.jpeg_quality must be 1-100, got {jpeg_quality}")

        sharpen_web = self.export.get("sharpen_web_amount", 0.35)
        if not isinstance(sharpen_web, (int, float)) or not (0 <= sharpen_web <= 2.0):
            errors.append(f"export.sharpen_web_amount must be 0-2.0, got {sharpen_web}")

        sharpen_print = self.export.get("sharpen_print_amount", 0.1)
        if not isinstance(sharpen_print, (int, float)) or not (0 <= sharpen_print <= 2.0):
            errors.append(f"export.sharpen_print_amount must be 0-2.0, got {sharpen_print}")

        # Validate consistency params
        target_median = self.consistency.get("target_median", 0.42)
        if not isinstance(target_median, (int, float)) or not (0.1 <= target_median <= 0.9):
            errors.append(f"consistency.target_median must be 0.1-0.9, got {target_median}")

        # Validate styles exist
        if not self.styles:
            errors.append("At least one style must be defined in styles section")

        # Validate style parameters
        for style_name, style_params in self.styles.items():
            if (
                not isinstance(style_name, str)
                or not style_name.strip()
                or style_name in {".", ".."}
                or "/" in style_name
                or "\\" in style_name
                or "\x00" in style_name
            ):
                errors.append(f"Style name must be a single directory name: {style_name!r}")
            if not isinstance(style_params, dict):
                errors.append(f"Style '{style_name}' must be a dictionary")
                continue

            # Check exposure
            exposure = style_params.get("exposure", 0.0)
            if not isinstance(exposure, (int, float)) or not (-3.0 <= exposure <= 3.0):
                errors.append(f"Style '{style_name}' exposure must be -3.0 to 3.0, got {exposure}")

            # Check contrast
            contrast = style_params.get("contrast", 0)
            if not isinstance(contrast, (int, float)) or not (-50 <= contrast <= 50):
                errors.append(f"Style '{style_name}' contrast must be -50 to 50, got {contrast}")

            # Check saturation
            saturation = style_params.get("saturation", 0)
            if not isinstance(saturation, (int, float)) or not (-100 <= saturation <= 100):
                errors.append(f"Style '{style_name}' saturation must be -100 to 100, got {saturation}")

        # Explicit paths must be usable and match the intended pixel contract.
        for key, prophoto in (("prophoto_path", True), ("srgb_path", False)):
            if self.icc.get(key):
                try:
                    profile = load_icc_bytes(Path(self.icc[key]).expanduser())
                    _validate_rgb_profile(profile, prophoto=prophoto)
                except (OSError, ValueError, ImageCms.PyCMSError) as exc:
                    errors.append(f"icc.{key}: {exc}")
        if self.processing.get("enable_hdr", False) and exifread is None:
            errors.append("HDR requires exifread and complete capture exposure metadata")
        if self.processing.get("enable_pano", False) and not self.processing.get("pano_groups"):
            errors.append("Panorama requires explicit processing.pano_groups")
        if (
            self.processing.get("enable_pano", False)
            or self.processing.get("auto_upright", True)
            or self.retouch.get("dust_remove", False)
        ) and cv2 is None:
            errors.append("Selected panorama/upright/dust operations require OpenCV")

        if errors:
            raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))

        LOG.info("Configuration validated successfully")


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------- directories --------------------------------- #


@dataclass
class Layout:
    RAW_ORIG: Path
    RAW_BACKUP: Optional[Path]
    WORK_BASE: Path
    WORK_HDR: Path
    WORK_PANO: Path
    WORK_ALIGN: Path
    WORK_VARIANTS: Dict[str, Path]
    EXPORT_PRINT: Dict[str, Path]
    EXPORT_WEB: Dict[str, Path]
    DOCS: Path
    DOCS_CONTACTS: Path
    DOCS_MANIFESTS: Path

    @staticmethod
    def build(cfg: PipelineConfig) -> "Layout":
        root = cfg.project_root
        # Use configured styles instead of hardcoded variants
        variants = list(cfg.styles.keys())
        work_variants = {v: root / "WORK" / "Variants" / v for v in variants}
        export_print = {v: root / "EXPORT" / "Print_TIFF" / v for v in variants}
        export_web = {v: root / "EXPORT" / "Web_JPEG" / v for v in variants}

        return Layout(
            RAW_ORIG=root / "RAW" / "Originals",
            RAW_BACKUP=cfg.backup_raw_dir if cfg.backup_raw_dir else None,
            WORK_BASE=root / "WORK" / "BaseTIFF",
            WORK_HDR=root / "WORK" / "HDR",
            WORK_PANO=root / "WORK" / "Pano",
            WORK_ALIGN=root / "WORK" / "Aligned",
            WORK_VARIANTS=work_variants,
            EXPORT_PRINT=export_print,
            EXPORT_WEB=export_web,
            DOCS=root / "DOCS",
            DOCS_CONTACTS=root / "DOCS" / "ContactSheets",
            DOCS_MANIFESTS=root / "DOCS" / "Manifests",
        )

    def create(self) -> None:
        dirs = [
            self.RAW_ORIG,
            self.WORK_BASE,
            self.WORK_HDR,
            self.WORK_PANO,
            self.WORK_ALIGN,
            self.DOCS,
            self.DOCS_CONTACTS,
            self.DOCS_MANIFESTS,
        ]
        dirs += list(self.WORK_VARIANTS.values()) + list(self.EXPORT_PRINT.values()) + list(self.EXPORT_WEB.values())
        ensure_dirs(dirs)
        if self.RAW_BACKUP:
            self.RAW_BACKUP.mkdir(parents=True, exist_ok=True)


# ----------------------------- I/O utils ----------------------------------- #

RAW_EXTS = {
    ".cr2",
    ".cr3",
    ".nef",
    ".ne",  # Retained legacy alias.
    ".arw",
    ".raf",
    ".ra",  # Retained legacy alias.
    ".rw2",
    ".dng",
    ".orf",
    ".or",  # Retained legacy alias.
    ".srw",
    ".crw",
}


def find_raws(folder: Path) -> List[Path]:
    files: List[Path] = []
    for ext in RAW_EXTS:
        files += list(folder.rglob(f"*{ext}"))
        files += list(folder.rglob(f"*{ext.upper()}"))
    files = sorted(set(files), key=human_sort_key)
    return files


def mirror_offload(cfg: PipelineConfig, lay: Layout) -> List[Path]:
    LOG.info("Offloading RAW from %s -> %s", cfg.input_raw_dir, lay.RAW_ORIG)
    raws = find_raws(cfg.input_raw_dir)
    if not raws:
        raise ValueError("No RAW files found in input_raw_dir.")
    # Offload intentionally flattens the input tree, so reject ambiguous names
    # before copying instead of silently replacing one capture with another.
    names = [raw.name.casefold() for raw in raws]
    if len(set(names)) != len(names):
        raise ValueError("RAW offload requires unique basenames across input subdirectories")

    ensure_dirs([lay.RAW_ORIG])
    out_paths = []
    for src in tqdm(raws, desc="Copy RAW"):
        rel = src.name
        dst = lay.RAW_ORIG / rel
        copy_and_verify(src, dst)
        out_paths.append(dst)

    if lay.RAW_BACKUP:
        LOG.info("Backing up RAW to %s", lay.RAW_BACKUP)
        for p in tqdm(out_paths, desc="Backup RAW"):
            bdst = lay.RAW_BACKUP / p.name
            copy_and_verify(p, bdst)

    return out_paths


# ----------------------------- renaming ------------------------------------ #


def rename_raws(cfg: PipelineConfig, lay: Layout, files: List[Path]) -> List[Path]:
    if not cfg.rename.get("enabled", False):
        return files

    pattern = cfg.rename.get("pattern", "{project}_{room}_{seq:03d}")
    rooms_by_folder: Dict[str, str] = cfg.rename.get("rooms_by_folder", {})

    index_by_room: Dict[str, int] = {}
    mapping: Dict[str, str] = {}
    renamed: List[Path] = []

    for f in sorted(files, key=human_sort_key):
        room = guess_room_for_file(f, lay, rooms_by_folder)
        index_by_room.setdefault(room, 0)
        index_by_room[room] += 1
        seq = index_by_room[room]

        new_name = pattern.format(project=safe_name(cfg.project_name), room=safe_name(room), seq=seq) + f.suffix.lower()
        dst = f.with_name(new_name)

        # Handle collision with unique suffix
        if dst.exists():
            stem = dst.stem
            k = 1
            while dst.exists():
                dst = dst.with_name(f"{stem}_{k}{f.suffix.lower()}")
                k += 1

        f.rename(dst)
        mapping[f.name] = dst.name
        renamed.append(dst)

    (cfg.project_root / "DOCS" / "rename_mapping.json").write_text(json.dumps(mapping, indent=2))
    return renamed


def guess_room_for_file(f: Path, lay: Layout, rooms_by_folder: Dict[str, str]) -> str:
    # Map by containing folder match; fallback "Room"
    for folder, room in rooms_by_folder.items():
        full = (lay.RAW_ORIG.parent.parent / folder).resolve()
        try:
            if full in f.resolve().parents:
                return room
        except Exception:
            pass
    return "Room"


# --------------------- selects & contact sheet ------------------------------ #


def ensure_selects_csv(cfg: PipelineConfig, lay: Layout, raws: List[Path]) -> Path:
    csv_path = project_relative_path(
        cfg.project_root,
        cfg.selects.get("csv_path"),
        lay.DOCS / "selects.csv",
    )
    if csv_path.exists():
        return csv_path

    ensure_dirs([csv_path.parent])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "keep", "notes"])
        for r in raws:
            w.writerow([r.name, 1, ""])

    return csv_path


def filter_selects(cfg: PipelineConfig, files: List[Path]) -> List[Path]:
    if not cfg.selects.get("use_csv", False):
        return files

    csv_path = project_relative_path(
        cfg.project_root,
        cfg.selects.get("csv_path"),
        cfg.project_root / "DOCS" / "selects.csv",
    )
    if not csv_path.exists():
        return files

    keep = set()
    with csv_path.open("r", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0 or not row:
                continue
            fn, k, *_ = row
            if str(k).strip() in {"1", "true", "True", "Y", "y"}:
                keep.add(fn.strip())

    return [p for p in files if p.name in keep]


def build_contact_sheet(
    images: List[Path],
    out_pdf: Path,
    thumbs_per_row: int = 4,
    page_size=A4,
    caption: str = "",
) -> None:
    if not images:
        raise ValueError("Contact sheet requires at least one exported image")
    atomic_write(out_pdf, _render_contact_sheet, images, thumbs_per_row, page_size, caption)


def _render_contact_sheet(out_pdf: Path, images: List[Path], thumbs_per_row: int, page_size, caption: str) -> None:
    c = canvas.Canvas(str(out_pdf), pagesize=page_size)
    W, H = page_size
    margin = 36
    cell_w = (W - 2 * margin) / thumbs_per_row
    cell_h = cell_w * 0.75

    x, y = margin, H - margin

    if caption:
        c.setFont("Helvetica", 10)
        c.drawString(margin, y, caption)
        y -= 18

    for img in images:
        try:
            im = Image.open(img)
            im.thumbnail((int(cell_w), int(cell_h)), Image.LANCZOS)  # pylint: disable=no-member
            bio = ImageOps.exif_transpose(im)
            iw, ih = bio.size

            if x + cell_w > W - margin:
                x = margin
                y -= cell_h + 28

            if y < margin + cell_h:
                c.showPage()
                y = H - margin
                x = margin

            c.drawImage(ImageReader(bio), x, y - ih, iw, ih)
            c.setFont("Helvetica", 7)
            c.drawString(x, y - ih - 10, img.name)
            x += cell_w

        except Exception as e:
            raise RuntimeError(f"Contact sheet failed to render {img.name}") from e

    c.showPage()
    c.save()


# ------------------------- RAW → 16-bit ProPhoto ---------------------------- #


def raw_to_prophoto_tiff(raw_path: Path) -> np.ndarray:
    with rawpy.imread(str(raw_path)) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1),
            output_color=rawpy.ColorSpace.ProPhoto,  # pylint: disable=no-member
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  # pylint: disable=no-member
            half_size=False,
            four_color_rgb=False,
            bright=1.0,
        )  # uint16 0..65535 in ProPhoto RGB, linear gamma

    arr = rgb16.astype(np.float32) / 65535.0
    return np.clip(arr, 0.0, 1.0)


# Color matrices from W3C CSS Color 4 sample conversions (D50 ProPhoto,
# Bradford D50 -> D65, then D65 XYZ -> linear sRGB):
# https://www.w3.org/TR/css-color-4/#color-conversion-code
PROPHOTO_TO_XYZ = np.array(
    [
        [0.7977666449, 0.1351812974, 0.0313477341],
        [0.2880748288, 0.7118352342, 0.0000899369],
        [0.0, 0.0, 0.8251046025],
    ],
    dtype=np.float64,
)
D50_TO_D65 = np.array(
    [
        [0.9554734215, -0.0230984549, 0.0632592432],
        [-0.0283697093, 1.0099953981, 0.0210414412],
        [0.0123140149, -0.0205076493, 1.3303659262],
    ]
)
XYZ_TO_SRGB = np.array(
    [
        [12831 / 3959, -329 / 214, -1974 / 3959],
        [-851781 / 878810, 1648619 / 878810, 36519 / 878810],
        [705 / 12673, -2585 / 12673, 705 / 667],
    ]
)


def linear_prophoto_icc() -> bytes:
    """Build a deterministic ICC v2 matrix/shaper profile for linear masters."""

    def xyz(values):
        return b"XYZ " + bytes(4) + struct.pack(">3i", *(round(float(v) * 65536) for v in values))

    description = b"Transformation Portal Linear ProPhoto RGB\0"
    tags = {
        b"desc": b"desc" + bytes(4) + struct.pack(">I", len(description)) + description + bytes(78),
        b"cprt": b"text" + bytes(4) + b"Public colorimetric definitions\0",
        b"wtpt": xyz([0.9642, 1.0, 0.8249]),
    }
    for index, channel in enumerate((b"r", b"g", b"b")):
        tags[channel + b"XYZ"] = xyz(PROPHOTO_TO_XYZ[:, index])
        tags[channel + b"TRC"] = b"curv" + bytes(4) + struct.pack(">I", 0)
    header = bytearray(128)
    header[8:24] = struct.pack(">I", 0x02100000) + b"mntrRGB XYZ "
    header[24:36] = struct.pack(">6H", 2026, 1, 1, 0, 0, 0)
    header[36:40] = b"acsp"
    header[68:80] = struct.pack(">3i", *(round(v * 65536) for v in (0.9642, 1.0, 0.8249)))
    offset = 132 + 12 * len(tags)
    table, payload = bytearray(), bytearray()
    for signature, value in tags.items():
        table += signature + struct.pack(">II", offset + len(payload), len(value))
        payload += value + bytes((-len(value)) % 4)
    result = header + struct.pack(">I", len(tags)) + table + payload
    result[:4] = struct.pack(">I", len(result))
    return bytes(result)


def _icc_curve(profile: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Admit bounded matrix/shaper RGB profiles with identical monotone TRCs."""
    if (
        not 132 <= len(profile) <= 1024 * 1024
        or profile[36:40] != b"acsp"
        or struct.unpack_from(">I", profile, 0)[0] != len(profile)
    ):
        raise ValueError("Invalid or oversized RGB ICC profile")
    if profile[16:24] != b"RGB XYZ ":
        raise ValueError("RGB matrix/shaper ICC profile requires XYZ connection space")
    illuminant = np.array(struct.unpack_from(">3i", profile, 68)) / 65536
    if not np.allclose(illuminant, [0.9642, 1.0, 0.8249], atol=1e-4, rtol=0):
        raise ValueError("ICC header must use the D50 connection-space illuminant")
    count = struct.unpack_from(">I", profile, 128)[0]
    if count > 128 or 132 + count * 12 > len(profile):
        raise ValueError("Invalid ICC tag directory")
    tags = {}
    for index in range(count):
        signature, offset, size = struct.unpack_from(">4sII", profile, 132 + index * 12)
        if offset < 132 + count * 12 or size < 12 or offset + size > len(profile) or signature in tags:
            raise ValueError("Invalid ICC tag bounds")
        if signature[:3] in (b"A2B", b"B2A", b"D2B", b"B2D"):
            raise ValueError("ICC LUT transforms are outside the matrix/shaper contract")
        tags[signature] = profile[offset : offset + size]
    curves = [tags.get(channel + b"TRC") for channel in (b"r", b"g", b"b")]
    if curves[0] is None or not all(curve == curves[0] for curve in curves):
        raise ValueError("RGB ICC profile requires identical channel transfer curves")
    curve = curves[0]
    grid = np.linspace(0, 1, 65536)
    if curve[:4] == b"curv":
        length = struct.unpack_from(">I", curve, 8)[0]
        if length == 0:
            values = grid
        elif length == 1 and len(curve) >= 14:
            gamma = struct.unpack_from(">H", curve, 12)[0] / 256
            values = grid**gamma
        elif 2 <= length <= 65536 and len(curve) >= 12 + length * 2:
            samples = np.frombuffer(curve, dtype=">u2", count=length, offset=12) / 65535
            values = np.interp(grid, np.linspace(0, 1, length), samples)
        else:
            raise ValueError("Unsupported ICC sampled curve")
    elif curve[:4] == b"para":
        kind = struct.unpack_from(">H", curve, 8)[0]
        lengths = (1, 3, 4, 5, 7)
        if kind > 4 or len(curve) < 12 + 4 * lengths[kind]:
            raise ValueError("Unsupported ICC parametric curve")
        params = np.frombuffer(curve, dtype=">i4", count=lengths[kind], offset=12) / 65536
        gamma = params[0]
        if kind == 0:
            values = grid**gamma
        else:
            aa, bb = params[1:3]
            if aa <= 0:
                raise ValueError("Invalid ICC curve slope")
            cc = params[3] if kind >= 2 else 0
            if kind <= 2:
                values = np.where(grid >= -bb / aa, np.maximum(aa * grid + bb, 0) ** gamma + cc, cc)
            else:
                dd = params[4]
                ee, ff = params[5:7] if kind == 4 else (0, 0)
                values = np.where(grid >= dd, np.maximum(aa * grid + bb, 0) ** gamma + ee, cc * grid + ff)
    else:
        raise ValueError("Unsupported ICC transfer curve")
    if not np.isfinite(values).all() or np.any(np.diff(values) < 0) or abs(values[0]) > 1e-4 or abs(values[-1] - 1) > 1e-4:
        raise ValueError("ICC transfer must monotonically span black to white")
    return grid, values


def _validate_rgb_profile(profile: bytes, *, prophoto: bool) -> tuple[np.ndarray, np.ndarray]:
    grid, values = _icc_curve(profile)
    parsed = ImageCms.ImageCmsProfile(io.BytesIO(profile)).profile
    if not parsed.is_matrix_shaper or parsed.xcolor_space.strip() != "RGB":
        raise ValueError("Only RGB matrix/shaper ICC profiles are supported")
    matrix = np.array([parsed.red_colorant[0], parsed.green_colorant[0], parsed.blue_colorant[0]]).T
    if prophoto:
        if parsed.media_white_point is None or not np.allclose(
            parsed.media_white_point[0], [0.9642, 1.0, 0.8249], atol=3e-4, rtol=0
        ):
            raise ValueError("ProPhoto RGB ICC profile requires a D50 media white point")
        expected = PROPHOTO_TO_XYZ
    else:
        reference = ImageCms.createProfile("sRGB")
        expected = np.array([reference.red_colorant[0], reference.green_colorant[0], reference.blue_colorant[0]]).T
    if not np.allclose(matrix, expected, atol=3e-4, rtol=0):
        raise ValueError("ICC primaries do not match " + ("ProPhoto RGB" if prophoto else "sRGB"))
    if not prophoto:
        expected_curve = np.where(grid <= 0.04045, grid / 12.92, ((grid + 0.055) / 1.055) ** 2.4)
        if not np.allclose(values, expected_curve, atol=3e-4, rtol=0):
            raise ValueError("ICC transfer does not match encoded sRGB")
    return grid, values


def _rgb_samples(img: np.ndarray) -> np.ndarray:
    samples = np.asarray(img, dtype=np.float32)
    if samples.ndim != 3 or samples.shape[-1] != 3 or 0 in samples.shape or not np.isfinite(samples).all():
        raise ValueError("Expected finite HWC RGB samples")
    return np.clip(samples, 0, 1)


def linear_prophoto_to_srgb(img: np.ndarray) -> np.ndarray:
    """Convert linear D50 ProPhoto to encoded sRGB, clipping destination gamut."""
    matrix = XYZ_TO_SRGB @ D50_TO_D65 @ PROPHOTO_TO_XYZ
    linear = np.clip(_rgb_samples(img) @ matrix.T, 0, 1)
    encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return encoded.astype(np.float32)


def load_image_float(path: Path) -> np.ndarray:
    """Load TIFF without RGB truncation; decode embedded ProPhoto transfer."""
    profile = None
    if path.suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tif:
            if len(tif.pages) != 1:
                raise ValueError("Editorial intermediates require one RGB TIFF page")
            arr = tif.pages[0].asarray()
            tag = tif.pages[0].tags.get(34675)
            profile = bytes(tag.value) if tag else None
    else:
        with Image.open(path) as image:
            arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    result = _rgb_samples(arr)
    if profile:
        grid, curve = _validate_rgb_profile(profile, prophoto=True)
        result = np.interp(result, grid, curve).astype(np.float32)
    return result


def save_tiff16_prophoto(img: np.ndarray, path: Path, icc_bytes: Optional[bytes]) -> None:
    """Atomically store real uint16 RGB, encoded to the accompanying ICC TRC."""
    profile = icc_bytes if icc_bytes is not None else linear_prophoto_icc()
    grid, curve = _validate_rgb_profile(profile, prophoto=True)
    encoded = np.interp(_rgb_samples(img), curve, grid)
    img16 = np.rint(encoded * 65535).astype(np.uint16)

    def _write(p: Path):
        tifffile.imwrite(p, img16, photometric="rgb", compression="lzw", metadata=None, iccprofile=profile)

    atomic_write(path, _write)


def _srgb_icc() -> bytes:
    profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    profile[24:36] = struct.pack(">6H", 2026, 1, 1, 0, 0, 0)
    profile[84:100] = bytes(16)
    return bytes(profile)


def save_jpeg_srgb(img: np.ndarray, path: Path, icc_bytes: Optional[bytes], quality: int = 96) -> None:
    """Save 8-bit JPEG with sRGB color space."""
    profile = icc_bytes if icc_bytes is not None else _srgb_icc()
    _validate_rgb_profile(profile, prophoto=False)
    img8 = np.rint(_rgb_samples(img) * 255.0).astype(np.uint8)
    im = Image.fromarray(img8)

    # Use atomic write
    def _write(p: Path):
        im.save(
            str(p),
            format="JPEG",
            quality=int(quality),
            subsampling=0,
            icc_profile=profile,
            optimize=True,
        )

    atomic_write(path, _write)


# ----------------------------- transforms ---------------------------------- #


def resize_long_edge(img: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= long_edge:
        return img

    if h >= w:
        new_h = long_edge
        new_w = int(w * (new_h / h))
    else:
        new_w = long_edge
        new_h = int(h * (new_w / w))

    out = np.stack(
        [
            np.asarray(
                Image.fromarray(img[..., channel].astype(np.float32)).resize(
                    (max(1, new_w), max(1, new_h)), Image.Resampling.LANCZOS
                )
            )
            for channel in range(3)
        ],
        axis=-1,
    )
    out = np.clip(out, 0, 1).astype(np.float32)
    return out


def auto_upright_small(img: np.ndarray, max_deg: float = 3.0) -> np.ndarray:
    if cv2 is None:
        return img

    g = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)

    if lines is None:
        return img

    angles = []
    for rho, theta in lines[:, 0]:
        deg = (theta * 180.0 / np.pi) - 90.0
        if -max_deg <= deg <= max_deg:
            angles.append(deg)

    if not angles:
        return img

    rot = float(np.median(angles))
    if abs(rot) < 0.1:
        return img

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1.0)
    out = cv2.warpAffine(
        (img * 65535).astype(np.uint16),
        M,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return (out.astype(np.float32) / 65535.0).clip(0, 1)


# ----------------------------- grading ------------------------------------- #


def _contrast_luminance(img: np.ndarray, exponent: float) -> np.ndarray:
    """Monotone contrast about 18% linear gray without subtracting shadow light."""
    pixels = _rgb_samples(img)
    luminance = np.sum(pixels * PROPHOTO_TO_XYZ[1], axis=-1, keepdims=True)
    pivot = 0.18
    mapped = np.where(
        luminance <= pivot,
        pivot * np.maximum(luminance / pivot, 0) ** exponent,
        1 - (1 - pivot) * np.maximum((1 - luminance) / (1 - pivot), 0) ** exponent,
    )
    gain = np.divide(mapped, luminance, out=np.ones_like(luminance), where=luminance > 0)
    return np.clip(pixels * gain, 0, 1).astype(np.float32)


def s_curve(img: np.ndarray, strength: float = 0.12) -> np.ndarray:
    return _contrast_luminance(img, 1 + max(0, min(0.45, strength)))


def adjust_exposure(img: np.ndarray, ev: float) -> np.ndarray:
    if abs(ev) < 1e-6:
        return img
    factor = 2.0**ev
    return np.clip(img * factor, 0, 1)


def adjust_contrast(img: np.ndarray, amount: float) -> np.ndarray:
    return _contrast_luminance(img, 1 + np.tanh(amount / 50) * 0.6)


def _rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """Float HSV; hue is degrees, saturation/value stay in [0, 1]."""
    values = _rgb_samples(img)
    high, low = values.max(axis=-1), values.min(axis=-1)
    chroma = high - low
    hue = np.zeros_like(high)
    dominant = values.argmax(axis=-1)
    for index, offset in ((0, 0), (1, 2), (2, 4)):
        delta = values[..., (index + 1) % 3] - values[..., (index + 2) % 3]
        ratio = np.divide(delta, chroma, out=np.zeros_like(chroma), where=chroma > 0)
        hue = np.where((dominant == index) & (chroma > 0), ratio + offset, hue)
    saturation = np.divide(chroma, high, out=np.zeros_like(high), where=high > 0)
    return np.stack([(hue % 6) * 60, saturation, high], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    hue, saturation, value = np.moveaxis(hsv, -1, 0)
    sector = (hue / 60).astype(np.int32) % 6
    fraction = (hue / 60) % 1
    pp = value * (1 - saturation)
    qq = value * (1 - saturation * fraction)
    tt = value * (1 - saturation * (1 - fraction))
    result = np.empty_like(hsv)
    for index, components in enumerate(
        ((value, tt, pp), (qq, value, pp), (pp, value, tt), (pp, qq, value), (tt, pp, value), (value, pp, qq))
    ):
        mask = sector == index
        for channel, component in enumerate(components):
            result[..., channel][mask] = component[mask]
    return result


def adjust_saturation(img: np.ndarray, delta: float) -> np.ndarray:
    if abs(delta) < 1e-6:
        return img
    hsv = _rgb_to_hsv(img)
    hsv[..., 1] = np.clip(hsv[..., 1] + delta / 100, 0, 1)
    return _hsv_to_rgb(hsv)


def split_tone(
    img: np.ndarray,
    sh_h: Optional[float],
    sh_s: float,
    hi_h: Optional[float],
    hi_s: float,
) -> np.ndarray:
    if (sh_h is None or sh_s <= 0) and (hi_h is None or hi_s <= 0):
        return img
    hsv = _rgb_to_hsv(img)
    luma = np.sum(img * PROPHOTO_TO_XYZ[1], axis=-1)
    for target, strength, mask in (
        (sh_h, sh_s, np.clip(1 - luma * 2, 0, 1)),
        (hi_h, hi_s, np.clip(luma * 2 - 1, 0, 1)),
    ):
        if target is not None and strength > 0:
            # Blend hue along the shortest circular path, preserving float precision.
            delta = (float(target) - hsv[..., 0] + 180) % 360 - 180
            hsv[..., 0] = (hsv[..., 0] + delta * mask) % 360
            hsv[..., 1] = np.clip(hsv[..., 1] + strength * mask, 0, 1)
    return _hsv_to_rgb(hsv)


def vignette(img: np.ndarray, strength: float = 0.08) -> np.ndarray:
    if strength <= 0:
        return img

    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r /= np.sqrt(cx**2 + cy**2)
    mask = 1.0 - (r**2) * strength
    mask = np.clip(mask, 0.8, 1.0)[..., None]
    return np.clip(img * mask, 0, 1)


def style_grade(img: np.ndarray, style: str, params: Dict) -> np.ndarray:
    p = params.get(style, {})
    out = img.copy()

    out = adjust_exposure(out, float(p.get("exposure", 0.0)))
    out = adjust_contrast(out, float(p.get("contrast", 0.0)))
    out = s_curve(out, 0.12 if style != "minimal" else 0.06)
    out = adjust_saturation(out, float(p.get("saturation", 0.0)))

    st = p.get("split_tone", {})
    out = split_tone(
        out,
        st.get("shadows_hue_deg"),
        float(st.get("shadows_sat", 0.0)),
        st.get("highs_hue_deg"),
        float(st.get("highs_sat", 0.0)),
    )

    if style == "cinematic":
        out = vignette(out, 0.08)

    return np.clip(out, 0, 1)


# ----------------------------- consistency --------------------------------- #


def median_luma(img: np.ndarray) -> float:
    luma = np.sum(img * PROPHOTO_TO_XYZ[1], axis=-1)
    return float(np.median(luma))


def normalize_exposure(imgs: List[np.ndarray], target_median: float = 0.42) -> List[np.ndarray]:
    out = []
    for im in imgs:
        m = median_luma(im) + 1e-6
        factor = np.clip(target_median / m, 0.6, 1.6)
        out.append(np.clip(im * factor, 0, 1))
    return out


def neutralize_wb_near_white(img: np.ndarray) -> np.ndarray:
    # Neutralizes only near-white areas to limit color cast without shifting palette.
    luma = np.sum(img * PROPHOTO_TO_XYZ[1], axis=-1)
    mask = (luma > 0.7).astype(np.float32)[..., None]

    if mask.sum() < 1000:
        return img

    sel = img * mask + (1 - mask) * 0
    mean = sel.sum(axis=(0, 1)) / (mask.sum() + 1e-6)
    gain = mean.mean() / (mean + 1e-6)
    gain = np.clip(gain, 0.8, 1.25)
    out = np.clip(img * (gain**0.5), 0, 1)
    return out


# ------------------------------- retouch ----------------------------------- #


def remove_dust_spots(img: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return img

    gray = np.rint(np.sum(_rgb_samples(img) * PROPHOTO_TO_XYZ[1], axis=-1) * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(255 - gray, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(tophat, 220, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 5)
    # Telea uses absolute intensity-scale terms: unit-range float input can ring
    # across the whole [0, 1] range. Use its 0..255 scale without quantizing, then
    # restore the linear working range. Only mask detection uses integer samples.
    inpainted = np.stack(
        [cv2.inpaint(img[..., channel].astype(np.float32) * 255.0, mask, 3, cv2.INPAINT_TELEA) for channel in range(3)],
        axis=-1,
    )
    return np.clip(inpainted / 255.0, 0, 1)


def reduce_hotspots(img: np.ndarray) -> np.ndarray:
    luma = np.sum(img * PROPHOTO_TO_XYZ[1], axis=-1)
    mask = (luma > 0.95).astype(np.float32)[..., None]
    softened = np.clip(img * (1 - 0.2 * mask), 0, 1)
    return softened


def unsharp_mask(img: np.ndarray, amount: float = 0.2, radius: float = 1.2, threshold: float = 0.0) -> np.ndarray:
    if amount <= 0:
        return img

    low = gaussian_filter(img, sigma=(radius, radius, 0), mode="reflect")
    high = img - low
    mask = np.where(np.abs(high) > threshold, high, 0.0)
    out = np.clip(img + amount * mask, 0, 1)
    return out


# ------------------------------ HDR / Pano --------------------------------- #


def group_hdr_candidates(files: List[Path], gap_sec: float = 2.0) -> List[List[Path]]:
    # Heuristic: group consecutive triples captured within a small time gap.
    # Requires exifread.
    if exifread is None:
        return []

    def dt(path: Path) -> Optional[float]:
        try:
            with path.open("rb") as f:
                tags = exifread.process_file(f, details=False, stop_tag="EXIF DateTimeOriginal")
                dt_str = str(tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime") or "")
                # "YYYY:MM:DD HH:MM:SS"
                from datetime import datetime, timezone

                # Only wall-time differences are used for bracket grouping.
                return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
        except Exception:
            return None

    files_sorted = sorted(files, key=lambda p: (p.parent.name, p.name))
    times = [dt(p) for p in files_sorted]

    groups: List[List[Path]] = []
    buf: List[Path] = []

    for i, p in enumerate(files_sorted):
        if times[i] is None:
            if len(buf) >= 3:
                groups.append(buf[:])
            buf = []
            continue
        if not buf:
            buf = [p]
            continue

        if abs(times[i] - times[i - 1]) <= gap_sec and len(buf) < 5:
            buf.append(p)
        else:
            if len(buf) >= 3:
                groups.append(buf[:])
            buf = [p]

    if len(buf) >= 3:
        groups.append(buf[:])

    return groups


def _hdr_capture_settings(path: Path) -> tuple[float, float, float]:
    if exifread is None:
        raise RuntimeError("HDR requires exifread to verify shutter, aperture and ISO metadata")
    with path.open("rb") as handle:
        tags = exifread.process_file(handle, details=False)
    settings = []
    for name in ("EXIF ExposureTime", "EXIF FNumber", "EXIF ISOSpeedRatings"):
        tag = tags.get(name)
        if tag is None or not getattr(tag, "values", None):
            raise ValueError(f"Missing HDR capture metadata: {name} for {path.name}")
        value = tag.values[0]
        number = float(value.num) / float(value.den) if hasattr(value, "num") else float(value)
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"Invalid HDR capture metadata: {name}")
        settings.append(number)
    return tuple(settings)


def hdr_merge_debvec(paths: List[Path]) -> np.ndarray:
    """Merge linear RAW radiance using measured exposure times.

    The legacy function name is retained. Linear RAWs need no estimated camera
    response curve; saturation-weighted radiance replaces 8-bit Debevec input
    with fictitious equal exposures. A shared logarithmic shoulder maps the
    merged radiance into the existing unit-range photographic output contract.
    """
    if len(paths) < 2:
        raise ValueError("HDR requires at least two distinct exposures")
    settings = [_hdr_capture_settings(path) for path in paths]
    if len({item[0] for item in settings}) < 2:
        raise ValueError("HDR requires distinct measured shutter times")
    if any(not np.allclose(item[1:], settings[0][1:], rtol=1e-4) for item in settings[1:]):
        raise ValueError("HDR bracket aperture and ISO must remain constant")
    images = [_rgb_samples(raw_to_prophoto_tiff(path)) for path in paths]
    if any(image.shape != images[0].shape for image in images[1:]):
        raise ValueError("HDR inputs must have identical geometry and be aligned")
    total = np.zeros_like(images[0], dtype=np.float64)
    weights = np.zeros_like(total)
    times = [setting[0] for setting in settings]
    for image, seconds in zip(images, times):
        weight = np.maximum(0, 1 - np.abs(image * 2 - 1))
        total += weight * image / seconds
        weights += weight
    # Fully clipped highlights use the shortest exposure; fully black stays black.
    fallback = images[int(np.argmin(times))] / min(times)
    radiance = np.divide(total, weights, out=fallback.astype(np.float64), where=weights > 1e-8)
    reference = radiance * float(np.median(times))
    peak = reference.max(axis=-1, keepdims=True)
    white = max(1.0, float(np.percentile(peak, 99.9)))
    mapped_peak = np.log1p(peak) / np.log1p(white)
    scale = np.divide(mapped_peak, peak, out=np.ones_like(peak), where=peak > 0)
    return np.clip(reference * scale, 0, 1).astype(np.float32)


def _panorama_registration(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Estimate current -> previous geometry from disposable 8-bit features."""
    detector = cv2.SIFT_create()
    features = []
    for image in (previous, current):
        proxy = np.rint(linear_prophoto_to_srgb(image) * 255).astype(np.uint8)
        gray = cv2.cvtColor(proxy, cv2.COLOR_RGB2GRAY)
        features.append(detector.detectAndCompute(gray, None))
    (previous_keys, previous_desc), (current_keys, current_desc) = features
    if previous_desc is None or current_desc is None:
        raise ValueError("Panorama requires overlapping textured photographs")
    pairs = cv2.BFMatcher().knnMatch(current_desc, previous_desc, k=2)
    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(matches) < 12:
        raise ValueError("Insufficient unambiguous panorama feature matches")
    source = np.float32([current_keys[match.queryIdx].pt for match in matches])
    target = np.float32([previous_keys[match.trainIdx].pt for match in matches])
    transform, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0, maxIters=2000, confidence=0.995)
    if transform is None or inliers is None or int(inliers.sum()) < 8 or not np.isfinite(transform).all():
        raise ValueError("Panorama registration failed geometric verification")
    return transform


def stitch_pano(paths: List[Path]) -> np.ndarray:
    """Register adjacent views, then warp/blend original float ProPhoto samples."""
    if cv2 is None:
        raise RuntimeError("OpenCV not available for panorama stitching")
    if len(paths) < 2:
        raise ValueError("Panorama requires at least two overlapping photographs")
    images = [_rgb_samples(raw_to_prophoto_tiff(path)) for path in paths]
    transforms = [np.eye(3)]
    for previous, current in zip(images, images[1:]):
        transforms.append(transforms[-1] @ _panorama_registration(previous, current))
    corners = []
    for image, transform in zip(images, transforms):
        height, width = image.shape[:2]
        points = np.float32([[[0, 0], [width, 0], [width, height], [0, height]]])
        denominator = np.c_[points[0], np.ones(4)] @ transform[2]
        if np.any(np.abs(denominator) < 1e-6) or not (np.all(denominator > 0) or np.all(denominator < 0)):
            raise ValueError("Panorama projective horizon crosses an input image")
        corners.append(cv2.perspectiveTransform(points, transform)[0])
    bounds = np.concatenate(corners)
    if not np.isfinite(bounds).all():
        raise ValueError("Panorama transform has unbounded extent")
    left, top = np.floor(bounds.min(axis=0)).astype(int)
    right, bottom = np.ceil(bounds.max(axis=0)).astype(int)
    width, height = int(right - left), int(bottom - top)
    budget = 4 * sum(image.shape[0] * image.shape[1] for image in images)
    if width <= 0 or height <= 0 or max(width, height) >= 32767 or width * height > budget:
        raise ValueError("Panorama canvas exceeds bounded warp geometry")
    translation = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=np.float64)
    total = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    for image, transform in zip(images, transforms):
        hh, ww = image.shape[:2]
        yy, xx = np.ogrid[:hh, :ww]
        feather = np.minimum(np.minimum(yy + 1, hh - yy), np.minimum(xx + 1, ww - xx)).astype(np.float32)
        matrix = translation @ transform
        warped_weight = cv2.warpPerspective(feather, matrix, (width, height), flags=cv2.INTER_LINEAR)
        warped = cv2.warpPerspective(image * feather[..., None], matrix, (width, height), flags=cv2.INTER_LINEAR)
        total += warped
        weights += warped_weight
    return np.clip(np.divide(total, weights[..., None], out=total, where=weights[..., None] > 0), 0, 1)


# ------------------------------ exports ------------------------------------ #


def export_assets(
    img: np.ndarray,
    print_path: Path,
    web_path: Path,
    icc_prophoto: Optional[bytes],
    icc_srgb: Optional[bytes],
    web_long_edge: int,
    jpeg_quality: int,
    sharpen_web_amt: float,
    sharpen_print_amt: float,
) -> None:
    # Print TIFF (16-bit ProPhoto)
    p_img = unsharp_mask(img, amount=float(sharpen_print_amt), radius=1.2, threshold=0.0)
    save_tiff16_prophoto(p_img, print_path, icc_prophoto)

    # Resize/sharpen in the linear working space, then convert primaries and transfer.
    w_img = resize_long_edge(img, int(web_long_edge))
    w_img = unsharp_mask(w_img, amount=float(sharpen_web_amt), radius=1.2, threshold=0.0)
    save_jpeg_srgb(linear_prophoto_to_srgb(w_img), web_path, icc_srgb, quality=int(jpeg_quality))


# ---------------------------- metadata IPTC -------------------------------- #


def read_metadata_csv(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}

    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            fn = row.get("filename")
            if fn:
                out[fn] = row
    return out


def embed_iptc_exiftool(img_path: Path, row: Dict[str, str]) -> None:
    args = ["exiftool", "-overwrite_original", "-charset", "iptc=UTF8"]

    def add(tag: str, value: Optional[str], key: str) -> None:
        if not value:
            return
        args.extend([f"-{tag}={value}"])

    add("XMP-dc:Title", row.get("title"), "title")
    add("IPTC:ObjectName", row.get("title"), "title")
    add("IPTC:Caption-Abstract", row.get("description"), "description")
    add("XMP-dc:Description", row.get("description"), "description")
    add("XMP-photoshop:Credit", row.get("credit"), "credit")
    add("XMP-dc:Creator", row.get("creator"), "creator")
    add("IPTC:CopyrightNotice", row.get("copyright"), "copyright")

    kw = row.get("keywords")
    if kw:
        # comma or semicolon separated
        for k in [k.strip() for k in kw.replace(";", ",").split(",") if k.strip()]:
            args.extend([f"-IPTC:Keywords={k}"])

    add("XMP-iptcCore:Location", row.get("location"), "location")

    def write_metadata(temp_path: Path) -> None:
        shutil.copyfile(img_path, temp_path)
        subprocess.run(args + [str(temp_path.resolve())], check=True, capture_output=True, timeout=60)

    atomic_write(img_path, write_metadata)


def embed_iptc_fallback_jpeg(img_path: Path, row: Dict[str, str]) -> None:
    if img_path.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("The piexif metadata fallback supports JPEG files only")
    if piexif is None:
        raise RuntimeError("JPEG metadata requires ExifTool or the optional piexif runtime")

    original = img_path.read_bytes()
    exif_dict = piexif.load(original)
    if row.get("copyright"):
        exif_dict["0th"][piexif.ImageIFD.Copyright] = row["copyright"].encode("utf-8")
    if row.get("creator"):
        exif_dict["0th"][piexif.ImageIFD.Artist] = row["creator"].encode("utf-8")

    # Insert an EXIF segment without decoding/re-encoding JPEG pixels or losing ICC.
    updated = io.BytesIO()
    piexif.insert(piexif.dump(exif_dict), original, updated)
    atomic_write(img_path, lambda temp_path: temp_path.write_bytes(updated.getvalue()))


# ------------------------------- pipeline ---------------------------------- #


def run_pipeline(config_path: Path, verbosity: int = 1) -> None:
    setup_logging(verbosity)
    cfg = PipelineConfig.from_yaml(config_path)
    lay = Layout.build(cfg)
    lay.create()
    failures = []

    # Offload + optional rename
    raws_copied = mirror_offload(cfg, lay)
    raws = rename_raws(cfg, lay, raws_copied)

    # Selects template + contact sheet from embedded previews (fast)
    _selects_csv = ensure_selects_csv(cfg, lay, raws)  # noqa: F841
    raws = filter_selects(cfg, raws)
    if not raws:
        raise ValueError("No selected RAW inputs; no editorial outputs were produced")
    stems = [raw.stem.casefold() for raw in raws]
    if len(set(stems)) != len(stems):
        raise ValueError("Selected RAW inputs need unique stems for TIFF/JPEG outputs; enable renaming or split the run")

    # Decode RAW → BaseTIFF
    LOG.info("Decoding RAW to 16-bit ProPhoto base TIFFs")
    base_outputs: List[Path] = []
    icc_prophoto = (
        load_icc_bytes(Path(cfg.icc.get("prophoto_path", "")).expanduser()) if cfg.icc.get("prophoto_path") else None
    )

    for rp in tqdm(raws, desc="RAW→TIFF"):
        try:
            img = raw_to_prophoto_tiff(rp)
            out = lay.WORK_BASE / tiff_filename(rp.stem)
            save_tiff16_prophoto(img, out, icc_prophoto)
            base_outputs.append(out)
        except Exception as e:
            LOG.error("RAW decode failed %s: %s", rp, e)
            failures.append(str(e))

    # Optional HDR
    hdr_paths: List[Path] = []
    if cfg.processing.get("enable_hdr", False):
        LOG.info("HDR merge enabled")
        groups = group_hdr_candidates(raws, float(cfg.processing.get("hdr_group_gap_sec", 2.0)))
        if not groups:
            raise ValueError("HDR requested but no complete timestamped exposure brackets were found")
        for g in tqdm(groups, desc="HDR groups"):
            try:
                hdr = hdr_merge_debvec(g)
                out = lay.WORK_HDR / tiff_filename(g[0].stem, "_HDR")
                save_tiff16_prophoto(hdr, out, icc_prophoto)
                hdr_paths.append(out)
            except Exception as e:
                LOG.warning("HDR merge failed for %s: %s", [p.name for p in g], e)
                failures.append(str(e))

    # Optional Pano
    pano_paths: List[Path] = []
    if cfg.processing.get("enable_pano", False):
        LOG.info("Panorama stitching enabled")
        for group in cfg.processing.get("pano_groups", []):
            try:
                files = [next((p for p in raws if p.name == fn), None) for fn in group]
                files = [p for p in files if p]
                if len(files) != len(group) or len(files) < 2:
                    raise ValueError("Panorama group must name at least two selected RAW inputs")

                pano = stitch_pano(files)
                out = lay.WORK_PANO / tiff_filename(files[0].stem, "_PANO")
                save_tiff16_prophoto(pano, out, icc_prophoto)
                pano_paths.append(out)
            except Exception as e:
                LOG.warning("Pano stitch failed for %s: %s", group, e)
                failures.append(str(e))

    # Collect sources for alignment/variants: base + hdr + pano
    sources = base_outputs + hdr_paths + pano_paths

    # Auto-upright
    aligned_paths: List[Path] = []
    if cfg.processing.get("auto_upright", True):
        LOG.info("Auto-upright small-angle correction")
        for p in tqdm(sources, desc="Upright"):
            try:
                img = load_image_float(p)
                img = auto_upright_small(img, float(cfg.processing.get("upright_max_deg", 3.0)))
                out = lay.WORK_ALIGN / p.name
                save_tiff16_prophoto(img, out, icc_prophoto)
                aligned_paths.append(out)
            except Exception as e:
                LOG.warning("Upright failed %s: %s", p, e)
                failures.append(str(e))
    else:
        aligned_paths = sources

    # Variants per style
    LOG.info("Creating style variants")
    variant_map: Dict[str, List[Path]] = {k: [] for k in lay.WORK_VARIANTS}

    for p in tqdm(aligned_paths, desc="Variants"):
        try:
            base = load_image_float(p)

            for style, style_path in lay.WORK_VARIANTS.items():
                graded = style_grade(base, style, cfg.styles)

                if cfg.consistency.get("wb_neutralize", True):
                    graded = neutralize_wb_near_white(graded)

                out = style_path / p.name
                save_tiff16_prophoto(graded, out, icc_prophoto)
                variant_map[style].append(out)

        except Exception as e:
            LOG.warning("Variants failed %s: %s", p, e)
            failures.append(str(e))

    # Per-style consistency normalization
    LOG.info("Normalizing per-style exposure to target median")
    target = float(cfg.consistency.get("target_median", 0.42))

    for style, paths in variant_map.items():
        imgs = [load_image_float(pt) for pt in paths]
        norm = normalize_exposure(imgs, target_median=target)
        for im, pt in zip(norm, paths):
            save_tiff16_prophoto(im, pt, icc_prophoto)

    # Optional automated retouch
    if cfg.retouch.get("dust_remove", False) or cfg.retouch.get("hotspot_reduce", False):
        LOG.info("Applying lightweight automated retouch")
        for style, paths in variant_map.items():
            for pt in tqdm(paths, desc=f"Retouch {style}"):
                im = load_image_float(pt)

                if cfg.retouch.get("dust_remove", False):
                    im = remove_dust_spots(im)

                if cfg.retouch.get("hotspot_reduce", False):
                    im = reduce_hotspots(im)

                save_tiff16_prophoto(im, pt, icc_prophoto)

    # Export print & web
    icc_srgb = load_icc_bytes(Path(cfg.icc.get("srgb_path", "")).expanduser()) if cfg.icc.get("srgb_path") else None

    manifest = {"project": cfg.project_name, "exports": []}

    for style, paths in variant_map.items():
        for pt in tqdm(paths, desc=f"Export {style}"):
            im = load_image_float(pt)

            print_out = lay.EXPORT_PRINT[style] / pt.name  # TIFF
            web_out = lay.EXPORT_WEB[style] / (pt.stem + ".jpg")

            export_assets(
                im,
                print_out,
                web_out,
                icc_prophoto,
                icc_srgb,
                int(cfg.export.get("web_long_edge_px", 2500)),
                int(cfg.export.get("jpeg_quality", 96)),
                float(cfg.export.get("sharpen_web_amount", 0.35)),
                float(cfg.export.get("sharpen_print_amount", 0.1)),
            )

            manifest["exports"].append(
                {
                    "style": style,
                    "print_tif": str(print_out.relative_to(cfg.project_root)),
                    "web_jpeg": str(web_out.relative_to(cfg.project_root)),
                }
            )

    # Contact sheet of web JPEGs
    for style in lay.EXPORT_WEB:
        imgs = sorted(list(lay.EXPORT_WEB[style].glob("*.jpg")), key=human_sort_key)
        if not imgs:
            continue

        out_pdf = lay.DOCS_CONTACTS / contact_sheet_filename(style)
        build_contact_sheet(imgs, out_pdf, caption=f"{cfg.project_name} — {style}")

    # IPTC/XMP embedding
    meta_map = read_metadata_csv(
        project_relative_path(
            cfg.project_root,
            cfg.metadata.get("csv_path"),
            cfg.project_root / "DOCS" / "metadata.csv",
        )
    )
    if meta_map:
        LOG.info("Embedding IPTC/XMP metadata")
        ef = has_exiftool()

        for style in lay.EXPORT_WEB:
            for img in tqdm(
                sorted(list(lay.EXPORT_WEB[style].glob("*.jpg")), key=human_sort_key),
                desc=f"Metadata {style}",
            ):
                row = meta_map.get(img.name) or meta_map.get(tiff_filename(img.stem)) or meta_map.get(img.stem + ".jpg")
                if not row:
                    continue

                if ef:
                    embed_iptc_exiftool(img, row)
                else:
                    embed_iptc_fallback_jpeg(img, row)

        for style in lay.EXPORT_PRINT:
            for img in tqdm(
                sorted(list(lay.EXPORT_PRINT[style].glob(f"*{TIFF_SUFFIX}")), key=human_sort_key),
                desc=f"Metadata {style}",
            ):
                row = meta_map.get(img.name) or meta_map.get(tiff_filename(img.stem))
                if not row:
                    continue

                if ef:
                    embed_iptc_exiftool(img, row)
                else:
                    LOG.warning("TIFF IPTC/XMP embedding requires ExifTool; no TIFF tags were added to %s", img)

    # Manifest + zip
    manifest_path = lay.DOCS_MANIFESTS / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if failures or not manifest["exports"]:
        raise RuntimeError(
            f"Editorial run incomplete: {len(failures)} processing failure(s), {len(manifest['exports'])} exports"
        )

    if cfg.deliver.get("zip", True):
        zip_path = cfg.project_root / f"{safe_name(cfg.project_name)}_EXPORT.zip"
        if zip_path.exists():
            zip_path.unlink()

        shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=cfg.project_root, base_dir="EXPORT")
        LOG.info("Deliverable zip: %s", zip_path)

    LOG.info("Done. Print TIFFs in EXPORT/Print_TIFF/**; Web JPEGs in EXPORT/Web_JPEG/**")


# ------------------------------ CLI ---------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AD Editorial Interior Post-Production Pipeline")
    ap.add_argument("run", nargs="?", help="Run the full pipeline", default="run")
    ap.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    ap.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv)")

    args = ap.parse_args(argv)

    try:
        run_pipeline(args.config, verbosity=args.verbose)
        return 0
    except KeyboardInterrupt:
        LOG.error("Interrupted")
        return 2
    except Exception as e:
        LOG.exception("Pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
