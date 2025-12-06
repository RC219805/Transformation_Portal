from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .logging_utils import setup_logging
from . import io_utils
from . import torch_ops
from . import weights as weights_mod
from . import upscaling
from .material_segmentation import create_material_segmenter
from . import material_profiles


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _find_depth(depth_dir: Optional[Path], stem: str) -> Optional[Path]:
    """Search for depth maps with multiple naming patterns."""
    if not depth_dir:
        return None
    # Try multiple depth naming patterns
    for pattern in [
        f"{stem}.tif", f"{stem}.tiff",
        f"{stem}_depth.tif", f"{stem}_depth.tiff",
        f"{stem}_depth_16bit.tif", f"{stem}_depth_16bit.tiff"
    ]:
        cand = depth_dir / pattern
        if cand.exists():
            return cand
    return None


def _find_zone_masks(depth_dir: Optional[Path], stem: str) -> Dict[str, np.ndarray]:
    """Optional manual zone masks. Convention: stem_foreground.png etc."""
    out: Dict[str, np.ndarray] = {}
    if not depth_dir:
        return out
    for k in ("foreground", "midground", "background"):
        for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
            cand = depth_dir / f"{stem}_{k}{ext}"
            if cand.exists():
                try:
                    out[k] = io_utils.read_mask_any(cand)
                except Exception:
                    pass
                break
    return out


def _resize_mods(mods: material_profiles.MaterialMods, size_hw: Tuple[int, int]) -> material_profiles.MaterialMods:
    """Resize mod maps to a new H,W."""
    h, w = int(size_hw[0]), int(size_hw[1])

    def rs(x):
        return torch_ops.resize(x, (h, w), mode="bilinear", autocast=True)

    return material_profiles.MaterialMods(
        temp_offset=rs(mods.temp_offset),
        sat_mult=rs(mods.sat_mult),
        exp_mult=rs(mods.exp_mult),
        con_mult=rs(mods.con_mult),
        detail_mult=rs(mods.detail_mult),
        clarity_mult=rs(mods.clarity_mult),
        sharpen_mult=rs(mods.sharpen_mult),
        highlight_compress=rs(mods.highlight_compress).clamp(0.0, 1.0),
        source=mods.source,
    )


def _slice_mods(mods: material_profiles.MaterialMods, ya0: int, xa0: int, ya1: int, xa1: int) -> material_profiles.MaterialMods:
    """Slice mod maps to a tile region."""
    return material_profiles.MaterialMods(
        temp_offset=mods.temp_offset[:, :, ya0:ya1, xa0:xa1],
        sat_mult=mods.sat_mult[:, :, ya0:ya1, xa0:xa1],
        exp_mult=mods.exp_mult[:, :, ya0:ya1, xa0:xa1],
        con_mult=mods.con_mult[:, :, ya0:ya1, xa0:xa1],
        detail_mult=mods.detail_mult[:, :, ya0:ya1, xa0:xa1],
        clarity_mult=mods.clarity_mult[:, :, ya0:ya1, xa0:xa1],
        sharpen_mult=mods.sharpen_mult[:, :, ya0:ya1, xa0:xa1],
        highlight_compress=mods.highlight_compress[:, :, ya0:ya1, xa0:xa1],
        source=mods.source,
    )


class LuxPipelineV2:
    """Gold Standard Lux Depth Pipeline V2 (GPU-accelerated, modular)."""

    def __init__(self, cfg: PipelineConfig, logger=None):
        self.cfg = cfg
        self.cfg.apply_preset()

        self.logger = logger or setup_logging("INFO", json_logs=False)

        torch_ops.require_torch()
        self.device = torch_ops.pick_device(cfg.device)
        torch_ops.configure_torch(cfg.cudnn_benchmark)

        self.autocast = (str(cfg.precision).lower() == "fp16" and self.device.type == "cuda")

        # Backends
        self.upscaler = upscaling.create_upscaler(cfg, self.device)
        self.segmenter = create_material_segmenter(cfg.segmentation, self.device)

        # Post tiler
        self.tiler = torch_ops.Tiler(tile=int(cfg.post_tile), overlap=int(cfg.post_overlap)) if int(cfg.post_tile) > 0 else None

        self.logger.info(
            f"PipelineV2 init | device={self.device} autocast={self.autocast} "
            f"upscaler={type(self.upscaler).__name__} seg={type(self.segmenter).__name__ if self.segmenter else 'None'}"
        )

    def process_one(self, img_path: Path, depth_path: Optional[Path] = None) -> Dict[str, object]:
        """Process a single image file and write outputs to output_dir."""
        t0 = time.time()
        cfg = self.cfg

        if not cfg.output_dir:
            raise ValueError("cfg.output_dir is required for file-based processing")

        img_path = Path(img_path)
        stem = img_path.stem
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Output paths
        master_path = out_dir / f"{stem}_master16.tif"
        up_path = out_dir / f"{stem}_upscaled16.tif"
        marketing_path = out_dir / f"{stem}_marketing.png"
        preview_path = out_dir / f"{stem}_preview.jpg"
        report_path = out_dir / f"{stem}_report.json"

        if cfg.skip_existing and master_path.exists() and up_path.exists() and (marketing_path.exists() or not cfg.save_marketing_png):
            self.logger.info(f"skip_existing: {img_path.name}")
            return {"status": "skipped", "image": str(img_path)}

        # Load image
        rgb01, info = io_utils.read_rgb_any(img_path)
        H, W = rgb01.shape[:2]
        float_gb = (H * W * 3 * 4) / 1e9
        if float_gb > float(cfg.warn_float_gb):
            self.logger.warning(
                f"Large image {W}x{H} may stress RAM/VRAM: ~{float_gb:.2f} GB per float32 RGB buffer"
            )

        # Depth and weights
        if depth_path is None:
            depth_path = _find_depth(cfg.depth_dir, stem)
        depth01 = None
        if depth_path and Path(depth_path).exists():
            depth01 = io_utils.read_depth_u16(Path(depth_path))
        else:
            if cfg.strict_depth:
                raise FileNotFoundError(f"Depth missing for {img_path.name} (strict_depth=True)")
            self.logger.warning(f"Depth missing for {img_path.name}; using uniform weights")

        zone_masks = _find_zone_masks(cfg.depth_dir, stem)
        w = weights_mod.weights_from_assets(H, W, self.device, depth01, zone_masks, cfg)

        # Convert to torch
        rgb_t = torch_ops.to_torch_rgb(rgb01, self.device)

        # Material segmentation (optional)
        mods0: Optional[material_profiles.MaterialMods] = None
        if cfg.enable_material and self.segmenter is not None:
            try:
                masks = self.segmenter.predict(rgb_t)
                mods0 = material_profiles.build_material_mods(masks, cfg)
            except Exception as e:
                self.logger.exception(f"Material segmentation failed for {img_path.name}: {e}")
                mods0 = None

        # Grade at original resolution
        with torch_ops.maybe_autocast(self.autocast, self.device):
            master_t = torch_ops.grade_core(rgb_t, w.wfg, w.wmid, w.wbg, cfg, mods=mods0)
            master_t = torch_ops.soft_clip01(master_t, cfg.soft_clip_knee)
            if mods0 is not None and cfg.enable_material:
                master_t = torch_ops.material_highlight_compress(master_t, mods0.highlight_compress, knee=0.85)

        master01 = torch_ops.from_torch_rgb(master_t)
        if cfg.save_master:
            io_utils.atomic_write_rgb16_tiff(master_path, master01)

        # Preview (small JPG) - robust with Pillow fallback
        if cfg.save_preview_jpg:
            try:
                scale = float(cfg.preview_scale)
                if 0 < scale < 1.0:
                    ph, pw = int(round(H * scale)), int(round(W * scale))
                    try:
                        # Try OpenCV for fast resizing
                        import cv2
                        prev = cv2.resize(master01, (pw, ph), interpolation=cv2.INTER_AREA)
                    except Exception:
                        # Fallback to scipy for resizing
                        from scipy.ndimage import zoom
                        prev = zoom(master01, (scale, scale, 1), order=1)
                else:
                    prev = master01
                io_utils.atomic_write_jpg8(preview_path, prev, quality=92)
            except Exception as e:
                self.logger.warning(f"Preview JPG generation failed: {e}")

        # Upscaling path
        # Base upsample: GPU bicubic
        with torch_ops.maybe_autocast(self.autocast, self.device):
            base_up = torch_ops.resize(master_t, (H * cfg.upscale, W * cfg.upscale), mode="bicubic", autocast=True).clamp(0.0, 1.0)

        ai_up = base_up
        ai_status = "none"
        if cfg.upscaler_backend != "none":
            try:
                ai_up = self.upscaler.upscale(master_t)
                ai_status = str(cfg.upscaler_backend)
            except Exception as e:
                self.logger.exception(f"Upscaler failed for {img_path.name}; falling back to bicubic: {e}")
                ai_up = base_up
                ai_status = "fallback_bicubic"

        # Validate AI upscaler drift (optional)
        use_ai_details = (cfg.upscaler_backend != "none")
        color_diff = None
        luma_diff = None
        if cfg.validate_ai and use_ai_details:
            try:
                color_diff = torch_ops.mean_abs_rgb(base_up, ai_up)
                luma_diff = torch_ops.mean_abs_luma(base_up, ai_up)
                if color_diff > cfg.ai_color_fail or luma_diff > cfg.ai_luma_fail:
                    self.logger.warning(
                        f"AI drift FAIL {img_path.name}: rgb={color_diff:.4f} luma={luma_diff:.4f}; skipping AI detail transfer"
                    )
                    use_ai_details = False
                elif color_diff > cfg.ai_color_warn or luma_diff > cfg.ai_luma_warn:
                    self.logger.warning(f"AI drift WARN {img_path.name}: rgb={color_diff:.4f} luma={luma_diff:.4f}")
            except Exception as e:
                self.logger.exception(f"AI drift validation failed: {e}")

        # Precompute final-res weights and (optional) material mods once; slice per-tile.
        fH, fW = H * cfg.upscale, W * cfg.upscale
        with torch_ops.maybe_autocast(self.autocast, self.device):
            wfgF = torch_ops.resize(w.wfg, (fH, fW), mode="bilinear", autocast=True)
            wmidF = torch_ops.resize(w.wmid, (fH, fW), mode="bilinear", autocast=True)
            wbgF = torch_ops.resize(w.wbg, (fH, fW), mode="bilinear", autocast=True)

        mods_final: Optional[material_profiles.MaterialMods] = None
        if mods0 is not None:
            mods_final = _resize_mods(mods0, (fH, fW))

        def post_fn(tile_rgb, ya0: int, xa0: int, ya1: int, xa1: int, y0: int, x0: int, y1: int, x1: int):
            wfgT = wfgF[:, :, ya0:ya1, xa0:xa1]
            wmidT = wmidF[:, :, ya0:ya1, xa0:xa1]
            wbgT = wbgF[:, :, ya0:ya1, xa0:xa1]
            tile_mods = _slice_mods(mods_final, ya0, xa0, ya1, xa1) if mods_final is not None else None

            out = tile_rgb
            if use_ai_details:
                ai_tile = ai_up[:, :, ya0:ya1, xa0:xa1]
                out = torch_ops.detail_transfer(out, ai_tile, wfgT, wmidT, wbgT, cfg, mods=tile_mods, autocast=self.autocast)

            out = torch_ops.apply_clarity(out, wfgT, wmidT, wbgT, cfg, mods=tile_mods, autocast=self.autocast)
            out = torch_ops.apply_sharpen(out, wfgT, wmidT, wbgT, cfg, mods=tile_mods, autocast=self.autocast)
            out = torch_ops.soft_clip01(out, cfg.soft_clip_knee)
            if tile_mods is not None and cfg.enable_material:
                out = torch_ops.material_highlight_compress(out, tile_mods.highlight_compress, knee=0.85)
            return out

        if self.tiler is not None:
            out_up = self.tiler.run(base_up, post_fn)
        else:
            out_up = post_fn(base_up, 0, 0, base_up.shape[2], base_up.shape[3], 0, 0, base_up.shape[2], base_up.shape[3])

        out01 = torch_ops.from_torch_rgb(out_up)

        if cfg.save_upscaled:
            io_utils.atomic_write_rgb16_tiff(up_path, out01)
        if cfg.save_marketing_png:
            io_utils.atomic_write_png8(marketing_path, out01)

        # Report
        report = {
            "status": "ok",
            "image": str(img_path),
            "depth": str(depth_path) if depth_path else None,
            "zone_weights": w.source,
            "material_mods": mods0.source if mods0 is not None else None,
            "upscaler": ai_status,
            "ai_color_diff": color_diff,
            "ai_luma_diff": luma_diff,
            "timing_s": round(time.time() - t0, 3),
            "config": _config_to_json(cfg),
        }

        try:
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except Exception:
            pass

        return report

    def process_directory(self) -> List[Dict[str, object]]:
        cfg = self.cfg
        if not cfg.input_dir or not cfg.output_dir:
            raise ValueError("cfg.input_dir and cfg.output_dir are required for directory processing")

        inp = Path(cfg.input_dir)
        files = [p for p in sorted(inp.iterdir()) if p.is_file() and _is_image_file(p)]
        if not files:
            self.logger.warning(f"No images found in {inp}")
            return []

        results: List[Dict[str, object]] = []
        try:
            from tqdm import tqdm  # type: ignore
            it = tqdm(files, desc="LuxDepthV2", unit="img")
        except Exception:
            it = files

        for p in it:
            try:
                results.append(self.process_one(p))
            except Exception as e:
                self.logger.exception(f"Failed: {p.name}: {e}")
                results.append({"status": "error", "image": str(p), "error": str(e)})
        return results


def _config_to_json(cfg: PipelineConfig) -> Dict[str, object]:
    d = asdict(cfg)

    def conv(o):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        return o

    return conv(d)
