from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
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

# Phase 2 Slice 2: ExportManager integration
try:
    import sys
    from pathlib import Path as _Path
    _repo_root = _Path(__file__).parent.parent
    if str(_repo_root / "src") not in sys.path:
        sys.path.insert(0, str(_repo_root / "src"))
    from transformation_portal.core.storage import ExportManager, ExportConfig
    EXPORT_MANAGER_AVAILABLE = True
except ImportError:
    ExportManager = None
    ExportConfig = None
    EXPORT_MANAGER_AVAILABLE = False

# Materials v2 imports (lazy load to avoid import errors if not used)
try:
    from .materials_v2 import MaterialsV2Engine, MaterialsV2Config, SegmentationResult
    from .cache_manager import MaskCacheManager
    MATERIALS_V2_AVAILABLE = True
except ImportError:
    MATERIALS_V2_AVAILABLE = False
    MaterialsV2Engine = None
    MaterialsV2Config = None
    SegmentationResult = None
    MaskCacheManager = None

# Materials v3 imports (lazy load, disabled by default)
try:
    from .materials_v3 import MaterialsV3Engine, MaterialsV3Config
    MATERIALS_V3_AVAILABLE = True
except ImportError:
    MATERIALS_V3_AVAILABLE = False
    MaterialsV3Engine = None
    MaterialsV3Config = None


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _find_depth(depth_dir: Optional[Path], stem: str) -> Optional[Path]:
    if not depth_dir:
        return None
    # Try both {stem}_depth.{ext} and {stem}.{ext} patterns
    for pattern in (f"{stem}_depth", f"{stem}"):
        for ext in (".tif", ".tiff", ".png"):
            cand = depth_dir / f"{pattern}{ext}"
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

        # PRODUCTION SAFETY: Validate dependencies before starting
        self._validate_dependencies()
        
        # PRODUCTION SAFETY: Warn if validate_ai is disabled
        if not cfg.validate_ai:
            self.logger.warning(
                "⚠️  validate_ai=False detected! This disables AI safety checks. "
                "Production presets should always have validate_ai=True."
            )

        torch_ops.require_torch()
        self.device = torch_ops.pick_device(cfg.device)
        torch_ops.configure_torch(cfg.cudnn_benchmark)

        self.autocast = (str(cfg.precision).lower() == "fp16" and self.device.type == "cuda")

        # Backends
        self.upscaler = upscaling.create_upscaler(cfg, self.device)
        self.segmenter = create_material_segmenter(cfg.segmentation, self.device)

        # Post tiler
        self.tiler = torch_ops.Tiler(tile=int(cfg.post_tile), overlap=int(cfg.post_overlap)) if int(cfg.post_tile) > 0 else None

        # Materials v2 integration
        self.materials_v2_engine = None
        self.mask_cache_manager = None
        if MATERIALS_V2_AVAILABLE and cfg.materials_v2 and cfg.materials_v2.enabled:
            try:
                self.materials_v2_engine = MaterialsV2Engine(
                    config=cfg.materials_v2,
                    device=str(self.device),
                    logger=self.logger
                )
                
                if cfg.materials_v2.cache_enabled and cfg.materials_v2.cache_dir:
                    self.mask_cache_manager = MaskCacheManager(
                        cache_dir=Path(cfg.materials_v2.cache_dir),
                        logger=self.logger
                    )
                
                self.logger.info(
                    f"Materials v2 enabled | "
                    f"backend={cfg.materials_v2.backend} "
                    f"confidence_threshold={cfg.materials_v2.confidence.confidence_threshold} "
                    f"cache={cfg.materials_v2.cache_enabled}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize Materials v2: {e}; continuing without")
                self.materials_v2_engine = None
                self.mask_cache_manager = None

        # Materials v3 integration (opt-in, disabled by default)
        self.materials_v3_engine = None
        if MATERIALS_V3_AVAILABLE and cfg.materials_v3 and cfg.materials_v3.enabled:
            try:
                self.materials_v3_engine = MaterialsV3Engine(
                    config=cfg.materials_v3
                )
                
                self.logger.info(
                    f"Materials V3 enabled | "
                    f"taxonomy={cfg.materials_v3.taxonomy} "
                    f"refinement={cfg.materials_v3.refine_edges} "
                    f"pixel_ops={cfg.materials_v3.apply_pixel_ops} "
                    f"max_mp={cfg.materials_v3.max_megapixels}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize Materials V3: {e}; continuing without")
                self.materials_v3_engine = None

        # Capture reproducibility metadata on init
        self._repro_metadata = self._collect_reproducibility_metadata()

        # Phase 2 Slice 2/3: ExportManager (deferred to JIT if autotune enabled)
        self.export_manager = None
        self._export_manager_autotune_enabled = (
            cfg.phase2 and cfg.phase2.autotune_export if cfg.phase2 else False
        )
        
        if not self._export_manager_autotune_enabled and EXPORT_MANAGER_AVAILABLE and cfg.output_dir:
            # Static config: build ExportManager at init (backward compatible)
            try:
                from transformation_portal.core.storage.export_manager import MarketingExportConfig
                marketing_cfg = MarketingExportConfig(
                    png_compression_level=cfg.marketing_png_compression
                )
                export_config = ExportConfig(
                    output_dir=Path(cfg.output_dir),
                    marketing_config=marketing_cfg
                )
                self.export_manager = ExportManager(export_config, io_utils)
                self.logger.info("ExportManager initialized (static config)")
            except Exception as e:
                self.logger.warning(f"ExportManager init failed, using direct I/O: {e}")
        elif self._export_manager_autotune_enabled:
            self.logger.info("ExportManager will be built JIT with autotune")

        self.logger.info(
            f"PipelineV2 init | device={self.device} autocast={self.autocast} "
            f"upscaler={type(self.upscaler).__name__} seg={type(self.segmenter).__name__ if self.segmenter else 'None'} "
            f"post_tile={cfg.post_tile} validate_ai={cfg.validate_ai} export_manager={self.export_manager is not None}"
        )

    def _validate_dependencies(self) -> None:
        """PRODUCTION SAFETY: Validate dependencies against vulnerable packages."""
        try:
            import importlib.metadata
            # Check for vulnerable packages
            vulnerable_packages = ["basicsr", "realesrgan", "gfpgan"]
            found_vulnerable = []
            
            for pkg in vulnerable_packages:
                try:
                    version = importlib.metadata.version(pkg)
                    found_vulnerable.append(f"{pkg}=={version}")
                except importlib.metadata.PackageNotFoundError:
                    pass
            
            if found_vulnerable:
                msg = (
                    f"⚠️  SECURITY WARNING: Vulnerable packages detected: {', '.join(found_vulnerable)}\n"
                    f"These packages have known CVE-2024-27763 vulnerabilities.\n"
                    f"Please use requirements-repo.txt for safe dependencies.\n"
                    f"Use --upscaler-backend torch instead of realesrgan."
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
                self.logger.error(msg)
        except Exception as e:
            self.logger.debug(f"Dependency validation check skipped: {e}")

    def _collect_reproducibility_metadata(self) -> Dict[str, object]:
        """Collect reproducibility metadata for production stamping."""
        metadata: Dict[str, object] = {}
        
        # Git commit hash
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=2, check=False
            )
            if result.returncode == 0:
                metadata["git_commit"] = result.stdout.strip()
        except Exception:
            pass
        
        # Config hash (deterministic)
        try:
            cfg_dict = asdict(self.cfg)
            cfg_json = json.dumps(cfg_dict, sort_keys=True)
            metadata["config_hash"] = hashlib.sha256(cfg_json.encode()).hexdigest()[:16]
        except Exception:
            pass
        
        # Device info
        try:
            import torch
            metadata["device"] = str(self.device)
            if torch.cuda.is_available() and self.device.type == "cuda":
                metadata["gpu_name"] = torch.cuda.get_device_name(self.device)
                metadata["cuda_version"] = torch.version.cuda
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                metadata["gpu_name"] = "Apple Silicon (MPS)"
        except Exception:
            metadata["device"] = str(getattr(self, "device", "unknown"))
        
        # Python version
        metadata["python_version"] = platform.python_version()
        
        # PyTorch version
        try:
            import torch
            metadata["torch_version"] = torch.__version__
        except Exception:
            pass
        
        # Model versions
        metadata["upscaler_backend"] = self.cfg.upscaler_backend
        if self.cfg.model_path:
            metadata["model_path"] = str(self.cfg.model_path)
            metadata["model_sha256"] = self.cfg.model_sha256 or "not_specified"
        
        # Tiling settings
        metadata["post_tile"] = self.cfg.post_tile
        metadata["post_overlap"] = self.cfg.post_overlap
        metadata["upscale_tile"] = self.cfg.tile
        metadata["upscale_tile_pad"] = self.cfg.tile_pad
        
        return metadata

    def _sync_for_timing(self) -> None:
        """Optional sync to make per-stage timings accurate on async devices."""
        if not getattr(self.cfg, "timing_sync_device", False):
            return
        try:
            import torch
            d = str(getattr(self, "device", "")).lower()
            if "cuda" in d and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif d.startswith("mps") and hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.synchronize()
        except Exception:
            # never let timing sync crash the pipeline
            return

    def _stage(self, report: dict, name: str):
        """
        Context manager recording wall time per stage into report['stage_times_sec'].
        Accumulates if the same stage is entered multiple times (e.g. tiled loops).
        """
        from contextlib import contextmanager
        from time import perf_counter

        @contextmanager
        def _stage_context():
            self._sync_for_timing()
            t0 = perf_counter()
            try:
                yield
            finally:
                self._sync_for_timing()
                dt = perf_counter() - t0
                st = report.setdefault("stage_times_sec", {})
                st[name] = float(st.get(name, 0.0)) + float(dt)

        return _stage_context()

    def _ensure_output_dir(self, out_dir: Path) -> None:
        """Create output dir only if writes are enabled."""
        if not getattr(self.cfg, "write_outputs", True):
            return
        out_dir.mkdir(parents=True, exist_ok=True)

    def _write_json(self, path: Path, obj: dict) -> None:
        """Write JSON only if writes are enabled."""
        if not getattr(self.cfg, "write_outputs", True):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2))

    def process_one(self, img_path: Path, depth_path: Optional[Path] = None) -> Dict[str, object]:
        """Process a single image file and write outputs to output_dir."""
        t0 = time.time()
        cfg = self.cfg

        # Initialize report early so stage_times_sec is always present
        report = {
            "status": "processing",
            "image": str(img_path),
            "stage_times_sec": {},
        }

        if not cfg.output_dir:
            raise ValueError("cfg.output_dir is required for file-based processing")

        img_path = Path(img_path)
        stem = img_path.stem
        out_dir = Path(cfg.output_dir)
        self._ensure_output_dir(out_dir)

        # Output paths - use ExportManager if available
        if self.export_manager:
            master_path = self.export_manager.get_master_path(stem)
            up_path = self.export_manager.get_upscaled_path(stem)
            marketing_path = self.export_manager.get_marketing_path(stem)
            preview_path = self.export_manager.get_preview_path(stem)
            report_path = self.export_manager.get_report_path(stem)
        else:
            master_path = out_dir / f"{stem}_master16.tif"
            up_path = out_dir / f"{stem}_upscaled16.tif"
            marketing_path = out_dir / f"{stem}_marketing.png"
            preview_path = out_dir / f"{stem}_preview.jpg"
            report_path = out_dir / f"{stem}_report.json"

        if cfg.skip_existing and master_path.exists() and up_path.exists() and (marketing_path.exists() or not cfg.save_marketing_png):
            self.logger.info(f"skip_existing: {img_path.name}")
            return {"status": "skipped", "image": str(img_path)}

        # Load image
        with self._stage(report, "io/read_input"):
            rgb01, info = io_utils.read_rgb_any(img_path)
            H, W = rgb01.shape[:2]
        float_gb = (H * W * 3 * 4) / 1e9
        if float_gb > float(cfg.warn_float_gb):
            self.logger.warning(
                f"Large image {W}x{H} may stress RAM/VRAM: ~{float_gb:.2f} GB per float32 RGB buffer"
            )
        
        # Phase 2 Slice 3: Build ExportManager JIT with autotune (if enabled)
        if self._export_manager_autotune_enabled and self.export_manager is None and EXPORT_MANAGER_AVAILABLE:
            with self._stage(report, "export/autotune"):
                try:
                    from transformation_portal.core.storage import (
                        autotune_export_config,
                        compute_image_stats,
                    )
                    
                    # Compute image stats
                    use_complexity = cfg.phase2.autotune_use_complexity if cfg.phase2 else True
                    image_stats = compute_image_stats(
                        img_path,
                        rgb_array=rgb01 if use_complexity else None
                    )
                    
                    # Autotune export config
                    export_config = autotune_export_config(
                        output_dir=Path(cfg.output_dir),
                        image_width=image_stats.width,
                        image_height=image_stats.height,
                        scene_complexity=image_stats.scene_complexity,
                        enable_adaptive=True,
                        marketing_png_compression=cfg.marketing_png_compression,
                    )
                    
                    self.export_manager = ExportManager(export_config, io_utils)
                    
                    # Store autotune metadata for report
                    report["export_autotune"] = {
                        "enabled": True,
                        "image_stats": {
                            "width": image_stats.width,
                            "height": image_stats.height,
                            "megapixels": image_stats.megapixels,
                            "scene_complexity": image_stats.scene_complexity,
                        },
                        "final_export_config": {
                            "tiff_tile_size": export_config.tiff_tile_size,
                            "tiff_compression": export_config.tiff_compression,
                            "use_atomic_image_writes": export_config.use_atomic_image_writes,
                            "use_atomic_report_writes": export_config.use_atomic_report_writes,
                        },
                    }
                    
                    complexity_str = f"{image_stats.scene_complexity:.3f}" if image_stats.scene_complexity is not None else "N/A"
                    self.logger.info(
                        f"ExportManager autotuned | {image_stats.megapixels:.1f}MP "
                        f"complexity={complexity_str} "
                        f"tiled={export_config.tiff_tile_size is not None} "
                        f"atomic={export_config.use_atomic_image_writes}"
                    )
                except Exception as e:
                    self.logger.warning(f"Autotune failed, using baseline config: {e}")
                    export_config = ExportConfig(output_dir=Path(cfg.output_dir))
                    self.export_manager = ExportManager(export_config, io_utils)
                    report["export_autotune"] = {"enabled": True, "error": str(e)}
        elif not self._export_manager_autotune_enabled:
            # Add metadata indicating autotune was disabled
            report["export_autotune"] = {"enabled": False}

        # Depth and weights
        with self._stage(report, "io/read_depth"):
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
        # Stage 3a: Legacy material segmentation
        with self._stage(report, "material/segmentation"):
            mods0: Optional[material_profiles.MaterialMods] = None
            if cfg.enable_material and self.segmenter is not None:
                try:
                    masks = self.segmenter.predict(rgb_t)
                    mods0 = material_profiles.build_material_mods(masks, cfg)
                except Exception as e:
                    self.logger.exception(f"Material segmentation failed for {img_path.name}: {e}")
                    mods0 = None
        
        # Stage 3b: Materials v2 integration (NEW)
        materials_v2_result = None
        materials_v2_metadata = {}
        
        if self.materials_v2_engine is not None:
            with self._stage(report, "material/materials_v2"):
                try:
                    # Generate task ID for caching
                    task_id = f"{stem}_materials_v2"
                    
                    # Check cache first
                    cached_result = None
                    if self.mask_cache_manager is not None:
                        try:
                            input_hash = self.mask_cache_manager.compute_input_hash(img_path)
                            if self.mask_cache_manager.is_cached(task_id, input_hash):
                                # Load from cache
                                cached_data = self.mask_cache_manager.load(task_id, input_hash)
                                if cached_data:
                                    self.logger.info(f"Materials v2 loaded from cache: {task_id}")
                                    # Reconstruct result from cache
                                    cached_result = cached_data
                        except Exception as e:
                            self.logger.debug(f"Cache lookup failed: {e}")
                    
                    # Generate if not cached
                    if cached_result is None:
                        # Perform segmentation
                        materials_v2_result = self.materials_v2_engine.segment_with_confidence(
                            image=rgb01,
                            task_id=task_id
                        )
                        
                        # Store metadata
                        materials_v2_metadata = {
                            'confidence_avg': materials_v2_result.metrics.confidence_avg,
                            'confidence_min': materials_v2_result.metrics.confidence_min,
                            'confidence_max': materials_v2_result.metrics.confidence_max,
                            'high_confidence_pct': materials_v2_result.metrics.high_confidence_pct,
                            'low_confidence_pct': materials_v2_result.metrics.low_confidence_pct,
                            'coverage_ratio': materials_v2_result.metrics.coverage_ratio,
                            'material_counts': materials_v2_result.metrics.material_counts,
                            'is_high_quality': materials_v2_result.metrics.is_high_quality(),
                            'original_size': materials_v2_result.original_size,
                            'segmentation_size': materials_v2_result.segmentation_size,
                            'upsampled': materials_v2_result.upsampled,
                        }
                        
                        # Save to cache if enabled
                        if self.mask_cache_manager is not None:
                            try:
                                input_hash = self.mask_cache_manager.compute_input_hash(img_path)
                                self.mask_cache_manager.save(
                                    task_id=task_id,
                                    input_hash=input_hash,
                                    masks=materials_v2_result.masks,
                                    confidences=materials_v2_result.confidences,
                                    metadata=materials_v2_metadata
                                )
                                self.logger.debug(f"Materials v2 saved to cache: {task_id}")
                            except Exception as e:
                                self.logger.debug(f"Cache save failed: {e}")
                    else:
                        # Use cached result
                        materials_v2_result = cached_result
                        materials_v2_metadata = cached_result.get('metadata', {}) if isinstance(cached_result, dict) else {}
                    
                except Exception as e:
                    # Graceful fallback: log warning and continue without Materials v2
                    self.logger.warning(f"Materials v2 failed for {img_path.name}: {e}; continuing without")
                    materials_v2_result = None
                    materials_v2_metadata = {'error': str(e), 'fallback': True}

        # Stage 3c: Materials V3 integration (NEW: PR-3A plan mode + PR-4B pixel ops)
        materials_v3_metadata = {}
        materials_v3_response_plan = {}
        materials_v3_pixel_ops = {}
        
        if self.materials_v3_engine is not None:
            with self._stage(report, "material/materials_v3"):
                try:
                    # Prepare segmentation result dict (matches MaterialsV3Engine.process() signature)
                    seg_result_for_v3 = {
                        'materials': {}
                    }
                    
                    # Populate with actual masks from Stage 3a segmenter output
                    if cfg.enable_material and self.segmenter is not None and 'masks' in locals():
                        # Convert torch masks (1,1,H,W) to numpy (H,W) float32
                        for material_name, mask_t in masks.items():
                            try:
                                # mask_t is torch.Tensor of shape (1,1,H,W) or (1,H,W)
                                mask_np = mask_t.cpu().numpy()
                                # Squeeze to (H,W)
                                if mask_np.ndim == 4:  # (1,1,H,W)
                                    mask_np = mask_np[0, 0]
                                elif mask_np.ndim == 3:  # (1,H,W)
                                    mask_np = mask_np[0]
                                seg_result_for_v3['materials'][material_name] = mask_np.astype(np.float32)
                            except Exception as e:
                                self.logger.debug(f"Failed to convert mask {material_name}: {e}")
                    
                    # Call Materials V3 engine (plan mode + optional pixel ops)
                    v3_result = self.materials_v3_engine.process(
                        image=rgb01,
                        segmentation_result=seg_result_for_v3,
                        depth_map=depth01 if depth01 is not None else None
                    )
                    
                    # Extract metadata from V3 result
                    # FIX: V3 engine emits 'materials_v3', not 'materials_v3_metadata'
                    if 'materials_v3' in v3_result:
                        materials_v3_metadata = v3_result['materials_v3']
                    elif 'materials_v3_metadata' in v3_result:
                        # Fallback for backward compatibility
                        materials_v3_metadata = v3_result['materials_v3_metadata']
                    
                    if 'materials_v3_response_plan' in v3_result:
                        materials_v3_response_plan = v3_result['materials_v3_response_plan']
                    
                    if 'materials_v3_pixel_ops' in v3_result:
                        materials_v3_pixel_ops = v3_result['materials_v3_pixel_ops']
                    
                    # Apply pixel operations if enabled (PR-4B)
                    # This modifies rgb01 in-place if glass response is applied
                    enhanced_rgb01, pixel_ops_stats = self.materials_v3_engine.apply_glass_response_if_enabled(
                        image=rgb01,
                        segmentation_result=v3_result,
                        response_plan=materials_v3_response_plan,
                    )
                    
                    # If pixel ops were applied, rebuild rgb_t for downstream grading/upscaling
                    if pixel_ops_stats.get('enabled', False):
                        rgb01 = enhanced_rgb01
                        rgb_t = torch_ops.to_torch_rgb(rgb01, self.device)
                        self.logger.info(f"Materials V3 pixel ops applied to {img_path.name}: {pixel_ops_stats.get('applied_to', [])}")
                        materials_v3_pixel_ops = pixel_ops_stats
                    
                    self.logger.debug(f"Materials V3 processed: {img_path.name}")
                    
                except Exception as e:
                    # Graceful fallback: log warning and continue without Materials V3
                    self.logger.warning(f"Materials V3 failed for {img_path.name}: {e}; continuing without")
                    materials_v3_metadata = {'error': str(e), 'fallback': True}

        # Grade at original resolution
        with self._stage(report, "grade/master"):
            with torch_ops.maybe_autocast(self.autocast, self.device):
                master_t = torch_ops.grade_core(rgb_t, w.wfg, w.wmid, w.wbg, cfg, mods=mods0)
                master_t = torch_ops.soft_clip01(master_t, cfg.soft_clip_knee)
                if mods0 is not None and cfg.enable_material:
                    master_t = torch_ops.material_highlight_compress(master_t, mods0.highlight_compress, knee=0.85)

            master01 = torch_ops.from_torch_rgb(master_t)
        
        # Write master and preview only if enabled
        if cfg.write_outputs:
            with self._stage(report, "export_master"):
                if cfg.save_master:
                    if self.export_manager:
                        self.export_manager.write_master(stem, master01)
                    else:
                        io_utils.atomic_write_rgb16_tiff(master_path, master01)

            # Preview (small JPG)
            with self._stage(report, "export_preview"):
                if cfg.save_preview_jpg:
                    try:
                        import cv2
                        scale = float(cfg.preview_scale)
                        if 0 < scale < 1.0:
                            ph, pw = int(round(H * scale)), int(round(W * scale))
                            prev = cv2.resize(master01, (pw, ph), interpolation=cv2.INTER_AREA)
                        else:
                            prev = master01
                        if self.export_manager:
                            self.export_manager.write_preview(stem, prev, quality=92)
                        else:
                            io_utils.atomic_write_jpg8(preview_path, prev, quality=92)
                    except Exception:
                        pass

        # VRAM cleanup before upscaling (critical for Materials v2)
        if self.materials_v2_engine is not None:
            with self._stage(report, "material/cleanup"):
                try:
                    self.materials_v2_engine.release_resources()
                    self.logger.debug("Materials v2 resources released before upscaling")
                except Exception as e:
                    self.logger.debug(f"Materials v2 cleanup failed: {e}")
        
        # Upscaling path
        with self._stage(report, "upscale/base"):
            # Base upsample: GPU bicubic
            with torch_ops.maybe_autocast(self.autocast, self.device):
                base_up = torch_ops.resize(master_t, (H * cfg.upscale, W * cfg.upscale), mode="bicubic", autocast=True).clamp(0.0, 1.0)

        ai_up = base_up
        ai_status = "none"
        if cfg.upscaler_backend != "none":
            with self._stage(report, f"upscale/{cfg.upscaler_backend}"):
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

        # Write upscaled outputs only if enabled
        if cfg.write_outputs:
            with self._stage(report, "export_upscaled"):
                if cfg.save_upscaled:
                    if self.export_manager:
                        self.export_manager.write_upscaled(stem, out01)
                    else:
                        io_utils.atomic_write_rgb16_tiff(up_path, out01)
            
            with self._stage(report, "export_marketing"):
                if cfg.save_marketing_png:
                    if self.export_manager:
                        self.export_manager.write_marketing_png(stem, out01)
                    else:
                        io_utils.atomic_write_png8(marketing_path, out01)

        # M0: Capture marketing export metadata if available
        marketing_metadata = None
        if self.export_manager and cfg.save_marketing_png:
            marketing_metadata = self.export_manager.get_marketing_metadata()
        
        # Update report with final status
        report.update({
            "status": "ok",
            "depth": str(depth_path) if depth_path else None,
            "zone_weights": w.source,
            "material_mods": mods0.source if mods0 is not None else None,
            "materials_v2_enabled": bool(self.materials_v2_engine),
            "materials_v2_metadata": materials_v2_metadata if materials_v2_metadata else None,
            "materials_v3_enabled": bool(self.materials_v3_engine),
            "materials_v3_metadata": materials_v3_metadata if materials_v3_metadata else None,
            "materials_v3_response_plan": materials_v3_response_plan if materials_v3_response_plan else None,
            "materials_v3_pixel_ops": materials_v3_pixel_ops if materials_v3_pixel_ops else None,
            "upscaler": ai_status,
            "ai_color_diff": color_diff,
            "ai_luma_diff": luma_diff,
            "timing_s": round(time.time() - t0, 3),
            "config": _config_to_json(cfg),
            "write_outputs": bool(cfg.write_outputs),
            "marketing_export": marketing_metadata,
        })

        # Backward compatibility alias
        report["stage_times"] = report.get("stage_times_sec", {})
        
        # Phase 2 timing instrumentation:
        # - timing_s: total execution time (float, preserved for backward compatibility)
        # - timing_stages_s: per-stage timings (dict[str, float])
        report["timing_stages_s"] = report.get("stage_times_sec", {})
        
        # Optional: structured timing object for future use
        report["timing"] = {
            "total_s": report["timing_s"],
            "stages_s": report.get("stage_times_sec", {}),
        }
        
        # PRODUCTION: Add reproducibility stamping
        report["reproducibility"] = self._repro_metadata.copy()
        report["reproducibility"]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        report["reproducibility"]["preset"] = str(cfg.preset.value)

        # Stage 6.5: Add EfficientSAM V3 fusion observability
        if hasattr(self.segmenter, "get_segmentation_v3_report"):
            report["segmentation_v3"] = self.segmenter.get_segmentation_v3_report()

        # Write report JSON only if enabled (after all fields added)
        if cfg.write_outputs:
            with self._stage(report, "export_report"):
                if self.export_manager:
                    self.export_manager.write_report(stem, report)
                else:
                    self._write_json(report_path, report)

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
