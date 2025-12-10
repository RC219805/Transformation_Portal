"""
Phase 2 Slice 3 PR-1: ExportManager with config validation and infrastructure.

This module provides a clean interface layer for all pipeline export operations,
with Slice 3 optimization knobs (all disabled by default).

Architecture Evolution:
- Slice 2: Isolated export operations (behavior-identical)
- Slice 3 PR-1: Config + infrastructure (behavior-neutral, validation only)
- Slice 3 PR-2: Tiled TIFF + atomic writes (opt-in optimizations)
- Slice 3 PR-3: Async flush (opt-in parallelism)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class ExportConfig:
    """
    Export configuration with Slice 3 optimization knobs.
    
    Slice 3 PR-1: All optimization flags default OFF/None for strict backward compatibility.
    Future PRs will wire the actual optimization implementations.
    
    Frozen dataclass ensures immutability for thread-safe usage.
    """
    output_dir: Path
    
    # Slice 2 fields (existing)
    master_prefix: str = ""
    upscaled_prefix: str = ""
    preview_prefix: str = ""
    report_suffix: str = "_report.json"
    master_suffix: str = "_master16"
    upscaled_suffix: str = "_upscaled16"
    marketing_suffix: str = "_marketing"
    preview_jpg_suffix: str = "_preview"
    
    # Slice 3 PR-1: Tiered storage / scratch (infrastructure only)
    enable_tiered_storage: bool = False
    scratch_dir: Optional[Path] = None
    require_scratch_on_enable: bool = True
    
    # Slice 3 PR-1: TIFF tiling / compression (bounds checked, not yet used)
    tiff_tile_size: Optional[int] = None
    tiff_tile_size_min: int = 128
    tiff_tile_size_max: int = 1024
    tiff_compression: Optional[str] = None  # e.g., "lzw", "zstd", "deflate"
    
    # Slice 3 PR-1: Atomic writes (validated, not yet active)
    use_atomic_image_writes: bool = False
    use_atomic_report_writes: bool = False
    
    # Slice 3 PR-1: Async flush (skeleton only, wired in PR-3)
    async_flush: bool = False
    max_async_workers: int = 2


class ExportManager:
    """
    Phase 2 Slice 3 PR-1: Config validation + infrastructure (behavior-neutral).
    
    All optimization knobs are wired but disabled by default.
    This PR only adds validation and skeleton helpers - no behavior changes.
    
    Architectural Goals:
    1. Isolate all export operations in a single layer
    2. Maintain exact compatibility with existing file naming
    3. Delegate to proven I/O implementations (io_utils)
    4. Validate configuration early (fail-fast on misconfiguration)
    5. Enable future optimizations without pipeline changes
    
    Non-Goals (Future Slices):
    - Scratch directory management
    - Async I/O
    - Chunked BigTIFF writing
    - Export queue management
    """
    
    def __init__(self, config: ExportConfig, io_utils_module: Any):
        """
        Initialize ExportManager with configuration and I/O backend.
        
        Slice 3 PR-1: Added config validation (fail-fast on misconfiguration).
        
        Args:
            config: Export configuration with paths and naming conventions
            io_utils_module: Module providing atomic_write_* functions
                            (dependency injection for testing)
        
        Raises:
            ValueError: If config validation fails (scratch requirement, tile bounds, etc.)
        """
        self.config = config
        self._io = io_utils_module
        
        # Slice 3 PR-1: Validate configuration early
        self._validate_config()
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Slice 3 PR-1: Async executor placeholder (wired in PR-3)
        self._executor = None
    
    # ------------------------------------------------------------------ #
    # Slice 3 PR-1: Config validation (fail-fast on misconfiguration)
    # ------------------------------------------------------------------ #
    
    def _validate_config(self) -> None:
        """
        Validate ExportConfig on initialization.
        
        Slice 3 PR-1: Ensures misconfiguration fails at init, not during exports.
        No behavior changes - only validation logic.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Tiered storage scratch requirement
        if self.config.enable_tiered_storage and self.config.require_scratch_on_enable:
            if self.config.scratch_dir is None:
                raise ValueError(
                    "ExportConfig.enable_tiered_storage=True requires "
                    "scratch_dir when require_scratch_on_enable=True. "
                    "Provide scratch_dir or set require_scratch_on_enable=False."
                )
        
        # TIFF tile size bounds validation
        if self.config.tiff_tile_size is not None:
            ts = int(self.config.tiff_tile_size)
            if ts < self.config.tiff_tile_size_min or ts > self.config.tiff_tile_size_max:
                raise ValueError(
                    f"tiff_tile_size={ts} must be between "
                    f"{self.config.tiff_tile_size_min} and {self.config.tiff_tile_size_max}. "
                    f"Valid range prevents pathological tile counts (too small) "
                    f"or inefficient 1x1-tile bugs (too large)."
                )
        
        # Async workers sanity check
        if self.config.max_async_workers < 1:
            raise ValueError("max_async_workers must be >= 1")
    
    # ------------------------------------------------------------------ #
    # Slice 3 PR-1: Skeleton helpers (infrastructure only, no optimization yet)
    # ------------------------------------------------------------------ #
    
    def _resolve_scratch_path(self, final_path: Path) -> Path:
        """
        Slice 3 PR-1: Skeleton for tiered storage path resolution.
        
        For now, returns final_path directly (tiered storage not yet active).
        PR-2 will wire actual scratch path logic when enable_tiered_storage=True.
        
        Args:
            final_path: Intended final output path
        
        Returns:
            Path where file should be written (scratch or final)
        """
        # PR-1: Always return final path (no behavior change)
        return final_path
    
    def _atomic_move(self, src: Path, dst: Path) -> None:
        """
        Slice 3 PR-1: Skeleton for atomic file move.
        
        Uses Path.replace() which is atomic on POSIX when src/dst on same filesystem.
        PR-2 will add error handling and logging.
        
        Args:
            src: Source path (typically in scratch dir)
            dst: Destination path (final output location)
        """
        # PR-1: Simple replace, same as current behavior
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)
    
    def cleanup_scratch(self) -> None:
        """
        Slice 3 PR-1: Placeholder for scratch directory cleanup.
        
        PR-2 will implement actual cleanup logic when scratch_dir is active.
        Operator/orchestrator can call this manually or on job completion.
        """
        # PR-1: No-op (scratch not yet active)
        return
    
    def close(self) -> None:
        """
        Slice 3 PR-1: Resource cleanup hook.
        
        MUST be called by:
        - Pipeline shutdown
        - Batch job cleanup
        - Error paths (try/finally blocks)
        
        PR-3 will wire async executor shutdown here.
        """
        if getattr(self, "_executor", None) is not None:
            try:
                self._executor.shutdown(wait=True)
                self._executor = None
            except Exception:
                # Swallow shutdown errors to avoid masking primary exceptions
                pass
    
    # ------------------------------------------------------------------ #
    # Write methods (unchanged from Slice 2 for PR-1)
    # ------------------------------------------------------------------ #
    
    def write_master(self, stem: str, master_arr: np.ndarray, compression: str = "deflate") -> Path:
        """
        Write 16-bit master TIFF (behavior-identical to pipeline.py:467).
        
        Args:
            stem: Base filename without extension
            master_arr: RGB float32 array in [0, 1]
            compression: TIFF compression (deflate, lzw, none)
        
        Returns:
            Path to written file
        
        Raises:
            RuntimeError: If I/O dependencies missing
            OSError: If write fails
        """
        filename = f"{self.config.master_prefix}{stem}{self.config.master_suffix}.tif"
        path = self.config.output_dir / filename
        
        # Delegate to existing atomic writer (bit-identical behavior)
        self._io.atomic_write_rgb16_tiff(path, master_arr, compression=compression)
        return path
    
    def write_upscaled(self, stem: str, upscaled_arr: np.ndarray, compression: str = "deflate") -> Path:
        """
        Write 16-bit upscaled TIFF (behavior-identical to pipeline.py:569).
        
        Args:
            stem: Base filename without extension
            upscaled_arr: RGB float32 array in [0, 1]
            compression: TIFF compression
        
        Returns:
            Path to written file
        """
        filename = f"{self.config.upscaled_prefix}{stem}{self.config.upscaled_suffix}.tif"
        path = self.config.output_dir / filename
        
        self._io.atomic_write_rgb16_tiff(path, upscaled_arr, compression=compression)
        return path
    
    def write_preview(self, stem: str, preview_arr: np.ndarray, quality: int = 92) -> Path:
        """
        Write preview JPG (behavior-identical to pipeline.py:480).
        
        Args:
            stem: Base filename without extension
            preview_arr: RGB float32 array in [0, 1]
            quality: JPEG quality (1-100)
        
        Returns:
            Path to written file
        """
        filename = f"{self.config.preview_prefix}{stem}{self.config.preview_jpg_suffix}.jpg"
        path = self.config.output_dir / filename
        
        self._io.atomic_write_jpg8(path, preview_arr, quality=quality)
        return path
    
    def write_marketing_png(self, stem: str, png_arr: np.ndarray) -> Path:
        """
        Write 8-bit marketing PNG (behavior-identical to pipeline.py:571).
        
        Args:
            stem: Base filename without extension
            png_arr: RGB float32 array in [0, 1]
        
        Returns:
            Path to written file
        """
        filename = f"{self.config.upscaled_prefix}{stem}{self.config.marketing_suffix}.png"
        path = self.config.output_dir / filename
        
        self._io.atomic_write_png8(path, png_arr)
        return path
    
    def write_report(self, stem: str, report_dict: Dict[str, Any]) -> Path:
        """
        Write processing report JSON (behavior-identical to pipeline.py:611).
        
        Args:
            stem: Base filename without extension
            report_dict: Report data structure
        
        Returns:
            Path to written file
        """
        filename = f"{stem}{self.config.report_suffix}"
        path = self.config.output_dir / filename
        
        # Match existing behavior: direct write with indent=2 (non-atomic)
        # Note: atomic writes (.tmp + replace) can be added in Slice 3
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report_dict, indent=2))
        
        return path
    
    def get_master_path(self, stem: str) -> Path:
        """Get expected master output path (for skip_existing check)."""
        filename = f"{self.config.master_prefix}{stem}{self.config.master_suffix}.tif"
        return self.config.output_dir / filename
    
    def get_upscaled_path(self, stem: str) -> Path:
        """Get expected upscaled output path (for skip_existing check)."""
        filename = f"{self.config.upscaled_prefix}{stem}{self.config.upscaled_suffix}.tif"
        return self.config.output_dir / filename
    
    def get_marketing_path(self, stem: str) -> Path:
        """Get expected marketing PNG path (for skip_existing check)."""
        filename = f"{self.config.upscaled_prefix}{stem}{self.config.marketing_suffix}.png"
        return self.config.output_dir / filename
    
    def get_preview_path(self, stem: str) -> Path:
        """Get expected preview JPG path."""
        filename = f"{self.config.preview_prefix}{stem}{self.config.preview_jpg_suffix}.jpg"
        return self.config.output_dir / filename
    
    def get_report_path(self, stem: str) -> Path:
        """Get expected report JSON path."""
        filename = f"{stem}{self.config.report_suffix}"
        return self.config.output_dir / filename
