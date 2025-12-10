"""
Phase 2 Slice 2: ExportManager - Behavior-identical abstraction for I/O operations.

This module provides a clean interface layer for all pipeline export operations,
delegating to existing I/O implementations to ensure bit-identical outputs.

Architecture Design:
- Single responsibility: coordinate export operations
- Zero semantic changes to file formats or content
- Stage timing integration for observability
- Foundation for future optimizations (scratch dirs, async I/O, chunked BigTIFF)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class ExportConfig:
    """
    Export configuration for file naming and output control.
    
    Frozen dataclass ensures immutability for thread-safe usage.
    """
    output_dir: Path
    master_prefix: str = ""
    upscaled_prefix: str = ""
    preview_prefix: str = ""
    report_suffix: str = "_report.json"
    
    # Suffixes for compatibility with existing naming
    master_suffix: str = "_master16"
    upscaled_suffix: str = "_upscaled16"
    marketing_suffix: str = "_marketing"
    preview_jpg_suffix: str = "_preview"


class ExportManager:
    """
    Phase 2 Slice 2: Behavior-identical wrapper around existing I/O.
    
    Architectural Goals:
    1. Isolate all export operations in a single layer
    2. Maintain exact compatibility with existing file naming
    3. Delegate to proven I/O implementations (io_utils)
    4. Enable future optimizations without pipeline changes
    
    Non-Goals (Future Slices):
    - Scratch directory management
    - Async I/O
    - Chunked BigTIFF writing
    - Export queue management
    """
    
    def __init__(self, config: ExportConfig, io_utils_module: Any):
        """
        Initialize ExportManager with configuration and I/O backend.
        
        Args:
            config: Export configuration with paths and naming conventions
            io_utils_module: Module providing atomic_write_* functions
                            (dependency injection for testing)
        """
        self.config = config
        self._io = io_utils_module
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
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
