"""V2 pipeline runner for legacy enhancement workflow.

STUB IMPLEMENTATION - Critical classes to enable package imports.
Full implementation pending.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any


class V2Runner:
    """Runner for V2 enhancement pipeline.

    STUB IMPLEMENTATION - Full implementation pending.
    """

    def __init__(self):
        """Initialize V2 runner."""
        pass

    def run(
        self,
        input_path: Path,
        depth_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        preset: str = "default",
        device: str = "cpu",
        upscaler_backend: str = "default",
        log_file: Optional[Path] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run V2 enhancement pipeline.

        STUB: Not implemented.

        Args:
            input_path: Path to input image
            depth_dir: Directory containing depth map (not depth_path)
            output_dir: Output directory
            preset: Enhancement preset
            device: Device to use (cpu/cuda/mps)
            upscaler_backend: Upscaler backend to use
            log_file: Path to log file
            timeout: Timeout in seconds
            **kwargs: Additional arguments

        Returns:
            Dictionary with results and metadata (must include 'runtime_s')

        Raises:
            NotImplementedError: This is a stub implementation
        """
        raise NotImplementedError(
            "V2Runner.run() is a stub - full implementation pending. "
            "This module was created to enable package imports."
        )


def find_v2_report(output_dir: Path, image_key: str) -> Optional[Path]:
    """Find V2 report file for a given image key.

    STUB: Not implemented.

    Args:
        output_dir: Output directory to search
        image_key: Image identifier key

    Returns:
        Path to report file if found, None otherwise

    Raises:
        NotImplementedError: This is a stub implementation
    """
    raise NotImplementedError(
        "find_v2_report() is a stub - full implementation pending. "
        "This module was created to enable package imports."
    )
