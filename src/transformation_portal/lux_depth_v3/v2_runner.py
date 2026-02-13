"""V2 legacy pipeline runner.

Wraps the legacy depth-aware enhancement CLI script via subprocess.
Provides safe, testable interface for invoking V2 enhancement pipeline.

Design:
- No heavy ML dependencies (subprocess only)
- Robust repo root resolution
- Comprehensive error handling with command context
- Report JSON discovery and merging
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class V2Runner:
    """Wrapper for legacy V2 depth-aware enhancement pipeline.

    Invokes scripts/enhance_image.py via subprocess with controlled
    argument passing and comprehensive error handling.

    Attributes:
        repo_root: Repository root directory
        script_path: Path to enhance_image.py script
    """

    def __init__(self):
        """Initialize V2 runner with repo root and script path resolution."""
        self.repo_root = self._find_repo_root()
        self.script_path = self.repo_root / "scripts" / "enhance_image.py"

        if not self.script_path.exists():
            logger.warning(
                f"V2 enhancement script not found: {self.script_path}. " f"run() will raise FileNotFoundError if called."
            )

    def _find_repo_root(self) -> Path:
        """Find repository root by walking parents.

        Searches for markers: .git directory, pyproject.toml, or README.md.

        Returns:
            Repository root path

        Raises:
            RuntimeError: If repo root cannot be determined
        """
        # Start from this file's location
        current = Path(__file__).resolve()

        # Walk up parent directories
        for parent in [current] + list(current.parents):
            # Check for common repo markers
            if any(
                [
                    (parent / ".git").exists(),
                    (parent / "pyproject.toml").exists(),
                    (parent / "README.md").exists() and (parent / "src").exists(),
                ]
            ):
                logger.debug(f"Found repo root: {parent}")
                return parent

        # Fallback: assume we're in src/transformation_portal/lux_depth_v3/
        # Go up 3 levels
        fallback = current.parents[3]
        logger.warning(f"Could not find repo root markers, falling back to: {fallback}")
        return fallback

    def run(
        self,
        input_path: Path,
        depth_dir: Optional[Path],
        output_dir: Path,
        preset: str = "default",
        device: str = "cpu",
        upscaler_backend: Optional[str] = None,
        log_file: Optional[Path] = None,
        timeout: Optional[float] = None,
        masks_dir: Optional[Path] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run V2 depth-aware enhancement pipeline.

        Args:
            input_path: Input image path
            depth_dir: Directory containing depth maps (optional)
            output_dir: Output directory (required, will be created)
            preset: Enhancement preset name
            device: Device for processing (cpu/cuda/mps)
            upscaler_backend: Upscaler backend (optional)
            log_file: Log file path (optional)
            timeout: Subprocess timeout in seconds (optional)
            masks_dir: Directory containing material masks (optional, NPZ format)
            **kwargs: Additional arguments (reserved)

        Returns:
            Dict containing:
                - runtime_s: Execution time
                - status: "success"
                - report_path: Path to JSON report (if found)
                - <report fields>: Merged from report JSON (if found)
                - stdout/stderr: Process output (if report not found)

        Raises:
            FileNotFoundError: If enhance_image.py script missing
            RuntimeError: If subprocess fails
            TimeoutError: If subprocess times out
        """
        # Verify script exists
        if not self.script_path.exists():
            raise FileNotFoundError(
                f"V2 enhancement script not found: {self.script_path}. "
                f"Expected location: scripts/enhance_image.py in repo root {self.repo_root}"
            )

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [sys.executable, str(self.script_path), str(input_path)]

        # Add optional arguments (only if provided)
        if depth_dir is not None:
            cmd.extend(["--depth-dir", str(depth_dir)])

        # output_dir is required
        cmd.extend(["--output-dir", str(output_dir)])

        cmd.extend(["--preset", preset])
        cmd.extend(["--device", device])

        if upscaler_backend is not None:
            cmd.extend(["--upscaler", upscaler_backend])

        if log_file is not None:
            cmd.extend(["--log-file", str(log_file)])

        # Add masks directory if provided (Materials V3 integration)
        if masks_dir is not None:
            cmd.extend(["--masks-dir", str(masks_dir)])

        logger.info(f"Running V2 enhancement: {' '.join(cmd)}")

        # Execute subprocess with timing
        start_time = time.perf_counter()

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
            runtime_s = time.perf_counter() - start_time

            logger.info(f"V2 enhancement completed in {runtime_s:.2f}s")

        except subprocess.CalledProcessError as e:
            runtime_s = time.perf_counter() - start_time

            # Extract error info
            error_msg = e.stderr if e.stderr else e.stdout
            cmd_str = " ".join(cmd)

            raise RuntimeError(
                f"V2 enhancement failed (returncode={e.returncode}, runtime={runtime_s:.2f}s)\n"
                f"Command: {cmd_str}\n"
                f"Error output:\n{error_msg}"
            ) from e

        except subprocess.TimeoutExpired as e:
            runtime_s = time.perf_counter() - start_time

            # Extract partial output if available
            partial_stdout = e.stdout if hasattr(e, "stdout") and e.stdout else ""
            partial_stderr = e.stderr if hasattr(e, "stderr") and e.stderr else ""

            raise TimeoutError(
                f"V2 enhancement timed out after {timeout}s (partial runtime={runtime_s:.2f}s)\n"
                f"Partial stdout: {partial_stdout[:500]}\n"
                f"Partial stderr: {partial_stderr[:500]}"
            ) from e

        # Try to find and merge report JSON
        report_path = find_v2_report(output_dir, input_path.stem)

        if report_path:
            logger.info(f"Found V2 report: {report_path}")
            try:
                with open(report_path, "r") as f:
                    report_data = json.load(f)

                # Merge report with runtime info
                return {**report_data, "runtime_s": runtime_s, "status": "success", "report_path": str(report_path)}
            except Exception as e:
                logger.warning(f"Failed to load report JSON: {e}")
                # Fall through to stdout/stderr return
        else:
            logger.info(f"No V2 report found for {input_path.stem} in {output_dir}")

        # Return basic success info with process output
        return {
            "runtime_s": runtime_s,
            "status": "success",
            "report_path": None,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


def find_v2_report(output_dir: Path, image_key: str) -> Optional[Path]:
    """Find V2 enhancement report JSON.

    Searches for {image_key}_report.json in output directory.
    Falls back to recursive search if not found at top level.

    Args:
        output_dir: Output directory to search
        image_key: Base image filename (stem)

    Returns:
        Path to report JSON or None if not found
    """
    output_dir = Path(output_dir)

    # Try direct match first (fast path)
    direct_match = output_dir / f"{image_key}_report.json"
    if direct_match.exists():
        return direct_match

    # Try recursive search (handles nested output structures)
    # Limit depth to avoid excessive scanning
    for report_path in output_dir.glob("**/*_report.json"):
        if report_path.stem == f"{image_key}_report":
            return report_path

    return None
