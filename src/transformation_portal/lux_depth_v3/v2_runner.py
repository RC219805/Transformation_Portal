"""V2 pipeline runner for legacy enhancement workflow.

Provides interface to run V2 enhancement subprocess with timeout and error handling.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import subprocess
import time

logger = logging.getLogger(__name__)


class V2Runner:
    """Runner for V2 enhancement pipeline.

    Executes legacy V2 enhancement as a subprocess with timeout control.
    """

    def __init__(self):
        """Initialize V2 runner."""
        # Determine V2 script path (relative to repository root)
        self.repo_root = Path(__file__).resolve().parent.parent.parent.parent

        # Common V2 script locations to try
        self.script_candidates = [
            self.repo_root / "scripts" / "enhance_image.py",
            self.repo_root / "scripts" / "pipelines" / "tiff_enhancement_pipeline_v2.py",
            self.repo_root / "lux_render_pipeline.py",
        ]

        # Find first available script
        self.v2_script = None
        for candidate in self.script_candidates:
            if candidate.exists():
                self.v2_script = candidate
                logger.info(f"Found V2 enhancement script: {self.v2_script}")
                break

        if self.v2_script is None:
            logger.warning(
                "No V2 enhancement script found. V2Runner will operate in mock mode. "
                f"Searched: {[str(c) for c in self.script_candidates]}"
            )

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
        """
        start_time = time.time()

        # If no V2 script found, return mock success
        if self.v2_script is None:
            logger.warning("V2Runner in mock mode - no actual processing performed")
            runtime_s = time.time() - start_time
            return {
                "status": "ok",
                "runtime_s": runtime_s,
                "mock": True,
                "message": "V2 script not found - mock mode",
            }

        # Ensure output directory exists
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        # Note: This is a generic command structure. Actual V2 scripts may have different APIs.
        # We'll try a common pattern and handle errors gracefully.
        cmd = [
            "python",
            str(self.v2_script),
            "--input", str(input_path),
        ]

        if output_dir:
            cmd.extend(["--output", str(output_dir)])

        if depth_dir:
            cmd.extend(["--depth-dir", str(depth_dir)])

        if preset and preset != "default":
            cmd.extend(["--preset", preset])

        if device and device != "cpu":
            cmd.extend(["--device", device])

        if upscaler_backend and upscaler_backend != "default":
            cmd.extend(["--upscaler", upscaler_backend])

        logger.info(f"Running V2 enhancement: {' '.join(cmd)}")

        try:
            # Run subprocess with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,  # Don't raise on non-zero exit
            )

            runtime_s = time.time() - start_time

            # Log output
            if log_file:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, 'w') as f:
                    f.write(f"Command: {' '.join(cmd)}\n")
                    f.write(f"Exit code: {result.returncode}\n")
                    f.write(f"Runtime: {runtime_s:.2f}s\n")
                    f.write("\n--- STDOUT ---\n")
                    f.write(result.stdout)
                    f.write("\n--- STDERR ---\n")
                    f.write(result.stderr)

            # Check for errors
            if result.returncode != 0:
                error_msg = f"V2 enhancement failed with exit code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr[:500]}"

                logger.error(error_msg)

                return {
                    "status": "error",
                    "runtime_s": runtime_s,
                    "error": error_msg,
                    "exit_code": result.returncode,
                    "stdout": result.stdout[:1000] if result.stdout else "",
                    "stderr": result.stderr[:1000] if result.stderr else "",
                }

            # Success
            logger.info(f"V2 enhancement completed in {runtime_s:.2f}s")

            return {
                "status": "ok",
                "runtime_s": runtime_s,
                "stdout": result.stdout[:1000] if result.stdout else "",
            }

        except subprocess.TimeoutExpired:
            runtime_s = time.time() - start_time
            error_msg = f"V2 enhancement timed out after {timeout}s"
            logger.error(error_msg)

            return {
                "status": "timeout",
                "runtime_s": runtime_s,
                "error": error_msg,
                "timeout": timeout,
            }

        except Exception as e:
            runtime_s = time.time() - start_time
            error_msg = f"V2 enhancement failed: {e}"
            logger.error(error_msg)

            return {
                "status": "error",
                "runtime_s": runtime_s,
                "error": error_msg,
            }


def find_v2_report(output_dir: Path, image_key: str) -> Optional[Path]:
    """Find V2 report file for a given image key.

    Args:
        output_dir: Output directory to search
        image_key: Image identifier key

    Returns:
        Path to report file if found, None otherwise
    """
    output_dir = Path(output_dir)

    if not output_dir.exists():
        return None

    # Common report file patterns
    patterns = [
        f"{image_key}_report.json",
        f"{image_key}.json",
        f"report_{image_key}.json",
        f"{image_key}_metadata.json",
    ]

    for pattern in patterns:
        report_path = output_dir / pattern
        if report_path.exists():
            logger.debug(f"Found V2 report: {report_path}")
            return report_path

    # Search recursively if not found at top level
    for json_file in output_dir.rglob("*.json"):
        if image_key in json_file.stem:
            logger.debug(f"Found V2 report (recursive): {json_file}")
            return json_file

    logger.debug(f"No V2 report found for {image_key} in {output_dir}")
    return None
