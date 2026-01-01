"""V2 subprocess runner with robust logging.

Handles subprocess invocation of lux_depth_v2 with proper stdout/stderr
capture, timeout handling, and log file management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import subprocess
import sys
import time
import logging

logger = logging.getLogger(__name__)


class V2RunnerError(Exception):
    """Error running V2 enhancement pipeline."""

    pass


class V2Runner:
    """Subprocess runner for lux_depth_v2 enhancement."""

    def __init__(
        self,
        v2_module_path: Optional[Path] = None,
        python_exe: Optional[str] = None,
    ):
        """Initialize V2 runner.

        Args:
            v2_module_path: Path to lux_depth_v2 module (default: auto-detect)
            python_exe: Python executable to use (default: sys.executable)
        """
        self.python_exe = python_exe or sys.executable
        self.v2_module_path = v2_module_path

        # Auto-detect V2 module path
        if self.v2_module_path is None:
            # Assume lux_depth_v2 is sibling to lux_depth_v3
            v3_path = Path(__file__).parent.parent
            v2_path = v3_path.parent / "lux_depth_v2"
            if v2_path.exists() and (v2_path / "__init__.py").exists():
                self.v2_module_path = v2_path
            else:
                # Try to import to verify it's available
                try:
                    import lux_depth_v2  # noqa: F401

                    self.v2_module_path = None  # Will use module import
                except ImportError:
                    raise V2RunnerError("Cannot find lux_depth_v2 module. Ensure it's installed or specify v2_module_path.")

    def run(
        self,
        input_path: Path,
        depth_dir: Path,
        output_dir: Path,
        preset: str = "production_ultra",
        device: str = "auto",
        upscaler_backend: str = "torch",
        log_file: Optional[Path] = None,
        timeout: Optional[float] = None,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run V2 enhancement on a single image.

        Args:
            input_path: Input image path
            depth_dir: Directory containing {stem}_depth.png
            output_dir: Output directory for V2 results
            preset: V2 preset name
            device: Device (auto, cuda, cpu)
            upscaler_backend: Upscaler backend
            log_file: Optional log file for stdout/stderr
            timeout: Optional timeout in seconds
            extra_args: Additional CLI arguments

        Returns:
            Dictionary with status, runtime, and output info

        Raises:
            V2RunnerError: If V2 fails
        """
        # Build command
        cmd = [
            self.python_exe,
            "-m",
            "lux_depth_v2.cli",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--preset",
            preset,
            "--device",
            device,
            "--upscaler-backend",
            upscaler_backend,
        ]

        # Only include --depth-dir if it's a valid path (not None)
        if depth_dir is not None:
            cmd.extend(["--depth-dir", str(depth_dir)])

        if extra_args:
            cmd.extend(extra_args)

        logger.info(f"Running V2: {' '.join(cmd)}")

        # Run subprocess with logging
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.v2_module_path,
            )
            runtime_s = time.time() - start_time

            # Write logs if requested
            if log_file:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, "w") as f:
                    f.write("=== STDOUT ===\n")
                    f.write(result.stdout)
                    f.write("\n=== STDERR ===\n")
                    f.write(result.stderr)
                    f.write(f"\n=== EXIT CODE: {result.returncode} ===\n")
                logger.info(f"Wrote V2 logs to {log_file}")

            # Check for errors
            if result.returncode != 0:
                error_msg = f"V2 failed with exit code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nStderr: {result.stderr[:500]}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "runtime_s": runtime_s,
                    "exit_code": result.returncode,
                    "error": error_msg,
                }

            logger.info(f"V2 completed successfully in {runtime_s:.1f}s")
            return {
                "status": "ok",
                "runtime_s": runtime_s,
                "exit_code": 0,
            }

        except subprocess.TimeoutExpired:
            runtime_s = time.time() - start_time
            error_msg = f"V2 timed out after {timeout}s"
            logger.error(error_msg)
            # Note: subprocess.run with timeout does not automatically kill the process
            # The process may still be running as a zombie process
            logger.warning("V2 process may still be running after timeout")
            return {
                "status": "error",
                "runtime_s": runtime_s,
                "error": error_msg,
            }

        except Exception as e:
            runtime_s = time.time() - start_time
            error_msg = f"V2 runner error: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "runtime_s": runtime_s,
                "error": error_msg,
            }


def find_v2_report(output_dir: Path, stem: str) -> Optional[Path]:
    """Find V2 report JSON in output directory.

    Args:
        output_dir: V2 output directory
        stem: Image stem

    Returns:
        Path to report JSON if found, None otherwise
    """
    # Common V2 report naming patterns
    patterns = [
        f"{stem}_report.json",
        f"{stem}_metadata.json",
        f"report_{stem}.json",
    ]

    for pattern in patterns:
        report_path = output_dir / pattern
        if report_path.exists():
            return report_path

    # Search recursively
    for json_file in output_dir.rglob("*.json"):
        if stem in json_file.stem:
            return json_file

    return None
