"""V2 legacy pipeline runner.

Wraps the legacy depth-aware enhancement CLI script via subprocess.
Provides safe, testable interface for invoking V2 enhancement pipeline.

Design:
- No heavy ML dependencies (subprocess only)
- Robust repo root resolution
- Comprehensive error handling with command context
- Report JSON discovery and merging
- Path validation to prevent traversal attacks
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from transformation_portal.core.security.path import safe_resolve_path
from transformation_portal.core.security.validation import ValidationError

from .path_aliasing import normalize_lexical_path

logger = logging.getLogger(__name__)


class V2Runner:
    """Wrapper for legacy V2 depth-aware enhancement pipeline.

    Invokes scripts/enhance_image.py via subprocess with controlled
    argument passing and comprehensive error handling.

    Attributes:
        repo_root: Repository root directory
        script_path: Path to enhance_image.py script
    """

    def __init__(self) -> None:
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
        masks_file: Optional[Path] = None,
        asset_key: Optional[str] = None,
        output_bit_depth: Optional[int] = None,
        **kwargs: Any,
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
            masks_file: Explicit path to material masks NPZ file (optional, Materials V3 integration)
            asset_key: Canonical asset key for depth/report resolution (optional,
                defaults to input_path.stem if not provided). When provided, aligns
                depth lookup and report naming with orchestrator's canonical identity.
            output_bit_depth: Explicit enhanced-image encoding depth (8 or 16).
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

        Note:
            Path validation is performed for all paths, but paths outside the
            repository root are allowed with a warning logged. This is intentional
            as user data directories may legitimately be outside the repository.
        """
        # Verify script exists
        if not self.script_path.exists():
            raise FileNotFoundError(
                f"V2 enhancement script not found: {self.script_path}. "
                f"Expected location: scripts/enhance_image.py in repo root {self.repo_root}"
            )

        # Validate all paths to prevent traversal attacks
        # Use repo_root as the allowed root for path validation
        try:
            validated_input = safe_resolve_path(input_path, allowed_root=self.repo_root)
            logger.debug(f"Validated input path: {validated_input}")
        except ValidationError:
            # Input paths may legitimately be outside repo (user data directories)
            # Fall back to resolving and logging
            validated_input = normalize_lexical_path(input_path)
            logger.debug(f"Input path outside repo root (allowed): {validated_input}")

        # Output directory validation - must be within repo or explicitly allowed
        try:
            validated_output = safe_resolve_path(output_dir, allowed_root=self.repo_root)
        except ValidationError:
            # Output paths may also be outside repo - resolve but warn
            validated_output = normalize_lexical_path(output_dir)
            logger.warning(
                f"Output directory outside repo root: {validated_output}. "
                "Consider using paths within the repository for better isolation."
            )

        # Validate depth_dir if provided
        validated_depth_dir = None
        if depth_dir is not None:
            try:
                validated_depth_dir = safe_resolve_path(depth_dir, allowed_root=self.repo_root)
            except ValidationError:
                validated_depth_dir = normalize_lexical_path(depth_dir)
                logger.debug(f"Depth directory outside repo root (allowed): {validated_depth_dir}")

        # Validate masks_file if provided
        validated_masks_file = None
        if masks_file is not None:
            try:
                validated_masks_file = safe_resolve_path(masks_file, allowed_root=self.repo_root)
            except ValidationError:
                validated_masks_file = normalize_lexical_path(masks_file)
                logger.debug(f"Masks file outside repo root (allowed): {validated_masks_file}")

        # Validate log_file if provided
        validated_log_file = None
        if log_file is not None:
            try:
                validated_log_file = safe_resolve_path(log_file, allowed_root=self.repo_root)
            except ValidationError:
                validated_log_file = normalize_lexical_path(log_file)
                logger.debug(f"Log file outside repo root (allowed): {validated_log_file}")

        # Ensure output directory exists
        validated_output.mkdir(parents=True, exist_ok=True)

        # Validate asset_key if provided (must be stem-like, not path-like)
        # Intentionally check for both / and \ on all platforms to prevent
        # cross-platform path injection (e.g., Windows-style path on Unix)
        validated_asset_key = None
        if asset_key is not None:
            validated_asset_key = str(asset_key).strip()
            if not validated_asset_key:
                validated_asset_key = None
            elif "/" in validated_asset_key or "\\" in validated_asset_key:
                raise ValueError(f"asset_key must be a stem-like identifier (no path separators), got: {asset_key!r}")

        # Build command using validated paths
        cmd = [sys.executable, str(self.script_path), str(validated_input)]

        # Add optional arguments (only if provided)
        if validated_depth_dir is not None:
            cmd.extend(["--depth-dir", str(validated_depth_dir)])

        # output_dir is required
        cmd.extend(["--output-dir", str(validated_output)])

        cmd.extend(["--preset", preset])
        cmd.extend(["--device", device])

        if upscaler_backend is not None:
            cmd.extend(["--upscaler", upscaler_backend])

        if validated_log_file is not None:
            cmd.extend(["--log-file", str(validated_log_file)])

        # Add explicit mask file path if provided (Materials V3 integration)
        # Uses explicit NPZ path to eliminate filename coupling
        if validated_masks_file is not None:
            cmd.extend(["--masks-file", str(validated_masks_file)])

        # Add canonical asset key if provided (depth/report identity alignment)
        # This aligns V2 depth lookup and report naming with orchestrator's canonical identity
        if validated_asset_key is not None:
            cmd.extend(["--asset-key", validated_asset_key])
        if output_bit_depth is not None:
            if isinstance(output_bit_depth, bool) or not isinstance(output_bit_depth, int) or output_bit_depth not in {8, 16}:
                raise ValueError("output_bit_depth must be 8 or 16")
            cmd.extend(["--output-bit-depth", str(output_bit_depth)])

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

            # Extract partial output if available (may be bytes or str)
            partial_stdout_raw = e.stdout if hasattr(e, "stdout") and e.stdout else b""
            partial_stderr_raw = e.stderr if hasattr(e, "stderr") and e.stderr else b""

            # Decode if bytes
            partial_stdout = (
                partial_stdout_raw.decode("utf-8", errors="replace")
                if isinstance(partial_stdout_raw, bytes)
                else str(partial_stdout_raw)
            )
            partial_stderr = (
                partial_stderr_raw.decode("utf-8", errors="replace")
                if isinstance(partial_stderr_raw, bytes)
                else str(partial_stderr_raw)
            )

            raise TimeoutError(
                f"V2 enhancement timed out after {timeout}s (partial runtime={runtime_s:.2f}s)\n"
                f"Partial stdout: {partial_stdout[:500]}\n"
                f"Partial stderr: {partial_stderr[:500]}"
            ) from e

        # Try to find and merge report JSON
        # Use validated_asset_key if provided for consistent report discovery with orchestrator
        report_lookup_key = validated_asset_key or validated_input.stem
        report_path = find_v2_report(validated_output, report_lookup_key)

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
            logger.info(f"No V2 report found for {report_lookup_key} in {validated_output}")

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
    prefixed_matches = []
    for report_path in output_dir.glob("**/*_report.json"):
        stem = report_path.stem
        if stem == f"{image_key}_report":
            return report_path
        if stem.startswith(f"{image_key}_") and stem.endswith("_report"):
            prefixed_matches.append(report_path)

    # Deterministic fallback for derived stems (e.g., *_materials_v3_enhanced_report.json)
    if prefixed_matches:
        return sorted(prefixed_matches)[0]

    return None
