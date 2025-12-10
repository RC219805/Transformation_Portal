"""
Processing report for reproducibility tracking.

Captures comprehensive metadata about each processing run.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from pathlib import Path
import subprocess
import platform
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class GitInfo:
    """Git repository state for reproducibility."""
    commit: str
    branch: str
    is_dirty: bool
    remote_url: Optional[str] = None

    @classmethod
    def capture(cls, repo_path: Optional[Path] = None) -> Optional[GitInfo]:
        """
        Capture current git state.

        Args:
            repo_path: Repository path (defaults to current directory)

        Returns:
            GitInfo or None if not in a git repository
        """
        try:
            cwd = str(repo_path) if repo_path else None

            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=cwd,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            is_dirty = subprocess.run(
                ["git", "diff-index", "--quiet", "HEAD"],
                cwd=cwd,
                check=False,
                stderr=subprocess.DEVNULL
            ).returncode != 0

            try:
                remote_url = subprocess.check_output(
                    ["git", "config", "--get", "remote.origin.url"],
                    cwd=cwd,
                    stderr=subprocess.DEVNULL
                ).decode().strip()
            except subprocess.CalledProcessError:
                remote_url = None

            return cls(
                commit=commit,
                branch=branch,
                is_dirty=is_dirty,
                remote_url=remote_url
            )

        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("Not in a git repository, skipping git info")
            return None


@dataclass
class DeviceInfo:
    """Hardware and software environment."""
    device_type: str
    device_name: Optional[str]
    torch_version: Optional[str]
    cuda_version: Optional[str]
    python_version: str
    platform: str
    cpu_count: int

    @classmethod
    def capture(cls) -> DeviceInfo:
        """Capture device information."""
        import os

        device_type = "cpu"
        device_name = None
        cuda_version = None
        torch_version = None

        if TORCH_AVAILABLE:
            torch_version = torch.__version__

            if torch.cuda.is_available():
                device_type = "cuda"
                try:
                    device_name = torch.cuda.get_device_name(0)
                    cuda_version = torch.version.cuda
                except Exception:
                    pass

            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_type = "mps"
                device_name = "Apple Silicon"

        if device_name is None:
            device_name = platform.processor() or "Unknown"

        return cls(
            device_type=device_type,
            device_name=device_name,
            torch_version=torch_version,
            cuda_version=cuda_version,
            python_version=platform.python_version(),
            platform=platform.platform(),
            cpu_count=os.cpu_count() or 1
        )


@dataclass
class ModelInfo:
    """Model checksums and versions."""
    model_name: str
    checkpoint_sha256: Optional[str] = None
    model_version: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_weights(cls, model_name: str, weights_path: Optional[Path] = None) -> ModelInfo:
        """
        Compute model checksum from weights file.

        Args:
            model_name: Name of the model
            weights_path: Path to weights file (optional)

        Returns:
            ModelInfo with checksum if weights_path provided
        """
        checkpoint_sha256 = None

        if weights_path and weights_path.exists():
            try:
                hasher = hashlib.sha256()
                with open(weights_path, "rb") as f:
                    # Read in chunks for large files
                    while chunk := f.read(8192):
                        hasher.update(chunk)
                checkpoint_sha256 = hasher.hexdigest()
            except Exception as e:
                logger.warning(f"Failed to compute checksum for {weights_path}: {e}")

        return cls(
            model_name=model_name,
            checkpoint_sha256=checkpoint_sha256
        )


@dataclass
class ProcessingReport:
    """
    Comprehensive processing report for reproducibility.

    Captures all information needed to reproduce a processing run.
    """
    git_info: Optional[GitInfo]
    device_info: DeviceInfo
    model_info: Optional[ModelInfo]
    config_hash: str
    preset: str
    input_path: str
    output_path: str
    timestamp: str
    duration_ms: float
    metrics: Dict[str, float]
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)

        # Handle None values
        if data["git_info"] is None:
            data["git_info"] = {}
        if data["model_info"] is None:
            data["model_info"] = {}
        if data["metadata"] is None:
            data["metadata"] = {}

        return data

    def save(self, path: Path):
        """
        Save report to JSON file.

        Args:
            path: Output path for report
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved processing report to {path}")

    @classmethod
    def load(cls, path: Path) -> ProcessingReport:
        """
        Load report from JSON file.

        Args:
            path: Path to report file

        Returns:
            ProcessingReport
        """
        with open(path) as f:
            data = json.load(f)

        # Reconstruct nested dataclasses
        if data.get("git_info"):
            data["git_info"] = GitInfo(**data["git_info"])

        if data.get("device_info"):
            data["device_info"] = DeviceInfo(**data["device_info"])

        if data.get("model_info"):
            data["model_info"] = ModelInfo(**data["model_info"])

        return cls(**data)

    @classmethod
    def create(
        cls,
        config: Dict[str, Any],
        input_path: Path,
        output_path: Path,
        duration_ms: float,
        metrics: Dict[str, float],
        model_info: Optional[ModelInfo] = None,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingReport:
        """
        Create report from processing run.

        Args:
            config: Processing configuration
            input_path: Input file path
            output_path: Output file path
            duration_ms: Processing duration in milliseconds
            metrics: Quality metrics
            model_info: Model information (optional)
            success: Whether processing succeeded
            error: Error message if failed
            metadata: Additional metadata

        Returns:
            ProcessingReport
        """
        from datetime import datetime

        # Compute config hash
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()

        return cls(
            git_info=GitInfo.capture(),
            device_info=DeviceInfo.capture(),
            model_info=model_info,
            config_hash=config_hash,
            preset=config.get("preset", "default"),
            input_path=str(input_path),
            output_path=str(output_path),
            timestamp=datetime.utcnow().isoformat() + "Z",
            duration_ms=duration_ms,
            metrics=metrics,
            success=success,
            error=error,
            metadata=metadata or {}
        )
