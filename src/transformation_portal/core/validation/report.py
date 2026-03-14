"""
Execution Reporting & Reproducibility.

Captures the full context of a processing run to ensure results can be
reproduced or debugged later.
"""

import json
import logging
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class GitInfo:
    commit_hash: str
    branch: str
    is_dirty: bool

    @classmethod
    def capture(cls) -> "GitInfo":
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
            status = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
            return cls(commit, branch, bool(status))
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            # Git not available or not in a git repository
            return cls("unknown", "unknown", False)


@dataclass
class DeviceInfo:
    system: str
    python_version: str
    pytorch_version: str
    cuda_version: Optional[str]
    gpu_name: Optional[str]

    @classmethod
    def capture(cls) -> "DeviceInfo":
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        cuda = torch.version.cuda if torch.cuda.is_available() else None
        return cls(
            system=platform.platform(),
            python_version=platform.python_version(),
            pytorch_version=torch.__version__,
            cuda_version=cuda,
            gpu_name=gpu,
        )


@dataclass
class ModelInfo:
    """Details about neural models used."""

    name: str
    variant: str
    checksum: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingReport:
    """The Master Report for a single execution."""

    # Metadata
    job_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0

    # Context
    git: GitInfo = field(default_factory=GitInfo.capture)
    device: DeviceInfo = field(default_factory=DeviceInfo.capture)

    # Execution
    parameters: Dict[str, Any] = field(default_factory=dict)
    models_used: List[ModelInfo] = field(default_factory=list)

    # Results
    metrics: Dict[str, float] = field(default_factory=dict)
    output_files: List[str] = field(default_factory=list)

    def save(self, path: str):
        """Save report to JSON."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
