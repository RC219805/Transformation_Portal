"""Execution Manifest for run-level provenance.

This module provides run-level manifests that capture:
- All node hashes from a pipeline run
- Merkle root of the entire execution
- Environment metadata (Python version, platform)
- Timestamps and duration

The manifest serves as a portable record of a complete run
that can be used for:
- Reproducibility verification
- Audit trails
- Run comparison

Example:
    >>> builder = ManifestBuilder(engine.dag)
    >>>
    >>> # After running nodes
    >>> manifest = builder.build(
    ...     node_hashes=["abc...", "def..."],
    ...     run_id="run_001",
    ... )
    >>>
    >>> # Export manifest
    >>> manifest.save(Path("manifest.json"))
    >>>
    >>> # Later: verify run
    >>> loaded = ExecutionManifest.load(Path("manifest.json"))
    >>> assert loaded.root_hash == manifest.root_hash
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformation_portal.storage.merkle_dag import MerkleDAG

logger = logging.getLogger(__name__)


def _hash_dict(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of a dictionary.

    Uses canonical JSON serialization for determinism.
    """
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EnvironmentInfo:
    """Environment metadata for reproducibility.

    Attributes:
        python_version: Python version string
        platform: Platform identifier
        hostname: Machine hostname
        user: Current user
        cwd: Working directory at manifest creation
    """

    python_version: str
    platform: str
    hostname: str
    user: str
    cwd: str

    @classmethod
    def capture(cls) -> "EnvironmentInfo":
        """Capture current environment information."""
        return cls(
            python_version=sys.version,
            platform=platform.platform(),
            hostname=platform.node(),
            user=os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
            cwd=os.getcwd(),
        )


@dataclass
class ExecutionManifest:
    """Manifest for a complete pipeline run.

    Captures all execution metadata needed for reproducibility
    and audit purposes.

    Attributes:
        run_id: Unique run identifier
        node_hashes: List of Merkle hashes for executed nodes
        root_hash: Merkle root of all node hashes
        created_at: ISO timestamp of manifest creation
        duration_seconds: Total run duration
        environment: Environment information
        metadata: Additional run metadata
    """

    run_id: str
    node_hashes: List[str]
    root_hash: str
    created_at: str
    duration_seconds: float
    environment: EnvironmentInfo
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "version": "1.0",
            "run_id": self.run_id,
            "node_hashes": self.node_hashes,
            "root_hash": self.root_hash,
            "created_at": self.created_at,
            "duration_seconds": self.duration_seconds,
            "environment": asdict(self.environment),
            "metadata": self.metadata,
        }

    def to_json(self, *, pretty: bool = True) -> str:
        """Convert manifest to JSON string.

        Args:
            pretty: If True, format with indentation

        Returns:
            JSON string
        """
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: Path) -> None:
        """Save manifest to file.

        Args:
            path: Output file path
        """
        path.write_text(self.to_json())
        logger.info("Saved execution manifest to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ExecutionManifest":
        """Load manifest from file.

        Args:
            path: Input file path

        Returns:
            Loaded ExecutionManifest
        """
        data = json.loads(path.read_text())

        env_data = data.get("environment", {})
        environment = EnvironmentInfo(
            python_version=env_data.get("python_version", ""),
            platform=env_data.get("platform", ""),
            hostname=env_data.get("hostname", ""),
            user=env_data.get("user", ""),
            cwd=env_data.get("cwd", ""),
        )

        manifest = cls(
            run_id=data["run_id"],
            node_hashes=data["node_hashes"],
            root_hash=data["root_hash"],
            created_at=data["created_at"],
            duration_seconds=data.get("duration_seconds", 0),
            environment=environment,
            metadata=data.get("metadata", {}),
        )

        logger.info("Loaded execution manifest from %s", path)
        return manifest

    def verify_root_hash(self) -> bool:
        """Verify the root hash is correct.

        Returns:
            True if root hash matches computed value
        """
        computed = _hash_dict({"nodes": sorted(self.node_hashes)})
        return computed == self.root_hash


class ManifestBuilder:
    """Builder for execution manifests.

    Creates manifests from completed pipeline runs with
    proper Merkle root computation.

    Example:
        >>> builder = ManifestBuilder(dag)
        >>> builder.start("run_001")
        >>>
        >>> # ... execute nodes ...
        >>>
        >>> manifest = builder.build(node_hashes)
    """

    def __init__(
        self,
        dag: Optional[MerkleDAG] = None,
    ) -> None:
        """Initialize manifest builder.

        Args:
            dag: Optional MerkleDAG for node verification
        """
        self.dag = dag
        self._start_time: Optional[float] = None
        self._run_id: Optional[str] = None

    def start(self, run_id: str) -> None:
        """Start tracking a run.

        Args:
            run_id: Unique run identifier
        """
        self._run_id = run_id
        self._start_time = time.time()
        logger.debug("Started manifest tracking for run: %s", run_id)

    def build(
        self,
        node_hashes: List[str],
        *,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionManifest:
        """Build execution manifest.

        Args:
            node_hashes: List of Merkle hashes for executed nodes
            run_id: Run identifier (uses tracked if not provided)
            metadata: Additional metadata to include

        Returns:
            ExecutionManifest
        """
        effective_run_id = run_id or self._run_id or f"run_{int(time.time())}"

        # Compute duration
        duration = 0.0
        if self._start_time:
            duration = time.time() - self._start_time

        # Compute root hash (sorted for determinism)
        root_hash = _hash_dict({"nodes": sorted(node_hashes)})

        manifest = ExecutionManifest(
            run_id=effective_run_id,
            node_hashes=list(node_hashes),
            root_hash=root_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=duration,
            environment=EnvironmentInfo.capture(),
            metadata=metadata or {},
        )

        logger.info(
            "Built execution manifest: run=%s, nodes=%d, root=%s",
            effective_run_id,
            len(node_hashes),
            root_hash[:8],
        )

        return manifest

    def build_from_dag(
        self,
        *,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionManifest:
        """Build manifest from all nodes in the DAG.

        Args:
            run_id: Run identifier
            metadata: Additional metadata

        Returns:
            ExecutionManifest
        """
        if not self.dag:
            raise ValueError("No DAG provided to ManifestBuilder")

        node_hashes = list(self.dag.nodes.keys())
        return self.build(node_hashes, run_id=run_id, metadata=metadata)
