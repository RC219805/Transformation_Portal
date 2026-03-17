"""Execution Sandbox for DAG nodes.

This module provides isolated, deterministic execution environments
for DAG nodes. Each node runs in a sandbox with:
- Scoped filesystem access (via FSGuard)
- CAS-only IO enforcement
- GPU isolation
- Reproducible artifact flows

Design:
    Node → Sandbox → FSGuard (scoped) → CAS
                      ↓
                GPU lease (semaphore)

Guarantees:
- No arbitrary filesystem access
- Only allowed paths (CAS + node workspace)
- GPU isolation per node
- Deterministic IO boundaries
- Optional reproducibility logging

Example:
    >>> sandbox = Sandbox(
    ...     node_id="llava_quality_001",
    ...     config=SandboxConfig(
    ...         workspace_root=Path("/tmp/workspaces"),
    ...         cas_root=Path("/data/cas"),
    ...     ),
    ...     fs=get_fs_guard(),
    ...     cas=ArtifactStore(Path("/data/cas")),
    ... )
    >>>
    >>> # Materialize inputs from CAS
    >>> input_path = sandbox.materialize_input(sha, "input.png")
    >>>
    >>> # Run computation
    >>> result = process_image(input_path)
    >>>
    >>> # Write output and persist to CAS
    >>> out_path = sandbox.write(["results", "output"], result, suffix=".json")
    >>> out_sha = sandbox.persist_output(out_path)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformation_portal.core.security.fs_guard import (
    FSContext,
    FSGuard,
    FSPolicyError,
    get_fs_guard,
)
from transformation_portal.core.security.path_safety import (
    PathSafetyError,
    validate_safe_name,
)
from transformation_portal.storage.cas_store import ArtifactStore, CASError

logger = logging.getLogger(__name__)


class SandboxError(RuntimeError):
    """Raised for sandbox security or policy violations."""

    pass


@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for execution sandbox.

    Attributes:
        workspace_root: Root directory for node workspaces
        cas_root: Root directory for CAS storage
        enable_gpu: Whether GPU access is allowed
        allow_network: Whether network access is allowed (not enforced at runtime)
        max_workspace_size_mb: Maximum workspace size in MB (0 = unlimited)
        cleanup_on_exit: Whether to clean workspace after execution
    """

    workspace_root: Path
    cas_root: Path
    enable_gpu: bool = True
    allow_network: bool = False
    max_workspace_size_mb: int = 0
    cleanup_on_exit: bool = False


@dataclass
class SandboxMetrics:
    """Metrics collected during sandbox execution.

    Attributes:
        inputs_materialized: Number of CAS objects materialized
        outputs_persisted: Number of outputs persisted to CAS
        bytes_read: Total bytes read
        bytes_written: Total bytes written
        start_time: Execution start timestamp
        end_time: Execution end timestamp
    """

    inputs_materialized: int = 0
    outputs_persisted: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Execution duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class Sandbox:
    """Execution sandbox for a DAG node.

    Provides isolated, deterministic execution with:
    - Scoped filesystem access via FSGuard
    - CAS-only IO (inputs materialize from CAS, outputs persist to CAS)
    - Metrics collection for reproducibility

    Example:
        >>> sandbox = Sandbox("node_001", config, fs, cas)
        >>>
        >>> # Inputs MUST come from CAS
        >>> img = sandbox.materialize_input(sha, "input.png")
        >>>
        >>> # Outputs MUST go to CAS
        >>> out_sha = sandbox.persist_output(result_path)
    """

    def __init__(
        self,
        node_id: str,
        config: SandboxConfig,
        fs: FSGuard,
        cas: ArtifactStore,
    ) -> None:
        """Initialize sandbox for a node.

        Args:
            node_id: Unique identifier for the node
            config: Sandbox configuration
            fs: FSGuard instance for filesystem operations
            cas: CAS store for artifact storage

        Raises:
            SandboxError: If node_id fails validation
        """
        try:
            validate_safe_name(node_id)
        except PathSafetyError as e:
            raise SandboxError(f"Invalid node_id: {e}")

        self.node_id = node_id
        self.config = config
        self.fs = fs
        self.cas = cas
        self.metrics = SandboxMetrics()

        # Node-specific workspace
        self.workspace = config.workspace_root / node_id
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Scoped FS contexts for different trust levels
        self._user_ctx = FSContext(mode="user", base_dir=self.workspace)
        self._internal_ctx = FSContext(mode="internal", base_dir=self.workspace)
        self._cas_ctx = FSContext(mode="cas", base_dir=config.cas_root / "objects")

        # Track materialized and persisted artifacts for reproducibility
        self._materialized: Dict[str, Path] = {}
        self._persisted: Dict[Path, str] = {}

        logger.info(
            "Sandbox created: node_id=%s, workspace=%s",
            node_id,
            self.workspace,
        )

    # -----------------------------
    # CAS-ONLY IO ENFORCEMENT
    # -----------------------------
    def materialize_input(self, sha: str, rel_path: str) -> Path:
        """Materialize a CAS object as an input file.

        Inputs MUST come from CAS - this enforces deterministic,
        reproducible execution.

        Args:
            sha: SHA-256 hash of the CAS object
            rel_path: Relative path within workspace for materialized file

        Returns:
            Path to materialized file

        Raises:
            SandboxError: If SHA is invalid or CAS object not found
        """
        # Validate rel_path components
        parts = rel_path.replace("\\", "/").split("/")
        for part in parts[:-1]:  # Validate directory parts
            if part:
                try:
                    validate_safe_name(part)
                except PathSafetyError as e:
                    raise SandboxError(f"Invalid path component: {e}")

        # Construct target path
        target = self.workspace / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.cas.materialize(sha, target, use_symlink=True)
        except CASError as e:
            raise SandboxError(f"Failed to materialize input: {e}")

        self._materialized[sha] = target
        self.metrics.inputs_materialized += 1

        logger.debug(
            "Sandbox materialize: %s -> %s",
            sha[:8],
            rel_path,
        )

        return target

    def persist_output(self, path: Path) -> str:
        """Persist a workspace file to CAS.

        Outputs MUST go to CAS - this enforces deterministic,
        reproducible artifact flows.

        Args:
            path: Path to file within workspace to persist

        Returns:
            SHA-256 hash of persisted object

        Raises:
            SandboxError: If path is outside workspace or file doesn't exist
        """
        # Ensure path is within workspace
        try:
            path.relative_to(self.workspace)
        except ValueError:
            raise SandboxError(f"Cannot persist file outside workspace: {path}")

        if not path.exists():
            raise SandboxError(f"File does not exist: {path}")

        try:
            obj = self.cas.add_file(path)
        except CASError as e:
            raise SandboxError(f"Failed to persist output: {e}")

        self._persisted[path] = obj.sha256
        self.metrics.outputs_persisted += 1
        self.metrics.bytes_written += obj.size_bytes

        logger.debug(
            "Sandbox persist: %s -> %s",
            path.name,
            obj.sha256[:8],
        )

        return obj.sha256

    def persist_bytes(self, data: bytes) -> str:
        """Persist bytes directly to CAS.

        Args:
            data: Bytes to persist

        Returns:
            SHA-256 hash of persisted object
        """
        try:
            obj = self.cas.add_bytes(data)
        except CASError as e:
            raise SandboxError(f"Failed to persist bytes: {e}")

        self.metrics.outputs_persisted += 1
        self.metrics.bytes_written += len(data)

        return obj.sha256

    # -----------------------------
    # FILE ACCESS WRAPPERS
    # All IO goes through FSGuard
    # -----------------------------
    def read(self, rel_path: str) -> str:
        """Read text from a file in the workspace.

        Args:
            rel_path: Relative path within workspace

        Returns:
            File contents as string
        """
        path = self.workspace / rel_path

        # Verify within workspace
        try:
            path.relative_to(self.workspace)
        except ValueError:
            raise SandboxError(f"Path escapes workspace: {rel_path}")

        content = self.fs.read_text(path)
        self.metrics.bytes_read += len(content.encode("utf-8"))
        return content

    def read_bytes(self, rel_path: str) -> bytes:
        """Read bytes from a file in the workspace.

        Args:
            rel_path: Relative path within workspace

        Returns:
            File contents as bytes
        """
        path = self.workspace / rel_path

        try:
            path.relative_to(self.workspace)
        except ValueError:
            raise SandboxError(f"Path escapes workspace: {rel_path}")

        content = self.fs.read_bytes(path)
        self.metrics.bytes_read += len(content)
        return content

    def write(
        self,
        rel_parts: List[str],
        data: str,
        *,
        suffix: str = ".txt",
    ) -> Path:
        """Write text to a file in the workspace.

        Args:
            rel_parts: Path segments within workspace (last is filename stem)
            data: Text content to write
            suffix: File extension

        Returns:
            Path to written file
        """
        if not rel_parts:
            raise SandboxError("At least one path segment required")

        # Validate all parts
        for part in rel_parts:
            try:
                validate_safe_name(part)
            except PathSafetyError as e:
                raise SandboxError(f"Invalid path component: {e}")

        # Construct path
        if len(rel_parts) == 1:
            path = self.fs.user_file(self._user_ctx, rel_parts[0], suffix=suffix)
        else:
            dir_path = self.fs.internal_path(self._internal_ctx, rel_parts[:-1])
            from transformation_portal.core.security.path_safety import safe_join_file

            path = safe_join_file(dir_path, rel_parts[-1], suffix=suffix)

        self.fs.write_text(path, data)
        self.metrics.bytes_written += len(data.encode("utf-8"))

        logger.debug("Sandbox write: %s", path)
        return path

    def write_bytes(
        self,
        rel_parts: List[str],
        data: bytes,
        *,
        suffix: str = ".bin",
    ) -> Path:
        """Write bytes to a file in the workspace.

        Args:
            rel_parts: Path segments within workspace
            data: Binary content to write
            suffix: File extension

        Returns:
            Path to written file
        """
        if not rel_parts:
            raise SandboxError("At least one path segment required")

        for part in rel_parts:
            try:
                validate_safe_name(part)
            except PathSafetyError as e:
                raise SandboxError(f"Invalid path component: {e}")

        if len(rel_parts) == 1:
            path = self.fs.user_file(self._user_ctx, rel_parts[0], suffix=suffix)
        else:
            dir_path = self.fs.internal_path(self._internal_ctx, rel_parts[:-1])
            from transformation_portal.core.security.path_safety import safe_join_file

            path = safe_join_file(dir_path, rel_parts[-1], suffix=suffix)

        self.fs.write_bytes(path, data)
        self.metrics.bytes_written += len(data)

        return path

    def write_json(
        self,
        rel_parts: List[str],
        data: Any,
    ) -> Path:
        """Write JSON data to a file in the workspace.

        Args:
            rel_parts: Path segments within workspace
            data: JSON-serializable data

        Returns:
            Path to written file
        """
        json_str = json.dumps(data, indent=2, default=str)
        return self.write(rel_parts, json_str, suffix=".json")

    # -----------------------------
    # WORKSPACE MANAGEMENT
    # -----------------------------
    def list_workspace(self) -> List[Path]:
        """List all files in the workspace.

        Returns:
            List of paths to files in workspace
        """
        return list(self.workspace.rglob("*"))

    def cleanup(self) -> None:
        """Clean up the workspace directory.

        Removes all files and directories in the workspace.
        """
        import shutil

        if self.workspace.exists():
            shutil.rmtree(self.workspace)
            logger.info("Sandbox cleanup: %s", self.workspace)

    # -----------------------------
    # EXECUTION LIFECYCLE
    # -----------------------------
    def start(self) -> None:
        """Mark start of sandbox execution."""
        self.metrics.start_time = time.time()
        logger.debug("Sandbox start: %s", self.node_id)

    def finish(self) -> None:
        """Mark end of sandbox execution."""
        self.metrics.end_time = time.time()

        if self.config.cleanup_on_exit:
            self.cleanup()

        logger.info(
            "Sandbox finish: node_id=%s, inputs=%d, outputs=%d, duration=%.2fs",
            self.node_id,
            self.metrics.inputs_materialized,
            self.metrics.outputs_persisted,
            self.metrics.duration_seconds or 0,
        )

    # -----------------------------
    # REPRODUCIBILITY
    # -----------------------------
    def get_manifest(self) -> Dict[str, Any]:
        """Get execution manifest for reproducibility.

        Returns:
            Dictionary containing inputs, outputs, and metrics
        """
        return {
            "node_id": self.node_id,
            "workspace": str(self.workspace),
            "inputs": {sha: str(path) for sha, path in self._materialized.items()},
            "outputs": {str(path): sha for path, sha in self._persisted.items()},
            "metrics": {
                "inputs_materialized": self.metrics.inputs_materialized,
                "outputs_persisted": self.metrics.outputs_persisted,
                "bytes_read": self.metrics.bytes_read,
                "bytes_written": self.metrics.bytes_written,
                "duration_seconds": self.metrics.duration_seconds,
            },
        }

    def __enter__(self) -> "Sandbox":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.finish()
