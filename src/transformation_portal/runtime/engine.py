"""Integrated Execution Engine with Merkle DAG provenance.

This module provides the top-level execution engine that integrates:
- Process-isolated execution (spawn-safe)
- GPU pooling with deterministic leasing
- CAS-backed artifact storage
- Merkle DAG for full provenance tracking

Design:
    ExecutionEngine
         ↓
    ProcessExecutor ─── GPUPool
         ↓
    Sandbox ─── FSGuard ─── CAS
         ↓
    MerkleDAG (provenance)

Example:
    >>> engine = ExecutionEngine(
    ...     workspace_root=Path("/tmp/workspaces"),
    ...     cas_root=Path("/data/cas"),
    ...     gpu_devices=[0, 1],
    ... )
    >>>
    >>> node_hash, outputs = engine.run_node(
    ...     ProcessImageNode,
    ...     inputs={"image_sha": "abc123..."},
    ...     node_id="process_001",
    ... )
    >>>
    >>> # Full provenance available
    >>> lineage = engine.get_lineage(node_hash)
    >>> engine.export_dag(Path("provenance.json"))
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from transformation_portal.core.security.fs_guard import FSGuard, get_fs_guard
from transformation_portal.runtime.gpu_pool import GPULease, GPUPool, GPUPoolError
from transformation_portal.runtime.process_executor import (
    ProcessExecutor,
    ProcessResult,
    ProcessTask,
)
from transformation_portal.storage.cas_store import ArtifactStore
from transformation_portal.storage.merkle_dag import MerkleDAG, MerkleNode

logger = logging.getLogger(__name__)


class ExecutionEngineError(RuntimeError):
    """Raised for execution engine errors."""


@dataclass
class ExecutionRecord:
    """Record of a node execution.

    Attributes:
        node_id: Unique execution identifier
        node_type: Class name of executed node
        merkle_hash: Hash in the Merkle DAG
        inputs: Input parameters
        outputs: Output CAS references
        duration_seconds: Execution time
        gpu_device: GPU device used (if any)
        success: Whether execution succeeded
        error: Error message if failed
    """

    node_id: str
    node_type: str
    merkle_hash: Optional[str]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    duration_seconds: float
    gpu_device: Optional[int] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class EngineConfig:
    """Configuration for execution engine.

    Attributes:
        workspace_root: Root directory for node workspaces
        cas_root: Root directory for CAS storage
        gpu_devices: List of GPU device IDs to manage
        process_timeout: Default timeout for process execution
        cleanup_workspaces: Whether to cleanup workspaces after execution
        enable_provenance: Whether to track provenance in Merkle DAG
    """

    workspace_root: Path
    cas_root: Path
    gpu_devices: Optional[List[int]] = None
    process_timeout: Optional[float] = None
    cleanup_workspaces: bool = True
    enable_provenance: bool = True


class ExecutionEngine:
    """Integrated execution engine with full provenance tracking.

    Provides:
    - Process-isolated node execution
    - GPU resource management
    - CAS-backed artifact storage
    - Merkle DAG provenance

    This is the top-level orchestrator for deterministic,
    reproducible pipeline execution.

    Example:
        >>> engine = ExecutionEngine(EngineConfig(
        ...     workspace_root=Path("/tmp/ws"),
        ...     cas_root=Path("/data/cas"),
        ...     gpu_devices=[0],
        ... ))
        >>>
        >>> # Execute a node
        >>> hash, outputs = engine.run_node(
        ...     MyProcessingNode,
        ...     inputs={"data_sha": "abc123..."},
        ...     node_id="proc_001",
        ...     use_gpu=True,
        ... )
        >>>
        >>> # Check provenance
        >>> lineage = engine.get_lineage(hash)
        >>> print(f"Lineage depth: {len(lineage)}")
    """

    def __init__(
        self,
        config: EngineConfig,
        *,
        fs: Optional[FSGuard] = None,
    ) -> None:
        """Initialize execution engine.

        Args:
            config: Engine configuration
            fs: FSGuard instance (uses global if not provided)
        """
        self.config = config
        self.fs = fs or get_fs_guard()

        # Initialize CAS
        self.cas = ArtifactStore(config.cas_root)

        # Initialize process executor
        self.executor = ProcessExecutor(timeout=config.process_timeout)

        # Initialize GPU pool if devices specified
        if config.gpu_devices:
            self.gpu_pool = GPUPool(devices=config.gpu_devices)
        else:
            self.gpu_pool = GPUPool(auto_detect=True)
            if self.gpu_pool.total_devices == 0:
                self.gpu_pool = None

        # Initialize Merkle DAG for provenance
        self.dag = MerkleDAG() if config.enable_provenance else None

        # Execution tracking
        self._executions: List[ExecutionRecord] = []
        self._input_artifacts: Dict[str, str] = {}  # sha -> merkle_hash

        # Create directories
        config.workspace_root.mkdir(parents=True, exist_ok=True)
        config.cas_root.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ExecutionEngine initialized: workspace=%s, cas=%s, gpus=%s",
            config.workspace_root,
            config.cas_root,
            self.gpu_pool.total_devices if self.gpu_pool else 0,
        )

    def register_input(
        self,
        sha: str,
        *,
        artifact_type: str = "input",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register an input artifact in the Merkle DAG.

        Call this before running nodes to establish provenance
        for input data.

        Args:
            sha: SHA-256 hash of the artifact (CAS reference)
            artifact_type: Type of artifact
            metadata: Additional metadata

        Returns:
            Merkle hash of the registered artifact
        """
        if not self.dag:
            return sha

        if sha in self._input_artifacts:
            return self._input_artifacts[sha]

        merkle_hash = self.dag.add_artifact(
            artifact_type=artifact_type,
            content_hash=sha,
            metadata=metadata or {},
        )

        self._input_artifacts[sha] = merkle_hash
        return merkle_hash

    def run_node(
        self,
        node_cls: Type,
        *,
        inputs: Dict[str, Any],
        node_id: str,
        node_kwargs: Optional[Dict[str, Any]] = None,
        use_gpu: bool = False,
        timeout: Optional[float] = None,
        input_artifact_shas: Optional[List[str]] = None,
    ) -> tuple[Optional[str], Dict[str, Any]]:
        """Execute a node and track provenance.

        Args:
            node_cls: DAG node class to instantiate
            inputs: Inputs to pass to node.run()
            node_id: Unique identifier for this execution
            node_kwargs: Keyword arguments for node constructor
            use_gpu: Whether to acquire GPU for execution
            timeout: Override default process timeout
            input_artifact_shas: List of input CAS SHAs for provenance

        Returns:
            Tuple of (merkle_hash, outputs)
            merkle_hash is None if provenance disabled

        Raises:
            ExecutionEngineError: If execution fails
        """
        start_time = time.time()
        gpu_lease: Optional[GPULease] = None
        gpu_device: Optional[int] = None

        # Acquire GPU if requested
        if use_gpu and self.gpu_pool:
            try:
                gpu_lease = self.gpu_pool.acquire(timeout=30)
                gpu_device = gpu_lease.device_id
                logger.debug("Acquired GPU %d for node %s", gpu_device, node_id)
            except GPUPoolError as e:
                logger.warning("Failed to acquire GPU: %s", e)

        try:
            # Build sandbox config
            sandbox_config = {
                "node_id": node_id,
                "workspace_root": str(self.config.workspace_root),
                "cas_root": str(self.config.cas_root),
                "cleanup_on_exit": self.config.cleanup_workspaces,
            }

            if gpu_device is not None:
                sandbox_config["gpu_id"] = gpu_device

            # Create task
            task = ProcessTask(
                node_cls=node_cls,
                node_kwargs=node_kwargs or {},
                inputs=inputs,
                sandbox_config=sandbox_config,
            )

            # Execute in process
            result = self.executor.run(task, timeout=timeout)

            duration = time.time() - start_time

            # Handle failure
            if not result.success:
                record = ExecutionRecord(
                    node_id=node_id,
                    node_type=node_cls.__name__,
                    merkle_hash=None,
                    inputs=inputs,
                    outputs={},
                    duration_seconds=duration,
                    gpu_device=gpu_device,
                    success=False,
                    error=result.error,
                )
                self._executions.append(record)

                raise ExecutionEngineError(f"Node {node_id} failed: {result.error}")

            # Register in Merkle DAG
            merkle_hash = None
            if self.dag:
                # Get input merkle hashes
                input_hashes = []
                for sha in input_artifact_shas or []:
                    if sha in self._input_artifacts:
                        input_hashes.append(self._input_artifacts[sha])
                    else:
                        # Auto-register input
                        input_hashes.append(self.register_input(sha))

                merkle_hash = self.dag.add_computation(
                    node_id=node_id,
                    inputs=input_hashes,
                    outputs=result.outputs,
                    metadata={
                        "node_type": node_cls.__name__,
                        "duration_seconds": duration,
                        "gpu_device": gpu_device,
                        "manifest": result.manifest,
                    },
                )

                # Register outputs as artifacts for downstream nodes
                for key, sha in result.outputs.items():
                    if isinstance(sha, str) and len(sha) == 64:
                        self._input_artifacts[sha] = merkle_hash

            # Record execution
            record = ExecutionRecord(
                node_id=node_id,
                node_type=node_cls.__name__,
                merkle_hash=merkle_hash,
                inputs=inputs,
                outputs=result.outputs,
                duration_seconds=duration,
                gpu_device=gpu_device,
                success=True,
            )
            self._executions.append(record)

            logger.info(
                "Node %s completed: outputs=%d, duration=%.2fs, merkle=%s",
                node_id,
                len(result.outputs),
                duration,
                merkle_hash[:8] if merkle_hash else "N/A",
            )

            return merkle_hash, result.outputs

        finally:
            # Release GPU
            if gpu_lease and self.gpu_pool:
                self.gpu_pool.release(gpu_lease)

    def run_pipeline(
        self,
        nodes: List[tuple[str, Type, Dict[str, Any], Dict[str, Any]]],
        *,
        use_gpu: bool = False,
    ) -> List[tuple[Optional[str], Dict[str, Any]]]:
        """Execute a sequence of nodes.

        Args:
            nodes: List of (node_id, node_cls, node_kwargs, inputs) tuples
            use_gpu: Whether to use GPU for all nodes

        Returns:
            List of (merkle_hash, outputs) tuples
        """
        results = []
        for node_id, node_cls, node_kwargs, inputs in nodes:
            hash, outputs = self.run_node(
                node_cls,
                inputs=inputs,
                node_id=node_id,
                node_kwargs=node_kwargs,
                use_gpu=use_gpu,
            )
            results.append((hash, outputs))
        return results

    def get_lineage(self, merkle_hash: str) -> List[MerkleNode]:
        """Get full lineage of a computation.

        Args:
            merkle_hash: Merkle hash of the computation

        Returns:
            List of MerkleNodes in topological order
        """
        if not self.dag:
            return []
        return self.dag.get_lineage(merkle_hash)

    def export_dag(self, path: Path) -> None:
        """Export provenance DAG to JSON.

        Args:
            path: Output file path
        """
        if self.dag:
            self.dag.export(path)

    def verify_integrity(self) -> List[str]:
        """Verify integrity of the provenance DAG.

        Returns:
            List of error messages (empty if valid)
        """
        if not self.dag:
            return []
        return self.dag.verify_integrity()

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary.

        Returns:
            Dictionary with execution statistics
        """
        successful = sum(1 for e in self._executions if e.success)
        failed = len(self._executions) - successful
        total_duration = sum(e.duration_seconds for e in self._executions)

        return {
            "total_executions": len(self._executions),
            "successful": successful,
            "failed": failed,
            "total_duration_seconds": total_duration,
            "gpu_enabled": self.gpu_pool is not None,
            "gpu_devices": self.gpu_pool.total_devices if self.gpu_pool else 0,
            "dag_nodes": len(self.dag.nodes) if self.dag else 0,
            "cas_root": str(self.config.cas_root),
        }

    @property
    def executions(self) -> List[ExecutionRecord]:
        """All execution records."""
        return list(self._executions)
