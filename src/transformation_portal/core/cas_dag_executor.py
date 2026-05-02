"""CAS-aware DAG executor for partial reuse.

This module integrates the CAS execution layer with the stage graph
for partial DAG reuse and deterministic replay.

Key Features:
    - Partial DAG reuse: Only recompute changed stages
    - Automatic dependency tracking: Upstream changes invalidate downstream
    - Parallel safety: FileLock prevents duplicate computation
    - Full provenance: Complete execution lineage in Merkle DAG

Example:
    [RAW] → [DEPTH] → [SEGMENTATION] → [SPLATTING] → [MESH]

    If only SEGMENTATION config changes:
        RAW ✔ (cached)
        DEPTH ✔ (cached)
        SEGMENTATION ❌ (recompute)
        SPLATTING ❌ (recompute - depends on changed SEGMENTATION)
        MESH ❌ (recompute - depends on changed SPLATTING)

Usage:
    >>> from transformation_portal.stage_graph import StageGraph
    >>> from transformation_portal.storage.cas_store import ArtifactStore
    >>> from transformation_portal.core.cas_dag_executor import CASDAGExecutor
    >>>
    >>> graph = StageGraph("my_pipeline")
    >>> graph.add_stage(stage1)
    >>> graph.add_stage(stage2, deps=[stage1.name])
    >>>
    >>> executor = CASDAGExecutor(
    ...     artifact_store=ArtifactStore(Path("/cache/cas")),
    ...     cache_dir=Path("/cache/results"),
    ... )
    >>> result = executor.execute(graph, context)
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from transformation_portal.core._cas_helpers import CASObjectMissingError
from transformation_portal.core._cas_helpers import atomic_write_json as _atomic_write_json
from transformation_portal.core._cas_helpers import compute_numpy_array_id as _compute_numpy_array_id
from transformation_portal.core._cas_helpers import load_serializable, make_serializable
from transformation_portal.core._cas_helpers import sanitize_cas_id_for_filename as _sanitize_cas_id_for_filename
from transformation_portal.core.execution_identity import (
    ArtifactMetadata,
    ExecutionIdentity,
    compute_cas_id,
    compute_code_hash,
    create_artifact_metadata,
    is_compatible,
    resolve_platform_lockfile,
)
from transformation_portal.core.execution_wrapper import (
    CASExecutor,
    ExecutorConfig,
    FileLock,
)
from transformation_portal.determinism.jcs import dumpb as jcs_dumpb
from transformation_portal.stage_graph.graph import GraphExecution, StageGraph
from transformation_portal.stage_graph.stage import Stage, StageContext, StageResult, StageStatus
from transformation_portal.storage.cas_store import ArtifactStore
from transformation_portal.storage.merkle_dag import MerkleDAG

logger = logging.getLogger(__name__)


@dataclass
class CASExecutionResult:
    """Result of CAS-aware DAG execution.

    Attributes:
        success: True if all stages completed successfully
        stage_results: Results for each stage
        execution_order: Order stages were processed
        cache_hits: Number of stages loaded from cache
        cache_misses: Number of stages that had to compute
        total_duration_ms: Total execution time
        merkle_dag: Provenance DAG for this execution
        identities: Execution identities for each stage
    """

    success: bool
    stage_results: Dict[str, StageResult]
    execution_order: List[str]
    cache_hits: int
    cache_misses: int
    total_duration_ms: float
    merkle_dag: Optional[MerkleDAG] = None
    identities: Dict[str, ExecutionIdentity] = field(default_factory=dict)
    error: Optional[str] = None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = float(self.cache_hits) / float(total) if total > 0 else 0.0

        # Estimate speedup from caching
        if hit_rate >= 1.0:
            speedup = float("inf")
        elif hit_rate > 0:
            speedup = 1.0 / (1.0 - hit_rate)
        else:
            speedup = 1.0

        return {
            "total_stages": total,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "speedup_estimate": speedup,
        }


@dataclass
class CASDAGConfig:
    """Configuration for CAS-aware DAG executor.

    Attributes:
        enable_caching: If False, always execute (no cache lookup)
        enable_provenance: If True, build Merkle DAG for lineage
        verify_on_load: If True, verify artifact integrity on cache hit
        allow_cross_platform: If True, allow cross-platform CPU artifacts
        parallel: If True, execute independent stages in parallel
        max_workers: Maximum parallel workers
        lock_timeout: Timeout for file locks (seconds)
        code_paths: Paths to include in code hash computation
        lockfile_path: Path to lockfile for deterministic builds (CI-required)
    """

    enable_caching: bool = True
    enable_provenance: bool = True
    verify_on_load: bool = True
    allow_cross_platform: bool = False
    parallel: bool = True
    max_workers: int = 4
    lock_timeout: float = 300.0
    code_paths: List[str] = field(default_factory=lambda: ["src/"])
    lockfile_path: Optional[str] = None


class CASDAGExecutor:
    """CAS-aware DAG executor with partial reuse.

    Executes a stage graph with content-addressable caching,
    enabling partial DAG reuse when only some stages change.

    The executor:
    1. Computes execution identity for each stage
    2. Checks CAS for cached results
    3. Executes only stages with cache misses
    4. Propagates cache invalidation to downstream stages
    5. Builds Merkle DAG for full provenance

    Example:
        >>> store = ArtifactStore(Path("/cache/cas"))
        >>> executor = CASDAGExecutor(store, Path("/cache"))
        >>>
        >>> # First run: all stages execute
        >>> result1 = executor.execute(graph, context)
        >>> assert result1.cache_misses == len(graph.stages)
        >>>
        >>> # Second run: all stages cached
        >>> result2 = executor.execute(graph, context)
        >>> assert result2.cache_hits == len(graph.stages)
        >>>
        >>> # Third run with modified config: partial recompute
        >>> context.config["depth"]["quality"] = "ultra"
        >>> result3 = executor.execute(graph, context)
        >>> assert result3.cache_hits < len(graph.stages)
    """

    def __init__(
        self,
        artifact_store: ArtifactStore,
        cache_dir: Path,
        config: Optional[CASDAGConfig] = None,
    ):
        """Initialize CAS DAG executor.

        Args:
            artifact_store: CAS store for artifact storage
            cache_dir: Directory for result caching
            config: Executor configuration
        """
        self.artifact_store = artifact_store
        self.cache_dir = Path(cache_dir)
        self.config = config or CASDAGConfig()

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir = self.cache_dir / ".locks"
        self.locks_dir.mkdir(exist_ok=True)

        # Pre-compute code hash
        self._code_hash = compute_code_hash(self.config.code_paths)

        # Resolve lockfile path for CI determinism (ADR-032)
        if self.config.lockfile_path:
            self._lockfile_path = str(self.config.lockfile_path)
        else:
            resolved = resolve_platform_lockfile()
            self._lockfile_path = str(resolved) if resolved else None

        # Stage executor
        executor_config = ExecutorConfig(
            enable_caching=self.config.enable_caching,
            verify_on_load=self.config.verify_on_load,
            allow_cross_platform=self.config.allow_cross_platform,
            lock_timeout=self.config.lock_timeout,
            code_paths=self.config.code_paths,
            lockfile_path=self._lockfile_path,
        )
        self._stage_executor = CASExecutor(
            artifact_store=artifact_store,
            cache_dir=cache_dir / "stages",
            config=executor_config,
        )

    def _compute_stage_identity(
        self,
        stage: Stage,
        context: StageContext,
        upstream_identities: Dict[str, ExecutionIdentity],
    ) -> ExecutionIdentity:
        """Compute execution identity for a stage.

        Input IDs include:
        - Context artifact hashes
        - Upstream stage CAS IDs (for cascade invalidation)

        Args:
            stage: Stage to compute identity for
            context: Execution context
            upstream_identities: Identities of upstream stages

        Returns:
            ExecutionIdentity for this stage execution
        """
        import numpy as np

        # Collect input IDs from context artifacts
        input_ids = []

        for dep_name in stage.get_dependencies():
            if dep_name in upstream_identities:
                # Use upstream CAS ID for cascade invalidation
                input_ids.append(upstream_identities[dep_name].cas_id)

            # Also include actual artifact hashes
            artifact = context.get_artifact(dep_name)
            if artifact is not None:
                if isinstance(artifact, np.ndarray):
                    # Use shared helper for consistent NumPy identity across executors
                    # This includes dtype, shape, and data hash to avoid false cache hits
                    input_ids.append(_compute_numpy_array_id(artifact))
                elif hasattr(artifact, "tobytes"):
                    # Non-NumPy types with tobytes (e.g., custom array-like)
                    input_ids.append(hashlib.sha256(artifact.tobytes()).hexdigest())
                elif isinstance(artifact, dict):
                    input_ids.append(hashlib.sha256(jcs_dumpb(artifact)).hexdigest())
                elif hasattr(artifact, "sha256"):
                    input_ids.append(artifact.sha256)

        # Compute stage config hash
        stage_config = context.config.get(stage.name, {})

        return compute_cas_id(
            stage_name=stage.name,
            input_ids=input_ids,
            config=stage_config,
            stage_version=stage.version,
            code_hash=self._code_hash,
            lockfile_path=self._lockfile_path,
        )

    @staticmethod
    def _sanitize_stage_name_for_filename(stage_name: str) -> str:
        """Strip path-separator and traversal characters from a stage name.

        ``stage_name`` is interpolated into a lock-file path. Internal callers
        only ever pass simple identifiers, but a defensive scrub keeps an
        accidental ``..`` or ``/`` from ever escaping ``locks_dir``.
        Replaces every disallowed character with ``_`` rather than raising,
        because the lock is best-effort coordination — corrupting the name is
        preferable to failing the run.
        """
        cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", stage_name)
        # Defuse traversal even if ``.`` and ``-`` survived the regex above.
        if cleaned in {"", ".", ".."} or cleaned.startswith(".."):
            cleaned = "_" + cleaned
        return cleaned

    def _get_lock(self, stage_name: str, cas_id: str) -> FileLock:
        """Get file lock for a stage execution."""
        safe_id = _sanitize_cas_id_for_filename(cas_id)
        safe_stage = self._sanitize_stage_name_for_filename(stage_name)
        lock_file = self.locks_dir / f"{safe_stage}_{safe_id[:16]}.lock"
        return FileLock(lock_file, timeout=self.config.lock_timeout)

    def _add_provenance_node(
        self,
        merkle_dag: MerkleDAG,
        merkle_node_hashes: Dict[str, str],
        stage_name: str,
        stage: Stage,
        identity: ExecutionIdentity,
        cached: bool,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Add provenance node for a stage execution.

        Helper method to avoid duplication between cache hit and miss paths.

        Args:
            merkle_dag: Merkle DAG to add node to
            merkle_node_hashes: Dict mapping stage names to merkle node hashes
            stage_name: Name of the stage
            stage: Stage instance
            identity: Execution identity
            cached: Whether result was from cache
            duration_ms: Execution duration (None for cache hits)
        """
        # Create metadata for the artifact node
        metadata: Dict[str, Any] = {"stage": stage_name, "cached": cached}
        if duration_ms is not None:
            metadata["duration_ms"] = duration_ms

        # Add artifact node for this stage's identity
        artifact_hash = merkle_dag.add_artifact(
            artifact_type="execution_identity",
            content_hash=identity.cas_id,
            metadata=metadata,
        )
        merkle_node_hashes[stage_name] = artifact_hash

        # Add computation node if there are upstream dependencies
        upstream_hashes = [merkle_node_hashes[d] for d in stage.get_dependencies() if d in merkle_node_hashes]
        if upstream_hashes:
            computation_metadata = {"version": stage.version}
            if duration_ms is not None:
                computation_metadata["duration_ms"] = duration_ms

            merkle_dag.add_computation(
                node_id=stage_name,
                inputs=upstream_hashes,
                outputs={"cas_id": identity.cas_id, "cached": cached},
                metadata=computation_metadata,
            )

    def execute(
        self,
        graph: StageGraph,
        context: StageContext,
        run_id: Optional[str] = None,
    ) -> CASExecutionResult:
        """Execute DAG with CAS-aware caching.

        Args:
            graph: Stage graph to execute
            context: Execution context
            run_id: Optional run identifier

        Returns:
            CASExecutionResult with cache statistics and provenance
        """
        import uuid

        if run_id is None:
            run_id = str(uuid.uuid4())

        start_time = time.time()
        context.run_id = run_id

        # Initialize result tracking
        stage_results: Dict[str, StageResult] = {}
        identities: Dict[str, ExecutionIdentity] = {}
        execution_order: List[str] = []
        cache_hits = 0
        cache_misses = 0

        # Initialize provenance DAG
        merkle_dag = MerkleDAG() if self.config.enable_provenance else None
        # Track merkle node hashes for each stage. MerkleDAG.add_computation() requires
        # that input node hashes exist in the DAG. This dict maps stage names to their
        # artifact node hashes, enabling us to reference upstream stages when adding
        # computation nodes for downstream stages.
        merkle_node_hashes: Dict[str, str] = {}

        try:
            # Get topological execution order
            topo_order = graph.get_execution_order()

            # Execute stages in order
            for stage_name in topo_order:
                stage = graph.stages[stage_name]

                # Compute identity including upstream dependencies
                identity = self._compute_stage_identity(
                    stage=stage,
                    context=context,
                    upstream_identities=identities,
                )
                identities[stage_name] = identity

                # Check cache
                cached_result = None
                if self.config.enable_caching:
                    cached_result = self._check_cache(identity)

                if cached_result is not None:
                    # Cache hit
                    logger.info(
                        "Stage %s: cache hit (%s)",
                        stage_name,
                        identity.cas_id[:16],
                    )
                    stage_results[stage_name] = cached_result
                    execution_order.append(stage_name)
                    cache_hits += 1

                    # Propagate artifacts
                    for name, value in cached_result.artifacts.items():
                        context.set_artifact(name, value)

                    # Add to provenance DAG
                    if merkle_dag:
                        self._add_provenance_node(
                            merkle_dag=merkle_dag,
                            merkle_node_hashes=merkle_node_hashes,
                            stage_name=stage_name,
                            stage=stage,
                            identity=identity,
                            cached=True,
                        )
                else:
                    # Cache miss - execute with lock
                    with self._get_lock(stage_name, identity.cas_id):
                        # Double-check cache after acquiring lock
                        if self.config.enable_caching:
                            cached_result = self._check_cache(identity)
                            if cached_result is not None:
                                logger.info(
                                    "Stage %s: cache hit (after lock) (%s)",
                                    stage_name,
                                    identity.cas_id[:16],
                                )
                                stage_results[stage_name] = cached_result
                                execution_order.append(stage_name)
                                cache_hits += 1

                                for name, value in cached_result.artifacts.items():
                                    context.set_artifact(name, value)
                                continue

                        # Execute stage
                        logger.info(
                            "Stage %s: executing (cache miss)",
                            stage_name,
                        )
                        result = stage.execute(context)
                        stage_results[stage_name] = result
                        execution_order.append(stage_name)
                        cache_misses += 1

                        # Store in cache
                        if self.config.enable_caching and result.is_success():
                            self._store_cache(identity, result)

                        # Propagate artifacts
                        for name, value in result.artifacts.items():
                            context.set_artifact(name, value)

                        # Add to provenance DAG
                        if merkle_dag:
                            self._add_provenance_node(
                                merkle_dag=merkle_dag,
                                merkle_node_hashes=merkle_node_hashes,
                                stage_name=stage_name,
                                stage=stage,
                                identity=identity,
                                cached=False,
                                duration_ms=result.duration_ms,
                            )

                        # Check for failure
                        if not result.is_success():
                            return CASExecutionResult(
                                success=False,
                                stage_results=stage_results,
                                execution_order=execution_order,
                                cache_hits=cache_hits,
                                cache_misses=cache_misses,
                                total_duration_ms=(time.time() - start_time) * 1000,
                                merkle_dag=merkle_dag,
                                identities=identities,
                                error=f"Stage {stage_name} failed: {result.error}",
                            )

        except Exception as e:
            logger.error("DAG execution failed: %s", e)
            return CASExecutionResult(
                success=False,
                stage_results=stage_results,
                execution_order=execution_order,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                total_duration_ms=(time.time() - start_time) * 1000,
                merkle_dag=merkle_dag,
                identities=identities,
                error=str(e),
            )

        return CASExecutionResult(
            success=True,
            stage_results=stage_results,
            execution_order=execution_order,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            total_duration_ms=(time.time() - start_time) * 1000,
            merkle_dag=merkle_dag,
            identities=identities,
        )

    def _check_cache(self, identity: ExecutionIdentity) -> Optional[StageResult]:
        """Check if stage result is cached.

        Uses recursive deserialization to handle nested numpy arrays and dicts,
        matching the single-stage executor's semantics.

        Args:
            identity: Execution identity to look up

        Returns:
            Cached StageResult if found and valid, None otherwise
        """
        cache_path = self._cache_path(identity.cas_id)
        if not cache_path.exists():
            return None

        try:
            import json

            data = json.loads(cache_path.read_text())

            # Validate schema version
            if data.get("schema_version") != identity.schema_version:
                logger.debug("Cache schema mismatch for %s", identity.cas_id[:16])
                return None

            # Validate platform compatibility
            metadata_dict = data.get("metadata", {})
            if metadata_dict:
                metadata = ArtifactMetadata.from_dict(metadata_dict)
                if not is_compatible(
                    metadata,
                    allow_cross_platform=self.config.allow_cross_platform,
                ):
                    logger.debug("Cache platform mismatch for %s", identity.cas_id[:16])
                    return None

            # Reconstruct artifacts using shared recursive deserialization
            # This handles nested dicts/lists with numpy arrays
            raw_artifacts = data.get("artifacts", {})
            artifacts = load_serializable(raw_artifacts, self.artifact_store)

            return StageResult(
                stage_name=data["stage_name"],
                stage_version=data["stage_version"],
                status=StageStatus.CACHED,
                artifacts=artifacts,
                cache_hit=True,
                cache_key=identity.cas_id,
                metadata=data.get("result_metadata", {}),
            )

        except CASObjectMissingError as e:
            # Missing CAS object - treat as cache miss for self-healing
            logger.warning("Missing CAS object for %s: %s", identity.cas_id[:16], e)
            return None
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("Failed to load cache for %s: %s", identity.cas_id[:16], e)
            return None

    def _store_cache(self, identity: ExecutionIdentity, result: StageResult) -> None:
        """Store stage result in cache atomically.

        Uses atomic writes to prevent partial writes visible to concurrent readers.
        Uses shared recursive serialization for consistent semantics with the
        single-stage executor - handles nested dicts/lists with numpy arrays.

        Args:
            identity: Execution identity
            result: Stage result to cache
        """
        cache_path = self._cache_path(identity.cas_id)

        # Use shared recursive serialization (same as single-stage executor)
        # This handles nested dicts/lists with numpy arrays
        base_path = cache_path.parent
        serialized_artifacts = make_serializable(
            result.artifacts,
            self.artifact_store,
            base_path,
            identity.cas_id,
        )

        # Create artifact metadata from SERIALIZED form (consistent with single-stage executor)
        # This ensures the output_hash reflects actual content, not just type signatures
        output_hash = hashlib.sha256(jcs_dumpb(serialized_artifacts)).hexdigest()
        metadata = create_artifact_metadata(output_hash, identity)

        # Build cache entry
        cache_data = {
            "cas_id": identity.cas_id,
            "stage_name": result.stage_name,
            "stage_version": result.stage_version,
            "schema_version": identity.schema_version,
            "artifacts": serialized_artifacts,
            "result_metadata": result.metadata,
            "metadata": metadata.to_dict(),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

        # Use atomic write (shared with single-stage executor)
        _atomic_write_json(cache_path, cache_data)
        logger.debug("Cached result for %s", identity.cas_id[:16])

    def _cache_path(self, cas_id: str) -> Path:
        """Get cache file path for a CAS ID."""
        safe_id = _sanitize_cas_id_for_filename(cas_id)
        prefix = safe_id[:2]
        return self.cache_dir / "dag_cache" / prefix / f"{safe_id}.json"

    def invalidate(
        self,
        stage_names: Optional[List[str]] = None,
        before: Optional[datetime] = None,
    ) -> int:
        """Invalidate cached results.

        Args:
            stage_names: Specific stages to invalidate (None = all)
            before: Invalidate entries older than this time

        Returns:
            Number of cache entries invalidated
        """
        import json

        count = 0
        dag_cache = self.cache_dir / "dag_cache"

        if not dag_cache.exists():
            return 0

        for prefix_dir in dag_cache.iterdir():
            if not prefix_dir.is_dir():
                continue

            for cache_file in prefix_dir.glob("*.json"):
                try:
                    data = json.loads(cache_file.read_text())

                    # Check stage name filter
                    if stage_names is not None:
                        if data.get("stage_name") not in stage_names:
                            continue

                    # Check time filter
                    if before is not None:
                        cached_at = data.get("cached_at", "")
                        if cached_at:
                            cache_time = datetime.fromisoformat(cached_at)
                            if cache_time >= before:
                                continue

                    # Invalidate by removing
                    cache_file.unlink()
                    count += 1

                except (json.JSONDecodeError, OSError):
                    continue

        logger.info("Invalidated %d cache entries", count)
        return count


def verify_dag_determinism(
    executor: CASDAGExecutor,
    graph: StageGraph,
    context: StageContext,
    runs: int = 2,
) -> tuple[bool, Dict[str, List[str]]]:
    """Verify DAG produces deterministic output.

    Runs the DAG multiple times and compares output hashes.

    Args:
        executor: CAS DAG executor
        graph: Stage graph to verify
        context: Execution context
        runs: Number of runs to compare

    Returns:
        Tuple of (is_deterministic, stage_name -> list of output hashes)
    """
    # Disable caching for determinism verification
    original_caching = executor.config.enable_caching
    executor.config.enable_caching = False

    try:
        all_hashes: Dict[str, List[str]] = {name: [] for name in graph.stages}

        for i in range(runs):
            # Create fresh context
            run_context = StageContext(
                artifacts=dict(context.artifacts),
                config=dict(context.config),
                device=context.device,
                cache_enabled=False,
            )

            result = executor.execute(graph, run_context)

            if not result.success:
                logger.warning("Run %d failed: %s", i + 1, result.error)
                return False, all_hashes

            for stage_name, identity in result.identities.items():
                all_hashes[stage_name].append(identity.cas_id)

            logger.debug("Run %d completed", i + 1)

        # Check all hashes are identical
        is_deterministic = True
        for stage_name, hashes in all_hashes.items():
            if len(set(hashes)) != 1:
                logger.warning(
                    "Non-deterministic stage %s: %d unique hashes",
                    stage_name,
                    len(set(hashes)),
                )
                is_deterministic = False

        return is_deterministic, all_hashes

    finally:
        executor.config.enable_caching = original_caching
