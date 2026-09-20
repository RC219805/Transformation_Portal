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
import json
import logging
import os
import re
import stat
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from transformation_portal.core._cas_helpers import CASObjectMissingError
from transformation_portal.core._cas_helpers import atomic_write_json as _atomic_write_json
from transformation_portal.core._cas_helpers import canonical_input_value, load_serializable, make_serializable
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
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.stage_graph.graph import StageGraph
from transformation_portal.stage_graph.stage import Stage, StageContext, StageResult, StageStatus
from transformation_portal.storage.cas_store import ArtifactStore, CASError
from transformation_portal.storage.merkle_dag import MerkleDAG

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthoritativeStageIdentity:
    """Trusted compiler adapter; no executable values cross the plan boundary.

    The caller validates its versioned identity factory before constructing
    this adapter. Cache reads bind both its key and the exact immutable payload.
    """

    stage_name: str
    stage_version: str
    cache_key: str
    canonical_identity_bytes: bytes

    def __post_init__(self) -> None:
        if not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", self.cache_key):
            raise ValueError("Authoritative cache keys must be SHA-256 digests")
        if type(self.canonical_identity_bytes) is not bytes or len(self.canonical_identity_bytes) > 1_048_576:
            raise ValueError("Authoritative identity must be bounded canonical bytes")
        payload = json.loads(self.canonical_identity_bytes)
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("schema"), str)
            or canonicalize_json(payload) != self.canonical_identity_bytes
        ):
            raise ValueError("Authoritative identity must be a canonical schema-bearing object")
        if not self.stage_name or not self.stage_version:
            raise ValueError("Authoritative identity requires stage name and version")

    @property
    def cas_id(self) -> str:
        return self.cache_key if self.cache_key.startswith("sha256:") else f"sha256:{self.cache_key}"

    @property
    def schema_version(self) -> str:
        return self.to_dict()["schema"]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_identity_bytes)


StageIdentity = Union[ExecutionIdentity, AuthoritativeStageIdentity]
IdentityProvider = Callable[[Stage, StageContext, Mapping[str, StageIdentity]], AuthoritativeStageIdentity]
ExecutionCheckpoint = Callable[[str, str, StageContext], None]
StageCachePolicy = Callable[[Stage, StageContext], bool]
_CACHE_MANIFEST_SCHEMA = "tp.cas.stage-cache.v1"
_MAX_CACHE_MANIFEST_BYTES = 4 * 1024**2
_CACHE_MANIFEST_KEYS = frozenset(
    {
        "cache_manifest_schema",
        "cas_id",
        "identity",
        "stage_name",
        "stage_version",
        "schema_version",
        "artifacts",
        "artifacts_sha256",
        "result_metadata",
        "metadata",
        "cached_at",
        "payload_sha256",
    }
)


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
    identities: Dict[str, StageIdentity] = field(default_factory=dict)
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
        upstream_identities: Dict[str, StageIdentity],
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
        # A stage may consume root artifacts without declaring stage dependencies.
        # Bind all stage-visible values conservatively unless a trusted compiler
        # supplies the exact consumed-input identity through identity_provider.
        artifact_payload = canonical_input_value(context.artifacts)
        upstream_payload = {
            name: upstream_identities[name].cas_id for name in stage.get_dependencies() if name in upstream_identities
        }
        input_ids = [hashlib.sha256(jcs_dumpb({"artifacts": artifact_payload, "upstream": upstream_payload})).hexdigest()]
        return compute_cas_id(
            stage_name=stage.name,
            input_ids=input_ids,
            config={"stage": context.config.get(stage.name, {}), "device": context.device},
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
        identity: StageIdentity,
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
        *,
        identity_provider: Optional[IdentityProvider] = None,
        checkpoint: Optional[ExecutionCheckpoint] = None,
        cache_policy: Optional[StageCachePolicy] = None,
    ) -> CASExecutionResult:
        """Execute trusted stages with one cache and compiler-owned authority.

        Callbacks are trusted in-process integrations, never deserialized from
        plan JSON. A checkpoint exception stops execution before subsequent
        propagation/publication. Process-level cancellation belongs to the
        caller's existing worker boundary.
        """
        import uuid

        started = time.monotonic()
        context = StageContext(
            artifacts=dict(context.artifacts),
            config=dict(context.config),
            device=context.device,
            cache_enabled=False,
            cache_dir=None,
            run_id=run_id or str(uuid.uuid4()),
            metadata=dict(context.metadata),
        )
        stage_results: Dict[str, StageResult] = {}
        identities: Dict[str, StageIdentity] = {}
        execution_order: List[str] = []
        cache_hits = cache_misses = 0
        merkle_dag = MerkleDAG() if self.config.enable_provenance else None
        merkle_node_hashes: Dict[str, str] = {}

        def check(stage_name: str, phase: str) -> None:
            if checkpoint is not None:
                checkpoint(stage_name, phase, context)

        def finish(error: Optional[str] = None) -> CASExecutionResult:
            return CASExecutionResult(
                success=error is None,
                stage_results=stage_results,
                execution_order=execution_order,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                total_duration_ms=(time.monotonic() - started) * 1000,
                merkle_dag=merkle_dag,
                identities=identities,
                error=error,
            )

        try:
            for stage_name in graph.get_execution_order():
                stage = graph.stages[stage_name]
                check(stage_name, "before_identity")
                identity = (
                    identity_provider(stage, context, dict(identities))
                    if identity_provider is not None
                    else self._compute_stage_identity(stage, context, identities)
                )
                if not isinstance(identity, (ExecutionIdentity, AuthoritativeStageIdentity)):
                    raise TypeError("Identity provider returned an unsupported authority type")
                if identity.stage_name != stage.name or identity.stage_version != stage.version:
                    raise ValueError("Stage identity does not match the executing stage")
                identities[stage_name] = identity
                selected_cache_policy = cache_policy(stage, context) if cache_policy is not None else True
                if type(selected_cache_policy) is not bool:
                    raise TypeError("Stage cache policy must return a boolean")
                cache_enabled = self.config.enable_caching and selected_cache_policy
                check(stage_name, "before_cache")
                result = self._check_cache(identity) if cache_enabled else None
                check(stage_name, "after_cache")
                if result is None:
                    with self._get_lock(stage_name, identity.cas_id):
                        check(stage_name, "before_cache")
                        result = self._check_cache(identity) if cache_enabled else None
                        check(stage_name, "after_cache")
                        if result is None:
                            check(stage_name, "before_compute")
                            result = stage.execute_uncached(context)
                            cache_misses += 1
                            check(stage_name, "after_compute")
                            if cache_enabled and result.is_success():
                                check(stage_name, "before_store")
                                self._store_cache(identity, result)
                if result.cache_hit:
                    cache_hits += 1
                stage_results[stage_name] = result
                execution_order.append(stage_name)
                if not result.is_success():
                    return finish(f"Stage {stage_name} failed: {result.error}")
                check(stage_name, "before_propagate")
                for name, value in result.artifacts.items():
                    context.set_artifact(name, value)
                check(stage_name, "after_propagate")
                if merkle_dag is not None:
                    self._add_provenance_node(
                        merkle_dag,
                        merkle_node_hashes,
                        stage_name,
                        stage,
                        identity,
                        cached=result.cache_hit,
                        duration_ms=None if result.cache_hit else result.duration_ms,
                    )
        except Exception as exc:
            logger.error("DAG execution failed: %s", exc)
            return finish(str(exc))
        return finish()

    def _read_cache_manifest(self, cache_path: Path) -> dict[str, Any]:
        """Read a bounded regular manifest without following cache symlinks."""
        if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
            raise OSError("Verified cache reads require no-follow directory descriptors")
        descriptors = []
        try:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            descriptor = os.open(self.cache_dir, flags)
            descriptors.append(descriptor)
            for component in cache_path.relative_to(self.cache_dir).parts[:-1]:
                descriptor = os.open(component, flags, dir_fd=descriptor)
                descriptors.append(descriptor)
            file_descriptor = os.open(cache_path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
            descriptors.append(file_descriptor)
            info = os.fstat(file_descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_CACHE_MANIFEST_BYTES:
                raise ValueError("Cache manifest is not a bounded regular file")
            chunks = []
            size = 0
            while chunk := os.read(file_descriptor, min(65536, _MAX_CACHE_MANIFEST_BYTES + 1 - size)):
                size += len(chunk)
                if size > _MAX_CACHE_MANIFEST_BYTES:
                    raise ValueError("Cache manifest exceeds byte limit")
                chunks.append(chunk)

            def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("Duplicate cache manifest key")
                    result[key] = value
                return result

            return json.loads(b"".join(chunks), object_pairs_hook=unique_pairs)
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def _check_cache(self, identity: StageIdentity) -> Optional[StageResult]:
        """Accept only identity-bound, closed manifests and verified CAS bytes."""
        try:
            data = self._read_cache_manifest(self._cache_path(identity.cas_id))
            if not isinstance(data, dict) or set(data) != _CACHE_MANIFEST_KEYS:
                return None
            if (
                data["cache_manifest_schema"] != _CACHE_MANIFEST_SCHEMA
                or data["cas_id"] != identity.cas_id
                or data["schema_version"] != identity.schema_version
                or data["identity"] != identity.to_dict()
                or data["stage_name"] != identity.stage_name
                or data["stage_version"] != identity.stage_version
                or not isinstance(data["artifacts"], dict)
                or not isinstance(data["result_metadata"], dict)
            ):
                return None
            digest = data.pop("payload_sha256")
            if hashlib.sha256(jcs_dumpb(data)).hexdigest() != digest:
                return None
            if hashlib.sha256(jcs_dumpb(data["artifacts"])).hexdigest() != data["artifacts_sha256"]:
                return None
            if isinstance(identity, ExecutionIdentity):
                metadata = ArtifactMetadata.from_dict(data["metadata"])
                if (
                    metadata.artifact_id != data["artifacts_sha256"]
                    or metadata.execution_identity != identity.cas_id
                    or not is_compatible(metadata, allow_cross_platform=self.config.allow_cross_platform)
                ):
                    return None
            elif data["metadata"] != {}:
                return None
            artifacts = load_serializable(data["artifacts"], self.artifact_store)
            return StageResult(
                stage_name=identity.stage_name,
                stage_version=identity.stage_version,
                status=StageStatus.CACHED,
                artifacts=artifacts,
                cache_hit=True,
                cache_key=identity.cas_id,
                metadata=data["result_metadata"],
            )
        except (CASObjectMissingError, CASError, ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
            logger.debug("Cache entry rejected for %s: %s", identity.cas_id[:16], exc)
            return None

    def _store_cache(self, identity: StageIdentity, result: StageResult) -> None:
        """Publish one closed identity-bound cache manifest atomically."""
        if result.stage_name != identity.stage_name or result.stage_version != identity.stage_version:
            raise ValueError("Cache result does not match its stage identity")
        cache_path = self._cache_path(identity.cas_id)
        serialized = make_serializable(result.artifacts, self.artifact_store, cache_path.parent, identity.cas_id)
        output_hash = hashlib.sha256(jcs_dumpb(serialized)).hexdigest()
        metadata = create_artifact_metadata(output_hash, identity).to_dict() if isinstance(identity, ExecutionIdentity) else {}
        data = {
            "cache_manifest_schema": _CACHE_MANIFEST_SCHEMA,
            "cas_id": identity.cas_id,
            "identity": identity.to_dict(),
            "stage_name": result.stage_name,
            "stage_version": result.stage_version,
            "schema_version": identity.schema_version,
            "artifacts": serialized,
            "artifacts_sha256": output_hash,
            "result_metadata": result.metadata,
            "metadata": metadata,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        data["payload_sha256"] = hashlib.sha256(jcs_dumpb(data)).hexdigest()
        if len(jcs_dumpb(data)) > _MAX_CACHE_MANIFEST_BYTES:
            raise ValueError("Cache manifest exceeds byte limit")
        _atomic_write_json(cache_path, data)

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
