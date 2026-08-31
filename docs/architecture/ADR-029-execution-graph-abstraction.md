# ADR-029: Execution Graph Abstraction for Spatial AI Orchestration

**Status:** IMPLEMENTED — conditional long-term stage-executor supersession recorded for ADR-051
**Date:** 2026-02-12
**Updated:** 2026-08-30
**Authority:** Transformation Portal Architect
**Supersedes:** None
**Conditional partial supersession:** [ADR-051](ADR-051-execution-artifact-authority-designation.md),
effective only when ADR-051 is Accepted and only for the long-term stage-executor designation
**Related:** ADR-027 (Phase 2 Extension), Phase 3 Foundation (v2.1.0-phase2-foundation)
**Enforcement:** CI gates (determinism, cache correctness), provenance validation

**Implementation Status:**
- Core abstractions (ExecutionGraph, Stage, Executor, ArtifactStore): ✅ Complete
- Stage adapters (IngestStage, SegmentationStage, MaterialsStage): ✅ Complete (2026-03-23)
- Pipeline integration (use_execution_graph flag): ✅ Complete (2026-03-23)
- Graph-based execution path: ✅ Complete (2026-03-23)
- MaterialsStage caching: ⚠️ Disabled (outputs contain nested PBRTextures objects not serializable by ArtifactStore)

---

## Executive Summary

**Decision:** Introduce an **execution graph abstraction** for Spatial AI orchestration that models pipeline execution as an explicit directed acyclic graph (DAG) with deterministic caching, automatic provenance tracking, and fail-fast resource budgeting.

**Design Principle:** Treat orchestration like a **physics engine for spatial AI**—explicit constraints, deterministic behavior, introspectable state, predictable scaling. Not a generic workflow system.

**Core Abstractions:**
1. **ExecutionGraph** — Explicit DAG with stage dependencies and resource annotations
2. **Stage** — Pure function protocol with input/output contracts and resource declarations
3. **Executor** — Runtime that schedules stages, enforces budgets, manages cache
4. **ArtifactStore** — Content-addressed storage with provenance metadata and atomic writes

**Why Phase 3 Matters:** As we add neural materials, 3DGS backends, NeRF adapters, and batch pipelines, the **orchestration layer becomes the entropy hotspot**—not the models, the execution graph. Phase 3 makes state complexity sustainable at scale.

---

## Context

### What Phase 2 Proved

Phase 2 (ADR-027) wasn't just about adding segmentation and materials. It proved four critical properties:

1. **Tier Separation Works**
   - ML tier isolated from core (zero import contamination via ADR-023)
   - Optional dependencies truly optional
   - Golden baselines stable under 25k-line expansion

2. **Governance Scales**
   - APEX policy gates held under feature growth pressure
   - Contract enforcement didn't collapse
   - Performance ledgering survived ML integration

3. **Determinism Survives ML**
   - HuggingFace revision pinning closes entropy channel (ADR-021)
   - Stable presets + pinned revisions = reproducible pipelines
   - Can now trust failures (no silent upstream drift)

4. **Mathematical Honesty**
   - Camera interpolation limitations documented (not hidden)
   - OBJ export scalability acknowledged (not shipped as time bomb)
   - Bias toward truth over polish

### Current Orchestration Limitations

**Current State:** `SpatialAIPipeline` (Phase 2.4) executes stages as hardcoded sequence:

```python
# src/transformation_portal/spatial_ai/orchestration/pipeline.py (lines 280-312)
if "ingest" in self.config.stages:
    result.linear_image = self._run_ingest(...)
    result.stages_completed.append("ingest")

if "segment" in self.config.stages:
    result.segmentation = self._run_segmentation(...)
    result.stages_completed.append("segment")

if "materials" in self.config.stages:
    result.materials = self._run_materials(...)
    result.stages_completed.append("materials")
```

**Problems This Creates:**

1. **No Introspection**
   - Cannot visualize DAG before execution
   - Cannot trace artifact lineage after execution
   - Cannot predict resource requirements

2. **No Caching**
   - Models re-load on every run (SAM2 @ 1024x1024 = 3s)
   - If input unchanged, repeating work is waste
   - No mechanism to skip completed stages

3. **No Provenance**
   - Can't trace: "Which model version produced this mask?"
   - Can't verify: "Is this texture from the golden baseline?"
   - Can't audit: "What config was used for this 3DGS?"

4. **No Resource Planning**
   - Multi-stage pipelines can OOM mid-execution
   - GPU memory budget not enforced upfront
   - No predictable memory footprint

5. **No Parallelism**
   - Sequential execution even when stages independent
   - Materials generation could parallelize across segments
   - Wastes multi-GPU resources

### The Phase 3 Problem Space

As we add:
- Neural materials (NVDIFFREC, MaterialGAN)
- 3DGS backends (Inria, MipNeRF360, InstantNGP)
- NeRF adapters
- Batch pipelines (process 1000 images)
- Multi-view reconstruction

The **orchestration layer becomes the limiting factor**:
- State explosion (100 stages × 10 checkpoints = 1000 artifacts)
- Memory pressure (3 models × 4GB = 12GB VRAM)
- Execution time (no caching = 10× slower than necessary)
- Debugging complexity (no lineage = can't reproduce failures)

**Phase 3 Core Insight:** This is a **distributed systems problem masquerading as a feature set**. We need runtime guarantees, not more features.

---

## Decision

### 1. Core Abstractions

#### 1.1 Stage Protocol

**Pure function with explicit contracts:**

```python
# src/transformation_portal/spatial_ai/orchestration/graph/stage.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Any, Dict, List
from pathlib import Path

@dataclass
class StageMetadata:
    """Metadata describing stage characteristics."""
    name: str
    version: str  # Semantic version for cache invalidation
    description: str
    resource_requirements: ResourceRequirements

    # Determinism guarantees
    deterministic: bool = True  # Same input → same output
    idempotent: bool = True     # Multiple runs safe

    # Execution hints
    can_parallelize: bool = False
    checkpoint_policy: CheckpointPolicy = CheckpointPolicy.ALWAYS

@dataclass
class ResourceRequirements:
    """Resource contract for stage execution."""
    gpu_memory_mb: int = 0
    cpu_memory_mb: int = 512
    min_disk_mb: int = 100
    gpu_required: bool = False
    estimated_time_ms: int = 1000  # For progress estimation

class Stage(Protocol):
    """Protocol for execution graph stages.

    Stages are pure functions:
    - No side effects (except output artifacts)
    - No global state access
    - No orchestration awareness (no caching, DAG, provenance logic)

    Orchestration is cross-cutting. Stages are domain functions.
    """

    @property
    def metadata(self) -> StageMetadata:
        """Stage metadata for resource planning and caching."""
        ...

    def execute(
        self,
        inputs: Dict[str, Any],
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Execute stage with inputs, return outputs.

        Args:
            inputs: Named inputs (from graph dependencies)
            context: Execution context (device, output_dir, etc.)

        Returns:
            Named outputs (available to downstream stages)

        Raises:
            StageExecutionError: On execution failure
        """
        ...

    def compute_cache_key(
        self,
        inputs: Dict[str, Any],
        context: ExecutionContext,
    ) -> str:
        """Compute content-addressed cache key.

        Key components:
        - Stage version (metadata.version)
        - Input fingerprints (hash of input data)
        - Execution context (device, config overrides)

        Returns:
            SHA256 hex string (64 chars)
        """
        ...
```

**Example Stage Implementation:**

```python
# src/transformation_portal/spatial_ai/segmentation/sam2_stage.py

class SAM2Stage:
    """SAM2 segmentation stage for execution graph."""

    @property
    def metadata(self) -> StageMetadata:
        return StageMetadata(
            name="sam2_segmentation",
            version="2.1.0",  # Bump on algorithm change
            description="SAM2 automatic mask generation",
            resource_requirements=ResourceRequirements(
                gpu_memory_mb=2048,
                cpu_memory_mb=512,
                min_disk_mb=100,
                gpu_required=False,  # Can fallback to CPU
                estimated_time_ms=3000,  # 3s @ 1024x1024
            ),
            deterministic=True,
            idempotent=True,
            can_parallelize=False,  # Single image
        )

    def execute(
        self,
        inputs: Dict[str, Any],
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Execute SAM2 segmentation."""
        # Extract inputs
        linear_rgb = inputs["linear_rgb"]  # (H, W, 3) float32
        gamma = inputs.get("gamma", 1.0)

        # Load model (executor handles caching)
        backend = SAM2Backend(
            model_size=context.config.get("model_size", "large"),
            device=context.device,
            repo_id=context.config.get("repo_id"),
            revision=context.config.get("revision"),
        )

        # Execute
        seg_input = SegmentationInput(
            image=linear_rgb,
            gamma=gamma,
            mode="auto",
        )
        result = backend.segment(seg_input)

        # Return outputs
        return {
            "masks": result.masks,  # (N, H, W) bool
            "scores": result.scores,  # (N,) float32
            "metadata": result.metadata,  # List[MaskMetadata]
        }

    def compute_cache_key(
        self,
        inputs: Dict[str, Any],
        context: ExecutionContext,
    ) -> str:
        """Compute cache key from inputs and config."""
        import hashlib

        # Input fingerprint
        input_hash = hashlib.sha256(inputs["linear_rgb"].tobytes()).hexdigest()[:16]

        # Config fingerprint
        config_parts = [
            context.config.get("model_size", "large"),
            context.config.get("revision", "unknown"),
            str(context.device),
        ]
        config_hash = hashlib.sha256(
            ":".join(config_parts).encode()
        ).hexdigest()[:16]

        # Cache key = stage_version:input_hash:config_hash
        return f"{self.metadata.version}:{input_hash}:{config_hash}"
```

#### 1.2 ExecutionGraph

**Explicit DAG with dependencies:**

```python
# src/transformation_portal/spatial_ai/orchestration/graph/execution_graph.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional
from pathlib import Path

@dataclass
class StageNode:
    """Node in execution graph."""
    stage_id: str  # Unique identifier
    stage: Stage
    inputs: Dict[str, str]  # input_name → source (e.g., "ingest.linear_rgb")
    optional: bool = False  # Skip if inputs unavailable

@dataclass
class ExecutionPlan:
    """Execution plan generated from graph."""
    stages: List[StageNode]  # Topologically sorted
    total_gpu_memory_mb: int
    total_cpu_memory_mb: int
    estimated_time_ms: int
    checkpoints: List[str]  # Stage IDs to checkpoint

class ExecutionGraph:
    """Directed acyclic graph of pipeline stages.

    Key features:
    - Explicit dependencies (data lineage)
    - Resource annotations (memory budgets)
    - Topological execution order
    - Parallel execution opportunities
    - Checkpoint semantics

    Example:
        >>> graph = ExecutionGraph()
        >>> graph.add_stage("ingest", IngestStage(), inputs={})
        >>> graph.add_stage(
        ...     "segment",
        ...     SAM2Stage(),
        ...     inputs={"linear_rgb": "ingest.linear_rgb"},
        ... )
        >>> plan = graph.plan()
        >>> print(f"Total GPU memory: {plan.total_gpu_memory_mb}MB")
    """

    def __init__(self):
        self._stages: Dict[str, StageNode] = {}
        self._execution_order: Optional[List[str]] = None

    def add_stage(
        self,
        stage_id: str,
        stage: Stage,
        inputs: Dict[str, str],
        optional: bool = False,
    ) -> None:
        """Add stage to graph.

        Args:
            stage_id: Unique identifier for this stage
            stage: Stage implementation
            inputs: Input mapping (input_name → "stage_id.output_name")
            optional: Skip if inputs unavailable

        Raises:
            ValueError: If stage_id already exists or creates cycle
        """
        if stage_id in self._stages:
            raise ValueError(f"Stage '{stage_id}' already exists")

        node = StageNode(
            stage_id=stage_id,
            stage=stage,
            inputs=inputs,
            optional=optional,
        )

        self._stages[stage_id] = node
        self._execution_order = None  # Invalidate cached order

    def plan(
        self,
        resource_limits: Optional[ResourceLimits] = None,
    ) -> ExecutionPlan:
        """Generate execution plan with resource validation.

        Args:
            resource_limits: Optional resource constraints

        Returns:
            ExecutionPlan with topologically sorted stages

        Raises:
            ResourceError: If resource limits exceeded
            GraphError: If graph has cycles or missing dependencies
        """
        # Topological sort
        execution_order = self._topological_sort()

        # Compute resource totals
        total_gpu_mb = 0
        total_cpu_mb = 0
        total_time_ms = 0
        checkpoints = []

        sorted_stages = []
        for stage_id in execution_order:
            node = self._stages[stage_id]
            reqs = node.stage.metadata.resource_requirements

            total_gpu_mb = max(total_gpu_mb, reqs.gpu_memory_mb)  # Peak, not sum
            total_cpu_mb += reqs.cpu_memory_mb
            total_time_ms += reqs.estimated_time_ms

            if node.stage.metadata.checkpoint_policy == CheckpointPolicy.ALWAYS:
                checkpoints.append(stage_id)

            sorted_stages.append(node)

        # Validate resource limits
        if resource_limits:
            if resource_limits.max_gpu_memory_gb:
                limit_mb = resource_limits.max_gpu_memory_gb * 1024
                if total_gpu_mb > limit_mb:
                    raise ResourceError(
                        f"Graph requires {total_gpu_mb}MB GPU memory, "
                        f"limit is {limit_mb}MB"
                    )

        return ExecutionPlan(
            stages=sorted_stages,
            total_gpu_memory_mb=total_gpu_mb,
            total_cpu_memory_mb=total_cpu_mb,
            estimated_time_ms=total_time_ms,
            checkpoints=checkpoints,
        )

    def visualize(self, output_path: Path) -> None:
        """Render graph as DOT/GraphViz diagram.

        Args:
            output_path: Output path for .dot or .png file
        """
        # Generate DOT format
        lines = ["digraph ExecutionGraph {"]
        lines.append('  rankdir=LR;')

        for stage_id, node in self._stages.items():
            # Node
            label = f"{stage_id}\\n{node.stage.metadata.resource_requirements.gpu_memory_mb}MB GPU"
            lines.append(f'  "{stage_id}" [label="{label}"];')

            # Edges
            for input_name, source in node.inputs.items():
                source_stage = source.split(".")[0]
                lines.append(f'  "{source_stage}" -> "{stage_id}" [label="{input_name}"];')

        lines.append("}")

        dot_content = "\n".join(lines)
        output_path.write_text(dot_content)

    def _topological_sort(self) -> List[str]:
        """Topological sort with cycle detection."""
        # Kahn's algorithm
        in_degree = {sid: 0 for sid in self._stages}

        # Compute in-degrees
        for node in self._stages.values():
            for source in node.inputs.values():
                source_stage = source.split(".")[0]
                if source_stage in in_degree:
                    in_degree[source_stage] += 1

        # Queue of stages with no dependencies
        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            stage_id = queue.pop(0)
            result.append(stage_id)

            # Decrease in-degree for downstream stages
            for other_id, node in self._stages.items():
                for source in node.inputs.values():
                    if source.startswith(f"{stage_id}."):
                        in_degree[other_id] -= 1
                        if in_degree[other_id] == 0:
                            queue.append(other_id)

        # Check for cycles
        if len(result) != len(self._stages):
            raise GraphError("Graph contains cycles")

        return result
```

#### 1.3 ArtifactStore

**Content-addressed storage with provenance:**

```python
# src/transformation_portal/spatial_ai/orchestration/graph/artifact_store.py

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
from pathlib import Path
import json
import hashlib
import tempfile
import shutil
import numpy as np

@dataclass
class ProvenanceMetadata:
    """Provenance metadata for cached artifacts."""
    cache_key: str
    stage_id: str
    stage_version: str
    input_fingerprints: Dict[str, str]
    config_snapshot: Dict[str, Any]
    timestamp: str  # ISO 8601
    hostname: str
    python_version: str
    numpy_version: str
    torch_version: Optional[str] = None
    device: str = "cpu"

    # Model provenance (if applicable)
    model_repo_id: Optional[str] = None
    model_revision: Optional[str] = None

class ArtifactStore:
    """Content-addressed artifact store with provenance tracking.

    Storage layout:
        .cache/spatial_ai/
        ├── artifacts/
        │   ├── <cache_key_prefix>/
        │   │   ├── <cache_key>.npz        # Artifact data
        │   │   └── <cache_key>.json       # Provenance metadata
        │   └── ...
        └── index.db  # SQLite index for queries (Phase 3 L3)

    Features:
    - Content-addressed (same inputs → same cache key)
    - Atomic writes (temp + rename)
    - Provenance metadata (input hashes, model revisions, timestamps)
    - LRU eviction (size-based limits)
    - Queryable lineage (SQL-like interface, L3)

    Example:
        >>> store = ArtifactStore(cache_dir=".cache/spatial_ai")
        >>>
        >>> # Check cache
        >>> cache_key = stage.compute_cache_key(inputs, context)
        >>> if store.exists(cache_key):
        ...     result = store.load(cache_key)
        >>> else:
        ...     result = stage.execute(inputs, context)
        ...     store.store(cache_key, result, provenance)
    """

    def __init__(
        self,
        cache_dir: Path,
        max_size_gb: float = 10.0,
        eviction_policy: str = "lru",
    ):
        """Initialize artifact store.

        Args:
            cache_dir: Base directory for cache
            max_size_gb: Maximum cache size (LRU eviction)
            eviction_policy: "lru" or "manual"
        """
        self.cache_dir = Path(cache_dir)
        self.artifacts_dir = self.cache_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_gb = max_size_gb
        self.eviction_policy = eviction_policy

    def exists(self, cache_key: str) -> bool:
        """Check if artifact exists in cache."""
        artifact_path = self._artifact_path(cache_key)
        return artifact_path.exists()

    def load(self, cache_key: str) -> Dict[str, Any]:
        """Load artifact from cache.

        Args:
            cache_key: Cache key (from stage.compute_cache_key)

        Returns:
            Artifact data (stage outputs)

        Raises:
            CacheMissError: If artifact not found
        """
        artifact_path = self._artifact_path(cache_key)
        metadata_path = self._metadata_path(cache_key)

        if not artifact_path.exists():
            raise CacheMissError(f"Artifact not found: {cache_key}")

        # Load artifact (assume .npz for now)
        data = dict(np.load(artifact_path, allow_pickle=False))

        # Load metadata
        if metadata_path.exists():
            with open(metadata_path) as f:
                provenance = json.load(f)
            data["_provenance"] = provenance

        # Touch for LRU
        artifact_path.touch()

        return data

    def store(
        self,
        cache_key: str,
        data: Dict[str, Any],
        provenance: ProvenanceMetadata,
    ) -> None:
        """Store artifact with provenance metadata.

        Atomic write: temp file → rename (no partial artifacts).

        Args:
            cache_key: Cache key (from stage.compute_cache_key)
            data: Artifact data (stage outputs)
            provenance: Provenance metadata
        """
        artifact_path = self._artifact_path(cache_key)
        metadata_path = self._metadata_path(cache_key)

        # Ensure directory exists
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        # Write artifact atomically
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=artifact_path.parent,
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)

            # Save as .npz (no pickle for security)
            np.savez_compressed(tmp, **data)

        # Atomic rename
        tmp_path.rename(artifact_path)

        # Write metadata
        with open(metadata_path, "w") as f:
            json.dump(asdict(provenance), f, indent=2)

        # Check size limits and evict if needed
        self._enforce_size_limit()

    def invalidate(self, cache_key: str) -> None:
        """Remove artifact from cache."""
        artifact_path = self._artifact_path(cache_key)
        metadata_path = self._metadata_path(cache_key)

        artifact_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    def query_provenance(
        self,
        stage_id: Optional[str] = None,
        model_revision: Optional[str] = None,
        since_timestamp: Optional[str] = None,
    ) -> List[ProvenanceMetadata]:
        """Query artifacts by provenance criteria.

        Phase 3 L3 feature: SQL-like lineage queries.

        Args:
            stage_id: Filter by stage ID
            model_revision: Filter by model revision
            since_timestamp: Filter by creation time

        Returns:
            List of matching provenance records
        """
        # Placeholder for L3: iterate all metadata files
        # Future: SQLite index for efficient queries
        results = []

        for metadata_path in self.artifacts_dir.rglob("*.json"):
            with open(metadata_path) as f:
                prov = json.load(f)

            # Apply filters
            if stage_id and prov.get("stage_id") != stage_id:
                continue
            if model_revision and prov.get("model_revision") != model_revision:
                continue
            if since_timestamp and prov.get("timestamp", "") < since_timestamp:
                continue

            results.append(ProvenanceMetadata(**prov))

        return results

    def _artifact_path(self, cache_key: str) -> Path:
        """Compute artifact file path from cache key."""
        # Shard by first 2 chars for filesystem efficiency
        prefix = cache_key[:2]
        return self.artifacts_dir / prefix / f"{cache_key}.npz"

    def _metadata_path(self, cache_key: str) -> Path:
        """Compute metadata file path from cache key."""
        return self._artifact_path(cache_key).with_suffix(".json")

    def _enforce_size_limit(self) -> None:
        """Evict oldest artifacts if cache exceeds size limit."""
        if self.eviction_policy != "lru":
            return

        # Compute total size
        total_size = sum(
            f.stat().st_size
            for f in self.artifacts_dir.rglob("*.npz")
        )

        limit_bytes = self.max_size_gb * 1e9
        if total_size <= limit_bytes:
            return

        # Evict oldest (by access time)
        artifacts = sorted(
            self.artifacts_dir.rglob("*.npz"),
            key=lambda p: p.stat().st_atime,
        )

        for artifact_path in artifacts:
            if total_size <= limit_bytes:
                break

            # Remove artifact and metadata
            size = artifact_path.stat().st_size
            artifact_path.unlink()
            artifact_path.with_suffix(".json").unlink(missing_ok=True)
            total_size -= size
```

#### 1.4 Executor

**Runtime with caching and resource enforcement:**

```python
# src/transformation_portal/spatial_ai/orchestration/graph/executor.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import time
import platform
import sys

@dataclass
class ExecutionContext:
    """Context passed to stages during execution."""
    device: str  # "cuda", "mps", "cpu"
    output_dir: Path
    config: Dict[str, Any]  # Stage-specific config
    artifact_store: ArtifactStore
    resource_manager: ResourceManager

@dataclass
class ExecutionResult:
    """Result from graph execution."""
    outputs: Dict[str, Any]  # Final outputs
    stages_executed: List[str]
    stages_cached: List[str]
    execution_time_ms: float
    peak_memory_mb: float
    artifacts_written: List[str]  # Cache keys
    provenance: List[ProvenanceMetadata]

class Executor:
    """Execution runtime for execution graphs.

    Features:
    - Topological execution order
    - Deterministic caching (content-addressed)
    - Resource reservation (fail-fast on OOM)
    - Provenance tracking (automatic)
    - Parallel execution (where dependencies allow, L2)

    Example:
        >>> graph = ExecutionGraph()
        >>> # ... add stages ...
        >>>
        >>> executor = Executor(
        ...     artifact_store=store,
        ...     resource_limits=limits,
        ... )
        >>> result = executor.execute(
        ...     graph=graph,
        ...     inputs={"input_path": "scene.tiff"},
        ...     output_dir=Path("output/"),
        ... )
    """

    def __init__(
        self,
        artifact_store: ArtifactStore,
        resource_limits: Optional[ResourceLimits] = None,
    ):
        self.artifact_store = artifact_store
        self.resource_limits = resource_limits or ResourceLimits()
        self.resource_manager = ResourceManager(resource_limits)

    def execute(
        self,
        graph: ExecutionGraph,
        inputs: Dict[str, Any],
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute graph with caching and provenance tracking.

        Args:
            graph: Execution graph to run
            inputs: Initial inputs (e.g., {"input_path": "..."})
            output_dir: Directory for outputs
            config: Optional stage-specific config overrides

        Returns:
            ExecutionResult with outputs and metadata

        Raises:
            ResourceError: If resource limits exceeded
            StageExecutionError: If stage execution fails
        """
        start_time = time.perf_counter()

        # Generate execution plan
        plan = graph.plan(self.resource_limits)

        # Reserve resources upfront (fail-fast)
        with self.resource_manager:
            device = self.resource_manager.select_device()

            # Create execution context
            context = ExecutionContext(
                device=device,
                output_dir=output_dir,
                config=config or {},
                artifact_store=self.artifact_store,
                resource_manager=self.resource_manager,
            )

            # Track execution state
            stage_outputs = {"_inputs": inputs}  # Namespace for initial inputs
            stages_executed = []
            stages_cached = []
            artifacts_written = []
            provenance_records = []

            # Execute stages in topological order
            for node in plan.stages:
                stage_id = node.stage_id
                stage = node.stage

                # Resolve inputs from previous stages
                resolved_inputs = {}
                for input_name, source in node.inputs.items():
                    if "." in source:
                        source_stage, output_name = source.split(".", 1)
                        if source_stage not in stage_outputs:
                            if node.optional:
                                # Skip optional stage if inputs missing
                                continue
                            else:
                                raise StageExecutionError(
                                    stage_id,
                                    f"Missing dependency: {source_stage}"
                                )
                        resolved_inputs[input_name] = stage_outputs[source_stage][output_name]
                    else:
                        # Direct input from initial inputs
                        resolved_inputs[input_name] = inputs.get(source)

                # Compute cache key
                cache_key = stage.compute_cache_key(resolved_inputs, context)

                # Check cache
                if self.artifact_store.exists(cache_key):
                    # Cache hit: load from store
                    outputs = self.artifact_store.load(cache_key)
                    stages_cached.append(stage_id)
                else:
                    # Cache miss: execute stage
                    outputs = stage.execute(resolved_inputs, context)

                    # Build provenance metadata
                    provenance = ProvenanceMetadata(
                        cache_key=cache_key,
                        stage_id=stage_id,
                        stage_version=stage.metadata.version,
                        input_fingerprints=self._compute_input_fingerprints(resolved_inputs),
                        config_snapshot=context.config,
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        hostname=platform.node(),
                        python_version=sys.version,
                        numpy_version=np.__version__,
                        device=context.device,
                    )

                    # Store in cache
                    self.artifact_store.store(cache_key, outputs, provenance)

                    stages_executed.append(stage_id)
                    artifacts_written.append(cache_key)
                    provenance_records.append(provenance)

                # Save outputs for downstream stages
                stage_outputs[stage_id] = outputs

            # Compute execution time
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Get peak memory
            peak_memory_mb = self.resource_manager.get_peak_memory_mb()

            return ExecutionResult(
                outputs=stage_outputs,
                stages_executed=stages_executed,
                stages_cached=stages_cached,
                execution_time_ms=execution_time_ms,
                peak_memory_mb=peak_memory_mb,
                artifacts_written=artifacts_written,
                provenance=provenance_records,
            )

    def _compute_input_fingerprints(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, str]:
        """Compute SHA256 fingerprints for all inputs."""
        import hashlib

        fingerprints = {}
        for name, value in inputs.items():
            if isinstance(value, np.ndarray):
                fingerprints[name] = hashlib.sha256(value.tobytes()).hexdigest()[:16]
            elif isinstance(value, (str, int, float, bool)):
                fingerprints[name] = hashlib.sha256(str(value).encode()).hexdigest()[:16]
            else:
                fingerprints[name] = "unknown"

        return fingerprints
```

---

### 2. Before/After Examples

#### 2.1 Current Approach (Phase 2.4)

**Hardcoded sequence, no caching, no provenance:**

```python
# Current: src/transformation_portal/spatial_ai/orchestration/pipeline.py

pipeline = SpatialAIPipeline.from_preset("spatial_ai_standard")

result = pipeline.process(
    input_path="scene.tiff",
    output_dir="output/",
)

# Problems:
# - Every run re-executes all stages (no cache)
# - No visibility into DAG before execution
# - No artifact lineage after execution
# - Memory OOM failures are late (mid-execution)
```

#### 2.2 Phase 3 Approach (ExecutionGraph)

**Explicit DAG, content-addressed cache, automatic provenance:**

```python
# Phase 3: Execution graph with caching

from transformation_portal.spatial_ai.orchestration.graph import (
    ExecutionGraph,
    Executor,
    ArtifactStore,
)
from transformation_portal.spatial_ai.ingest import IngestStage
from transformation_portal.spatial_ai.segmentation import SAM2Stage
from transformation_portal.spatial_ai.materials import MaterialStage

# 1. Build execution graph
graph = ExecutionGraph()

graph.add_stage(
    "ingest",
    IngestStage(),
    inputs={
        "input_path": "_inputs.input_path",
    },
)

graph.add_stage(
    "segment",
    SAM2Stage(),
    inputs={
        "linear_rgb": "ingest.linear_rgb",
        "gamma": "ingest.gamma",
    },
)

graph.add_stage(
    "materials",
    MaterialStage(),
    inputs={
        "linear_rgb": "ingest.linear_rgb",
        "masks": "segment.masks",
    },
    optional=True,  # Skip if segment failed
)

# 2. Validate resource requirements BEFORE execution
plan = graph.plan(resource_limits=ResourceLimits(max_gpu_memory_gb=8.0))
print(f"Total GPU memory: {plan.total_gpu_memory_mb}MB")
print(f"Estimated time: {plan.estimated_time_ms / 1000:.1f}s")

# 3. Visualize DAG
graph.visualize(Path("output/dag.dot"))

# 4. Execute with caching
store = ArtifactStore(cache_dir=Path(".cache/spatial_ai"))
executor = Executor(artifact_store=store, resource_limits=ResourceLimits())

result = executor.execute(
    graph=graph,
    inputs={"input_path": "scene.tiff"},
    output_dir=Path("output/"),
)

print(f"Executed: {result.stages_executed}")
print(f"Cached: {result.stages_cached}")
print(f"Time: {result.execution_time_ms / 1000:.1f}s")

# 5. Query provenance
for prov in result.provenance:
    print(f"Stage: {prov.stage_id}")
    print(f"  Version: {prov.stage_version}")
    print(f"  Model: {prov.model_repo_id}@{prov.model_revision}")
    print(f"  Device: {prov.device}")
    print(f"  Timestamp: {prov.timestamp}")

# Benefits:
# ✅ Second run is instant (cache hit)
# ✅ Resource failure is immediate (plan validation)
# ✅ DAG visualization shows dependencies
# ✅ Provenance is automatic (no manual logging)
```

#### 2.3 Migration Example

**Existing SpatialAIPipeline continues to work:**

```python
# Option 1: Existing API (unchanged)
pipeline = SpatialAIPipeline.from_preset("spatial_ai_standard")
result = pipeline.process("scene.tiff", "output/")

# Option 2: New graph API (opt-in)
from transformation_portal.spatial_ai.orchestration import graph_from_preset

graph = graph_from_preset("spatial_ai_standard")
executor = Executor(artifact_store=store)
result = executor.execute(
    graph=graph,
    inputs={"input_path": "scene.tiff"},
    output_dir=Path("output/"),
)

# Migration strategy:
# - Phase 2.4 pipeline wraps graph executor internally
# - Existing presets generate compatible graphs
# - No breaking changes to CLI or API
```

---

### 3. Resource Budgeting API

**Fail-fast resource reservation:**

```python
# Reserve resources upfront (fail before execution)

from transformation_portal.spatial_ai.orchestration.graph import ResourceRequirements

# Stage declares requirements
class SAM2Stage:
    @property
    def metadata(self) -> StageMetadata:
        return StageMetadata(
            resource_requirements=ResourceRequirements(
                gpu_memory_mb=2048,  # Hard requirement
                cpu_memory_mb=512,
                min_disk_mb=100,
                gpu_required=False,  # Can fallback to CPU
            ),
        )

# Executor validates BEFORE execution
executor = Executor(
    resource_limits=ResourceLimits(max_gpu_memory_gb=4.0),
)

try:
    plan = graph.plan(resource_limits=ResourceLimits(max_gpu_memory_gb=4.0))
except ResourceError as e:
    # Fails immediately with clear message:
    # "Graph requires 6144MB GPU memory, limit is 4096MB"
    print(f"Cannot execute: {e}")
    sys.exit(1)

# Benefits:
# - No silent OOM mid-execution
# - Clear resource requirements in logs
# - Enables auto-scaling (reserve GPU in CI, CPU locally)
```

---

### 4. Determinism Testing Strategy

#### 4.1 Cache Correctness

**Test: Cache hit produces bitwise identical output**

```python
# tests/spatial_ai/orchestration/graph/test_cache_correctness.py

def test_cache_produces_identical_output(tmp_path):
    """Verify cache hit = bitwise identical output."""

    # Create graph
    graph = ExecutionGraph()
    graph.add_stage("test", TestStage(), inputs={})

    # Create store
    store = ArtifactStore(cache_dir=tmp_path / "cache")
    executor = Executor(artifact_store=store)

    # First execution (cache miss)
    result1 = executor.execute(
        graph=graph,
        inputs={"data": np.random.rand(100, 100)},
        output_dir=tmp_path / "out1",
    )

    # Second execution (cache hit)
    result2 = executor.execute(
        graph=graph,
        inputs={"data": np.random.rand(100, 100)},  # Same data
        output_dir=tmp_path / "out2",
    )

    # Verify cache hit
    assert len(result1.stages_executed) == 1
    assert len(result2.stages_cached) == 1

    # Verify bitwise identical
    np.testing.assert_array_equal(
        result1.outputs["test"]["output"],
        result2.outputs["test"]["output"],
    )
```

#### 4.2 Provenance Validation

**Test: Provenance metadata is complete and queryable**

```python
def test_provenance_metadata_completeness(tmp_path):
    """Verify provenance includes all required fields."""

    store = ArtifactStore(cache_dir=tmp_path / "cache")
    executor = Executor(artifact_store=store)

    # Execute graph
    result = executor.execute(graph, inputs, tmp_path / "out")

    # Verify provenance records
    assert len(result.provenance) > 0

    for prov in result.provenance:
        # Required fields
        assert prov.cache_key
        assert prov.stage_id
        assert prov.stage_version
        assert prov.timestamp
        assert prov.input_fingerprints
        assert prov.config_snapshot

    # Query by stage
    sam2_artifacts = store.query_provenance(stage_id="segment")
    assert len(sam2_artifacts) == 1
    assert sam2_artifacts[0].stage_id == "segment"
```

#### 4.3 DAG Execution Determinism

**Test: Topological sort is stable**

```python
def test_execution_order_deterministic():
    """Verify topological sort is stable across runs."""

    graph = ExecutionGraph()
    graph.add_stage("a", StageA(), inputs={})
    graph.add_stage("b", StageB(), inputs={"x": "a.x"})
    graph.add_stage("c", StageC(), inputs={"y": "a.y"})
    graph.add_stage("d", StageD(), inputs={"b": "b.z", "c": "c.z"})

    # Execute 10 times
    orders = []
    for _ in range(10):
        plan = graph.plan()
        orders.append([node.stage_id for node in plan.stages])

    # Verify all orders are identical
    assert all(order == orders[0] for order in orders)
```

---

### 5. Migration Strategy

#### 5.1 Backward Compatibility

**SpatialAIPipeline wraps ExecutionGraph internally:**

```python
# src/transformation_portal/spatial_ai/orchestration/pipeline.py (updated)

class SpatialAIPipeline:
    """Backward-compatible wrapper around ExecutionGraph."""

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Build execution graph from config
        self._graph = self._build_graph_from_config(config)

        # Create artifact store
        self._store = ArtifactStore(
            cache_dir=Path.home() / ".cache/transformation_portal/spatial_ai",
        )

        # Create executor
        self._executor = Executor(
            artifact_store=self._store,
            resource_limits=config.resource_limits,
        )

    def process(self, input_path, output_dir, save_intermediates=True):
        """Execute pipeline (delegates to graph executor)."""

        # Execute graph
        result = self._executor.execute(
            graph=self._graph,
            inputs={"input_path": input_path},
            output_dir=output_dir,
        )

        # Convert to PipelineResult (backward compatibility)
        return self._graph_result_to_pipeline_result(result)

    def _build_graph_from_config(self, config: PipelineConfig) -> ExecutionGraph:
        """Build graph from preset config."""
        graph = ExecutionGraph()

        if "ingest" in config.stages:
            graph.add_stage("ingest", IngestStage(), inputs={
                "input_path": "_inputs.input_path",
            })

        if "segment" in config.stages:
            graph.add_stage("segment", SAM2Stage(), inputs={
                "linear_rgb": "ingest.linear_rgb",
            })

        # ... etc

        return graph
```

#### 5.2 Preset Migration

**Existing presets work without changes:**

```yaml
# config/presets/spatial_ai/spatial_ai_standard.yaml (unchanged)

tier: standard
pipeline:
  ingest:
    strict_ingest: false
  segmentation:
    backend: sam2
    model:
      revision: "e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251"
  materials:
    backend: heuristic

# Phase 3 runtime converts this to ExecutionGraph internally
# No preset changes required
```

#### 5.3 CLI Migration

**CLI works unchanged:**

```bash
# Existing command (works as before)
transformation_portal spatial-ai process \
  --preset spatial_ai_standard \
  --input scene.tiff \
  --output output/

# New command (opt-in to graph features)
transformation_portal spatial-ai graph \
  --preset spatial_ai_standard \
  --input scene.tiff \
  --output output/ \
  --visualize dag.dot \
  --cache-dir .cache/
```

#### 5.4 Golden Baseline Preservation

**Phase 2 golden outputs remain valid:**

```python
# Test: Phase 3 produces identical outputs to Phase 2

def test_phase3_matches_phase2_golden_baseline():
    """Verify Phase 3 executor produces Phase 2 golden outputs."""

    # Phase 2 golden artifacts (frozen)
    golden_dir = Path("tests/spatial_ai/golden/phase2")

    # Phase 3 execution
    graph = graph_from_preset("spatial_ai_standard")
    executor = Executor(artifact_store=store)
    result = executor.execute(
        graph=graph,
        inputs={"input_path": golden_dir / "input.tiff"},
        output_dir=tmp_path,
    )

    # Compare outputs (bitwise identical)
    phase2_masks = np.load(golden_dir / "segmentation_masks.npz")
    phase3_masks = result.outputs["segment"]["masks"]

    np.testing.assert_array_equal(phase2_masks, phase3_masks)
```

---

## Consequences

### Positive

✅ **Execution is introspectable**
- Can visualize DAG before execution (`.dot` export)
- Can trace artifact lineage after execution (provenance queries)
- Can predict resource requirements (plan validation)

✅ **Caching works without compromising determinism**
- Cache hit = bitwise identical output (content-addressed keys)
- Cache miss = deterministic recomputation (pure functions)
- Cache invalidation automatic (version + inputs change → new key)

✅ **Resource limits are enforceable**
- OOM failures are predictable (fail-fast on plan validation)
- Memory scaling is documented (resource requirements in metadata)
- Can reserve budgets upfront (no mid-execution surprises)

✅ **Provenance is automatic**
- Every artifact has metadata (input hash, model rev, config, timestamp)
- Can reproduce any artifact from metadata alone
- Queryable lineage (SQL-like interface in L3)

✅ **Backward compatibility preserved**
- Existing `SpatialAIPipeline` API unchanged
- Existing presets work without modification
- Golden baselines remain valid (bitwise identical outputs)

✅ **Testing rigor**
- Cache correctness testable (determinism + bitwise equality)
- Provenance completeness testable (metadata validation)
- DAG execution testable (topological sort stability)

### Negative

⚠️ **Implementation complexity**
- New abstractions (Stage, Graph, Executor, ArtifactStore)
- ~2000 lines of new code
- More surface area to maintain

**Mitigation:** Phased rollout (L1 → L2 → L3 → L4), comprehensive tests, clear documentation.

⚠️ **Cache storage overhead**
- Artifacts consume disk space (10GB default limit)
- Provenance metadata overhead (~1-2KB per artifact)
- LRU eviction required

**Mitigation:** Configurable limits, LRU eviction, manual cache clear command.

⚠️ **Learning curve**
- New APIs for graph construction
- Stage protocol requirements (compute_cache_key, metadata)
- Executor semantics (context, provenance)

**Mitigation:** Clear examples, migration guide, backward-compatible wrapper.

### Neutral

🔄 **Cache invalidation strategy**
- Version bumps invalidate cache (semantic versioning)
- Input changes invalidate cache (content-addressed keys)
- Config changes invalidate cache (config in key)
- Manual invalidation available (clear command)

🔄 **Parallel execution (L2)**
- DAG enables parallelism where dependencies allow
- Requires thread-safe stages (current stages are)
- Memory budgeting more complex (concurrent peak)

---

## Alternatives Considered

### Alternative 1: Adopt Airflow/Prefect/Luigi/Dagster

**Generic workflow orchestrators.**

**Rejected:**
- **Heavyweight dependencies:** 50-100+ dependencies, complex setup
- **Cloud-native bias:** Designed for distributed clusters, not local execution
- **Generality tax:** Features we don't need (retries, sensors, schedules)
- **Integration friction:** Our stages are spatial AI, not ETL tasks
- **Debugging complexity:** Remote execution, opaque logs, containerization

**Architect Assessment:** This is a **domain-specific execution layer** for spatial AI pipelines, not a generic workflow system. Keep it focused. Don't chase generality.

### Alternative 2: Keep Hardcoded Sequence + Add Manual Caching

**Add caching to existing `SpatialAIPipeline` without graph abstraction.**

**Rejected:**
- **No introspection:** Can't visualize dependencies or plan resources
- **No provenance:** Manual logging is error-prone
- **No parallelism:** Sequential execution forever
- **Technical debt:** Band-aid on architectural problem
- **Scaling failure:** Doesn't address state complexity at scale

**Architect Assessment:** Kicks the can down the road. Phase 3 problems (state explosion, resource pressure, debugging) remain unsolved.

### Alternative 3: External Task Queue (Celery)

**Distributed task execution with Celery/Redis.**

**Rejected:**
- **Premature distribution:** Local execution doesn't work yet
- **Operational overhead:** Redis broker, worker processes, monitoring
- **Determinism risk:** Network failures, worker restarts, message loss
- **Anti-pattern:** Don't build distributed systems before local works (Phase 3 Foundation)

**Deferred:** Consider for L4 (multi-node execution) after L1-L3 prove local execution works.

### Alternative 4: Lazy Evaluation (Dask-like)

**Lazy computation graphs with automatic parallelism.**

**Rejected:**
- **Complexity mismatch:** Spatial AI stages are expensive (seconds), not cheap (milliseconds)
- **Debugging nightmare:** Lazy evaluation hides execution order
- **Caching friction:** Dask caching doesn't fit content-addressed model
- **Memory pressure:** Automatic parallelism can exceed budgets

**Architect Assessment:** Eager execution with explicit caching is simpler and more predictable for our workload.

---

## Implementation Plan

### Phase 3 Deliverables (Aligned with PHASE3_FOUNDATION.md)

#### L1: Foundation (4-6 weeks)

**Milestone:** Core abstractions working for 2-3 stages.

| Deliverable | Description | Success Criteria |
|-------------|-------------|------------------|
| **Stage Protocol** | `Stage`, `StageMetadata`, `ResourceRequirements` | SAM2Stage, IngestStage, MaterialStage implement protocol |
| **ExecutionGraph** | DAG construction, topological sort, cycle detection | Can build 3-stage graph (ingest → segment → materials) |
| **ArtifactStore** | Filesystem-based, content-addressed, atomic writes | Cache hit = bitwise identical, provenance metadata attached |
| **Executor** | Sequential execution, cache integration | Executes graph, respects cache, tracks provenance |
| **Tests** | Unit + integration tests | ≥85% coverage, determinism verified |

**Deliverables:**
- [ ] `src/transformation_portal/spatial_ai/orchestration/graph/stage.py`
- [ ] `src/transformation_portal/spatial_ai/orchestration/graph/execution_graph.py`
- [ ] `src/transformation_portal/spatial_ai/orchestration/graph/artifact_store.py`
- [ ] `src/transformation_portal/spatial_ai/orchestration/graph/executor.py`
- [ ] `tests/spatial_ai/orchestration/graph/` (comprehensive test suite)
- [ ] Update `SpatialAIPipeline` to wrap graph executor (backward compatibility)

#### L2: Optimization (3-4 weeks)

**Milestone:** Performance wins without sacrificing determinism.

| Deliverable | Description | Success Criteria |
|-------------|-------------|------------------|
| **Parallel Execution** | Execute independent stages concurrently | Materials generation parallelizes across segments |
| **Resource Reservation** | Upfront budget validation | Fail-fast on insufficient GPU memory |
| **LRU Eviction** | Size-based cache limits | Cache stays under 10GB, LRU eviction works |
| **Model Lifecycle** | Lazy loading, unloading between stages | Peak memory = max single stage, not sum |

**Deliverables:**
- [ ] Parallel executor (thread pool, dependency resolution)
- [ ] Resource reservation API (reserve upfront, fail-fast)
- [ ] LRU cache eviction (size-based limits)
- [ ] Model lifecycle integration (lazy load, unload)

#### L3: Observability (2-3 weeks)

**Milestone:** Execution is fully introspectable.

| Deliverable | Description | Success Criteria |
|-------------|-------------|------------------|
| **DAG Visualization** | Render graph as DOT/GraphViz | Can visualize before execution |
| **Provenance Queries** | SQL-like lineage interface | Can query artifacts by stage, model, timestamp |
| **Cost Prediction** | Time + memory estimates | Predict runtime within 20% |
| **Execution Dashboard** | Real-time progress UI (optional) | Live progress for batch workloads |

**Deliverables:**
- [ ] `ExecutionGraph.visualize()` (DOT export)
- [ ] `ArtifactStore.query_provenance()` (SQLite index)
- [ ] Cost prediction (estimate runtime from metadata)
- [ ] CLI commands (`graph visualize`, `cache query`)

#### L4: Scale (4-6 weeks)

**Milestone:** Multi-GPU, multi-node execution.

| Deliverable | Description | Success Criteria |
|-------------|-------------|------------------|
| **Distributed Execution** | Multi-GPU, multi-node support | 10× throughput on 10-GPU cluster |
| **Incremental Recomputation** | Only re-run invalidated stages | Cache-aware re-execution |
| **Remote ArtifactStore** | S3-compatible backends | Cloud artifact storage |
| **Batch Optimization** | Process 1000 images efficiently | Amortized overhead, smart batching |

**Deliverables:**
- [ ] Distributed executor (Ray, Dask, or custom)
- [ ] Incremental recomputation (invalidation tracking)
- [ ] S3 artifact store backend
- [ ] Batch pipeline optimization

---

## Success Metrics

### Phase 3 is successful when:

#### Technical Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Cache hit rate** | ≥80% on repeat runs | Log cache hits/misses |
| **Cache correctness** | 100% bitwise identical | Determinism tests |
| **Resource prediction** | ±20% of actual | Plan estimates vs actual |
| **Provenance coverage** | 100% of artifacts | Every artifact has metadata |
| **Test coverage** | ≥85% | `pytest --cov` |
| **Backward compatibility** | 100% | All Phase 2 tests pass |

#### Performance Metrics

| Workload | Phase 2 (no cache) | Phase 3 (cached) | Improvement |
|----------|-------------------|------------------|-------------|
| Single image, cold | 10s | 10s (first run) | —|
| Single image, warm | 10s | <1s (cache hit) | **10×** |
| 100 images, batch | 1000s | <100s (partial cache) | **10×** |
| Re-run after config change | 1000s | ~300s (partial invalidation) | **3×** |

#### Governance Metrics

| Policy | Compliance Check | Enforcement |
|--------|------------------|-------------|
| **Determinism** | Cache = bitwise identical | CI test gate |
| **Provenance** | All artifacts have metadata | ArtifactStore enforces |
| **Resource limits** | Fail-fast on OOM | Executor validates plan |
| **Tier separation** | No ML imports in core | ADR-023 isolation checker |

---

## References

- **Phase 3 Foundation:** Session file `PHASE3_FOUNDATION.md`
- **Current Pipeline:** `src/transformation_portal/spatial_ai/orchestration/pipeline.py`
- **ADR-027:** Phase 2 Spatial AI Extension Architecture
- **ADR-023:** Pipeline isolation requirements
- **ADR-021:** HuggingFace revision pinning policy
- **Resource Manager:** `src/transformation_portal/spatial_ai/orchestration/resource_manager.py`
- **Benchmark Tests:** `tests/benchmarks/README.md` (determinism measurement practices)

---

## Approval

**Status:** Implemented
**Review Date:** 2026-02-12
**Approver:** Transformation Portal Architect (required)
**Implementation Start:** Upon approval
**Review Interval:** 6 months (2026-08-12)

---

## Open Questions

1. **SQLite vs File-based Provenance Index?**
   - L1-L2: File-based (simple, no dependencies)
   - L3: SQLite index (efficient queries)
   - Decision: Defer to L3 implementation

2. **Parallel Execution Semantics?**
   - Thread pool vs process pool vs async?
   - Decision: Start with thread pool (simpler), evaluate process pool if GIL becomes bottleneck

3. **Cache Warmup Strategy?**
   - Pre-populate cache for common inputs?
   - Decision: Manual warmup command, not automatic

4. **Distributed Backend Choice?**
   - Ray vs Dask vs custom?
   - Decision: Defer to L4, evaluate based on L1-L3 learnings

---

**Amendments:**
- 2026-08-30 — If and when Accepted, ADR-051 partially supersedes only this record's long-term
  executor designation.
  The shipped Spatial AI graph API, presets, adapters, cache behavior, golden outputs, and public
  single-view and multi-view contracts remain supported migration constraints. This includes typed
  camera/request validation, research-tier/license gates, result shapes, PLY/sidecar export, and
  provenance. Spatial stages and cache metadata become adapters to the designated plan, executor,
  identity, and CAS authorities; this amendment does not withdraw those public contracts. The
  approval footer was also corrected from stale `Proposed` to `Implemented` to match the completed
  implementation recorded at the top of this ADR.

---

*This ADR is binding. Deviations require explicit superseding ADR with migration plan.*
