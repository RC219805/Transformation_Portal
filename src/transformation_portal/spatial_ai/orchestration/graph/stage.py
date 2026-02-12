"""Stage protocol for execution graph (Phase 3 L1).

Defines the core abstraction for execution stages in the Spatial AI orchestration system.

Design Principles (ADR-029):
- Pure function semantics (no side effects, no global state)
- Deterministic by default (same inputs → same outputs)
- Explicit resource declarations (GPU memory, CPU memory, time estimates)
- Content-addressable caching (compute_cache_key from inputs + config)
- Introspectable metadata (name, version, resource requirements)

Stage Contract:
1. Metadata: Declare name, version, resource requirements
2. Execute: Transform inputs → outputs (pure function)
3. Cache Key: Compute content-addressed key from inputs + context

Anti-patterns:
- Implicit dependencies (stages must declare inputs explicitly)
- Global state (stages are stateless value transformations)
- Non-determinism (same inputs must produce same outputs)
- Resource dishonesty (declare actual requirements, not aspirational)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Protocol, runtime_checkable


class CheckpointPolicy(str, Enum):
    """Checkpoint policy for stage outputs."""

    NEVER = "never"  # Never checkpoint (cheap to recompute)
    ALWAYS = "always"  # Always checkpoint (expensive to recompute)
    ON_FAILURE = "on_failure"  # Checkpoint only on downstream failure
    AUTO = "auto"  # Executor decides based on resource cost


@dataclass(frozen=True)
class ResourceRequirements:
    """Resource requirements for stage execution.

    Attributes:
        gpu_memory_mb: Peak GPU memory in megabytes (0 if CPU-only).
        cpu_memory_mb: Peak CPU memory in megabytes.
        min_disk_mb: Minimum disk space required for outputs (MB).
        gpu_required: If True, stage fails on CPU-only systems.
        estimated_time_ms: Estimated execution time in milliseconds.
        can_parallelize: If True, stage can execute in parallel with others.

    Design notes:
    - Use peak memory, not average (fail-fast on resource limits).
    - Estimate conservatively (better to over-estimate than OOM).
    - Time estimates inform progress tracking, not scheduling (L1).
    """

    gpu_memory_mb: int = 0
    cpu_memory_mb: int = 512
    min_disk_mb: int = 100
    gpu_required: bool = False
    estimated_time_ms: int = 1000
    can_parallelize: bool = False

    def __post_init__(self):
        """Validate resource requirements."""
        if self.gpu_memory_mb < 0:
            raise ValueError(f"gpu_memory_mb must be >= 0, got {self.gpu_memory_mb}")
        if self.cpu_memory_mb <= 0:
            raise ValueError(f"cpu_memory_mb must be > 0, got {self.cpu_memory_mb}")
        if self.min_disk_mb < 0:
            raise ValueError(f"min_disk_mb must be >= 0, got {self.min_disk_mb}")
        if self.estimated_time_ms <= 0:
            raise ValueError(f"estimated_time_ms must be > 0, got {self.estimated_time_ms}")


@dataclass(frozen=True)
class StageMetadata:
    """Metadata for execution stage.

    Attributes:
        name: Human-readable stage name (e.g., "sam2_segmentation").
        version: Stage version (bump on algorithm change, affects cache key).
        description: Human-readable description.
        resource_requirements: Resource declarations for planning.
        deterministic: If True, same inputs → same outputs (always).
        idempotent: If True, executing twice = executing once.
        checkpoint_policy: When to checkpoint stage outputs.

    Design notes:
    - Version is semantic: bump on any algorithm change that affects outputs.
    - Deterministic = True is the default; False requires justification.
    - Idempotent stages can be safely retried without side effects.
    """

    name: str
    version: str
    description: str
    resource_requirements: ResourceRequirements
    deterministic: bool = True
    idempotent: bool = True
    checkpoint_policy: CheckpointPolicy = CheckpointPolicy.AUTO

    def __post_init__(self):
        """Validate metadata."""
        if not self.name:
            raise ValueError("Stage name cannot be empty")
        if not self.version:
            raise ValueError("Stage version cannot be empty")
        if not self.description:
            raise ValueError("Stage description cannot be empty")


@runtime_checkable
class Stage(Protocol):
    """Protocol for execution stages in the spatial AI orchestration system.

    A Stage is a pure function transformation:
        inputs: Dict[str, Any] → outputs: Dict[str, Any]

    Key contracts:
    1. Determinism: Same inputs + context → same outputs (if metadata.deterministic)
    2. Isolation: No global state, no side effects (except I/O declared in metadata)
    3. Introspection: Metadata describes resource requirements and behavior
    4. Cacheability: compute_cache_key enables content-addressed caching

    Example:
        >>> class SAM2Stage:
        ...     @property
        ...     def metadata(self) -> StageMetadata:
        ...         return StageMetadata(
        ...             name="sam2_segmentation",
        ...             version="2.1.0",
        ...             description="SAM2 automatic mask generation",
        ...             resource_requirements=ResourceRequirements(
        ...                 gpu_memory_mb=2048,
        ...                 cpu_memory_mb=512,
        ...                 estimated_time_ms=3000,
        ...             ),
        ...         )
        ...
        ...     def execute(self, inputs, context):
        ...         linear_rgb = inputs["linear_rgb"]
        ...         # ... segmentation logic ...
        ...         return {"masks": masks, "scores": scores}
        ...
        ...     def compute_cache_key(self, inputs, context):
        ...         import hashlib
        ...         input_hash = hashlib.sha256(inputs["linear_rgb"].tobytes()).hexdigest()
        ...         return f"{self.metadata.version}:{input_hash[:16]}"
    """

    @property
    def metadata(self) -> StageMetadata:
        """Stage metadata (name, version, resource requirements).

        Returns:
            StageMetadata describing this stage's behavior and requirements.
        """
        ...

    def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Execute stage transformation.

        Args:
            inputs: Stage inputs (key-value pairs).
            context: Execution context (device, config, output_dir).

        Returns:
            Stage outputs (key-value pairs).

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If execution fails.
        """
        ...

    def compute_cache_key(self, inputs: Dict[str, Any], context: ExecutionContext) -> str:
        """Compute content-addressed cache key.

        The cache key uniquely identifies this stage's outputs given:
        - Stage version (metadata.version)
        - Input fingerprints (hash of input data)
        - Execution context (device, config overrides)

        Design:
        - Same inputs + context → same cache key (deterministic)
        - Different inputs/context → different cache key (collision-resistant)
        - Format: "{stage_version}:{input_hash}:{config_hash}"

        Args:
            inputs: Stage inputs.
            context: Execution context.

        Returns:
            SHA256-based cache key (hex string, typically 64 chars or truncated).

        Example:
            >>> key = stage.compute_cache_key(inputs, context)
            >>> print(key)
            "2.1.0:a3f5e8b2c1d4:9f7e6d5c4b3a"
        """
        ...


# Forward declaration for type hints (actual implementation in executor.py)
class ExecutionContext(Protocol):
    """Execution context passed to stages.

    Attributes:
        device: Execution device ("cuda", "cpu", "mps").
        config: Stage-specific configuration overrides.
        output_dir: Output directory for artifacts.
        enable_caching: Whether caching is enabled.
    """

    device: str
    config: Dict[str, Any]
    output_dir: Any  # Path
    enable_caching: bool
