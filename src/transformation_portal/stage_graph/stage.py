"""
Core stage abstraction for pipeline processing.

A Stage represents a discrete, cacheable unit of work with:
- Deterministic input → output transformation
- Content-addressed caching
- Dependency tracking
- Observable execution
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import json
import time
import logging

import numpy as np

logger = logging.getLogger(__name__)


class StageStatus(str, Enum):
    """Stage execution status."""
    PENDING = "pending"
    RUNNING = "running"
    CACHED = "cached"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageContext:
    """
    Execution context passed to stages.

    Contains input artifacts, configuration, and runtime environment.
    """
    # Input artifacts from previous stages
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # Configuration parameters
    config: Dict[str, Any] = field(default_factory=dict)

    # Runtime environment
    device: str = "cpu"
    cache_enabled: bool = True
    cache_dir: Optional[Path] = None

    # Execution metadata
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_artifact(self, name: str, default: Any = None) -> Any:
        """Get artifact from context."""
        return self.artifacts.get(name, default)

    def set_artifact(self, name: str, value: Any):
        """Set artifact in context."""
        self.artifacts[name] = value

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)


@dataclass
class StageResult:
    """
    Result of stage execution.

    Contains output artifacts, metrics, and execution metadata.
    """
    # Stage identification
    stage_name: str
    stage_version: str

    # Execution status
    status: StageStatus

    # Output artifacts
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # Execution metrics
    duration_ms: float = 0.0
    cache_hit: bool = False
    cache_key: Optional[str] = None

    # Error information
    error: Optional[str] = None
    error_traceback: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status in (StageStatus.COMPLETED, StageStatus.CACHED)

    def get_artifact(self, name: str, default: Any = None) -> Any:
        """Get output artifact."""
        return self.artifacts.get(name, default)


class Stage(ABC):
    """
    Abstract base class for pipeline stages.

    Subclasses must implement:
    - compute(): Core processing logic
    - get_cache_key(): Cache key generation
    - get_dependencies(): Declare stage dependencies
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize stage.

        Args:
            name: Unique stage name
            version: Semantic version for cache invalidation
        """
        self.name = name
        self.version = version
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def compute(self, context: StageContext) -> StageResult:
        """
        Execute stage computation.

        Args:
            context: Execution context with input artifacts

        Returns:
            Stage result with output artifacts
        """
        pass

    @abstractmethod
    def get_cache_key(self, context: StageContext) -> str:
        """
        Generate cache key for this execution.

        Must be deterministic based on inputs and configuration.

        Args:
            context: Execution context

        Returns:
            Cache key string
        """
        pass

    def get_dependencies(self) -> List[str]:
        """
        Get list of stage names this stage depends on.

        Returns:
            List of stage names
        """
        return []

    def execute(self, context: StageContext) -> StageResult:
        """
        Execute stage with caching and error handling.

        Args:
            context: Execution context

        Returns:
            Stage result
        """
        start_time = time.time()

        try:
            # Generate cache key
            cache_key = self.get_cache_key(context)

            # Check cache if enabled
            if context.cache_enabled and context.cache_dir:
                cached_result = self._load_from_cache(cache_key, context.cache_dir)
                if cached_result:
                    self.logger.info(f"Cache hit for {self.name}: {cache_key[:16]}...")
                    cached_result.duration_ms = (time.time() - start_time) * 1000
                    return cached_result

            # Execute computation
            self.logger.info(f"Executing {self.name}...")
            result = self.compute(context)

            # Update result metadata
            result.stage_name = self.name
            result.stage_version = self.version
            result.duration_ms = (time.time() - start_time) * 1000
            result.cache_key = cache_key
            result.cache_hit = False

            if result.status == StageStatus.PENDING:
                result.status = StageStatus.COMPLETED

            # Save to cache if successful
            if context.cache_enabled and context.cache_dir and result.is_success():
                self._save_to_cache(result, cache_key, context.cache_dir)

            return result

        except Exception as e:
            import traceback
            duration_ms = (time.time() - start_time) * 1000

            self.logger.error(f"Stage {self.name} failed: {e}")

            return StageResult(
                stage_name=self.name,
                stage_version=self.version,
                status=StageStatus.FAILED,
                duration_ms=duration_ms,
                error=str(e),
                error_traceback=traceback.format_exc(),
            )

    def _load_from_cache(self, cache_key: str, cache_dir: Path) -> Optional[StageResult]:
        """Load result from cache."""
        cache_path = cache_dir / f"{cache_key}.json"
        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)

            # Reconstruct artifacts, loading numpy arrays
            artifacts = {}
            for key, value in data.get("artifacts", {}).items():
                if isinstance(value, dict):
                    if "__numpy__" in value:
                        # Load numpy array
                        npy_path = cache_dir / value["__numpy__"]
                        artifacts[key] = np.load(npy_path)
                    else:
                        # Handle nested dicts
                        nested_dict = {}
                        for k, v in value.items():
                            if isinstance(v, dict) and "__numpy__" in v:
                                npy_path = cache_dir / v["__numpy__"]
                                nested_dict[k] = np.load(npy_path)
                            else:
                                nested_dict[k] = v
                        artifacts[key] = nested_dict
                else:
                    artifacts[key] = value

            # Reconstruct result
            result = StageResult(
                stage_name=data["stage_name"],
                stage_version=data["stage_version"],
                status=StageStatus.CACHED,
                artifacts=artifacts,
                cache_hit=True,
                cache_key=cache_key,
                metadata=data.get("metadata", {}),
            )

            return result

        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(self, result: StageResult, cache_key: str, cache_dir: Path):
        """Save result to cache."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}.json"

        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_artifacts = {}
            for key, value in result.artifacts.items():
                if isinstance(value, np.ndarray):
                    # Save large arrays as separate .npy files
                    npy_path = cache_dir / f"{cache_key}_{key}.npy"
                    np.save(npy_path, value)
                    serializable_artifacts[key] = {"__numpy__": str(npy_path.name)}
                elif isinstance(value, dict):
                    # Handle nested dicts with numpy arrays
                    serializable_dict = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            npy_path = cache_dir / f"{cache_key}_{key}_{k}.npy"
                            np.save(npy_path, v)
                            serializable_dict[k] = {"__numpy__": str(npy_path.name)}
                        else:
                            serializable_dict[k] = v
                    serializable_artifacts[key] = serializable_dict
                else:
                    serializable_artifacts[key] = value

            data = {
                "stage_name": result.stage_name,
                "stage_version": result.stage_version,
                "status": result.status.value,
                "artifacts": serializable_artifacts,
                "duration_ms": result.duration_ms,
                "metadata": result.metadata,
                "timestamp": result.timestamp,
            }

            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute stable hash of configuration data."""
        # Sort keys for deterministic serialization
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
