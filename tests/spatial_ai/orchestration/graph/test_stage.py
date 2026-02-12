"""Tests for Stage protocol and metadata (Phase 3 L1).

Test Coverage:
- StageMetadata validation
- ResourceRequirements validation
- Stage protocol compliance
- Cache key computation
- Determinism verification
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

import numpy as np
import pytest

from transformation_portal.spatial_ai.orchestration.graph.stage import (
    CheckpointPolicy,
    ResourceRequirements,
    Stage,
    StageMetadata,
)


class TestResourceRequirements:
    """Tests for ResourceRequirements dataclass."""

    def test_defaults(self):
        """Test default resource requirements."""
        reqs = ResourceRequirements()
        assert reqs.gpu_memory_mb == 0
        assert reqs.cpu_memory_mb == 512
        assert reqs.min_disk_mb == 100
        assert reqs.gpu_required is False
        assert reqs.estimated_time_ms == 1000
        assert reqs.can_parallelize is False

    def test_custom_values(self):
        """Test custom resource requirements."""
        reqs = ResourceRequirements(
            gpu_memory_mb=2048,
            cpu_memory_mb=1024,
            min_disk_mb=500,
            gpu_required=True,
            estimated_time_ms=5000,
            can_parallelize=True,
        )
        assert reqs.gpu_memory_mb == 2048
        assert reqs.cpu_memory_mb == 1024
        assert reqs.min_disk_mb == 500
        assert reqs.gpu_required is True
        assert reqs.estimated_time_ms == 5000
        assert reqs.can_parallelize is True

    def test_validation_negative_gpu_memory(self):
        """Test validation rejects negative GPU memory."""
        with pytest.raises(ValueError, match="gpu_memory_mb must be >= 0"):
            ResourceRequirements(gpu_memory_mb=-1)

    def test_validation_zero_cpu_memory(self):
        """Test validation rejects zero CPU memory."""
        with pytest.raises(ValueError, match="cpu_memory_mb must be > 0"):
            ResourceRequirements(cpu_memory_mb=0)

    def test_validation_negative_disk(self):
        """Test validation rejects negative disk space."""
        with pytest.raises(ValueError, match="min_disk_mb must be >= 0"):
            ResourceRequirements(min_disk_mb=-100)

    def test_validation_zero_time(self):
        """Test validation rejects zero estimated time."""
        with pytest.raises(ValueError, match="estimated_time_ms must be > 0"):
            ResourceRequirements(estimated_time_ms=0)

    def test_immutability(self):
        """Test ResourceRequirements is frozen (immutable)."""
        reqs = ResourceRequirements(gpu_memory_mb=2048)
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            reqs.gpu_memory_mb = 4096


class TestStageMetadata:
    """Tests for StageMetadata dataclass."""

    def test_valid_metadata(self):
        """Test valid stage metadata."""
        metadata = StageMetadata(
            name="test_stage",
            version="1.0.0",
            description="Test stage for unit tests",
            resource_requirements=ResourceRequirements(),
        )
        assert metadata.name == "test_stage"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test stage for unit tests"
        assert metadata.deterministic is True
        assert metadata.idempotent is True
        assert metadata.checkpoint_policy == CheckpointPolicy.AUTO

    def test_validation_empty_name(self):
        """Test validation rejects empty stage name."""
        with pytest.raises(ValueError, match="Stage name cannot be empty"):
            StageMetadata(
                name="",
                version="1.0.0",
                description="Test",
                resource_requirements=ResourceRequirements(),
            )

    def test_validation_empty_version(self):
        """Test validation rejects empty version."""
        with pytest.raises(ValueError, match="Stage version cannot be empty"):
            StageMetadata(
                name="test",
                version="",
                description="Test",
                resource_requirements=ResourceRequirements(),
            )

    def test_validation_empty_description(self):
        """Test validation rejects empty description."""
        with pytest.raises(ValueError, match="Stage description cannot be empty"):
            StageMetadata(
                name="test",
                version="1.0.0",
                description="",
                resource_requirements=ResourceRequirements(),
            )

    def test_custom_checkpoint_policy(self):
        """Test custom checkpoint policy."""
        metadata = StageMetadata(
            name="expensive_stage",
            version="1.0.0",
            description="Expensive stage that should always checkpoint",
            resource_requirements=ResourceRequirements(),
            checkpoint_policy=CheckpointPolicy.ALWAYS,
        )
        assert metadata.checkpoint_policy == CheckpointPolicy.ALWAYS

    def test_immutability(self):
        """Test StageMetadata is frozen (immutable)."""
        metadata = StageMetadata(
            name="test",
            version="1.0.0",
            description="Test",
            resource_requirements=ResourceRequirements(),
        )
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            metadata.name = "modified"


class TestStageProtocol:
    """Tests for Stage protocol compliance."""

    def test_simple_stage_implementation(self):
        """Test simple stage implementation."""

        class SimpleStage:
            """Simple stage for testing."""

            @property
            def metadata(self) -> StageMetadata:
                return StageMetadata(
                    name="simple_stage",
                    version="1.0.0",
                    description="Simple test stage",
                    resource_requirements=ResourceRequirements(cpu_memory_mb=256, estimated_time_ms=100),
                )

            def execute(self, inputs: Dict[str, Any], context: Any) -> Dict[str, Any]:
                """Execute simple transformation."""
                value = inputs.get("value", 0)
                return {"result": value * 2}

            def compute_cache_key(self, inputs: Dict[str, Any], context: Any) -> str:
                """Compute cache key."""
                value = inputs.get("value", 0)
                input_hash = hashlib.sha256(str(value).encode()).hexdigest()[:16]
                return f"{self.metadata.version}:{input_hash}"

        # Verify protocol compliance
        stage = SimpleStage()
        assert isinstance(stage, Stage)

        # Test execution
        result = stage.execute({"value": 42}, None)
        assert result["result"] == 84

        # Test cache key
        key = stage.compute_cache_key({"value": 42}, None)
        assert key.startswith("1.0.0:")
        assert len(key) > 6  # version + ":" + hash

    def test_deterministic_cache_keys(self):
        """Test cache keys are deterministic."""

        class DeterministicStage:
            """Stage with deterministic cache keys."""

            @property
            def metadata(self) -> StageMetadata:
                return StageMetadata(
                    name="deterministic_stage",
                    version="1.0.0",
                    description="Deterministic test stage",
                    resource_requirements=ResourceRequirements(),
                )

            def execute(self, inputs: Dict[str, Any], context: Any) -> Dict[str, Any]:
                return {"output": inputs["input"] * 2}

            def compute_cache_key(self, inputs: Dict[str, Any], context: Any) -> str:
                # Deterministic hash from numpy array
                data = inputs["input"]
                if isinstance(data, np.ndarray):
                    data_hash = hashlib.sha256(data.tobytes()).hexdigest()[:16]
                else:
                    data_hash = hashlib.sha256(str(data).encode()).hexdigest()[:16]
                return f"{self.metadata.version}:{data_hash}"

        stage = DeterministicStage()

        # Same input → same cache key
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        key1 = stage.compute_cache_key({"input": arr}, None)
        key2 = stage.compute_cache_key({"input": arr}, None)
        assert key1 == key2

        # Different input → different cache key
        arr2 = np.array([1, 2, 3, 4, 6], dtype=np.float32)
        key3 = stage.compute_cache_key({"input": arr2}, None)
        assert key3 != key1

    def test_stage_with_numpy_arrays(self):
        """Test stage that processes numpy arrays."""

        class NumpyStage:
            """Stage that processes numpy arrays."""

            @property
            def metadata(self) -> StageMetadata:
                return StageMetadata(
                    name="numpy_stage",
                    version="1.0.0",
                    description="NumPy array processing stage",
                    resource_requirements=ResourceRequirements(cpu_memory_mb=512),
                )

            def execute(self, inputs: Dict[str, Any], context: Any) -> Dict[str, Any]:
                image = inputs["image"]
                # Simple transformation: normalize
                normalized = (image - image.mean()) / (image.std() + 1e-8)
                return {"normalized": normalized}

            def compute_cache_key(self, inputs: Dict[str, Any], context: Any) -> str:
                image = inputs["image"]
                img_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]
                return f"{self.metadata.version}:{img_hash}"

        stage = NumpyStage()

        # Test with random image
        image = np.random.rand(256, 256, 3).astype(np.float32)
        result = stage.execute({"image": image}, None)

        assert "normalized" in result
        assert result["normalized"].shape == image.shape
        assert np.abs(result["normalized"].mean()) < 1e-5  # Near zero mean
        assert np.abs(result["normalized"].std() - 1.0) < 0.1  # Near unit std


class TestCheckpointPolicy:
    """Tests for CheckpointPolicy enum."""

    def test_checkpoint_policy_values(self):
        """Test checkpoint policy enum values."""
        assert CheckpointPolicy.NEVER == "never"
        assert CheckpointPolicy.ALWAYS == "always"
        assert CheckpointPolicy.ON_FAILURE == "on_failure"
        assert CheckpointPolicy.AUTO == "auto"

    def test_checkpoint_policy_in_metadata(self):
        """Test checkpoint policy usage in metadata."""
        for policy in [
            CheckpointPolicy.NEVER,
            CheckpointPolicy.ALWAYS,
            CheckpointPolicy.ON_FAILURE,
            CheckpointPolicy.AUTO,
        ]:
            metadata = StageMetadata(
                name="test",
                version="1.0.0",
                description="Test",
                resource_requirements=ResourceRequirements(),
                checkpoint_policy=policy,
            )
            assert metadata.checkpoint_policy == policy
