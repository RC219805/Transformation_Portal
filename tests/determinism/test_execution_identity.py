"""Tests for Phase 2 Deterministic Execution Layer.

Test Coverage:
- ExecutionIdentity computation and serialization
- CAS ID determinism
- Platform compatibility checks
- Code hash computation
- Config hash computation
- Execution gate (should_execute)
- Determinism verification
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transformation_portal.core.execution_identity import (
    ArtifactMetadata,
    CAS_IDENTITY_VERSION,
    ExecutionIdentity,
    compute_cas_id,
    compute_code_hash,
    compute_config_hash,
    create_artifact_metadata,
    is_compatible,
    should_execute,
    verify_determinism,
)
from transformation_portal.core.platform_matrix import PlatformMatrix


class TestExecutionIdentity:
    """Tests for ExecutionIdentity dataclass."""

    def test_basic_creation(self):
        """Test basic ExecutionIdentity creation."""
        identity = ExecutionIdentity(
            stage_name="depth_estimation",
            stage_version="1.0.0",
            input_ids=("sha256:abc123", "sha256:def456"),
            code_hash="sha256:code123",
            config_hash="sha256:config456",
            env_fingerprint="sha256:env789",
            platform_id="darwin-arm64-mps",
            cas_id="sha256:final123",
        )

        assert identity.stage_name == "depth_estimation"
        assert identity.stage_version == "1.0.0"
        assert len(identity.input_ids) == 2
        assert identity.schema_version == CAS_IDENTITY_VERSION

    def test_immutability(self):
        """Test ExecutionIdentity is frozen (immutable)."""
        identity = ExecutionIdentity(
            stage_name="test",
            stage_version="1.0.0",
            input_ids=(),
            code_hash="sha256:x",
            config_hash="sha256:y",
            env_fingerprint="sha256:z",
            platform_id="linux-x86_64-cpu",
            cas_id="sha256:final",
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            identity.stage_name = "modified"

    def test_to_dict_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        identity = ExecutionIdentity(
            stage_name="segmentation",
            stage_version="2.0.0",
            input_ids=("sha256:a", "sha256:b"),
            code_hash="sha256:code",
            config_hash="sha256:config",
            env_fingerprint="sha256:env",
            platform_id="darwin-arm64-cpu",
            cas_id="sha256:final",
        )

        # Roundtrip
        data = identity.to_dict()
        restored = ExecutionIdentity.from_dict(data)

        assert restored.stage_name == identity.stage_name
        assert restored.stage_version == identity.stage_version
        assert restored.input_ids == identity.input_ids
        assert restored.cas_id == identity.cas_id


class TestArtifactMetadata:
    """Tests for ArtifactMetadata dataclass."""

    def test_basic_creation(self):
        """Test basic ArtifactMetadata creation."""
        metadata = ArtifactMetadata(
            artifact_id="sha256:artifact123",
            stage="depth_estimation",
            inputs=("sha256:input1",),
            code_hash="sha256:code",
            config_hash="sha256:config",
            env_fingerprint="sha256:env",
            platform_id="darwin-arm64-mps",
            created_at="2026-03-18T12:00:00Z",
        )

        assert metadata.artifact_id == "sha256:artifact123"
        assert metadata.stage == "depth_estimation"
        assert metadata.version == CAS_IDENTITY_VERSION

    def test_to_dict_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        metadata = ArtifactMetadata(
            artifact_id="sha256:x",
            stage="test",
            inputs=("sha256:a", "sha256:b"),
            code_hash="sha256:code",
            config_hash="sha256:cfg",
            env_fingerprint="sha256:env",
            platform_id="linux-x86_64-cuda",
            created_at="2026-03-18T12:00:00Z",
            execution_identity="sha256:exec123",
        )

        data = metadata.to_dict()
        restored = ArtifactMetadata.from_dict(data)

        assert restored.artifact_id == metadata.artifact_id
        assert restored.inputs == metadata.inputs
        assert restored.execution_identity == metadata.execution_identity


class TestComputeCodeHash:
    """Tests for compute_code_hash function."""

    def test_code_hash_deterministic(self):
        """Test code hash is deterministic across calls."""
        hash1 = compute_code_hash(use_git=False, paths=[])
        hash2 = compute_code_hash(use_git=False, paths=[])

        # With no paths, should return consistent placeholder
        assert hash1 == hash2

    def test_code_hash_with_files(self, tmp_path):
        """Test code hash includes file contents."""
        # Create test files
        (tmp_path / "test.py").write_text("def foo(): pass")
        (tmp_path / "test2.py").write_text("def bar(): pass")

        hash1 = compute_code_hash(paths=[str(tmp_path)], use_git=False)
        assert hash1.startswith("sha256:")

        # Modify file
        (tmp_path / "test.py").write_text("def foo(): return 42")
        hash2 = compute_code_hash(paths=[str(tmp_path)], use_git=False)

        # Hash should change
        assert hash1 != hash2


class TestComputeConfigHash:
    """Tests for compute_config_hash function."""

    def test_config_hash_deterministic(self):
        """Test config hash is deterministic."""
        config = {"model": "DA3-Large", "quality": "high", "device": "cpu"}

        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)

        assert hash1 == hash2
        assert hash1.startswith("sha256:")

    def test_config_hash_key_order_independent(self):
        """Test config hash is independent of dict key order."""
        config1 = {"a": 1, "b": 2, "c": 3}
        config2 = {"c": 3, "a": 1, "b": 2}

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)

        assert hash1 == hash2

    def test_config_hash_value_sensitive(self):
        """Test config hash changes with values."""
        config1 = {"model": "DA3-Large"}
        config2 = {"model": "DA3-Small"}

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)

        assert hash1 != hash2

    def test_config_hash_none_config(self):
        """Test config hash handles None."""
        hash1 = compute_config_hash(None)
        hash2 = compute_config_hash({})

        assert hash1 == hash2

    def test_config_hash_nested_dict(self):
        """Test config hash handles nested dicts."""
        config = {
            "depth": {"model": "DA3", "device": "mps"},
            "segmentation": {"threshold": 0.5},
        }

        hash1 = compute_config_hash(config)
        assert hash1.startswith("sha256:")


class TestComputeCasId:
    """Tests for compute_cas_id function."""

    def test_cas_id_deterministic(self):
        """Test CAS ID is deterministic."""
        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                id1 = compute_cas_id(
                    stage_name="test",
                    input_ids=["sha256:abc"],
                    config={"x": 1},
                )
                id2 = compute_cas_id(
                    stage_name="test",
                    input_ids=["sha256:abc"],
                    config={"x": 1},
                )

                assert id1.cas_id == id2.cas_id

    def test_cas_id_changes_with_inputs(self):
        """Test CAS ID changes when inputs change."""
        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                id1 = compute_cas_id(
                    stage_name="test",
                    input_ids=["sha256:abc"],
                    config={},
                )
                id2 = compute_cas_id(
                    stage_name="test",
                    input_ids=["sha256:xyz"],
                    config={},
                )

                assert id1.cas_id != id2.cas_id

    def test_cas_id_changes_with_config(self):
        """Test CAS ID changes when config changes."""
        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                id1 = compute_cas_id(
                    stage_name="test",
                    input_ids=[],
                    config={"quality": "low"},
                )
                id2 = compute_cas_id(
                    stage_name="test",
                    input_ids=[],
                    config={"quality": "high"},
                )

                assert id1.cas_id != id2.cas_id

    def test_cas_id_input_order_normalized(self):
        """Test CAS ID normalizes input order."""
        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:fixed_code"
                mock_env.return_value = "sha256:fixed_env"

                id1 = compute_cas_id(
                    stage_name="test",
                    input_ids=["sha256:a", "sha256:b", "sha256:c"],
                    config={},
                )
                id2 = compute_cas_id(
                    stage_name="test",
                    input_ids=["sha256:c", "sha256:a", "sha256:b"],
                    config={},
                )

                # Same inputs in different order should produce same CAS ID
                assert id1.cas_id == id2.cas_id

    def test_cas_id_contains_all_components(self):
        """Test ExecutionIdentity contains all required components."""
        with patch(
            "transformation_portal.core.execution_identity.compute_code_hash"
        ) as mock_code:
            with patch(
                "transformation_portal.core.execution_identity.get_env_fingerprint"
            ) as mock_env:
                mock_code.return_value = "sha256:test_code"
                mock_env.return_value = "sha256:test_env"

                identity = compute_cas_id(
                    stage_name="my_stage",
                    input_ids=["sha256:input1"],
                    config={"key": "value"},
                    stage_version="2.0.0",
                )

                assert identity.stage_name == "my_stage"
                assert identity.stage_version == "2.0.0"
                assert identity.input_ids == ("sha256:input1",)
                assert identity.code_hash == "sha256:test_code"
                assert identity.env_fingerprint == "sha256:test_env"
                assert identity.cas_id.startswith("sha256:")


class TestShouldExecute:
    """Tests for should_execute function."""

    def test_should_execute_cache_miss(self):
        """Test should_execute returns True on cache miss."""
        mock_store = MagicMock()
        mock_store.has_object.return_value = False

        identity = ExecutionIdentity(
            stage_name="test",
            stage_version="1.0.0",
            input_ids=(),
            code_hash="sha256:x",
            config_hash="sha256:y",
            env_fingerprint="sha256:z",
            platform_id="linux-x86_64-cpu",
            cas_id="sha256:notfound",
        )

        assert should_execute(identity, mock_store) is True
        mock_store.has_object.assert_called_once_with("sha256:notfound")

    def test_should_execute_cache_hit(self):
        """Test should_execute returns False on cache hit."""
        mock_store = MagicMock()
        mock_store.has_object.return_value = True

        identity = ExecutionIdentity(
            stage_name="test",
            stage_version="1.0.0",
            input_ids=(),
            code_hash="sha256:x",
            config_hash="sha256:y",
            env_fingerprint="sha256:z",
            platform_id="linux-x86_64-cpu",
            cas_id="sha256:found",
        )

        assert should_execute(identity, mock_store) is False


class TestIsCompatible:
    """Tests for is_compatible function."""

    def test_compatible_same_platform_same_env(self):
        """Test compatibility with same platform and env."""
        # Get current platform for a realistic test
        current_platform = PlatformMatrix.detect(accel="cpu")

        metadata = ArtifactMetadata(
            artifact_id="sha256:x",
            stage="test",
            inputs=(),
            code_hash="sha256:c",
            config_hash="sha256:cfg",
            env_fingerprint="sha256:matching_env",
            platform_id=current_platform.canonical_target,  # Same as current
            created_at="2026-03-18T12:00:00Z",
        )

        result = is_compatible(
            metadata,
            current_platform=current_platform,
            current_env_fingerprint="sha256:matching_env",
        )

        assert result is True

    def test_incompatible_different_platform(self):
        """Test incompatibility with different platform."""
        metadata = ArtifactMetadata(
            artifact_id="sha256:x",
            stage="test",
            inputs=(),
            code_hash="sha256:c",
            config_hash="sha256:cfg",
            env_fingerprint="sha256:env",
            platform_id="linux-x86_64-cuda",
            created_at="2026-03-18T12:00:00Z",
        )

        platform = PlatformMatrix.detect(accel="cpu")

        result = is_compatible(
            metadata,
            current_platform=platform,
            current_env_fingerprint="sha256:env",
        )

        assert result is False

    def test_incompatible_different_env(self):
        """Test incompatibility with different environment."""
        metadata = ArtifactMetadata(
            artifact_id="sha256:x",
            stage="test",
            inputs=(),
            code_hash="sha256:c",
            config_hash="sha256:cfg",
            env_fingerprint="sha256:old_env",
            platform_id="darwin-arm64-cpu",
            created_at="2026-03-18T12:00:00Z",
        )

        platform = PlatformMatrix.detect(accel="cpu")

        result = is_compatible(
            metadata,
            current_platform=platform,
            current_env_fingerprint="sha256:new_env",
        )

        assert result is False

    def test_cpu_fallback_allowed(self):
        """Test CPU fallback mode allows cross-platform reuse."""
        metadata = ArtifactMetadata(
            artifact_id="sha256:x",
            stage="test",
            inputs=(),
            code_hash="sha256:c",
            config_hash="sha256:cfg",
            env_fingerprint="sha256:env",
            platform_id="linux-x86_64-cpu",
            created_at="2026-03-18T12:00:00Z",
        )

        platform = PlatformMatrix.detect(accel="cpu")

        result = is_compatible(
            metadata,
            current_platform=platform,
            current_env_fingerprint="sha256:env",
            allow_cpu_fallback=True,
        )

        # Both are CPU platforms, should allow
        assert result is True

    def test_cpu_fallback_not_allowed_for_gpu(self):
        """Test CPU fallback does not allow GPU artifacts."""
        metadata = ArtifactMetadata(
            artifact_id="sha256:x",
            stage="test",
            inputs=(),
            code_hash="sha256:c",
            config_hash="sha256:cfg",
            env_fingerprint="sha256:env",
            platform_id="linux-x86_64-cuda",  # GPU platform
            created_at="2026-03-18T12:00:00Z",
        )

        platform = PlatformMatrix.detect(accel="cpu")

        result = is_compatible(
            metadata,
            current_platform=platform,
            current_env_fingerprint="sha256:env",
            allow_cpu_fallback=True,
        )

        # CUDA is not CPU, should not allow
        assert result is False


class TestCreateArtifactMetadata:
    """Tests for create_artifact_metadata function."""

    def test_creates_complete_metadata(self):
        """Test creates complete artifact metadata."""
        identity = ExecutionIdentity(
            stage_name="depth_estimation",
            stage_version="1.0.0",
            input_ids=("sha256:input1", "sha256:input2"),
            code_hash="sha256:code",
            config_hash="sha256:config",
            env_fingerprint="sha256:env",
            platform_id="darwin-arm64-mps",
            cas_id="sha256:exec123",
        )

        metadata = create_artifact_metadata(
            artifact_id="sha256:artifact456",
            execution_identity=identity,
        )

        assert metadata.artifact_id == "sha256:artifact456"
        assert metadata.stage == "depth_estimation"
        assert metadata.inputs == ("sha256:input1", "sha256:input2")
        assert metadata.code_hash == "sha256:code"
        assert metadata.config_hash == "sha256:config"
        assert metadata.env_fingerprint == "sha256:env"
        assert metadata.platform_id == "darwin-arm64-mps"
        assert metadata.execution_identity == "sha256:exec123"
        assert metadata.created_at  # Should have timestamp


class TestVerifyDeterminism:
    """Tests for verify_determinism function."""

    def test_deterministic_function(self):
        """Test verification passes for deterministic function."""

        def deterministic_fn(inputs: dict, config: Any) -> dict:
            return {"result": inputs["value"] * 2}

        is_det, hashes = verify_determinism(
            stage_fn=deterministic_fn,
            inputs={"value": 42},
            config={},
            runs=3,
        )

        assert is_det is True
        assert len(hashes) == 3
        assert len(set(hashes)) == 1  # All identical

    def test_non_deterministic_function(self):
        """Test verification fails for non-deterministic function."""
        import random

        def non_deterministic_fn(inputs: dict, config: Any) -> dict:
            return {"result": random.random()}

        is_det, hashes = verify_determinism(
            stage_fn=non_deterministic_fn,
            inputs={},
            config={},
            runs=3,
        )

        assert is_det is False
        assert len(hashes) == 3
        # Hashes should differ (with high probability)

    def test_numpy_array_determinism(self):
        """Test verification handles numpy arrays."""

        def numpy_fn(inputs: dict, config: Any) -> np.ndarray:
            return inputs["arr"] * 2

        arr = np.array([1, 2, 3], dtype=np.float32)

        is_det, hashes = verify_determinism(
            stage_fn=numpy_fn,
            inputs={"arr": arr},
            config={},
            runs=2,
        )

        assert is_det is True
