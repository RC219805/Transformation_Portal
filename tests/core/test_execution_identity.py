"""Tests for core.execution_identity — CAS identity and config hashing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from transformation_portal.core.execution_identity import (
    ArtifactMetadata,
    ExecutionIdentity,
    compute_cas_id,
    compute_config_hash,
    create_artifact_metadata,
    explain_cache_miss,
    is_ci_environment,
)

pytestmark = pytest.mark.unit


class TestIsCIEnvironment:
    def test_detects_github_actions(self, monkeypatch):
        """CI=true is detected as CI."""
        monkeypatch.setenv("CI", "true")
        assert is_ci_environment() is True

    def test_detects_github_actions_specific(self, monkeypatch):
        """GITHUB_ACTIONS env var is detected."""
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        assert is_ci_environment() is True

    def test_detects_gitlab_ci(self, monkeypatch):
        """GITLAB_CI env var is detected."""
        monkeypatch.setenv("GITLAB_CI", "true")
        assert is_ci_environment() is True

    def test_local_env_returns_false(self, monkeypatch):
        """No CI env vars → returns False."""
        for var in ["CI", "GITHUB_ACTIONS", "JENKINS_URL", "GITLAB_CI", "CIRCLECI", "TRAVIS", "BUILDKITE"]:
            monkeypatch.delenv(var, raising=False)
        assert is_ci_environment() is False


class TestComputeConfigHash:
    def test_same_config_same_hash(self):
        """Identical dicts produce identical hashes."""
        h1 = compute_config_hash({"a": 1, "b": 2})
        h2 = compute_config_hash({"a": 1, "b": 2})
        assert h1 == h2

    def test_different_config_different_hash(self):
        """Changed value produces different hash."""
        h1 = compute_config_hash({"a": 1})
        h2 = compute_config_hash({"a": 2})
        assert h1 != h2

    def test_hash_is_deterministic(self):
        """Called twice returns the same hash."""
        cfg = {"model": "DA3-Large", "quantization": "none"}
        assert compute_config_hash(cfg) == compute_config_hash(cfg)

    def test_nested_config_hashed(self):
        """Nested dicts are included in hash."""
        h1 = compute_config_hash({"outer": {"inner": 1}})
        h2 = compute_config_hash({"outer": {"inner": 2}})
        assert h1 != h2

    def test_hash_starts_with_sha256_prefix(self):
        """Hash string uses 'sha256:' prefix."""
        h = compute_config_hash({"x": 1})
        assert h.startswith("sha256:")

    def test_none_config_does_not_raise(self):
        """None config is treated as empty dict without raising."""
        h = compute_config_hash(None)
        assert h.startswith("sha256:")


class TestComputeCasId:
    def test_returns_execution_identity(self):
        """compute_cas_id returns an ExecutionIdentity instance."""
        identity = compute_cas_id(
            stage_name="test_stage",
            input_ids=["sha256:abc"],
            config={"model": "test"},
            code_hash="sha256:code",
            env_fingerprint="sha256:env",
            lockfile_hash="sha256:lock",
        )
        assert isinstance(identity, ExecutionIdentity)

    def test_same_inputs_same_cas_id(self):
        """Identical inputs produce same CAS ID."""
        kwargs = dict(
            stage_name="depth",
            input_ids=["sha256:abc"],
            config={"model": "DA3"},
            code_hash="sha256:code",
            env_fingerprint="sha256:env",
            lockfile_hash="sha256:lock",
        )
        i1 = compute_cas_id(**kwargs)
        i2 = compute_cas_id(**kwargs)
        assert i1.cas_id == i2.cas_id

    def test_different_config_different_cas_id(self):
        """Different config produces different CAS ID."""
        base = dict(
            stage_name="depth",
            input_ids=["sha256:abc"],
            code_hash="sha256:code",
            env_fingerprint="sha256:env",
            lockfile_hash="sha256:lock",
        )
        i1 = compute_cas_id(**base, config={"model": "A"})
        i2 = compute_cas_id(**base, config={"model": "B"})
        assert i1.cas_id != i2.cas_id

    def test_stage_name_stored(self):
        """ExecutionIdentity.stage_name matches input."""
        identity = compute_cas_id(
            stage_name="my_stage",
            input_ids=[],
            config={},
            code_hash="sha256:c",
            env_fingerprint="sha256:e",
            lockfile_hash="sha256:lock",
        )
        assert identity.stage_name == "my_stage"


class TestExplainCacheMiss:
    def _make_identity(self, lockfile_hash: str = "sha256:cfg") -> ExecutionIdentity:
        return compute_cas_id(
            stage_name="s",
            input_ids=[],
            config={},
            code_hash="sha256:code",
            env_fingerprint="sha256:env",
            lockfile_hash=lockfile_hash,
        )

    def test_returns_dict_with_reason(self):
        """explain_cache_miss returns dict with 'reason' key."""
        i = self._make_identity()
        result = explain_cache_miss(i, i)
        assert "reason" in result

    def test_no_cached_identity_returns_no_cache(self):
        """cached_identity=None → reason includes 'No cached'."""
        i = self._make_identity()
        result = explain_cache_miss(i, None)
        assert "No cached" in result["reason"]

    def test_identifies_config_change(self):
        """Different lockfile_hash → difference listed."""
        i1 = self._make_identity(lockfile_hash="sha256:aaa")
        i2 = self._make_identity(lockfile_hash="sha256:bbb")
        result = explain_cache_miss(i1, i2)
        assert "lockfile_hash" in result["differences"]

    def test_matching_identities_empty_differences(self):
        """Same identity → differences list is empty."""
        i = self._make_identity()
        result = explain_cache_miss(i, i)
        assert not result["differences"]


class TestCreateArtifactMetadata:
    def _make_identity(self) -> ExecutionIdentity:
        return compute_cas_id(
            stage_name="depth",
            input_ids=["sha256:inp"],
            config={"model": "DA3"},
            code_hash="sha256:code",
            env_fingerprint="sha256:env",
            lockfile_hash="sha256:lock",
        )

    def test_returns_artifact_metadata(self):
        """create_artifact_metadata returns ArtifactMetadata."""
        identity = self._make_identity()
        meta = create_artifact_metadata("sha256:artifact", identity)
        assert isinstance(meta, ArtifactMetadata)

    def test_artifact_id_stored(self):
        """artifact_id matches the provided value."""
        identity = self._make_identity()
        meta = create_artifact_metadata("sha256:xyz", identity)
        assert meta.artifact_id == "sha256:xyz"

    def test_stage_name_propagated(self):
        """stage field matches identity.stage_name."""
        identity = self._make_identity()
        meta = create_artifact_metadata("sha256:art", identity)
        assert meta.stage == "depth"

    def test_to_dict_contains_expected_keys(self):
        """to_dict() includes artifact_id, stage, code_hash, config_hash."""
        identity = self._make_identity()
        meta = create_artifact_metadata("sha256:art", identity)
        d = meta.to_dict()
        for key in ("artifact_id", "stage", "code_hash", "config_hash"):
            assert key in d
