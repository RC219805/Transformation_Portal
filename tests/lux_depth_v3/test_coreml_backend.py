"""Unit tests for ``transformation_portal.lux_depth_v3.coreml_backend``.

The CoreML backend is the runtime path for published Apple-Silicon depth
artifacts and is governed (Apple-Silicon-only). Its happy paths require
``coremltools`` plus an ``.mlpackage`` on disk and cannot run in the
core CI environment, but the surrounding contract — supported model
allowlist, platform/import gating, cache stats, cache clear — is pure
Python and must be exercised offline. These tests cover that surface;
end-to-end inference belongs to the Apple-Silicon ML lane.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit]

from transformation_portal.lux_depth_v3 import coreml_backend  # noqa: E402


class TestSupportedModelGuard:
    """The CoreML backend only accepts a curated allowlist of published artifacts."""

    def test_supported_set_is_frozen_to_published_artifacts(self):
        # The allowlist is part of the CoreML release contract: adding a model
        # here must be accompanied by a published .mlpackage and a runtime
        # validation entry. Lock the membership so changes show up in review.
        assert coreml_backend._SUPPORTED_PUBLISHED_COREML_MODEL_IDS == {
            "apple/coreml-depth-anything-v2-small",
        }

    def test_unsupported_model_id_is_rejected_when_coremltools_missing(self, monkeypatch):
        # Even with coremltools unavailable, the allowlist gate must fire
        # before any conversion attempt.
        monkeypatch.setattr(coreml_backend, "COREML_AVAILABLE", True)

        with pytest.raises(ValueError, match="published CoreML artifacts"):
            coreml_backend.CoreMLDepthEstimator("some-org/unsupported-model")

    def test_init_fails_closed_when_coremltools_unavailable(self, monkeypatch):
        monkeypatch.setattr(coreml_backend, "COREML_AVAILABLE", False)

        with pytest.raises(RuntimeError, match="coremltools not available"):
            coreml_backend.CoreMLDepthEstimator(
                "apple/coreml-depth-anything-v2-small",
            )


class TestShouldUseCoreML:
    """``should_use_coreml`` is the gate every caller goes through."""

    def _make_config(self, **kwargs):
        return SimpleNamespace(**kwargs)

    def test_disabled_when_user_did_not_opt_in(self, monkeypatch):
        # Even if everything else is true, the user must explicitly
        # set use_coreml=True. ``force=True`` is the documented test escape.
        monkeypatch.setattr(coreml_backend, "COREML_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend, "TORCH_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend, "TRANSFORMERS_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(coreml_backend.platform, "machine", lambda: "arm64")

        config = self._make_config(use_coreml=False)
        assert coreml_backend.should_use_coreml(config) is False

    def test_disabled_off_apple_silicon(self, monkeypatch):
        monkeypatch.setattr(coreml_backend, "COREML_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend, "TORCH_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend, "TRANSFORMERS_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend.platform, "system", lambda: "Linux")
        monkeypatch.setattr(coreml_backend.platform, "machine", lambda: "x86_64")

        config = self._make_config(use_coreml=True)
        assert coreml_backend.should_use_coreml(config) is False

    def test_disabled_on_macos_intel(self, monkeypatch):
        monkeypatch.setattr(coreml_backend, "COREML_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend, "TORCH_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend, "TRANSFORMERS_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(coreml_backend.platform, "machine", lambda: "x86_64")

        config = self._make_config(use_coreml=True)
        assert coreml_backend.should_use_coreml(config) is False

    @pytest.mark.parametrize(
        "missing_flag",
        ["COREML_AVAILABLE", "TORCH_AVAILABLE", "TRANSFORMERS_AVAILABLE"],
    )
    def test_disabled_when_required_dep_missing(self, monkeypatch, missing_flag):
        # Every required ML flag has to be true; if any is false, gate closes.
        monkeypatch.setattr(coreml_backend, "COREML_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend, "TORCH_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend, "TRANSFORMERS_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend, missing_flag, False)
        monkeypatch.setattr(coreml_backend.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(coreml_backend.platform, "machine", lambda: "arm64")

        config = self._make_config(use_coreml=True)
        assert coreml_backend.should_use_coreml(config) is False

    def test_enabled_on_apple_silicon_when_all_deps_present(self, monkeypatch):
        monkeypatch.setattr(coreml_backend, "COREML_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend, "TORCH_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend, "TRANSFORMERS_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(coreml_backend.platform, "machine", lambda: "arm64")

        config = self._make_config(use_coreml=True)
        assert coreml_backend.should_use_coreml(config) is True

    def test_force_overrides_user_optin_but_not_platform_or_deps(self, monkeypatch):
        # ``force=True`` should NOT bypass platform/dependency requirements:
        # those checks are invariants of the runtime, not user policy.
        monkeypatch.setattr(coreml_backend, "COREML_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend, "TORCH_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend, "TRANSFORMERS_AVAILABLE", True)
        monkeypatch.setattr(coreml_backend.platform, "system", lambda: "Linux")
        monkeypatch.setattr(coreml_backend.platform, "machine", lambda: "x86_64")

        config = self._make_config(use_coreml=False)
        assert coreml_backend.should_use_coreml(config, force=True) is False


class TestCoreMLCacheStats:
    def test_stats_when_cache_dir_does_not_exist(self, tmp_path: Path):
        result = coreml_backend.get_coreml_cache_stats(tmp_path / "missing")
        assert result["exists"] is False
        assert result["model_count"] == 0
        assert result["total_size_mb"] == 0

    def test_stats_count_only_mlpackage_dirs(self, tmp_path: Path):
        # An empty cache dir should report zero models.
        result = coreml_backend.get_coreml_cache_stats(tmp_path)
        assert result["exists"] is True
        assert result["model_count"] == 0
        assert "models" in result

    def test_stats_aggregates_size_across_mlpackage_files(self, tmp_path: Path):
        package = tmp_path / "DepthAnything.mlpackage"
        package.mkdir()
        (package / "weights.bin").write_bytes(b"x" * 1024)
        (package / "manifest.json").write_text("{}", encoding="utf-8")

        result = coreml_backend.get_coreml_cache_stats(tmp_path)
        assert result["model_count"] == 1
        assert result["models"] == ["DepthAnything.mlpackage"]
        # Size is a positive number of MB even for tiny test fixtures.
        assert result["total_size_mb"] > 0


class TestCoreMLCacheClear:
    def test_clear_when_cache_does_not_exist_returns_zero(self, tmp_path: Path):
        assert coreml_backend.clear_coreml_cache(tmp_path / "missing") == 0

    def test_clear_removes_mlpackage_directories_only(self, tmp_path: Path):
        # Two .mlpackage dirs and one unrelated file; only the packages
        # should be removed.
        for name in ("a.mlpackage", "b.mlpackage"):
            d = tmp_path / name
            d.mkdir()
            (d / "weights.bin").write_bytes(b"weights")
        unrelated = tmp_path / "other.txt"
        unrelated.write_text("keep me", encoding="utf-8")

        removed = coreml_backend.clear_coreml_cache(tmp_path)

        assert removed == 2
        remaining = sorted(p.name for p in tmp_path.iterdir())
        assert remaining == ["other.txt"]
        # Re-clearing an already-empty cache must be idempotent.
        assert coreml_backend.clear_coreml_cache(tmp_path) == 0


class TestCoreMLDepthEstimatorCachePath:
    def test_cache_path_sanitizes_model_id_and_includes_revision(self, monkeypatch, tmp_path: Path):
        # Skip live CoreML loading by stopping init before _load_or_convert.
        monkeypatch.setattr(coreml_backend, "COREML_AVAILABLE", True)

        def _no_load(self, force_reconvert: bool = False):
            return None

        monkeypatch.setattr(
            coreml_backend.CoreMLDepthEstimator,
            "_load_or_convert",
            _no_load,
        )

        estimator = coreml_backend.CoreMLDepthEstimator(
            "apple/coreml-depth-anything-v2-small",
            cache_dir=tmp_path,
            revision="abc-123",
        )
        cache_path = estimator._get_cache_path()
        assert cache_path.parent == tmp_path
        # Model id slashes/hyphens get normalized to underscores so the path
        # is filesystem-safe regardless of the source repo namespace.
        assert "/" not in cache_path.name
        assert cache_path.name.endswith(".mlpackage")
        assert "abc_123" in cache_path.name

    def test_cache_path_omits_revision_token_when_unpinned(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(coreml_backend, "COREML_AVAILABLE", True)
        monkeypatch.setattr(
            coreml_backend.CoreMLDepthEstimator,
            "_load_or_convert",
            lambda self, force_reconvert=False: None,
        )

        estimator = coreml_backend.CoreMLDepthEstimator(
            "apple/coreml-depth-anything-v2-small",
            cache_dir=tmp_path,
        )
        cache_path = estimator._get_cache_path()
        assert cache_path.parent == tmp_path
        assert cache_path.name == "apple_coreml_depth_anything_v2_small.mlpackage"
