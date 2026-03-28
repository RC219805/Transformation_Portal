"""Tests for APEX + LLaVA integration (ADR-026 §5).

These tests validate the integration between ApexEvaluationHarness
and LLaVA quality validation backend.

Implementation Status:
    The integration IS implemented in:
    src/transformation_portal/evals/apex_llava_integration.py

Coverage:
- ApexLlavaConfig dataclass
- Factory functions for creating harnesses
- Manifest loading and model tier resolution
- Quality dimension prompt selection
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [
    pytest.mark.unit,
]


class TestApexLlavaConfig:
    """Test ApexLlavaConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from transformation_portal.evals import ApexLlavaConfig

        config = ApexLlavaConfig()

        assert config.model_tier == "quality_validation_primary"
        assert config.quality_dimension == "architectural"
        assert config.threshold == 0.70  # Default for architectural
        assert config.metric_weight == 0.5  # Default for architectural
        assert config.include_standard_metrics is True
        assert config.fail_on_vlm_error is False
        assert config.device_map == "auto"

    def test_threshold_defaults_by_dimension(self):
        """Test that threshold defaults vary by quality dimension."""
        from transformation_portal.evals import ApexLlavaConfig

        configs = {
            "segmentation": ApexLlavaConfig(quality_dimension="segmentation"),
            "architectural": ApexLlavaConfig(quality_dimension="architectural"),
            "depth": ApexLlavaConfig(quality_dimension="depth"),
            "material": ApexLlavaConfig(quality_dimension="material"),
        }

        assert configs["segmentation"].threshold == 0.75
        assert configs["architectural"].threshold == 0.70
        assert configs["depth"].threshold == 0.75
        assert configs["material"].threshold == 0.70

    def test_metric_weight_defaults_by_dimension(self):
        """Test that metric weights vary by quality dimension."""
        from transformation_portal.evals import ApexLlavaConfig

        configs = {
            "segmentation": ApexLlavaConfig(quality_dimension="segmentation"),
            "architectural": ApexLlavaConfig(quality_dimension="architectural"),
            "depth": ApexLlavaConfig(quality_dimension="depth"),
            "material": ApexLlavaConfig(quality_dimension="material"),
        }

        assert configs["segmentation"].metric_weight == 0.4
        assert configs["architectural"].metric_weight == 0.5
        assert configs["depth"].metric_weight == 0.5
        assert configs["material"].metric_weight == 0.4

    def test_explicit_threshold_overrides_default(self):
        """Test that explicit threshold overrides dimension default."""
        from transformation_portal.evals import ApexLlavaConfig

        config = ApexLlavaConfig(
            quality_dimension="segmentation",
            threshold=0.85,
        )

        assert config.threshold == 0.85  # Explicit, not default 0.75

    def test_explicit_metric_weight_overrides_default(self):
        """Test that explicit metric_weight overrides dimension default."""
        from transformation_portal.evals import ApexLlavaConfig

        config = ApexLlavaConfig(
            quality_dimension="segmentation",
            metric_weight=0.7,
        )

        assert config.metric_weight == 0.7  # Explicit, not default 0.4

    def test_model_tiers(self):
        """Test all supported model tiers."""
        from transformation_portal.evals import ApexLlavaConfig

        tiers = ["ci_smoke", "quality_validation_primary", "quality_max", "legacy_fallback"]

        for tier in tiers:
            config = ApexLlavaConfig(model_tier=tier)
            assert config.model_tier == tier

    @pytest.mark.parametrize("field_name", ["threshold", "metric_weight"])
    @pytest.mark.parametrize("bad_value", [-0.1, 1.1])
    def test_probability_fields_must_be_in_range(self, field_name, bad_value):
        """Test that probability-like config values are validated."""
        from transformation_portal.evals import ApexLlavaConfig, ApexLlavaIntegrationError

        kwargs = {field_name: bad_value}
        with pytest.raises(ApexLlavaIntegrationError, match=field_name):
            ApexLlavaConfig(**kwargs)


class TestCreateApexHarnessWithoutLlava:
    """Test create_apex_harness_without_llava factory."""

    def test_creates_harness_with_metrics(self):
        """Test creating metrics-only harness."""
        from transformation_portal.evals import create_apex_harness_without_llava

        harness = create_apex_harness_without_llava()

        assert harness.llava_backend is None
        assert len(harness.metric_fns) == 3  # sharpness, contrast, brightness
        assert harness.threshold == 0.70
        assert harness.metric_weight == 1.0  # All weight to metrics

    def test_custom_threshold(self):
        """Test creating harness with custom threshold."""
        from transformation_portal.evals import create_apex_harness_without_llava

        harness = create_apex_harness_without_llava(threshold=0.85)

        assert harness.threshold == 0.85

    def test_without_standard_metrics(self):
        """Test creating harness without standard metrics."""
        from transformation_portal.evals import create_apex_harness_without_llava

        harness = create_apex_harness_without_llava(include_standard_metrics=False)

        assert len(harness.metric_fns) == 0

    def test_with_additional_metrics(self):
        """Test creating harness with additional custom metrics."""
        from transformation_portal.evals import create_apex_harness_without_llava

        def custom_metric(paths):
            return 0.5

        harness = create_apex_harness_without_llava(
            include_standard_metrics=False,
            additional_metrics=[custom_metric],
        )

        assert len(harness.metric_fns) == 1
        assert harness.metric_fns[0] == custom_metric


class TestBuildMaterialQualityPrompt:
    """Test build_material_quality_prompt function."""

    def test_prompt_spec_structure(self):
        """Test that prompt spec has correct structure."""
        from transformation_portal.evals import build_material_quality_prompt

        prompt = build_material_quality_prompt()

        assert prompt.name == "material_pbr_quality"
        assert "PBR" in prompt.system_text
        assert "albedo" in prompt.user_text
        assert "normal" in prompt.user_text
        assert "roughness" in prompt.user_text
        assert "metallic" in prompt.user_text

    def test_prompt_with_context(self):
        """Test that context is included in prompt."""
        from transformation_portal.evals import build_material_quality_prompt
        from transformation_portal.ingest.canonical_json import dumps_json

        context = {"material_type": "marble", "expected_roughness": 0.3}
        prompt = build_material_quality_prompt(context=context)

        assert "Additional context" in prompt.user_text
        assert dumps_json(context, sort_keys=True, indent=2) in prompt.user_text

    def test_prompt_json_schema(self):
        """Test that prompt requests JSON schema output."""
        from transformation_portal.evals import build_material_quality_prompt

        prompt = build_material_quality_prompt()

        assert "passes_basic_quality" in prompt.user_text
        assert "summary_score" in prompt.user_text
        assert "issues" in prompt.user_text
        assert "issue_type" in prompt.user_text
        assert "severity" in prompt.user_text
        assert 'one of "low", "medium", "high"' in prompt.user_text


class TestManifestLoading:
    """Test manifest loading functionality."""

    def test_load_manifest_entry_for_tiers(self):
        """Test that manifest entries can be loaded for all tiers."""
        from transformation_portal.evals.apex_llava_integration import _load_model_manifest_entry

        tiers = ["ci_smoke", "quality_validation_primary", "quality_max", "legacy_fallback"]

        for tier in tiers:
            entry = _load_model_manifest_entry(tier)

            assert "repo_id" in entry
            assert "revision" in entry
            assert entry["revision"] is not None
            assert len(entry["revision"]) == 40  # SHA length

    def test_load_manifest_entry_unknown_tier_raises(self):
        """Test that unknown tier raises error."""
        from transformation_portal.evals import ApexLlavaIntegrationError
        from transformation_portal.evals.apex_llava_integration import _load_model_manifest_entry

        with pytest.raises(ApexLlavaIntegrationError, match="Unknown model tier"):
            _load_model_manifest_entry("nonexistent_tier")


class TestCreateApexHarnessWithLlava:
    """Test create_apex_harness_with_llava factory."""

    def test_creates_harness_with_llava_backend(self):
        """Test creating harness with LLaVA backend (mocked)."""
        from transformation_portal.evals import ApexLlavaConfig, create_apex_harness_with_llava

        config = ApexLlavaConfig(model_tier="ci_smoke")

        with patch("transformation_portal.evals.apex_llava_integration.create_llava_backend") as mock_create_backend:
            mock_backend = MagicMock()
            mock_create_backend.return_value = mock_backend

            harness = create_apex_harness_with_llava(config)

            assert harness.llava_backend is mock_backend
            assert len(harness.metric_fns) == 3  # Standard metrics
            mock_create_backend.assert_called_once_with(config)

    def test_creates_harness_with_default_config(self):
        """Test creating harness with default config."""
        from transformation_portal.evals import create_apex_harness_with_llava

        with patch("transformation_portal.evals.apex_llava_integration.create_llava_backend") as mock_create_backend:
            mock_backend = MagicMock()
            mock_create_backend.return_value = mock_backend

            create_apex_harness_with_llava()

            # Should use default config
            call_args = mock_create_backend.call_args[0][0]
            assert call_args.model_tier == "quality_validation_primary"
            assert call_args.quality_dimension == "architectural"

    def test_preload_model_option(self):
        """Test preload_model option calls load()."""
        from transformation_portal.evals import ApexLlavaConfig, create_apex_harness_with_llava

        config = ApexLlavaConfig(model_tier="ci_smoke")

        with patch("transformation_portal.evals.apex_llava_integration.create_llava_backend") as mock_create_backend:
            mock_backend = MagicMock()
            mock_create_backend.return_value = mock_backend

            create_apex_harness_with_llava(config, preload_model=True)

            mock_backend.load.assert_called_once()

    def test_no_preload_by_default(self):
        """Test that model is not preloaded by default."""
        from transformation_portal.evals import ApexLlavaConfig, create_apex_harness_with_llava

        config = ApexLlavaConfig(model_tier="ci_smoke")

        with patch("transformation_portal.evals.apex_llava_integration.create_llava_backend") as mock_create_backend:
            mock_backend = MagicMock()
            mock_create_backend.return_value = mock_backend

            create_apex_harness_with_llava(config, preload_model=False)

            mock_backend.load.assert_not_called()


class TestConvenienceFactories:
    """Test convenience factory functions."""

    def test_create_ci_smoke_harness(self):
        """Test create_ci_smoke_harness factory."""
        from transformation_portal.evals import create_ci_smoke_harness

        with patch("transformation_portal.evals.apex_llava_integration.create_llava_backend") as mock_create_backend:
            mock_backend = MagicMock()
            mock_create_backend.return_value = mock_backend

            harness = create_ci_smoke_harness()

            # Should use ci_smoke tier
            call_args = mock_create_backend.call_args[0][0]
            assert call_args.model_tier == "ci_smoke"
            assert harness.threshold == 0.60  # Lower for CI

    def test_create_production_harness(self):
        """Test create_production_harness factory."""
        from transformation_portal.evals import create_production_harness

        with patch("transformation_portal.evals.apex_llava_integration.create_llava_backend") as mock_create_backend:
            mock_backend = MagicMock()
            mock_create_backend.return_value = mock_backend

            create_production_harness(quality_dimension="depth")

            # Should use quality_validation_primary tier
            call_args = mock_create_backend.call_args[0][0]
            assert call_args.model_tier == "quality_validation_primary"
            assert call_args.quality_dimension == "depth"

    def test_create_quality_max_harness(self):
        """Test create_quality_max_harness factory."""
        from transformation_portal.evals import create_quality_max_harness

        with patch("transformation_portal.evals.apex_llava_integration.create_llava_backend") as mock_create_backend:
            mock_backend = MagicMock()
            mock_create_backend.return_value = mock_backend

            harness = create_quality_max_harness(quality_dimension="material")

            # Should use quality_max tier
            call_args = mock_create_backend.call_args[0][0]
            assert call_args.model_tier == "quality_max"
            assert call_args.quality_dimension == "material"
            assert harness.threshold == 0.75  # Higher for quality-max


class TestPromptBuilderSelection:
    """Test quality dimension to prompt builder mapping."""

    def test_get_prompt_builder_for_all_dimensions(self):
        """Test that prompt builders exist for all dimensions."""
        from transformation_portal.evals.apex_llava_integration import (
            _get_prompt_builder_for_dimension,
        )

        dimensions = ["segmentation", "architectural", "depth", "material"]

        for dimension in dimensions:
            builder = _get_prompt_builder_for_dimension(dimension)
            assert callable(builder)

            # Each builder should return a LlavaPromptSpec
            prompt = builder()
            assert hasattr(prompt, "name")
            assert hasattr(prompt, "system_text")
            assert hasattr(prompt, "user_text")

    def test_unknown_dimension_falls_back_to_architectural(self):
        """Test that unknown dimension falls back to architectural prompt."""
        from transformation_portal.evals.apex_llava_integration import (
            _get_prompt_builder_for_dimension,
        )
        from transformation_portal.evals.vision_language import (
            build_architectural_quality_prompt,
        )

        builder = _get_prompt_builder_for_dimension("unknown_dimension")
        expected = build_architectural_quality_prompt

        # Should return architectural prompt builder
        assert builder == expected


class TestHarnessEvaluationFlow:
    """Test the evaluation flow through the harness."""

    @pytest.mark.parametrize(
        ("quality_dimension", "expected_prompt_name"),
        [
            ("segmentation", "segmentation_mask_quality"),
            ("architectural", "architectural_quality"),
            ("depth", "depth_map_quality"),
            ("material", "material_pbr_quality"),
        ],
    )
    def test_evaluate_with_mocked_backend_uses_dimension_prompt(
        self,
        quality_dimension,
        expected_prompt_name,
    ):
        """Test evaluation flow wires the selected prompt into the backend call."""
        from transformation_portal.evals import ApexLlavaConfig, create_apex_harness_with_llava
        from transformation_portal.ingest.canonical_json import dumps_json
        from transformation_portal.evals.vision_language import VQAResult

        config = ApexLlavaConfig(
            model_tier="ci_smoke",
            quality_dimension=quality_dimension,
            include_standard_metrics=False,
        )

        # Create mock backend
        mock_backend = MagicMock()
        mock_result = VQAResult(
            passes_basic_quality=True,
            summary_score=0.85,
            issues=[],
            model_key="test_model",
        )
        mock_backend.evaluate_images.return_value = mock_result
        mock_backend.is_loaded.return_value = True

        with patch("transformation_portal.evals.apex_llava_integration.create_llava_backend") as mock_create:
            mock_create.return_value = mock_backend

            harness = create_apex_harness_with_llava(config)

            # Mock image paths (don't need real files for mocked backend)
            image_paths = [Path("/tmp/test_image.png")]
            context = {"scene": "kitchen", "material": "oak"}

            # Evaluate
            result = harness.evaluate(image_paths=image_paths, context=context)

            # Check result structure
            assert hasattr(result, "score")
            assert hasattr(result, "passes")
            assert hasattr(result, "vlm_score")
            assert result.vlm_score == 0.85

            # Backend should have been called
            mock_backend.evaluate_images.assert_called_once()
            call_kwargs = mock_backend.evaluate_images.call_args.kwargs
            assert call_kwargs["image_paths"] == image_paths
            assert call_kwargs["context"] == context
            assert call_kwargs["prompt_spec"].name == expected_prompt_name

            if quality_dimension == "material":
                assert dumps_json(context, sort_keys=True, indent=2) in call_kwargs["prompt_spec"].user_text
