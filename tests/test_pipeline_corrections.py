"""Tests for pipeline corrections C1–C3 (post-PR #975).

C1: Stage naming + config alias correctness
    - "segment" in pipeline config should be treated as "segmentation"
C2: Decouple emit_exr / emit_provenance from save_intermediates for Ultra
    - apex_research_ultra always emits EXR + provenance (contract artifacts)
C3: Materials stage resource lifecycle
    - Materials stage uses resource_manager.register_model() / unload_model()
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from transformation_portal.spatial_ai.orchestration.pipeline import PipelineConfig, SpatialAIPipeline

# ---------------------------------------------------------------------------
# C1: Stage naming + config alias correctness
# ---------------------------------------------------------------------------


class TestStageAliasNormalization:
    """'segment' in pipeline dict must populate segmentation config."""

    def test_segment_alias_populates_segmentation_config(self):
        """Config dict with 'segment' key should populate segmentation field."""
        data = {
            "tier": "standard",
            "pipeline": {
                "ingest": {},
                "segment": {"backend": "sam2", "model": {"size": "large"}},
            },
        }
        config = SpatialAIPipeline._dict_to_config(data)

        assert "segmentation" in config.stages
        assert "segment" not in config.stages
        assert config.segmentation == {"backend": "sam2", "model": {"size": "large"}}

    def test_segmentation_key_still_works(self):
        """Config dict with 'segmentation' key should still work as before."""
        data = {
            "tier": "standard",
            "pipeline": {
                "ingest": {},
                "segmentation": {"backend": "sam2"},
            },
        }
        config = SpatialAIPipeline._dict_to_config(data)

        assert "segmentation" in config.stages
        assert config.segmentation == {"backend": "sam2"}

    def test_segment_alias_preferred_over_empty_segmentation(self):
        """When only 'segment' is present, its data is used for segmentation."""
        data = {
            "tier": "standard",
            "pipeline": {
                "segment": {"backend": "sam2", "confidence_threshold": 0.9},
            },
        }
        config = SpatialAIPipeline._dict_to_config(data)

        assert config.segmentation.get("confidence_threshold") == 0.9

    def test_stages_completed_uses_canonical_name(self, tmp_path):
        """process() should record 'segmentation' (not 'segment') in stages_completed."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest", "segmentation"],
        )
        pipeline = SpatialAIPipeline(config)

        input_file = tmp_path / "test.tiff"
        input_file.write_bytes(b"dummy")
        output_dir = tmp_path / "output"

        mock_ingest = MagicMock()
        mock_seg = MagicMock()

        with (
            patch.object(pipeline, "_run_ingest", return_value=mock_ingest),
            patch.object(pipeline, "_run_segmentation", return_value=mock_seg),
        ):
            result = pipeline.process(input_file, output_dir, save_intermediates=False)

        assert "segmentation" in result.stages_completed
        assert "segment" not in result.stages_completed


# ---------------------------------------------------------------------------
# C2: Decouple emit_exr / emit_provenance from save_intermediates for Ultra
# ---------------------------------------------------------------------------


class TestUltraEmissionDecoupling:
    """Ultra tier always emits EXR + provenance as contract artifacts."""

    def _run_ingest_capture(self, pipeline, tmp_path, save_intermediates):
        """Helper: run pipeline and capture the emit_exr/emit_provenance values passed to decoder."""
        import numpy as np

        input_file = tmp_path / "test.tiff"
        input_file.write_bytes(b"dummy")
        output_dir = tmp_path / "output"

        captured_calls = {}

        def mock_decode(**kwargs):
            captured_calls.update(kwargs)
            result = MagicMock()
            result.input_size = (100, 200)
            result.linear_rgb = MagicMock()
            result.linear_rgb.min.return_value = 0.0
            result.linear_rgb.max.return_value = 1.0
            return result

        mock_decoder = MagicMock()
        mock_decoder.decode = mock_decode

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.LinearDecoder", return_value=mock_decoder):
            pipeline.process(input_file, output_dir, save_intermediates=save_intermediates)

        return captured_calls

    def test_ultra_emits_exr_when_save_intermediates_false(self, tmp_path):
        """apex_research_ultra should emit EXR even when save_intermediates=False."""
        config = PipelineConfig(
            tier="apex_research_ultra",
            stages=["ingest"],
            ingest={"emit_exr": True, "emit_provenance": True},
        )
        pipeline = SpatialAIPipeline(config)

        captured = self._run_ingest_capture(pipeline, tmp_path, save_intermediates=False)

        assert captured.get("emit_exr") is True
        assert captured.get("emit_provenance") is True

    def test_standard_tier_respects_save_intermediates(self, tmp_path):
        """Standard tier should NOT emit EXR when save_intermediates=False."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest"],
            ingest={"emit_exr": True, "emit_provenance": True},
        )
        pipeline = SpatialAIPipeline(config)

        captured = self._run_ingest_capture(pipeline, tmp_path, save_intermediates=False)

        assert captured.get("emit_exr") is False
        assert captured.get("emit_provenance") is False

    def test_standard_tier_emits_when_save_intermediates_true(self, tmp_path):
        """Standard tier should emit EXR when save_intermediates=True."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest"],
            ingest={"emit_exr": True, "emit_provenance": True},
        )
        pipeline = SpatialAIPipeline(config)

        captured = self._run_ingest_capture(pipeline, tmp_path, save_intermediates=True)

        assert captured.get("emit_exr") is True
        assert captured.get("emit_provenance") is True


# ---------------------------------------------------------------------------
# C3: Materials stage resource lifecycle
# ---------------------------------------------------------------------------


class TestMaterialsResourceLifecycle:
    """Materials stage must register and unload its backend via resource_manager."""

    def test_materials_registers_and_unloads_model(self, tmp_path):
        """Materials stage should call register_model and unload_model."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest", "segmentation", "materials"],
        )
        pipeline = SpatialAIPipeline(config)

        input_file = tmp_path / "test.tiff"
        input_file.write_bytes(b"dummy")
        output_dir = tmp_path / "output"

        mock_ingest = MagicMock()
        mock_seg = MagicMock()
        mock_seg.masks = []
        mock_seg.metadata = []

        with (
            patch.object(pipeline, "_run_ingest", return_value=mock_ingest),
            patch.object(pipeline, "_run_segmentation", return_value=mock_seg),
            patch.object(pipeline.resource_manager, "register_model") as mock_register,
            patch.object(pipeline.resource_manager, "unload_model") as mock_unload,
            patch.object(pipeline.resource_manager, "select_device", return_value="cpu"),
            patch("transformation_portal.spatial_ai.orchestration.pipeline.MaterialBackend") as MockBackend,
        ):
            MockBackend.return_value = MagicMock()
            pipeline.process(input_file, output_dir, save_intermediates=False)

        mock_register.assert_any_call("materials", MockBackend.return_value)
        mock_unload.assert_any_call("materials")


# Pytest markers
pytestmark = [
    pytest.mark.unit,
]
