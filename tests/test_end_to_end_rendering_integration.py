#!/usr/bin/env python3
"""End-to-end integration tests for the complete rendering pipeline.

Tests the full integration of:
- Context extraction → Strategy derivation
- Strategy → Depth processing configuration
- Strategy → Material response configuration
- Strategy → Color grading configuration
- Complete pipeline execution with all stages

These tests verify that all components work together correctly,
without requiring actual ML models or heavy dependencies.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Add scripts directory to path for importing context_aware_rendering
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))


# Import shared mock context classes
from tests.test_helpers import MockRoomContext, MockProjectContext  # noqa: E402


# ---------------------------------------------------------------------------
# Mock Processors
# ---------------------------------------------------------------------------


class MockAdjustmentSettings:
    """Mock adjustment settings for color grading."""

    def __init__(self, **kwargs):
        self.exposure = kwargs.get('exposure', 0.0)
        self.white_balance_temp = kwargs.get('white_balance_temp', 6500.0)
        self.white_balance_tint = kwargs.get('white_balance_tint', 0.0)
        self.shadow_lift = kwargs.get('shadow_lift', 0.0)
        self.highlight_recovery = kwargs.get('highlight_recovery', 0.0)
        self.midtone_contrast = kwargs.get('midtone_contrast', 0.0)
        self.vibrance = kwargs.get('vibrance', 0.0)
        self.saturation = kwargs.get('saturation', 0.0)
        self.clarity = kwargs.get('clarity', 0.0)
        self.chroma_denoise = kwargs.get('chroma_denoise', 0.0)
        self.glow = kwargs.get('glow', 0.0)


def mock_apply_adjustments(image_arr: np.ndarray, settings: MockAdjustmentSettings) -> np.ndarray:
    """Mock color grading application."""
    # Apply simulated adjustments
    result = image_arr.copy()

    # Exposure
    result = result * (1.0 + settings.exposure * 0.5)

    # Saturation (simplified)
    if settings.saturation != 0:
        gray = np.mean(result, axis=2, keepdims=True)
        result = gray + (result - gray) * (1.0 + settings.saturation)

    # Contrast
    if settings.midtone_contrast != 0:
        result = (result - 0.5) * (1.0 + settings.midtone_contrast) + 0.5

    return np.clip(result, 0, 1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_context_module():
    """Create mock for architectural_context_extractor module."""
    mock_module = MagicMock()
    mock_module.ProjectContext = MockProjectContext
    mock_module.RoomContext = MockRoomContext
    mock_module.ArchitecturalContextExtractor = MagicMock()
    return mock_module


@pytest.fixture
def luxury_estate_context():
    """Create a comprehensive luxury estate project context."""
    rooms = {
        'kitchen_main': MockRoomContext(
            name='Gourmet Kitchen',
            dimensions=(20.0, 15.0),
            ceiling_height=10.0,
            materials=['marble', 'stainless steel', 'walnut', 'glass'],
            features=['island', 'wine storage', 'professional appliances'],
        ),
        'living_great': MockRoomContext(
            name='Great Room',
            dimensions=(30.0, 25.0),
            ceiling_height=12.0,
            materials=['oak', 'limestone', 'leather', 'silk'],
            features=['fireplace', 'built-in entertainment', 'floor-to-ceiling windows'],
        ),
        'bedroom_primary': MockRoomContext(
            name='Primary Suite',
            dimensions=(25.0, 20.0),
            ceiling_height=10.0,
            materials=['walnut', 'cashmere', 'silk', 'marble'],
            features=['sitting area', 'private terrace', 'spa bathroom'],
        ),
        'bathroom_spa': MockRoomContext(
            name='Spa Bathroom',
            dimensions=(15.0, 12.0),
            ceiling_height=10.0,
            materials=['carrara marble', 'chrome', 'glass', 'teak'],
            features=['soaking tub', 'rain shower', 'heated floors'],
        ),
        'outdoor_pool': MockRoomContext(
            name='Pool Terrace',
            dimensions=(50.0, 30.0),
            materials=['travertine', 'teak', 'stainless steel'],
            features=['infinity pool', 'outdoor kitchen', 'fire pit'],
        ),
    }

    return MockProjectContext(
        project_name='Montecito Estate',
        project_number='ME-2024-001',
        address='123 Ocean View Drive, Montecito, CA',
        architect='Studio Arch',
        total_sqft=12500.0,
        floors=['Lower Level', 'Main Level', 'Upper Level'],
        rooms=rooms,
        materials_palette=['marble', 'oak', 'walnut', 'limestone', 'stainless steel', 'glass', 'leather'],
        design_style='Modern Mediterranean',
    )


@pytest.fixture
def test_images(tmp_path):
    """Create test images for various room types."""
    images = {}
    rng = np.random.default_rng(seed=42)

    room_types = ['kitchen', 'living_room', 'bedroom_master', 'bathroom_spa', 'outdoor_pool', 'unknown_space']

    for room in room_types:
        img_path = tmp_path / f"{room}_render.jpg"
        # Create distinct images for each room (different base colors)
        base_color = rng.integers(80, 200, 3)
        arr = np.full((256, 256, 3), base_color, dtype=np.uint8)
        # Add some variation
        noise = rng.integers(-20, 20, (256, 256, 3))
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(img_path, quality=95)
        images[room] = img_path

    return images


@pytest.fixture
def pipeline_with_mocks(mock_context_module, luxury_estate_context, tmp_path):
    """Create ContextAwareRenderingPipeline with all processors mocked."""
    with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
        from context_aware_rendering import ContextAwareRenderingPipeline

        pipeline = ContextAwareRenderingPipeline(
            project_context=luxury_estate_context,
            output_dir=tmp_path / 'output',
        )
        return pipeline


# ---------------------------------------------------------------------------
# End-to-End Integration Tests
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration:
    """End-to-end tests for complete pipeline execution."""

    def test_complete_pipeline_all_stages_enabled(
        self, pipeline_with_mocks, test_images, mock_context_module
    ):
        """Test complete pipeline with context → depth → material → color."""
        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            # Test with color grading available (depth will be mocked as unavailable
            # since the actual processor requires ML dependencies)
            result = pipeline_with_mocks.process_render(
                test_images['kitchen'],
                apply_depth=True,
                apply_material=True,
                apply_color=True,
            )

        # Verify complete result structure
        assert 'output_path' in result
        assert 'strategy_path' in result
        assert 'strategy' in result
        assert 'processing_applied' in result
        assert 'depth_config' in result
        assert 'material_config' in result
        assert 'color_config' in result

        # Verify output files exist
        assert result['output_path'].exists()
        assert result['strategy_path'].exists()

        # Verify strategy was correctly derived
        assert result['strategy'].room_type == 'kitchen'
        assert result['strategy'].lighting_style == 'bright'

        # Verify configs were generated (configs are generated even if processors unavailable)
        assert result['depth_config'] is not None
        assert result['material_config'] is not None
        assert result['color_config'] is not None

    def test_pipeline_preserves_processing_chain_order(
        self, pipeline_with_mocks, test_images, mock_context_module
    ):
        """Verify processing stages execute in correct order."""
        processing_order = []

        def track_depth(*args, **kwargs):
            processing_order.append('depth')
            return np.random.rand(256, 256, 3).astype(np.float32)

        def track_color(*args, **kwargs):
            processing_order.append('color')
            return args[0]  # Return input unchanged

        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            with patch.object(
                pipeline_with_mocks, '_apply_depth_processing', side_effect=track_depth
            ):
                with patch.object(
                    pipeline_with_mocks, '_apply_color_grading', side_effect=track_color
                ):
                    pipeline_with_mocks.process_render(
                        test_images['kitchen'],
                        apply_depth=True,
                        apply_material=True,
                        apply_color=True,
                    )

        # Depth should be processed before color
        assert processing_order == ['depth', 'color']

    def test_pipeline_handles_partial_processing(
        self, pipeline_with_mocks, test_images, mock_context_module
    ):
        """Test pipeline with only some stages enabled."""
        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            # Only color grading, no depth
            with patch('context_aware_rendering._check_tiff_processor', return_value=True):
                mock_tiff_module = MagicMock()
                mock_tiff_module.AdjustmentSettings = MockAdjustmentSettings
                mock_tiff_module.apply_adjustments = mock_apply_adjustments

                with patch.dict(sys.modules, {
                    'luxury_tiff_batch_processor.adjustments': mock_tiff_module,
                }):
                    result = pipeline_with_mocks.process_render(
                        test_images['living_room'],
                        apply_depth=False,
                        apply_material=False,
                        apply_color=True,
                    )

        assert result['output_path'].exists()
        assert result['depth_config'] is None
        assert result['material_config'] is None
        assert result['color_config'] is not None

    def test_pipeline_graceful_degradation_missing_depth(
        self, pipeline_with_mocks, test_images, mock_context_module
    ):
        """Test pipeline continues when depth pipeline unavailable."""
        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            with patch('context_aware_rendering._check_depth_pipeline', return_value=False):
                result = pipeline_with_mocks.process_render(
                    test_images['bedroom_master'],
                    apply_depth=True,  # Requested but unavailable
                    apply_material=True,
                    apply_color=True,
                )

        # Pipeline should still produce output even when depth is unavailable
        assert result['output_path'].exists()
        # Processing log should not include depth (unavailable)
        assert 'depth_processing' not in result['processing_applied']
        # Color grading may or may not be applied depending on TIFF processor availability
        # The key test is that pipeline completes successfully


class TestStrategyToConfigIntegration:
    """Test integration between strategy derivation and config generation."""

    def test_kitchen_strategy_produces_correct_depth_config(
        self, pipeline_with_mocks, test_images, mock_context_module
    ):
        """Verify kitchen strategy generates appropriate depth config."""
        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            strategy = pipeline_with_mocks.derive_strategy(test_images['kitchen'])
            depth_config = pipeline_with_mocks.generate_depth_config(strategy)

        # Kitchen should use bright lighting → reinhard tone mapping
        assert depth_config['tone_map'] == 'reinhard'
        # Balanced depth emphasis
        assert depth_config['zone_weights']['midground'] == 1.0

    def test_bedroom_strategy_produces_warm_color_config(
        self, mock_context_module, tmp_path
    ):
        """Verify bedroom strategy generates warm color config for traditional style."""
        # Use traditional style to get warm temperature
        context = MockProjectContext(
            project_name='Traditional Estate',
            design_style='Traditional Colonial',
            rooms={'bedroom_primary': MockRoomContext(name='Primary Suite')},
            materials_palette=['mahogany', 'silk', 'velvet'],
        )

        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            from context_aware_rendering import ContextAwareRenderingPipeline

            pipeline = ContextAwareRenderingPipeline(
                project_context=context,
                output_dir=tmp_path / 'output',
            )

            img_path = tmp_path / 'bedroom_test.jpg'
            Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8)).save(img_path)

            strategy = pipeline.derive_strategy(img_path)
            color_config = pipeline.generate_color_config(strategy)

        # Traditional bedroom should have warm tint
        assert color_config['tint'] == 5
        assert color_config['saturation'] == 1.08

    def test_outdoor_strategy_produces_atmospheric_depth(
        self, pipeline_with_mocks, test_images, mock_context_module
    ):
        """Verify outdoor strategy emphasizes background depth."""
        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            strategy = pipeline_with_mocks.derive_strategy(test_images['outdoor_pool'])
            depth_config = pipeline_with_mocks.generate_depth_config(strategy)

        # Outdoor should emphasize atmospheric/background
        assert depth_config['zone_weights']['background'] == 1.0
        assert depth_config['zone_weights']['foreground'] == 0.6

    def test_material_config_reflects_room_materials(
        self, mock_context_module, tmp_path
    ):
        """Verify material config uses room-appropriate materials from project palette."""
        # Create context with materials that match bathroom defaults
        context = MockProjectContext(
            project_name='Spa Retreat',
            rooms={'bathroom_main': MockRoomContext(name='Main Bathroom')},
            materials_palette=['stone', 'glass', 'metal', 'tile', 'marble'],
        )

        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            from context_aware_rendering import ContextAwareRenderingPipeline

            pipeline = ContextAwareRenderingPipeline(
                project_context=context,
                output_dir=tmp_path / 'output',
            )

            img_path = tmp_path / 'bathroom_test.jpg'
            Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8)).save(img_path)

            strategy = pipeline.derive_strategy(img_path)
            material_config = pipeline.generate_material_config(strategy)

        # Bathroom materials should include stone, glass from matching project palette
        assert 'stone' in material_config['enabled_surfaces']
        assert 'glass' in material_config['enabled_surfaces']


class TestMultiRoomProcessing:
    """Test processing multiple rooms in sequence."""

    def test_process_multiple_rooms_maintains_context(
        self, pipeline_with_mocks, test_images, mock_context_module
    ):
        """Verify context is preserved across multiple room processing."""
        results = {}

        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            for room_type in ['kitchen', 'living_room', 'bedroom_master']:
                result = pipeline_with_mocks.process_render(
                    test_images[room_type],
                    apply_depth=False,
                    apply_material=False,
                    apply_color=False,
                )
                results[room_type] = result

        # Each room should have different strategy
        assert results['kitchen']['strategy'].room_type == 'kitchen'
        assert results['living_room']['strategy'].room_type == 'living'
        assert results['bedroom_master']['strategy'].room_type == 'bedroom'

        # Each should produce separate output files
        output_stems = {r['output_path'].stem for r in results.values()}
        assert len(output_stems) == 3

    def test_unknown_room_uses_default_strategy(
        self, pipeline_with_mocks, test_images, mock_context_module
    ):
        """Verify unknown rooms get sensible default strategy."""
        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            result = pipeline_with_mocks.process_render(
                test_images['unknown_space'],
                apply_depth=False,
                apply_material=False,
                apply_color=False,
            )

        assert result['strategy'].room_type == 'unknown'
        assert result['strategy'].lighting_style == 'ambient'
        assert result['strategy'].depth_emphasis == 'balanced'


class TestConfigFileIntegration:
    """Test strategy/config JSON file generation and structure."""

    def test_strategy_json_contains_all_fields(
        self, pipeline_with_mocks, test_images, mock_context_module
    ):
        """Verify strategy JSON has complete structure."""
        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            result = pipeline_with_mocks.process_render(
                test_images['kitchen'],
                apply_depth=True,
                apply_material=True,
                apply_color=True,
            )

        with open(result['strategy_path']) as f:
            strategy_data = json.load(f)

        # Check strategy section
        assert 'strategy' in strategy_data
        strategy = strategy_data['strategy']
        assert 'room_type' in strategy
        assert 'primary_materials' in strategy
        assert 'lighting_style' in strategy
        assert 'depth_emphasis' in strategy
        assert 'color_temperature' in strategy
        assert 'enhancement_strength' in strategy

        # Check config sections
        assert 'depth_config' in strategy_data
        assert 'material_config' in strategy_data
        assert 'color_config' in strategy_data

    def test_depth_config_json_structure(
        self, pipeline_with_mocks, test_images, mock_context_module
    ):
        """Verify depth config has correct structure for pipeline."""
        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            result = pipeline_with_mocks.process_render(
                test_images['living_room'],
                apply_depth=True,
                apply_material=False,
                apply_color=False,
            )

        with open(result['strategy_path']) as f:
            strategy_data = json.load(f)

        depth_config = strategy_data['depth_config']
        assert 'zone_weights' in depth_config
        assert 'foreground' in depth_config['zone_weights']
        assert 'midground' in depth_config['zone_weights']
        assert 'background' in depth_config['zone_weights']
        assert 'tone_map' in depth_config


class TestDesignStyleIntegration:
    """Test how design style affects processing decisions."""

    def test_modern_style_affects_color_temperature(
        self, mock_context_module, tmp_path
    ):
        """Verify Modern style produces neutral temperature."""
        context = MockProjectContext(
            project_name='Modern Villa',
            design_style='Ultra Modern',
            rooms={'kitchen_main': MockRoomContext(name='Kitchen')},
            materials_palette=['concrete', 'steel', 'glass'],
        )

        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            from context_aware_rendering import ContextAwareRenderingPipeline

            pipeline = ContextAwareRenderingPipeline(
                project_context=context,
                output_dir=tmp_path / 'output',
            )

            # Create test image
            img_path = tmp_path / 'kitchen_test.jpg'
            Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8)).save(img_path)

            strategy = pipeline.derive_strategy(img_path)

        assert strategy.color_temperature == 'neutral'

    def test_traditional_style_affects_enhancement(
        self, mock_context_module, tmp_path
    ):
        """Verify Traditional style reduces enhancement strength."""
        context = MockProjectContext(
            project_name='Traditional Estate',
            design_style='Traditional Georgian',
            rooms={'living_main': MockRoomContext(name='Living Room')},
            materials_palette=['mahogany', 'marble', 'brass'],
        )

        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            from context_aware_rendering import ContextAwareRenderingPipeline

            pipeline = ContextAwareRenderingPipeline(
                project_context=context,
                output_dir=tmp_path / 'output',
            )

            img_path = tmp_path / 'living_room_test.jpg'
            Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8)).save(img_path)

            strategy = pipeline.derive_strategy(img_path)

        assert strategy.color_temperature == 'warm'
        # Traditional should have reduced enhancement
        assert strategy.enhancement_strength <= 0.7


class TestDepthPipelineConfigBuilding:
    """Test depth pipeline configuration building from strategy."""

    def test_build_depth_pipeline_config_complete(
        self, pipeline_with_mocks, mock_context_module
    ):
        """Verify full depth pipeline config is properly built."""
        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            depth_config = {
                'model_size': 'small',
                'device': 'pytorch_cpu',
                'tone_map': 'agx',
                'depth_emphasis': 'balanced',
                'zone_weights': {
                    'foreground': 0.8,
                    'midground': 1.0,
                    'background': 0.8,
                },
            }

            full_config = pipeline_with_mocks._build_depth_pipeline_config(depth_config)

        # Check depth model config
        assert 'depth_model' in full_config
        assert full_config['depth_model']['variant'] == 'small'
        assert full_config['depth_model']['backend'] == 'pytorch_cpu'

        # Check processing config
        assert 'processing' in full_config
        proc = full_config['processing']
        assert proc['zone_tone_mapping']['enabled'] is True
        assert proc['zone_tone_mapping']['method'] == 'agx'
        assert proc['depth_aware_denoise']['enabled'] is True
        assert proc['depth_guided_filters']['enabled'] is True

    def test_atmospheric_depth_enables_atmospheric_effects(
        self, pipeline_with_mocks, mock_context_module
    ):
        """Verify atmospheric depth emphasis enables atmospheric effects."""
        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            depth_config = {
                'depth_emphasis': 'atmospheric',
                'tone_map': 'agx',
                'zone_weights': {
                    'foreground': 0.6,
                    'midground': 0.8,
                    'background': 1.0,
                },
            }

            full_config = pipeline_with_mocks._build_depth_pipeline_config(depth_config)

        assert full_config['processing']['atmospheric_effects']['enabled'] is True

    def test_foreground_depth_disables_atmospheric_effects(
        self, pipeline_with_mocks, mock_context_module
    ):
        """Verify foreground depth emphasis disables atmospheric effects."""
        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            depth_config = {
                'depth_emphasis': 'foreground',
                'tone_map': 'reinhard',
                'zone_weights': {
                    'foreground': 1.0,
                    'midground': 0.6,
                    'background': 0.3,
                },
            }

            full_config = pipeline_with_mocks._build_depth_pipeline_config(depth_config)

        assert full_config['processing']['atmospheric_effects']['enabled'] is False


class TestOutputQualityVerification:
    """Test output image quality and format handling."""

    def test_output_preserves_input_format(
        self, pipeline_with_mocks, tmp_path, mock_context_module
    ):
        """Verify output format matches input format."""
        # Create PNG input
        png_path = tmp_path / 'test_kitchen.png'
        Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8)).save(png_path)

        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            result = pipeline_with_mocks.process_render(
                png_path,
                apply_depth=False,
                apply_material=False,
                apply_color=False,
            )

        assert result['output_path'].suffix == '.png'

    def test_output_image_dimensions_preserved(
        self, pipeline_with_mocks, tmp_path, mock_context_module
    ):
        """Verify output dimensions match input."""
        img_path = tmp_path / 'test_living.jpg'
        original_size = (400, 300)
        Image.fromarray(
            np.full((300, 400, 3), 128, dtype=np.uint8)
        ).save(img_path, quality=95)

        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            result = pipeline_with_mocks.process_render(
                img_path,
                apply_depth=False,
                apply_material=False,
                apply_color=False,
            )

        with Image.open(result['output_path']) as output_img:
            assert output_img.size == original_size

    def test_output_image_valid_pixel_range(
        self, pipeline_with_mocks, test_images, mock_context_module
    ):
        """Verify output pixels are in valid range [0, 255]."""
        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            result = pipeline_with_mocks.process_render(
                test_images['kitchen'],
                apply_depth=False,
                apply_material=False,
                apply_color=False,
            )

        with Image.open(result['output_path']) as img:
            arr = np.array(img)
            assert arr.min() >= 0
            assert arr.max() <= 255


class TestErrorHandling:
    """Test error handling and recovery scenarios."""

    def test_handles_missing_project_materials(
        self, mock_context_module, tmp_path
    ):
        """Verify pipeline handles missing materials palette."""
        context = MockProjectContext(
            project_name='Minimal Project',
            rooms={'living_main': MockRoomContext(name='Living')},
            materials_palette=[],  # Empty palette
        )

        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            from context_aware_rendering import ContextAwareRenderingPipeline

            pipeline = ContextAwareRenderingPipeline(
                project_context=context,
                output_dir=tmp_path / 'output',
            )

            img_path = tmp_path / 'living_test.jpg'
            Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8)).save(img_path)

            result = pipeline.process_render(
                img_path,
                apply_depth=False,
                apply_material=False,
                apply_color=False,
            )

        assert result['output_path'].exists()

    def test_handles_processor_exceptions_gracefully(
        self, pipeline_with_mocks, test_images, mock_context_module
    ):
        """Verify pipeline continues after processor exceptions via internal handling."""
        with patch.dict(sys.modules, {'architectural_context_extractor': mock_context_module}):
            # Mock depth as available but have the internal processor raise
            with patch('context_aware_rendering._check_depth_pipeline', return_value=True):
                # Mock the depth pipeline import to raise when instantiated
                mock_depth_module = MagicMock()
                mock_depth_module.ArchitecturalDepthPipeline.side_effect = RuntimeError(
                    "Simulated processor failure"
                )

                with patch.dict(sys.modules, {
                    'transformation_portal.depth.pipeline': mock_depth_module
                }):
                    # The internal try/except in _apply_depth_processing should catch this
                    result = pipeline_with_mocks.process_render(
                        test_images['kitchen'],
                        apply_depth=True,
                        apply_material=False,
                        apply_color=False,
                    )

        # Pipeline should still produce output despite depth failure
        assert result['output_path'].exists()
        # Depth processing should have been skipped due to error
        assert 'depth_processing' not in result['processing_applied']


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
