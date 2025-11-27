#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for concrete async pipeline stages.

Tests cover:
- ImageLoadStage: Image loading functionality
- ImageSaveStage: Image saving functionality
- DepthEstimationStage: Depth estimation
- MaterialResponseStage: Material enhancement
- ColorGradingStage: Color grading
- ResizeStage: Image resizing
- DenoiseStage: Denoising
- Factory function: create_luxury_pipeline_stages
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import stages - may require optional dependencies
STAGES_AVAILABLE = False
IMPORT_ERROR = ""

try:
    from transformation_portal.streaming.stages import (
        ColorGradingStage,
        DenoiseStage,
        DepthEstimationStage,
        ImageData,
        ImageLoadStage,
        ImageSaveStage,
        MaterialResponseStage,
        ResizeStage,
        create_luxury_pipeline_stages,
    )
    STAGES_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR = str(e)


# Skip all tests if stages module dependencies not available
pytestmark = pytest.mark.skipif(
    not STAGES_AVAILABLE,
    reason=f"Stages module dependencies not available: {IMPORT_ERROR}"
)


# ============================================================================
# ImageData Tests
# ============================================================================

class TestImageData:
    """Tests for ImageData container."""

    def test_image_data_creation(self):
        """Test ImageData creation."""
        try:
            import numpy as np
            array = np.zeros((100, 100, 3), dtype=np.uint8)
        except ImportError:
            pytest.skip("numpy not available")

        data = ImageData(
            array=array,
            path=Path("/test/image.jpg"),
            metadata={"format": "JPEG"}
        )

        assert data.path == Path("/test/image.jpg")
        assert data.shape == (100, 100, 3)
        assert data.depth_map is None
        assert data.metadata["format"] == "JPEG"

    def test_image_data_with_depth(self):
        """Test ImageData with depth map."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        array = np.zeros((100, 100, 3))
        depth = np.zeros((100, 100))

        data = ImageData(
            array=array,
            path=Path("/test/image.jpg"),
            depth_map=depth
        )

        assert data.depth_map is not None
        assert data.depth_map.shape == (100, 100)


# ============================================================================
# ImageLoadStage Tests
# ============================================================================

class TestImageLoadStage:
    """Tests for ImageLoadStage."""

    def test_stage_initialization(self):
        """Test stage initialization."""
        stage = ImageLoadStage(
            max_concurrent=4,
            load_exif=True,
            convert_16bit=True
        )

        assert stage.name == "image_load"
        assert stage.max_concurrent == 4
        assert stage._load_exif is True
        assert stage._convert_16bit is True

    @pytest.mark.asyncio
    async def test_load_jpeg_image(self):
        """Test loading JPEG image."""
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            pytest.skip("PIL/numpy not available")

        # Create temp JPEG
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
            img.save(f.name, "JPEG")
            temp_path = Path(f.name)

        stage = ImageLoadStage()
        await stage.startup()

        try:
            result = await stage(temp_path)

            assert result.success
            assert result.data is not None
            assert result.data.path == temp_path
            assert result.data.array.shape == (100, 100, 3)
            assert result.data.metadata.get("format") == "JPEG"
        finally:
            await stage.shutdown()
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_load_png_image(self):
        """Test loading PNG image."""
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            pytest.skip("PIL/numpy not available")

        # Create temp PNG
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))
            img.save(f.name, "PNG")
            temp_path = Path(f.name)

        stage = ImageLoadStage()
        await stage.startup()

        try:
            result = await stage(temp_path)

            assert result.success
            assert result.data.metadata.get("format") == "PNG"
        finally:
            await stage.shutdown()
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_load_nonexistent_raises(self):
        """Test that loading nonexistent file fails."""
        stage = ImageLoadStage()
        await stage.startup()

        try:
            result = await stage(Path("/nonexistent/image.jpg"))

            assert not result.success
            assert result.error is not None
        finally:
            await stage.shutdown()


# ============================================================================
# ImageSaveStage Tests
# ============================================================================

class TestImageSaveStage:
    """Tests for ImageSaveStage."""

    def test_stage_initialization(self):
        """Test stage initialization."""
        stage = ImageSaveStage(
            output_dir="/tmp/output",
            output_format="JPEG",
            quality=90,
            suffix="_out"
        )

        assert stage.name == "image_save"
        assert stage._format == "JPEG"
        assert stage._quality == 90
        assert stage._suffix == "_out"

    @pytest.mark.asyncio
    async def test_save_jpeg(self):
        """Test saving as JPEG."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            stage = ImageSaveStage(
                output_dir=tmpdir,
                output_format="JPEG",
                quality=85
            )
            await stage.startup()

            try:
                image_data = ImageData(
                    array=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
                    path=Path("/original/test_image.jpg")
                )

                result = await stage(image_data)

                assert result.success
                output_path = Path(result.data.metadata["output_path"])
                assert output_path.exists()
                assert output_path.suffix == ".jpg"
            finally:
                await stage.shutdown()

    @pytest.mark.asyncio
    async def test_save_float_array(self):
        """Test saving float32 array."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            stage = ImageSaveStage(
                output_dir=tmpdir,
                output_format="PNG"
            )
            await stage.startup()

            try:
                # Float32 array in 0-1 range
                image_data = ImageData(
                    array=np.random.rand(100, 100, 3).astype(np.float32),
                    path=Path("/original/test.png")
                )

                result = await stage(image_data)

                assert result.success
                output_path = Path(result.data.metadata["output_path"])
                assert output_path.exists()
            finally:
                await stage.shutdown()


# ============================================================================
# DepthEstimationStage Tests
# ============================================================================

class TestDepthEstimationStage:
    """Tests for DepthEstimationStage."""

    def test_stage_initialization(self):
        """Test stage initialization."""
        from transformation_portal.streaming.async_pipeline import DeviceType

        stage = DepthEstimationStage(
            device=DeviceType.CPU,
            model_size="base",
            max_concurrent=1
        )

        assert stage.name == "depth_estimation"
        assert stage._model_size == "base"
        assert stage.required is False  # Optional by default

    @pytest.mark.asyncio
    async def test_depth_estimation_mock(self):
        """Test depth estimation with mock model."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        stage = DepthEstimationStage(cache_model=False)
        await stage.startup()

        try:
            image_data = ImageData(
                array=np.random.rand(100, 100, 3).astype(np.float32),
                path=Path("/test/image.jpg")
            )

            result = await stage(image_data)

            assert result.success
            assert result.data.depth_map is not None
            assert result.data.depth_map.shape == (100, 100)
            assert result.data.metadata.get("depth_estimated") is True
        finally:
            await stage.shutdown()


# ============================================================================
# MaterialResponseStage Tests
# ============================================================================

class TestMaterialResponseStage:
    """Tests for MaterialResponseStage."""

    def test_stage_initialization(self):
        """Test stage initialization."""
        stage = MaterialResponseStage(
            materials=["wood", "metal"],
            intensity=0.8,
            use_depth=True
        )

        assert stage.name == "material_response"
        assert stage._materials == ["wood", "metal"]
        assert stage._intensity == 0.8
        assert stage._use_depth is True

    @pytest.mark.asyncio
    async def test_material_enhancement(self):
        """Test material enhancement processing."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        stage = MaterialResponseStage(intensity=0.5)
        await stage.startup()

        try:
            image_data = ImageData(
                array=np.random.rand(100, 100, 3).astype(np.float32),
                path=Path("/test/image.jpg")
            )

            result = await stage(image_data)

            assert result.success
            assert result.data.metadata.get("material_enhanced") is True
        finally:
            await stage.shutdown()


# ============================================================================
# ColorGradingStage Tests
# ============================================================================

class TestColorGradingStage:
    """Tests for ColorGradingStage."""

    def test_stage_initialization(self):
        """Test stage initialization."""
        stage = ColorGradingStage(
            lut_path=None,
            intensity=0.8
        )

        assert stage.name == "color_grading"
        assert stage._intensity == 0.8

    @pytest.mark.asyncio
    async def test_color_grading_no_lut(self):
        """Test color grading without LUT (fallback enhancement)."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        stage = ColorGradingStage(intensity=0.5)
        await stage.startup()

        try:
            image_data = ImageData(
                array=np.random.rand(100, 100, 3).astype(np.float32),
                path=Path("/test/image.jpg")
            )

            result = await stage(image_data)

            assert result.success
            assert result.data.metadata.get("color_graded") is True
        finally:
            await stage.shutdown()

    @pytest.mark.asyncio
    async def test_color_grading_with_lut(self):
        """Test color grading with LUT file."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        # Create a minimal .cube LUT file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cube', delete=False) as f:
            f.write("# Test LUT\n")
            f.write("LUT_3D_SIZE 2\n")
            # Write 8 values for 2x2x2 LUT
            for r in range(2):
                for g in range(2):
                    for b in range(2):
                        f.write(f"{r} {g} {b}\n")
            lut_path = Path(f.name)

        stage = ColorGradingStage(lut_path=lut_path, intensity=0.5)
        await stage.startup()

        try:
            image_data = ImageData(
                array=np.random.rand(100, 100, 3).astype(np.float32),
                path=Path("/test/image.jpg")
            )

            result = await stage(image_data)

            assert result.success
        finally:
            await stage.shutdown()
            lut_path.unlink()


# ============================================================================
# ResizeStage Tests
# ============================================================================

class TestResizeStage:
    """Tests for ResizeStage."""

    def test_stage_initialization(self):
        """Test stage initialization."""
        stage = ResizeStage(
            target_size=(1920, 1080),
            method="lanczos",
            maintain_aspect=True
        )

        assert stage.name == "resize"
        assert stage._target_size == (1920, 1080)
        assert stage._method == "lanczos"
        assert stage._maintain_aspect is True

    @pytest.mark.asyncio
    async def test_resize_by_target(self):
        """Test resizing to target size."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        stage = ResizeStage(
            target_size=(50, 50),
            maintain_aspect=False
        )
        await stage.startup()

        try:
            image_data = ImageData(
                array=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
                path=Path("/test/image.jpg")
            )

            result = await stage(image_data)

            assert result.success
            assert result.data.array.shape[:2] == (50, 50)
            assert result.data.metadata.get("resized") is True
        finally:
            await stage.shutdown()

    @pytest.mark.asyncio
    async def test_resize_by_scale(self):
        """Test resizing by scale factor."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        stage = ResizeStage(
            scale_factor=0.5,
            maintain_aspect=True
        )
        await stage.startup()

        try:
            image_data = ImageData(
                array=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
                path=Path("/test/image.jpg")
            )

            result = await stage(image_data)

            assert result.success
            assert result.data.array.shape[:2] == (50, 50)
        finally:
            await stage.shutdown()


# ============================================================================
# DenoiseStage Tests
# ============================================================================

class TestDenoiseStage:
    """Tests for DenoiseStage."""

    def test_stage_initialization(self):
        """Test stage initialization."""
        stage = DenoiseStage(
            strength=0.5,
            use_depth=True
        )

        assert stage.name == "denoise"
        assert stage._strength == 0.5
        assert stage._use_depth is True

    @pytest.mark.asyncio
    async def test_denoise_without_depth(self):
        """Test denoising without depth map."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        stage = DenoiseStage(strength=0.3, use_depth=False)
        await stage.startup()

        try:
            # Create noisy image
            image_data = ImageData(
                array=np.random.rand(100, 100, 3).astype(np.float32),
                path=Path("/test/image.jpg")
            )

            result = await stage(image_data)

            assert result.success
            assert result.data.metadata.get("denoised") is True
        finally:
            await stage.shutdown()

    @pytest.mark.asyncio
    async def test_denoise_with_depth(self):
        """Test depth-adaptive denoising."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        stage = DenoiseStage(strength=0.5, use_depth=True)
        await stage.startup()

        try:
            image_data = ImageData(
                array=np.random.rand(100, 100, 3).astype(np.float32),
                path=Path("/test/image.jpg"),
                depth_map=np.random.rand(100, 100).astype(np.float32)
            )

            result = await stage(image_data)

            assert result.success
            assert result.data.metadata.get("denoised") is True
        finally:
            await stage.shutdown()


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestCreateLuxuryPipelineStages:
    """Tests for create_luxury_pipeline_stages factory function."""

    def test_default_stages(self):
        """Test creating default pipeline stages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stages = create_luxury_pipeline_stages(output_dir=tmpdir)

            # Should have: load, depth, material, color_grading, save
            assert len(stages) == 5

            stage_names = [s.name for s in stages]
            assert "image_load" in stage_names
            assert "depth_estimation" in stage_names
            assert "material_response" in stage_names
            assert "color_grading" in stage_names
            assert "image_save" in stage_names

    def test_minimal_stages(self):
        """Test creating minimal pipeline (load + save only)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stages = create_luxury_pipeline_stages(
                output_dir=tmpdir,
                enable_depth=False,
                enable_material=False,
                enable_color_grading=False
            )

            # Should have only: load, save
            assert len(stages) == 2

            stage_names = [s.name for s in stages]
            assert "image_load" in stage_names
            assert "image_save" in stage_names

    def test_with_lut_path(self):
        """Test creating stages with LUT path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stages = create_luxury_pipeline_stages(
                output_dir=tmpdir,
                enable_depth=False,
                enable_material=False,
                lut_path="/path/to/lut.cube"
            )

            # Find color grading stage and check LUT path
            color_stage = next(
                (s for s in stages if s.name == "color_grading"),
                None
            )
            assert color_stage is not None
            assert color_stage._lut_path == Path("/path/to/lut.cube")


# ============================================================================
# Integration Tests
# ============================================================================

class TestStagesIntegration:
    """Integration tests for pipeline stages."""

    @pytest.mark.asyncio
    async def test_load_process_save_flow(self):
        """Test complete load -> process -> save flow."""
        try:
            import numpy as np
            from PIL import Image
            from transformation_portal.streaming.async_pipeline import AsyncPipeline
        except ImportError:
            pytest.skip("Required dependencies not available")

        # Create temp input image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            img.save(f.name, "JPEG")
            input_path = Path(f.name)

        with tempfile.TemporaryDirectory() as output_dir:
            pipeline = AsyncPipeline()
            pipeline.add_stage(ImageLoadStage())
            pipeline.add_stage(ResizeStage(scale_factor=0.5))
            pipeline.add_stage(ImageSaveStage(output_dir=output_dir, output_format="PNG"))

            async with pipeline:
                result = await pipeline.process_item(input_path)

                assert result.data is not None
                output_path = Path(result.data.metadata["output_path"])
                assert output_path.exists()

        input_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
