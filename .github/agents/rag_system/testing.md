# Testing Template

**Use this template for**: Adding comprehensive tests for new features, regression tests, integration tests, and performance benchmarks

---

## Test Plan Overview

**Feature/Module**: `{FEATURE_NAME}`

**Test Scope**:
- [ ] Unit Tests (individual functions/classes)
- [ ] Integration Tests (full pipeline workflows)
- [ ] Property-Based Tests (using hypothesis)
- [ ] Performance Benchmarks
- [ ] Regression Tests (for bug fixes)
- [ ] Edge Case Tests

**Test Files to Create/Modify**:
- [ ] `tests/test_{module_name}.py` - Unit tests
- [ ] `tests/integration/test_{pipeline}_integration.py` - Integration tests
- [ ] `tests/test_{feature_name}_properties.py` - Property-based tests
- [ ] `tests/benchmarks/test_{feature}_performance.py` - Performance tests

---

## Repository Testing Conventions

### Test File Structure
```
tests/
├── __init__.py                          # Shared fixtures and utilities
├── conftest.py                          # pytest configuration
├── test_{module_name}.py                # Unit tests
├── test_{module_name}_properties.py     # Property-based tests (hypothesis)
├── integration/
│   ├── test_{pipeline}_integration.py   # Full pipeline tests
│   └── test_{workflow}_end_to_end.py    # End-to-end workflows
├── benchmarks/
│   ├── test_{feature}_performance.py    # Performance benchmarks
│   └── test_throughput.py              # Batch processing benchmarks
└── fixtures/
    ├── sample_images/                   # Test images
    ├── sample_videos/                   # Test videos
    └── configs/                         # Test configurations
```

### Pytest Markers

Use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.fast
def test_quick_function():
    """Fast tests run in < 100ms, run during development."""
    pass

@pytest.mark.slow
def test_expensive_operation():
    """Slow tests take > 1 second, run before commits."""
    pass

@pytest.mark.integration
def test_full_pipeline():
    """Integration tests require multiple components."""
    pass

@pytest.mark.requires_gpu
def test_cuda_processing():
    """Tests requiring GPU, skipped on CPU-only systems."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pass

@pytest.mark.requires_ffmpeg
def test_video_processing():
    """Tests requiring FFmpeg, skipped if not installed."""
    if not shutil.which('ffmpeg'):
        pytest.skip("FFmpeg not available")
    pass
```

### Running Tests
```bash
# Fast tests only (development)
pytest -m fast

# All tests except slow
pytest -m "not slow"

# Integration tests
pytest tests/integration/

# Specific test file with verbose output
pytest tests/test_depth_pipeline.py -v

# With coverage
pytest --cov={module_name} tests/

# Parallel execution (if pytest-xdist installed)
pytest -n auto

# Stop on first failure
pytest -x
```

---

## Unit Tests

### Test Class Structure

```python
# tests/test_{module_name}.py
"""
Unit tests for {MODULE_NAME}.

Test coverage:
- Core functionality: {PERCENTAGE}%
- Edge cases: {LIST_EDGE_CASES}
- Error handling: {LIST_ERROR_SCENARIOS}
"""

import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from {module_path} import {ClassOrFunctionToTest}


class Test{ClassName}:
    """Test suite for {ClassName}."""
    
    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a sample test image."""
        image = Image.new('RGB', (512, 512), color='red')
        image_path = tmp_path / "test_image.jpg"
        image.save(image_path)
        return image_path
    
    @pytest.fixture
    def instance(self):
        """Create an instance of the class under test."""
        return {ClassName}(param1="value1", param2="value2")
    
    def test_initialization(self, instance):
        """Test that class initializes correctly with default parameters."""
        assert instance is not None
        assert instance.param1 == "value1"
        assert instance.param2 == "value2"
    
    def test_initialization_with_invalid_params_raises_error(self):
        """Test that invalid initialization parameters raise ValueError."""
        with pytest.raises(ValueError, match="param1 must be"):
            {ClassName}(param1="invalid_value")
    
    def test_basic_processing(self, instance, sample_image):
        """Test basic processing workflow with valid input."""
        result = instance.process(sample_image)
        
        assert result is not None
        assert isinstance(result, {ExpectedType})
        # Add specific assertions about the result
    
    def test_preserves_image_dimensions(self, instance, sample_image):
        """Test that processing preserves image dimensions."""
        original = Image.open(sample_image)
        original_size = original.size
        
        result = instance.process(sample_image)
        
        if isinstance(result, Image.Image):
            assert result.size == original_size
    
    def test_preserves_metadata(self, instance, tmp_path):
        """Test that IPTC/XMP metadata is preserved."""
        # Create image with metadata
        image = Image.new('RGB', (100, 100))
        image.info['dpi'] = (300, 300)
        image.info['custom_field'] = 'test_value'
        
        image_path = tmp_path / "with_metadata.jpg"
        image.save(image_path)
        
        # Process
        result = instance.process(image_path)
        
        # Verify metadata
        if isinstance(result, Image.Image):
            assert result.info.get('dpi') == (300, 300)
            assert result.info.get('custom_field') == 'test_value'
    
    @pytest.mark.parametrize("intensity", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_different_intensity_values(self, intensity, sample_image):
        """Test processing with different intensity parameters."""
        instance = {ClassName}(intensity=intensity)
        result = instance.process(sample_image)
        
        assert result is not None
        # Verify intensity affects output appropriately
    
    @pytest.mark.parametrize("image_size", [
        (10, 10),      # Very small
        (512, 512),    # Standard
        (2048, 2048),  # 2K
        (4096, 4096),  # 4K
    ])
    def test_various_image_sizes(self, instance, image_size, tmp_path):
        """Test that processing works with various image sizes."""
        width, height = image_size
        image = Image.new('RGB', (width, height), color='blue')
        image_path = tmp_path / f"test_{width}x{height}.jpg"
        image.save(image_path)
        
        result = instance.process(image_path)
        
        assert result is not None
    
    @pytest.mark.parametrize("mode", ['RGB', 'RGBA', 'L'])
    def test_different_image_modes(self, instance, mode, tmp_path):
        """Test processing with different image color modes."""
        image = Image.new(mode, (100, 100), color='red' if mode != 'L' else 128)
        image_path = tmp_path / f"test_{mode}.png"
        image.save(image_path)
        
        result = instance.process(image_path)
        
        assert result is not None
    
    def test_missing_input_file_raises_error(self, instance):
        """Test that missing input file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            instance.process(Path("/nonexistent/file.jpg"))
    
    def test_invalid_file_format_raises_error(self, instance, tmp_path):
        """Test that invalid file format raises appropriate error."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("not an image")
        
        with pytest.raises((ValueError, IOError)):
            instance.process(invalid_file)
    
    def test_corrupted_image_raises_error(self, instance, tmp_path):
        """Test that corrupted image data raises appropriate error."""
        corrupted = tmp_path / "corrupted.jpg"
        corrupted.write_bytes(b"fake image data")
        
        with pytest.raises((ValueError, IOError)):
            instance.process(corrupted)


class Test{FunctionName}:
    """Test suite for {function_name} function."""
    
    def test_basic_functionality(self):
        """Test basic function behavior with valid inputs."""
        result = {function_name}(param1="value", param2=42)
        
        assert result == expected_value
    
    def test_edge_case_zero_input(self):
        """Test function with zero/empty input."""
        result = {function_name}(param1="", param2=0)
        
        assert result is not None  # Or specific expected behavior
    
    def test_edge_case_negative_values(self):
        """Test function with negative numeric values."""
        result = {function_name}(param1="value", param2=-10)
        
        # Define expected behavior for negative inputs
        assert result >= 0  # Example assertion
    
    def test_invalid_type_raises_typeerror(self):
        """Test that passing invalid type raises TypeError."""
        with pytest.raises(TypeError):
            {function_name}(param1=123, param2="should_be_int")
    
    def test_out_of_range_raises_valueerror(self):
        """Test that out-of-range values raise ValueError."""
        with pytest.raises(ValueError, match="must be between"):
            {function_name}(param1="value", param2=999)
```

---

## Integration Tests

### Full Pipeline Test

```python
# tests/integration/test_{pipeline}_integration.py
"""
Integration tests for {PIPELINE_NAME} full workflow.

Tests the complete pipeline from input to output with real data.
"""

import pytest
from pathlib import Path
from PIL import Image
import yaml

from {pipeline_module} import {PipelineClass}


class Test{Pipeline}Integration:
    """Integration tests for {PIPELINE_NAME}."""
    
    @pytest.fixture
    def config_file(self, tmp_path):
        """Create a test configuration file."""
        config = {
            'depth_model': {'name': 'vits', 'backend': 'cpu'},
            'output': {'format': 'jpg', 'quality': 90},
        }
        
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    @pytest.fixture
    def sample_images(self, tmp_path):
        """Create sample test images."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        
        for i in range(3):
            image = Image.new('RGB', (512, 512), 
                            color=(i*80, 100, 200-i*60))
            image.save(images_dir / f"test_{i}.jpg")
        
        return images_dir
    
    def test_single_image_processing(self, config_file, sample_images, tmp_path):
        """Test processing a single image through the full pipeline."""
        pipeline = {PipelineClass}.from_config(config_file)
        
        input_image = list(sample_images.glob("*.jpg"))[0]
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        result = pipeline.process(input_image, output_dir)
        
        assert result.success
        assert result.output_path.exists()
        assert result.processing_time_ms > 0
        
        # Verify output image
        output_image = Image.open(result.output_path)
        assert output_image.size == (512, 512)
    
    @pytest.mark.slow
    def test_batch_processing(self, config_file, sample_images, tmp_path):
        """Test batch processing multiple images."""
        pipeline = {PipelineClass}.from_config(config_file)
        
        output_dir = tmp_path / "batch_output"
        output_dir.mkdir()
        
        results = pipeline.batch_process(sample_images, output_dir)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        assert all(r.output_path.exists() for r in results)
        
        # Verify all outputs
        output_images = list(output_dir.glob("*.jpg"))
        assert len(output_images) == 3
    
    def test_pipeline_with_all_features_enabled(self, tmp_path, sample_images):
        """Test pipeline with all features/processors enabled."""
        config = {
            'depth_model': {'name': 'vits', 'backend': 'cpu'},
            'tone_mapping': {'enabled': True},
            'denoising': {'enabled': True, 'strength': 0.5},
            'clarity': {'enabled': True, 'strength': 0.2},
            'sharpening': {'enabled': True, 'amount': 0.5},
        }
        
        config_path = tmp_path / "full_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        pipeline = {PipelineClass}.from_config(config_path)
        
        input_image = list(sample_images.glob("*.jpg"))[0]
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        result = pipeline.process(input_image, output_dir)
        
        assert result.success
        assert result.output_path.exists()
    
    def test_error_recovery_invalid_image(self, config_file, tmp_path):
        """Test that pipeline handles invalid images gracefully."""
        pipeline = {PipelineClass}.from_config(config_file)
        
        # Create invalid image file
        invalid_image = tmp_path / "invalid.jpg"
        invalid_image.write_bytes(b"not an image")
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Should not crash, should report error
        result = pipeline.process(invalid_image, output_dir)
        
        assert not result.success
        assert result.error_message is not None
```

---

## Property-Based Tests (Hypothesis)

```python
# tests/test_{module}_properties.py
"""
Property-based tests using Hypothesis.

These tests verify that certain properties always hold across a wide
range of randomly generated inputs.
"""

import pytest
from hypothesis import given, strategies as st, assume
import numpy as np
from PIL import Image

from {module_path} import {function_or_class}


@given(
    width=st.integers(min_value=10, max_value=2048),
    height=st.integers(min_value=10, max_value=2048),
    intensity=st.floats(min_value=0.0, max_value=1.0),
)
def test_output_size_matches_input_size(width, height, intensity):
    """Property: Output image size should always match input size."""
    # Create random input image
    image = Image.new('RGB', (width, height), color='red')
    
    # Process
    processor = {ClassName}(intensity=intensity)
    result = processor.process(image)
    
    # Property: size preserved
    assert result.size == (width, height)


@given(
    image_array=st.lists(
        st.lists(
            st.tuples(
                st.integers(0, 255),  # R
                st.integers(0, 255),  # G
                st.integers(0, 255),  # B
            ),
            min_size=10, max_size=100
        ),
        min_size=10, max_size=100
    ),
    strength=st.floats(min_value=0.0, max_value=1.0)
)
def test_pixel_values_stay_in_valid_range(image_array, strength):
    """Property: Processed pixel values should stay in [0, 255]."""
    # Convert list to numpy array
    arr = np.array(image_array, dtype=np.uint8)
    image = Image.fromarray(arr, mode='RGB')
    
    # Process
    result = {function_name}(image, strength=strength)
    result_array = np.array(result)
    
    # Property: values in valid range
    assert result_array.min() >= 0
    assert result_array.max() <= 255


@given(
    intensity=st.floats(min_value=0.0, max_value=1.0)
)
def test_zero_intensity_preserves_input(intensity):
    """Property: Zero intensity should return unmodified input."""
    assume(intensity == 0.0)  # Only test when intensity is exactly 0
    
    image = Image.new('RGB', (100, 100), color=(123, 45, 67))
    
    processor = {ClassName}(intensity=intensity)
    result = processor.process(image)
    
    # Property: output equals input when intensity is 0
    assert np.array_equal(np.array(result), np.array(image))


@given(
    depth_values=st.lists(
        st.floats(min_value=0.0, max_value=1.0),
        min_size=100, max_size=1000
    )
)
def test_depth_normalization_properties(depth_values):
    """Property: Normalized depth should be in [0, 1] with full range."""
    depth_array = np.array(depth_values, dtype=np.float32)
    
    # Skip degenerate cases
    assume(depth_array.max() > depth_array.min())
    
    normalized = {normalize_depth_function}(depth_array)
    
    # Properties
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert np.isclose(normalized.min(), 0.0, atol=1e-6)
    assert np.isclose(normalized.max(), 1.0, atol=1e-6)
```

---

## Performance Benchmarks

```python
# tests/benchmarks/test_{feature}_performance.py
"""
Performance benchmarks for {FEATURE_NAME}.

Measures processing time, memory usage, and throughput.
"""

import pytest
import time
import psutil
import os
from PIL import Image
import numpy as np

from {module_path} import {ClassName}


class Test{Feature}Performance:
    """Performance benchmarks for {FEATURE}."""
    
    @pytest.fixture
    def process_monitor(self):
        """Monitor process memory usage."""
        process = psutil.Process(os.getpid())
        return process
    
    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_processing_time_2k_image(self, process_monitor):
        """Benchmark: Processing time for 2K image."""
        image = Image.new('RGB', (2048, 2048), color='blue')
        processor = {ClassName}()
        
        # Warmup
        _ = processor.process(image)
        
        # Benchmark
        iterations = 10
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            result = processor.process(image)
        
        elapsed_time = time.perf_counter() - start_time
        avg_time_ms = (elapsed_time / iterations) * 1000
        
        print(f"\nAvg processing time (2K): {avg_time_ms:.2f}ms")
        
        # Assert performance target
        assert avg_time_ms < 100, f"Too slow: {avg_time_ms:.2f}ms > 100ms"
    
    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_processing_time_4k_image(self):
        """Benchmark: Processing time for 4K image."""
        image = Image.new('RGB', (4096, 4096), color='green')
        processor = {ClassName}()
        
        start_time = time.perf_counter()
        result = processor.process(image)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        print(f"\nProcessing time (4K): {elapsed_ms:.2f}ms")
        
        assert elapsed_ms < 500, f"Too slow for 4K: {elapsed_ms:.2f}ms"
    
    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_memory_usage(self, process_monitor):
        """Benchmark: Memory usage during processing."""
        processor = {ClassName}()
        
        # Baseline memory
        baseline_mb = process_monitor.memory_info().rss / 1024 / 1024
        
        # Process large image
        image = Image.new('RGB', (4096, 4096), color='red')
        result = processor.process(image)
        
        # Peak memory
        peak_mb = process_monitor.memory_info().rss / 1024 / 1024
        memory_increase_mb = peak_mb - baseline_mb
        
        print(f"\nMemory increase: {memory_increase_mb:.2f}MB")
        
        # Assert memory target (< 500MB for 4K image)
        assert memory_increase_mb < 500, \
            f"Memory usage too high: {memory_increase_mb:.2f}MB"
    
    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_batch_throughput(self, tmp_path):
        """Benchmark: Batch processing throughput (images/second)."""
        # Create test images
        num_images = 20
        images = []
        for i in range(num_images):
            img = Image.new('RGB', (1024, 1024), 
                          color=(i*10, 100, 200-i*5))
            img_path = tmp_path / f"test_{i}.jpg"
            img.save(img_path)
            images.append(img_path)
        
        processor = {ClassName}()
        
        start_time = time.perf_counter()
        
        for img_path in images:
            result = processor.process(img_path)
        
        elapsed_sec = time.perf_counter() - start_time
        throughput = num_images / elapsed_sec
        
        print(f"\nThroughput: {throughput:.2f} images/sec")
        print(f"Total time: {elapsed_sec:.2f}s for {num_images} images")
        
        # Assert minimum throughput
        assert throughput > 2.0, \
            f"Throughput too low: {throughput:.2f} images/sec"
```

---

## Mock-Based Tests

```python
# tests/test_{module}_mocked.py
"""
Tests using mocks to isolate components and avoid heavy dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from {module_path} import {ClassName}


class Test{Class}WithMocks:
    """Tests using mocks for heavy dependencies."""
    
    @patch('{module_path}.load_depth_model')
    def test_initialization_loads_model(self, mock_load_model):
        """Test that initialization loads the depth model."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        processor = {ClassName}(model_name="vits")
        
        mock_load_model.assert_called_once_with("vits")
        assert processor.model == mock_model
    
    @patch('{module_path}.subprocess.run')
    def test_ffmpeg_command_execution(self, mock_subprocess):
        """Test that FFmpeg command is executed correctly."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
        
        processor = {ClassName}()
        processor.process_video(Path("input.mp4"), Path("output.mp4"))
        
        # Verify subprocess was called
        mock_subprocess.assert_called_once()
        
        # Verify command structure
        called_args = mock_subprocess.call_args[0][0]
        assert 'ffmpeg' in called_args
        assert 'input.mp4' in str(called_args)
    
    def test_lut_loading_caching(self, monkeypatch, tmp_path):
        """Test that LUT files are cached after first load."""
        # Create fake LUT file
        lut_file = tmp_path / "test.cube"
        lut_file.write_text("FAKE LUT DATA")
        
        load_count = 0
        
        def mock_load_lut(path):
            nonlocal load_count
            load_count += 1
            return np.random.rand(32, 32, 32, 3)
        
        monkeypatch.setattr('{module_path}.load_lut_file', mock_load_lut)
        
        processor = {ClassName}(lut_path=str(lut_file))
        
        # Load LUT twice
        lut1 = processor._get_lut()
        lut2 = processor._get_lut()
        
        # Should only load once (cached)
        assert load_count == 1
```

---

## Repository-Specific Test Patterns

### Pattern 1: Testing with Optional Dependencies

```python
# Gracefully skip tests when optional dependencies unavailable
try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

@pytest.mark.skipif(not TIFFFILE_AVAILABLE, 
                    reason="tifffile not installed")
def test_16bit_tiff_processing():
    """Test processing 16-bit TIFF files (requires tifffile)."""
    # Test code using tifffile
    pass
```

### Pattern 2: Testing Metadata Preservation

```python
def test_preserves_gps_coordinates(tmp_path):
    """Test that GPS coordinates are preserved through processing."""
    from PIL import Image
    import piexif
    
    # Create image with GPS data
    image = Image.new('RGB', (100, 100))
    
    exif_dict = {
        "GPS": {
            piexif.GPSIFD.GPSLatitude: ((34, 1), (3, 1), (0, 1)),
            piexif.GPSIFD.GPSLongitude: ((118, 1), (15, 1), (0, 1)),
        }
    }
    
    exif_bytes = piexif.dump(exif_dict)
    image_path = tmp_path / "with_gps.jpg"
    image.save(image_path, exif=exif_bytes)
    
    # Process
    result = process_image(image_path)
    
    # Verify GPS preserved
    exif_data = piexif.load(result.info.get('exif', b''))
    assert piexif.GPSIFD.GPSLatitude in exif_data["GPS"]
```

### Pattern 3: Testing FFmpeg Integration

```python
@pytest.mark.requires_ffmpeg
def test_ffmpeg_video_processing(tmp_path):
    """Test video processing with FFmpeg."""
    import shutil
    import subprocess
    
    if not shutil.which('ffmpeg'):
        pytest.skip("FFmpeg not available")
    
    # Create test video (using FFmpeg)
    input_video = tmp_path / "test.mp4"
    subprocess.run([
        'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=640x480:rate=30',
        '-pix_fmt', 'yuv420p', str(input_video)
    ], check=True, capture_output=True)
    
    # Process
    output_video = tmp_path / "output.mp4"
    process_video(input_video, output_video, preset="test")
    
    # Verify output exists and is valid
    assert output_video.exists()
    
    # Verify with ffprobe
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-show_streams', str(output_video)
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
```

---

## CI Test Configuration

### pytest.ini

```ini
[pytest]
# Minimum pytest version
minversion = 6.0

# Test discovery patterns
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Markers
markers =
    fast: Fast tests (< 100ms)
    slow: Slow tests (> 1 second)
    integration: Integration tests
    benchmark: Performance benchmarks
    requires_gpu: Tests requiring CUDA/MPS
    requires_ffmpeg: Tests requiring FFmpeg
    requires_tifffile: Tests requiring tifffile

# Ignore directories
norecursedirs = .git .tox build dist *.egg-info deprecated

# Output options
addopts =
    -ra
    --strict-markers
    --tb=short
    --disable-warnings

# Coverage
[coverage:run]
source = .
omit =
    tests/*
    setup.py
    deprecated/*
```

### Run in CI

```yaml
# .github/workflows/test.yml snippet
- name: Run fast tests
  run: pytest -m fast --cov --cov-report=xml

- name: Run full test suite
  run: pytest -m "not benchmark" --cov --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

---

**Template Version**: 1.0  
**Last Updated**: 2025-11-06  
**Maintained By**: Transformation Portal RAG System
