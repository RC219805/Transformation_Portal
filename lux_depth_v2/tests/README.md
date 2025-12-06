# Lux Depth V2 Test Suite

Comprehensive test suite with 115+ tests covering all core modules.

## Quick Start

```bash
# Run all tests
pytest tests/

# Fast tests only (recommended for development)
pytest tests/ -m "not slow and not gpu"

# With coverage report
pytest tests/ --cov=lux_depth_v2 --cov-report=html
```

## Test Structure

```
tests/
├── conftest.py               # Shared fixtures and pytest configuration
├── test_config.py            # Configuration tests (20 tests)
├── test_torch_ops.py         # Torch operations (35+ tests)
├── test_io_utils.py          # I/O utilities (30+ tests)
├── test_weights.py           # Weight generation (10 tests)
├── test_material_profiles.py # Material profiles (15 tests)
└── test_pipeline.py          # Integration tests (5+ tests)
```

## Test Categories

### Unit Tests
Fast, focused tests for individual functions and classes.

```bash
pytest tests/ -m "unit"
```

### Integration Tests
End-to-end tests for complete workflows.

```bash
pytest tests/ -m "integration"
```

### Slow Tests
Tests that take >5 seconds (large images, ML models).

```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Run only slow tests
pytest tests/ -m "slow"
```

### GPU Tests
Tests requiring CUDA GPU.

```bash
# Skip GPU tests
pytest tests/ -m "not gpu"
```

## Fixtures

Common fixtures available in `conftest.py`:

- **torch_device** - PyTorch device (CUDA if available, else CPU)
- **sample_rgb_array** - 64x64 test RGB image
- **sample_depth_array** - 64x64 test depth map
- **sample_mask_array** - 64x64 test binary mask
- **temp_dir** - Temporary directory for test outputs
- **sample_image_file** - PNG test image file
- **sample_tiff_file** - 16-bit TIFF test file
- **sample_depth_file** - 16-bit depth TIFF file
- **mock_config** - Pre-configured PipelineConfig

## Running Specific Tests

```bash
# Single test file
pytest tests/test_config.py -v

# Single test class
pytest tests/test_config.py::TestPipelineConfig -v

# Single test method
pytest tests/test_config.py::TestPipelineConfig::test_default_values -v

# Tests matching pattern
pytest tests/ -k "test_preset" -v
```

## Coverage

Target: 80%+ code coverage for all modules.

```bash
# Generate HTML coverage report
pytest tests/ --cov=lux_depth_v2 --cov-report=html

# View report
open htmlcov/index.html

# Terminal report with missing lines
pytest tests/ --cov=lux_depth_v2 --cov-report=term-missing
```

## Test Markers

Tests are marked with pytest markers for selective execution:

- **@pytest.mark.slow** - Slow tests (>5 seconds)
- **@pytest.mark.gpu** - Requires CUDA GPU
- **@pytest.mark.integration** - Integration tests
- **@pytest.mark.unit** - Unit tests (default)

## Writing Tests

### Example Unit Test

```python
# tests/test_my_module.py
import pytest
from lux_depth_v2 import my_module

class TestMyFunction:
    """Test my_function behavior."""
    
    def test_basic_case(self):
        """Test basic functionality."""
        result = my_module.my_function(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            my_module.my_function(invalid_input)
```

### Example Integration Test

```python
@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline(temp_dir, sample_image_file, mock_config):
    """Test complete pipeline processing."""
    mock_config.output_dir = temp_dir
    
    pipeline = LuxPipelineV2(mock_config)
    result = pipeline.process_one(sample_image_file)
    
    assert result["status"] == "ok"
```

## Dependencies

Core test dependencies (installed with package):
- pytest>=7.0
- pytest-cov>=4.0

Optional dependencies:
- pytest-xdist (parallel test execution)
- pytest-timeout (test timeouts)
- hypothesis (property-based testing)

Install all dev dependencies:

```bash
pip install -e ".[dev]"
```

## CI/CD Integration

Tests run automatically on:
- Pull requests
- Commits to main branch
- Manual workflow dispatch

CI matrix:
- Python: 3.10, 3.11, 3.12
- Platforms: Linux, macOS, Windows
- Configs: CPU-only, GPU (where available)

## Debugging

### Verbose Output

```bash
pytest tests/test_module.py -vv
```

### Show Print Statements

```bash
pytest tests/ -s
```

### Drop into Debugger on Failure

```bash
pytest tests/ --pdb
```

### Show Slowest Tests

```bash
pytest tests/ --durations=10
```

## Best Practices

1. **Isolation** - Tests should not depend on each other
2. **Mocking** - Mock external resources (files, network, heavy models)
3. **Fast Feedback** - Keep unit tests fast (<1s each)
4. **Clear Names** - Test names should describe what they test
5. **Fixtures** - Reuse common setup via fixtures
6. **Edge Cases** - Test boundary conditions and error paths
7. **Documentation** - Use docstrings to explain complex scenarios

## Test Statistics

- **Total Tests**: 115+
- **Unit Tests**: 90+
- **Integration Tests**: 10+
- **Slow Tests**: 5+
- **Target Coverage**: 80%+
- **Average Test Runtime**: <0.5s (unit tests)
- **Total Suite Runtime**: ~30s (fast tests)

## Support

For test-related issues:
1. Check test output for detailed error messages
2. Run with `-vv` for verbose output
3. Check fixtures in `conftest.py`
4. Review test documentation in `docs/development/testing.rst`
