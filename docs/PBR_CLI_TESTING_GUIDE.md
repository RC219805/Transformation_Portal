# PBR CLI Testing Guide

## Overview

The PBR CLI (`transformation_portal.lux_depth_v3.pbr_cli`) now has comprehensive test coverage across three tiers:

1. **Unit/CLI Tests** (`tests/test_pbr_cli.py`) - Fast, run in CI
2. **Stress Tests** (`tests/stress/test_stress_large_batch.py`) - Slower, on-demand
3. **Integration Tests** (existing depth_canonical tests)

## Running Tests

### Quick Test (CI Equivalent)

Run the fast CLI tests that execute in CI:

```bash
# All CLI tests
pytest tests/test_pbr_cli.py -v

# Specific test class
pytest tests/test_pbr_cli.py::TestValidInvocations -v

# Specific test
pytest tests/test_pbr_cli.py::TestValidInvocations::test_single_file_basic -v
```

### Stress Tests

Stress tests validate behavior under load (100+ images, memory monitoring):

```bash
# Run all stress tests
pytest tests/stress/ -v -m stress

# Run specific stress test
pytest tests/stress/test_stress_large_batch.py::TestLargeBatchProcessing::test_100_image_batch -v

# Run with output (see performance metrics)
pytest tests/stress/ -v -s -m stress

# Skip very slow tests
pytest tests/stress/ -v -m "stress and not slow"
```

### Full Test Suite

Run all tests including ML and slow tests:

```bash
# All tests
pytest tests/ -v

# Exclude stress tests
pytest tests/ -v -m "not stress"

# Only fast tests (exclude ML and stress)
pytest tests/ -v -m "not ml and not stress and not slow"
```

## Test Coverage

### CLI Tests (`test_pbr_cli.py`)

**Coverage**: 80%+ of CLI code paths

**Test Classes**:
- `TestValidInvocations` - Valid CLI usage patterns
- `TestEdgeCases` - Boundary conditions and corner cases
- `TestErrorHandling` - Error recovery and graceful failures
- `TestOutputValidation` - Output correctness verification
- `TestParameterValidation` - Parameter type/range validation
- `TestExitCodes` - Proper exit code behavior

**Key Test Cases**:
- ✅ Single file processing with all presets
- ✅ Batch directory processing
- ✅ Custom parameter overrides
- ✅ Mixed-case file extensions (.JPG, .PNG, .JPEG)
- ✅ Nonexistent paths (graceful error)
- ✅ Empty directories (clear message)
- ✅ Corrupt files (per-file error, batch continues)
- ✅ Output directory auto-creation
- ✅ Invalid preset names (helpful error)
- ✅ Parameter validation (types, ranges)
- ✅ Exit codes (0=success, 1=error)

### Stress Tests (`test_stress_large_batch.py`)

**Coverage**: Performance, scalability, resource management

**Test Classes**:
- `TestLargeBatchProcessing` - 100+ image batches
- `TestPerformanceBenchmarks` - Throughput and timing
- `TestResourceLimits` - Resource constraint handling

**Key Test Cases**:
- ✅ 100-image batch processing
- ✅ Memory usage bounded (no runaway growth)
- ✅ Repeated batches (no memory leaks)
- ✅ Mixed file sizes (realistic workload)
- ✅ Partial failures (batch continues)
- ✅ Throughput by preset (draft < standard < premium)
- ✅ 4K image handling
- ✅ Empty batch graceful failure

## Performance Baselines

Established performance baselines for regression detection:

### Single Image (1024×1024)
- **Draft preset**: ~0.5-1.0s
- **Standard preset**: ~0.8-1.5s
- **Premium preset**: ~1.0-2.0s

### Batch Processing (100 images, mixed sizes)
- **Expected throughput**: 50-100+ images/sec (draft preset)
- **Memory growth**: < 2GB for 20× 2048×2048 images

### Quality Ordering
Draft ≤ Standard ≤ Premium (processing time)

## CLI Robustness Improvements

### P1 Enhancements Implemented

1. **Input Validation**
   - ✅ Check if input path exists before processing
   - ✅ Validate preset names early with helpful errors
   - ✅ Clear error messages for common mistakes

2. **Output Directory Handling**
   - ✅ Auto-create output directory with `parents=True, exist_ok=True`
   - ✅ Supports nested paths (`output/nested/dir`)

3. **File Extension Support**
   - ✅ Case-insensitive extensions (`.jpg`, `.JPG`, `.jpeg`, `.JPEG`, `.png`, `.PNG`, `.npy`, `.NPY`)
   - ✅ Multiple extensions in batch mode

4. **Error Handling**
   - ✅ Per-image try/except in batch mode
   - ✅ Batch continues on individual failures
   - ✅ Summary reports success/error counts
   - ✅ List of failed files with error details
   - ✅ Proper exit codes (0=all success, 1=any error)

5. **User Feedback**
   - ✅ Clear progress indication
   - ✅ Detailed batch summary
   - ✅ Failed file listing with reasons

## Common Testing Patterns

### Testing CLI with Fixtures

```python
def test_my_cli_scenario(cli_runner, sample_depth_npy, tmp_path):
    """Test description."""
    output_dir = tmp_path / "output"

    result = cli_runner.invoke(app, [
        "generate",
        "--depth", str(sample_depth_npy),
        "--preset", "premium",
        "--output", str(output_dir),
    ])

    assert result.exit_code == 0
    assert "Generated PBR maps" in result.stdout
    assert output_dir.exists()
```

### Available Fixtures

- `cli_runner` - Typer CLI test runner
- `sample_depth_npy` - 512×512 .npy depth file
- `sample_depth_png` - 512×512 .png depth file
- `sample_depth_batch` - Directory with 5 mixed depth files
- `empty_directory` - Empty directory for edge cases
- `corrupt_depth_file` - Invalid file for error testing

### Creating Custom Test Data

```python
# Create synthetic depth map
depth = np.random.rand(1024, 1024).astype(np.float32)
depth_path = tmp_path / "test_depth.npy"
np.save(depth_path, depth)

# Create structured depth (gradient)
x, y = np.meshgrid(np.linspace(0, 1, 512), np.linspace(0, 1, 512))
depth = (x + y) / 2
```

## Troubleshooting Tests

### Test Failures

```bash
# Run single failing test with verbose output
pytest tests/test_pbr_cli.py::TestName::test_name -v -s

# Show local variables on failure
pytest tests/test_pbr_cli.py -l

# Stop on first failure
pytest tests/test_pbr_cli.py -x
```

### Memory Tests Failing

Memory tests require `psutil`:

```bash
pip install psutil
```

Skip memory tests if not available:

```bash
pytest tests/stress/ -v -k "not memory"
```

### Stress Tests Too Slow

Reduce batch sizes in stress tests or skip:

```bash
# Skip slow stress tests
pytest tests/stress/ -v -m "stress and not slow"
```

## CI Integration

### Current CI Coverage

CLI tests are included in the standard test suite:

```yaml
# .github/workflows/ci.yml
- name: Run tests
  run: |
    pytest tests/ -v -m "not ml and not stress and not slow"
```

### Optional: Nightly Stress Tests

To add nightly stress testing:

```yaml
# .github/workflows/nightly.yml
name: Nightly Stress Tests
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily
  workflow_dispatch:  # Manual trigger

jobs:
  stress-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest psutil
      - name: Run stress tests
        run: |
          pytest tests/stress/ -v -m stress --tb=short
```

## Test Maintenance

### Adding New Tests

1. **CLI tests**: Add to `tests/test_pbr_cli.py`
2. **Stress tests**: Add to `tests/stress/test_stress_large_batch.py`
3. Mark appropriately: `@pytest.mark.stress`, `@pytest.mark.slow`
4. Use existing fixtures where possible
5. Document expected behavior

### Updating Baselines

If performance characteristics change (e.g., optimization):

1. Run stress tests: `pytest tests/stress/ -v -s -m stress`
2. Review performance output
3. Update baseline assertions if justified
4. Document changes in commit message

### Test Dependencies

- **Core**: `pytest`, `typer`, `Pillow`, `numpy`
- **Optional**: `psutil` (for memory tests)
- **ML**: Not required for CLI tests (mocked)

## Known Limitations

1. **Case-insensitive extensions**: Now supported (.JPG, .PNG, etc.)
2. **Output directory**: Auto-created (no longer a limitation)
3. **Batch failures**: Continue processing (no longer a limitation)
4. **ML model mocking**: CLI tests don't load actual models (intentional for speed)

## Future Enhancements

### v2.1.0 Candidates

- [ ] Parallel batch processing (multithreading)
- [ ] Progress bar for large batches (tqdm integration)
- [ ] Resume interrupted batches (checkpoint support)
- [ ] Dry-run mode (validate without processing)
- [ ] JSON output mode (machine-readable results)
- [ ] Preset validation command (`--validate-preset`)

### v2.2.0+ Ideas

- [ ] GPU memory monitoring
- [ ] Distributed batch processing
- [ ] Performance regression CI gates
- [ ] Automatic preset selection based on image analysis
- [ ] Output format options (EXR, TIFF 16-bit)

## References

- **Source**: `src/transformation_portal/lux_depth_v3/pbr_cli.py`
- **Tests**: `tests/test_pbr_cli.py`, `tests/stress/test_stress_large_batch.py`
- **Architecture**: `docs/architecture/ADR-001-PBR-Integration-Architecture.md`
- **Review**: `docs/architecture/V2_0_0_RELEASE_REVIEW.md`
