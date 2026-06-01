# Pipeline Test Fixtures

This directory contains **synthetic test fixtures** for pipeline integration testing.

## Purpose

These fixtures enable testing of image processing pipelines without requiring:
- Large production files in version control
- Actual client data (privacy/confidentiality)
- External data dependencies

## Structure

```
tests/fixtures/pipelines/
└── 750_picacho_lane/
    └── input/
        ├── 750Picacho_Pool_UltraQuality.tif
        └── 750Picacho_GreatRoom_UltraQuality.tif
```

The `750_picacho_lane` fixture directory is a legacy test path retained for
stable fixture references. Use `750_picacho` for new repository paths.

## Fixture Characteristics

### Technical Specifications
- **Format**: 16-bit RGB TIFF
- **Dimensions**: 800×600 pixels
- **Size**: ~2.8 MB per file
- **Color Profile**: sRGB with realistic interior lighting gradients

### Why These Work for Testing
1. **Bit Depth**: 16-bit matches production workflow requirements
2. **Format**: TIFF preserves full precision through pipeline stages
3. **Size**: Small enough for CI/CD, large enough for realistic processing
4. **Pattern**: Gradients expose color casts and tone mapping issues

## Creating New Fixtures

Use the fixture generator:

```bash
python scripts/utilities/create_test_fixtures.py
```

### Guidelines for Test Fixtures
- **Keep them small**: Target < 5MB per file
- **Use synthetic data**: Never commit client images
- **Represent edge cases**: Test boundary conditions (dark/bright, saturated/neutral)
- **Document purpose**: Explain what each fixture tests

## Using Fixtures in Tests

### Recommended Pattern

```python
from pathlib import Path

# Repository-scoped path (works from any directory)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "pipelines" / "750_picacho_lane" / "input"

def test_pipeline():
    test_image = FIXTURE_PATH / "750Picacho_Pool_UltraQuality.tif"

    if not test_image.exists():
        pytest.skip(f"Test fixture not found: {test_image}")

    # Process with pipeline...
```

### Error Handling

Always handle missing fixtures gracefully:
- Use `pytest.skip()` for optional integration tests
- Provide clear error messages indicating which fixture is missing
- Document how to regenerate fixtures

## Maintenance

### When to Update Fixtures
- Pipeline format requirements change (e.g., 32-bit float needed)
- New edge cases discovered (HDR, wide gamut, extreme aspect ratios)
- Performance benchmarks need larger/smaller reference files

### Size Constraints
- **Per-file maximum**: 5 MB (enforced by pre-commit hooks)
- **Total directory limit**: 50 MB
- Consider downsampling or JPEG 2000 if larger samples needed

## Migration from `projects/`

**Historical Context**: Previously, test scripts referenced files in `projects/750_picacho_lane/Final_Production_UltraQuality/`, which were:
1. Client-specific production files (not appropriate for version control)
2. Very large (~176MB total) causing repository bloat
3. Blocked by `.gitignore`, breaking clean checkouts

**Solution**: Synthetic fixtures in `tests/fixtures/` provide:
- ✅ Version-controlled test data
- ✅ No client data exposure
- ✅ Fast CI/CD execution
- ✅ Reproducible test environments

## Related Documentation

- `docs/testing/pipeline_testing.md` - Pipeline test strategies
- `CONTRIBUTING.md` - Test fixture contribution guidelines
- `.gitignore` - What files are excluded from version control

## Questions?

If fixtures are missing or broken:
1. Run `python scripts/utilities/create_test_fixtures.py`
2. Check `.gitignore` isn't blocking fixtures
3. See maintainer guidelines in `CONTRIBUTING.md`

---

**Important**: These are **synthetic test fixtures only**. Never commit actual client images or production files to version control.
