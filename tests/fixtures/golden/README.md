# Golden Test Fixtures

This directory contains the **golden regression test suite** - a small, curated set of fixtures that prevent production regressions.

## Purpose

Golden tests validate that core pipeline behavior remains stable across changes. Each fixture represents a real-world failure mode that has occurred in production.

## Current Fixtures (3)

1. `edge_case_thin_structure.exr` - Thin architectural elements (columns, railings)
2. `low_light_interior.exr` - Shadow detail and exposure recovery
3. `sky_water_gradient.exr` - Smooth gradients and atmospheric depth

See `CURATION.md` for detailed rationale.

## Usage

```bash
# Run golden tests
pytest -m golden

# Update expectations (when intentional changes made)
pytest -m golden --update-golden
```

## Storage

⚠️ **Do NOT commit large EXR files to git**

Fixtures are stored externally:
- Development: `~/test_fixtures/golden/`
- CI: Downloaded from artifact storage during setup

If fixtures are missing, tests will skip with clear message.

## Adding New Fixtures

1. Identify real regression (not arbitrary test)
2. Create minimal fixture (<10MB if possible)
3. Document in `CURATION.md`
4. Link to GitHub issue showing the bug
5. PR must show fixture prevents the regression

**Keep set small** - target 3-10 fixtures total.
