# Repository Organization

This document describes the high-level structure of the Transformation_Portal repository.

## Top-level layout
- `src/` — Python packages and application code
- `tests/` — Test suite
- `docs/` — Documentation (guides, migration notes, reference)
- `.github/` — CI workflows and repo automation

## Key packages
- `transformation_portal/` — Core library and pipelines
- `transformation_portal/depth_canonical/` — Canonical depth pipeline APIs
- `transformation_portal/lux_depth_v3/` — Deprecated legacy pipeline (kept for migration)

## Notes
This file is intentionally concise; it exists to provide a stable, discoverable orientation point.
