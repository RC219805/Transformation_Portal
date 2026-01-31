# Copilot Instructions - Transformation Portal

You are working in a production-oriented image/video processing repository. Optimize for correctness, repeatability, and safe change management.

## Primary Goals
1) Preserve stable contracts and preset behavior (v2.0.0 "Golden Path").
2) Keep CI green: fast feedback, no large-model downloads in tests, minimal disk usage.
3) Make changes easy to review: small PRs, clear diffs, updated docs/tests.

## Repo Map (where things go)
- `src/` - installable package code (preferred for new production logic)
- `scripts/` - operational runners / orchestration scripts (thin wrappers)
- `config/` - YAML presets and configuration
- `assets/` - LUTs and look assets
- `docs/` - architecture, guides, reports, changelogs
- `tools/` - developer/ops utilities (audits, manifests, dashboards)
- `tests/` - pytest suite

Avoid creating new root-level scripts unless there is a strong reason.

## Coding Standards
- Python 3.10+
- Keep lines ≤ 127 chars
- Prefer `pathlib.Path`, type hints, dataclasses for configs
- Keep functions single-purpose; separate CLI parsing from business logic
- Add docstrings for non-trivial algorithms (tone mapping, depth logic, material detection)
- Validate inputs early (fail fast) before long work (model load, IO, FFmpeg execution)

## Tests & CI Constraints (important)
- CI runs:
  - lint on Python 3.12
  - core tests on Python 3.10 and 3.12
  - ML tests on Python 3.11
- CI uses `requirements-lint.txt` for lint and `requirements-ci.txt` for tests.
- ML tests must not download models; honor `TRANSFORMERS_OFFLINE=1` patterns.

When adding tests:
- Default to unit tests with mocks for FFmpeg, file IO, and model inference.
- Mark heavy tests as `ml` or `slow` and ensure they're isolated.
- Keep deterministic outputs (seed randomness where relevant).

## Presets & Versioning
- Presets have stability taxonomy: stable / canary / experimental.
- If you add or modify presets:
  - Document intent + expected impact
  - Add/adjust tests that validate preset parameters
  - Keep preset naming consistent and human-meaningful

Version alignment rules:
- Contract schema version, package version, and runtime `__version__` should remain aligned for contract-impacting changes.

## Dependency Changes
- Root requirement files are convenience pins:
  - `requirements.txt`, `requirements-ci.txt`, `requirements-dev.txt`, `requirements-lint.txt`
- Maintainers update source-of-truth in `requirements/*.in` and recompile outputs:
  - `cd requirements && make compile`
- Do not add large ML dependencies to core runtime unless strictly necessary.

## Documentation Expectations
Any change that affects user workflows must update:
- README (if public-facing behavior changes)
- Relevant docs in `docs/`
- Examples (if they exist for that workflow)

## Before You Finalize
- Run `pytest -v tests/ -ra -m "not ml and not slow" --maxfail=1` for core behavior (mirrors CI)
- Run `pytest -v tests/ -ra -m "ml and not slow" --maxfail=1` only when ML dependencies are installed
- Ensure linting does not regress (`flake8`, `pylint` where applicable)
- Keep diffs tight and reviewable
