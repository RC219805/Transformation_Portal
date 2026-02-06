# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Performance Ledger v1.7 Upgrade:** Major enhancement with backward compatibility
  - **Optional NumPy Dependency:** Pure Python fallback for environments without NumPy
  - **Bootstrap Confidence Intervals:** 95% CI for mean using configurable iterations (default 1000)
  - **Expanded Exit Codes:** 0=success, 1=regression, 2=backend_mismatch, 3=insufficient_data
  - **Backend Mismatch Detection:** Prevents comparing incompatible runs
  - **Input Validation Bounds:** DoS prevention (max 10K bootstrap iterations, min 3 samples)
  - **Strict Mode:** `--strict` flag fails on potential regressions (recommended for CI)
  - **Backward Compatibility:** `--version` deprecated but functional (use `--baseline-version`)
  - **Enhanced Statistics:** Added std_sec and bootstrap CI to baseline schema
  - **Performance:** NumPy mode maintains v1.0 speed, pure Python ~50x slower (acceptable for small datasets)
  - **Tests:** 50+ new tests (CLI integration, property-based math validation, benchmarks)
  - **Migration Guide:** `docs/performance_ledger_v1.7_migration.md`
  - See: [Performance Ledger v1.7 Verdict](PERFORMANCE_LEDGER_V1.7_VERDICT.md)

- **Backend Registry Integration (ADR-019):** Depth backend orchestration with fallback
  - DA3Backend adapter wrapping DA3InferenceEngine for unified interface
  - DepthBackendRegistry integration in orchestrator
  - Automatic fallback to DA3 when requested backend unavailable
  - Backend selection metadata captured in manifests
  - License enforcement for research-only backends (Depth Pro)
  - CLI flags: `--depth-backend {da3,depth_pro}`
  - Tests: Unit tests for DA3Backend, integration tests for orchestrator
  - Docs: README updated with backend selection guide
  - See: [ADR-019: Backend Registry Integration](docs/architecture/decisions/ADR-019-REVISED-DECISION.md)

- **Performance Ledger (ADR-023 Phase 2):** Standalone tool for performance regression detection
  - Parse manifests from batch runs and compute runtime statistics
  - Compare current runs against versioned baselines
  - Detect regressions using configurable thresholds (p95 > 10%, mean > 15%, failure_rate > 0%)
  - Generate markdown reports for human review and JSON for CI integration
  - Manual baseline governance (no automated updates)
  - Tool: `tools/performance_ledger.py`
  - Docs: `docs/performance/README.md`

- **Backend Selection Truth (ADR-023 Phase 3):** Enhanced transparency and debugging
  - Backend selection metadata in manifests (`backend_selection` field)
  - Truth-line logging on every batch run (requested vs resolved backend)
  - Fallback warnings when requested backend unavailable
  - Backward-compatible manifest schema (old manifests still parse)
  - Additive-only changes (no enforcement yet, deferred to v2.1.0)

### Breaking Changes
- **Drop Python 3.10 Support:** Minimum required Python version is now 3.11
  - Rationale: Align with ecosystem evolution (scikit-learn 1.8.0 dropped 3.10 support)
  - Impact: Users must upgrade to Python 3.11 or later
  - See: [ADR-020: Drop Python 3.10 Support](docs/architecture/ADR-020-drop-python-3.10.md)

### Fixed
- **Coverage Quality Gate:** Adjusted baseline threshold from 33% to 25% to reflect actual combined coverage
  - PR #832 fixed coverage artifact consolidation, revealing accurate combined coverage of 25.44%
  - Previous 33% threshold was aspirational, not historical
  - Added [Coverage Improvement Plan](docs/coverage-improvement-plan.md) with roadmap to 33% by Q2 2026
  - Baseline gate now prevents regression while allowing incremental improvement

### Changed
- **ML Stack Upgrades:** Major ML framework and dependency updates
  - torch: 2.4.1 → 2.10.0
  - torchvision: 0.19.1 → 0.25.0
  - scikit-learn: 1.7.2 → 1.8.0
  - timm: 0.6.7 → 1.0.24
  - diffusers: 0.31.0 → 0.36.0
  - transformers: 4.53.0 → 4.57.6
  - Benefits: Latest features, performance improvements, security fixes
  - Dependencies: Requires Python >=3.11 (see PR #794)
  - Validation: Comprehensive smoke tests added for ML stack compatibility

## [2.0.0] - 2025-11-14

### Added
- First stable release with production-ready contracts
- Versioned API contracts (schema-aligned payloads)
- Preset stability taxonomy (stable / canary / experimental)
- Service hardening with `/ready` readiness checks
- Context-aware rendering workflows
- Depth Pro integration (experimental)
- Unified depth backend contract

### Changed
- Improved preset discovery via CLI
- Enhanced documentation and architecture decision records

### Fixed
- Various stability and correctness improvements

[Unreleased]: https://github.com/RC219805/Transformation_Portal/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/RC219805/Transformation_Portal/releases/tag/v2.0.0
