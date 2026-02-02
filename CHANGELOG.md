# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking Changes
- **Drop Python 3.10 Support:** Minimum required Python version is now 3.11
  - Rationale: Align with ecosystem evolution (scikit-learn 1.8.0 dropped 3.10 support)
  - Impact: Users must upgrade to Python 3.11 or later
  - See: [ADR-020: Drop Python 3.10 Support](docs/architecture/ADR-020-drop-python-3.10.md)

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
