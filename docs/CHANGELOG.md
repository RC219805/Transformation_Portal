# Changelog

## Depth Zones Feature & Dependency Updates — 2026-01-22

### New Features
- **PR #680**: Depth Zones Port - Advanced depth-based zoning system
  - Added `lux_depth_v3/enhance/depth_zones.py` with foreground/midground/background segmentation
  - New CLI tool: `scripts/da3_depth_zones.py` for Depth Anything V3 integration
  - Zone-based processing enables depth-aware effects and selective enhancements
  - Diagnostic outputs: zone masks, preview composites, and depth statistics
  - Supports sky heuristics for far-field saturation handling

### Dependency Updates
- **PR #679**: Bump transformers from 4.57.5 to 4.57.6
- **PR #678**: Bump opencv-python from 4.12.0.88 to 4.13.0.90
- **PR #676**: Bump opencv-python-headless from 4.12.0.88 to 4.13.0.90
- **PR #675**: Bump bandit from 1.9.2 to 1.9.3 (security linting)
- **PR #674**: Bump component-detection-action from 0.1.0 to 0.1.1

### Code Quality Improvements
- **PR #677**: Bump black from 25.12.0 to 26.1.0
  - Applied black 26.1.0 2026 stable style across entire codebase
  - Reformatted 832 files for consistent code formatting
  - Enhanced pre-commit hooks with ruff-format integration

### Status
- **Python Compatibility**: 3.10-3.12 ✅
- **CI Status**: All critical checks passing ✅
- **Test Suite**: 1,348+ tests passing
- **New Capabilities**: Depth zoning for architectural visualization

---

## Dependency Updates & Python 3.10 Compatibility Maintenance — 2026-01-05/06

### Dependency Security & Compatibility Updates
- **PR #658**: Enhanced dependency update workflow with validation gates and artifact uploads
  - Added comprehensive CI validation for automated dependency updates
  - Created `docs/DEPENDENCY_UPDATES.md` with governance guidelines
  - Supersedes PR #656 with improved automation
- **PR #662**: Bump imagecodecs from 2024.12.30 to 2026.1.1
  - New codecs: HTJ2K, MESHOPT, UltraHDR support
  - Updated Cython 3.2 compatibility
  - Bug fixes for TIFF/WebP processing
- **PR #659**: Bump tifffile from 2024.12.12 to 2025.12.20
  - Performance optimizations for large TIFF processing
  - Python 3.10+ compatible, no breaking API changes
- **PR #663**: Bump scikit-learn from 1.7.2 to 1.8.0
  - Constraint updated: `scikit-learn<1.9` (maintains Python 3.10 compatibility)
  - Compatible with existing depth estimation workflows

### Python 3.10 Compatibility Preservation
- **Closed PR #660** (scipy 1.16.3): Requires Python 3.11+, blocks Python 3.10 users
- **Closed PR #661** (Pillow 12.1.0): Requires Python 3.11+, blocks Python 3.10 users
- **Dependabot Configuration**: Added ignore rules for Python 3.11+ dependencies
  - `scipy>=1.16` blocked until Python 3.11 migration
  - `Pillow>=12.0` blocked until Python 3.11 migration
  - Python 3.10 EOL: October 2026 (migration planning in progress)

### Depth Estimation Documentation & Quality Improvements
- **PR #655**: Comprehensive depth estimation capabilities analysis
  - Added `docs/DEPTH_ESTIMATION_ANALYSIS.md` with optimal configuration guide
  - Documented output artifact contracts (PNG/EXR/NPY/PFM formats)
  - Added quality validation metrics and troubleshooting guide
  - Fixed CLI command collisions in `lux_depth_v3/cli.py`
  - Removed unverifiable performance claims, added provable test coverage
  - Created architectural review documentation in `docs/architecture/PR_655_*`

### Infrastructure & Process Improvements
- Created comprehensive PR merge strategy documentation
  - `docs/architecture/PR_MERGE_STRATEGY_2026-01-05.md`
  - `docs/architecture/PR_MERGE_EXECUTION_SUMMARY_2026-01-05.md`
- Established dependency governance framework in `docs/DEPENDENCY_UPDATES.md`

### Architectural Decision Records
- **ADR-005**: Python 3.11 Migration Strategy
  - Proposed phased migration: Preparation (Q1 2026) → Migration (Q2 2026) → Validation
  - Target: Python 3.11 minimum version by May 2026 (5 months before Python 3.10 EOL)
  - Benefits: 10-25% performance improvement, unlock scipy 1.16+/Pillow 12+ updates
  - Migration guide planned for Phase 1 (Feb-March 2026)
  - See: `docs/architecture/adrs/ADR-005-PYTHON-311-MIGRATION.md`

### Current Status
- **Python Compatibility**: 3.10-3.12 ✅
- **CI Status**: All checks passing ✅
- **Open PRs**: 0 (all resolved)
- **Test Suite**: 1,348+ tests passing
- **Planned Migration**: Python 3.11+ (Q2 2026, pending approval)

---

## Comprehensive Codebase Review — 2025-12-04

### Documentation & Status Update
- Created [COMPREHENSIVE_CODEBASE_UPDATE_2025.md](COMPREHENSIVE_CODEBASE_UPDATE_2025.md) documenting:
  - Infrastructure improvements (CI/CD consolidation, security hardening)
  - New capabilities (async pipeline, context-aware rendering, RAG engine)
  - Performance enhancements (3-5x throughput, 92% smaller repo)
  - Codebase structure and test status

### Verified Status
- Test suite: **1,348 passed**, 257 skipped (ML dependencies)
- Linting: **0 critical errors** (flake8), minor pylint suggestions
- Security: CVE-2024-27763 mitigation verified
- CI/CD: Consolidated pipeline operational

---

## Phase 2 Knowledge Engine Activation — 2025-12-01

- RAG cache initialized (21.23 MB, 2,201 chunks, 544 files)
- Vector search enabled (all-MiniLM-L6-v2, 384 dimensions)
- Git hooks operational (post-commit, post-merge, post-checkout, pre-push)
- BM25 + semantic hybrid retrieval active
- PYTHONPATH hook resolution implemented
- Quality Trend Dashboard baseline established

## 2025-10-03

- Enhanced `.github/copilot-instructions.md` with best practice sections
- Added Repository Structure, Getting Started, Troubleshooting sections
- Added Code Examples with practical snippets for common tasks

## 2025-07-04

- Standardized README anchors and terminology for tooling sections

## 2025-07-03

- Reconciled README guidance with merged tooling pull requests

## 2025-07-02

- Added integrated comprehensive dataset for Picacho Lane project
