# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Ingest Contract v1.0.1:** Bumped ingest schema version from `1.0.0` to `1.0.1` for metadata normalization semantics hardening.
- **EXIF Normalization Semantics:** `ExifMetadata` now normalizes real-world EXIF string forms (for example `"4.5 mm"` and `"8 8 8"`) into numeric schema types before strict validation.
- **rawpy/libraw Toolchain Capture:** Ingest provenance now tolerates `rawpy` version shape differences (`rawpy.version.version`, `rawpy.__version__`, tuple-style `libraw_version`) without contract shape changes.
- **Contract Surface:** No JSON envelope or field shape changes; machine-mode contract (`tp.meta.machine.v1`) remains unchanged.

### Added
- **Machine-Readable JSON Output Mode (tp.meta.machine.v1):** Deterministic JSON API for metadata CLI automation
  - **`--json` Flag:** Emit structured JSON with stable envelope (schema, command, success, exit_code, data, error)
  - **Deterministic Serialization:** `sort_keys=True` ensures consistent key ordering across runs and platforms
  - **Typed Error Handling:** Structured error objects with exit code enums for programmatic parsing
  - **Exit Code Semantics:** Clear success/failure signaling (0=success, 1-5=specific failure modes)
  - **Per-Command Payloads:** Stable data schemas for `extract`, `validate`, `extract-batch`, `check-system`
  - **Golden Master Tests:** Contract enforcement via byte-exact output validation
  - **CI Contract Gate:** `.github/workflows/machine_mode_contract_validation.yml` blocks schema drift
  - **Reference Parser:** `tools/parse_machine_json.py` (Python) and `tools/parse_machine_json_examples.sh` (bash/jq)
  - **Contract Documentation:** `docs/api/MACHINE_MODE_CONTRACT.md` defines binding guarantees and versioning policy
  - **Optional Pretty-Print:** `--json-pretty` for human-readable JSON (2-space indent)
  - **File Output:** `--json-output <path>` writes JSON to file, keeps stdout clean
  - See: [Machine Mode Contract](docs/api/MACHINE_MODE_CONTRACT.md), PR #1024

### Added
- **FP Probe Version Governance (ADR-030):** Production-grade probe versioning for cross-ISA determinism
  - **Governance Contract:** `probe_version` is now a semantic contract with explicit bump criteria
  - **Locking Test:** `test_probe_version_locked()` enforces conscious version increments
  - **Manifest Schema v3:** Promotes `probe_version` and `probe_policy` to first-class fields in fpstate section
  - **Documentation:** Module docstring documents increment vs. no-increment criteria for probe changes
  - See: `src/transformation_portal/determinism/fp_probe.py` (PROBE VERSION GOVERNANCE section)

- **Phase 5: Material PBR Integration (v5.0.0):** Production-ready physics-based rendering for luxury real estate post-production
  - **Stable Preset (`material_pbr.yaml`):** Deterministic CPU-only heuristic backend, zero ML dependencies
  - **Canary Preset (`material_pbr_canary.yaml`):** Optional PBRFusion GPU backend (Apache 2.0) with auto-fallback
  - **Enhanced Heuristic Backend:** Bilateral filtering, depth-aware normals (5× scale), concavity-based AO (70%/30% blend)
  - **8 PBR-Accurate Material Presets:** Metal, glass, wood, stone, fabric, concrete, plastic, ceramic with validated roughness/metallic ranges
  - **PBR Texture Generation:** 6 maps (albedo, normal, roughness, metallic, AO, height) with `MaterialProperties` metadata
  - **Artifact Fingerprinting:** `PBRGenerationMetadata` tracks backend version, parameters, depth usage for reproducibility
  - **Backend Protocol:** `PBRBackendProtocol` formal interface for Phase 6 Gaussian Splatting integration
  - **CI Preset Stability Guard:** SHA256 hash enforcement prevents unintended stable preset modifications
  - **Performance:** 4.28s/MP @ 12MP (meets <5s/MP Quality Firewall target), <500MB memory
  - **62 Material Tests:** Comprehensive coverage including backend fallback, device placement, contract validation
  - See: [Performance Baselines](docs/performance/PHASE5_PBR_BASELINES.md)

### Removed
- **Archived Obsolete Module:** `depth_canonical` module superseded by ADR-019 backend architecture
  - Moved to `archive/depth_canonical/` with full git history preserved
  - Replacement: `src/transformation_portal/depth/backends/` (implemented in PR #906)
  - Associated tests moved to `archive/depth_canonical_tests/` and `archive/test_depth_canonical_yaml.py`
  - See: `archive/depth_canonical/ARCHIVE_README.md` for migration path and rationale

### Added
- **Ingest Contract v1.0.0:** Audit-grade provenance and schema validation for RAW/TIFF ingest
  - **Versioned Schemas:** Pydantic models for ProvenanceSidecar (v1.0.0) and IngestManifest (v1.0.0)
  - **Complete Metadata Extraction:** exiftool integration captures all EXIF tags + groups
  - **Provenance Capture:** Toolchain versions, git SHA, CLI args, timestamps, host/OS metadata
  - **Deterministic Output:** Sorted JSON keys, stable serialization (except run_id UUID)
  - **Schema Validation:** Hard-fail on missing fields, type mismatches, unknown fields (drift detection)
  - **Quality Firewall:** 8-bit conversion detection, gamma correction detection, dtype/range validation
  - **CI Enforcement:** `.github/workflows/ingest_contract_validation.yml` gates PRs on violations
  - **Atomic Writes:** Temp file + rename pattern prevents corruption
  - **30 Comprehensive Tests:** Schema validation, drift detection, determinism, gamma/8-bit checks
  - **Exit Codes:** 0=pass, 1=schema_fail, 2=8bit, 3=gamma, 4=drift, 5=other
  - **Contract Documentation:** `docs/apex/ingest_contract.md` defines binding guarantees
  - **Dependencies:** Added pydantic>=2.0 to core requirements
  - See: [Ingest Contract v1.0.0](docs/apex/ingest_contract.md)

### Fixed
- **CRITICAL: Lux Depth V3 Pipeline Bug Fixes (6 issues):**
  - **Fix #1:** Double EXIF rotation in v2_enhance.py - Strip EXIF data after `exif_transpose()` to prevent viewers from rotating twice (pixels already rotated + EXIF tag says rotate again)
  - **Fix #2:** Dimension mismatch in preprocessing/orchestrator - Resize depth maps back to original dimensions after multiple-of-14 padding/cropping required by Depth Anything V3
  - **Fix #3:** Quadratic complexity in batch_stats.py - Pre-compute median once for outlier detection (O(n²) → O(n log n) for large batches)
  - **Fix #4:** Redundant processing in parallel mode - Pass pre-computed paths to avoid duplicate manifest reads and hash computation (~15-20% I/O reduction)
  - **Fix #5:** Alpha channel safety in v2_enhance.py - Resize alpha channel if V2 processing changes resolution to prevent shape mismatch crashes
  - **Fix #6:** Output directory trap in input_discovery.py - Explicitly exclude output_dir when scanning to prevent processing own outputs
  - Impact: Data integrity (EXIF, dimensions, alpha), performance (batch stats, parallel I/O), robustness (output exclusion)
  - Tests: 15 new regression tests, all 83 lux_depth_v3 tests passing
  - See: [CRITICAL_FIXES_SUMMARY.md](CRITICAL_FIXES_SUMMARY.md)

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
- **PBR Texture Generation API (`generate_pbr_textures`)** now returns a `PBRTextures` dataclass instead of a 7-tuple
  - Old tuple-unpacking call sites now raise: `TypeError: cannot unpack non-iterable PBRTextures object`
  - Migrate to attribute access (`result.albedo`, `result.normal`, etc.)
  - This entry is authoritative for release behavior and supersedes earlier draft "zero breaking changes" wording
  - See: [Material PBR Migration Guide](docs/guides/MATERIAL_PBR_MIGRATION.md)
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
