# Governed Scripts Reference

This reference describes the maintained script topology for Transformation
Portal. It is not an exhaustive inventory of every experimental or project-local
helper; use it to find the current governed home for setup, pipeline, utility,
validation, and compatibility entrypoints.

For end-to-end operator commands, use
[PIPELINE_OPERATIONS_GUIDE.md](../pipeline_docs/PIPELINE_OPERATIONS_GUIDE.md).
For script placement enforcement, run
`python3 scripts/governance/check_script_topology.py --verbose`.

## Placement Contract

| Category | Governed home | Examples |
|----------|---------------|----------|
| Setup and runtime installers | `scripts/setup/` | `install_da3_runtime.sh`, `install_depth_pro_runtime.sh`, `install_raw_runtime.sh`, `install_fastvlm_runtime.sh` |
| Pipeline runners and processors | `scripts/pipelines/` | `run_montecito_apex_full.sh`, `process_source_tiffs_apex.sh`, `lux_render_pipeline.py`, `run_aerial_enhancement.py`, `run_depth_estimation.py` |
| Utilities and local tools | `scripts/utilities/` | `visualize_material_assignments.py`, `luxury_video_master_grader.py`, `create_board_textures.py` |
| Validation, CI, and governance gates | `scripts/validation/`, `scripts/governance/`, `scripts/ci/` | `check_script_topology.py`, `lint_runner.sh`, `check_per_package_coverage.py` |
| Path-stable contract tools | `tools/` | `verify_phase4_chain.py`, `build_machine_evidence.py`, `run_apex_eval.py` |
| Reusable domain code | `src/transformation_portal/` or another package under `src/` | `transformation_portal.pipelines.lux_render_pipeline`, `luxury_tiff_batch_processor.cli` |
| Legacy compatibility wrappers | `scripts/` root wrappers only when contract-preserving | `scripts/run_aerial_enhancement.py`, `scripts/run_depth_estimation.py`, `scripts/visualize_material_assignments.py`, `scripts/synthetic_viewer.py` |
| Contract-bound root entrypoints | `scripts/` root only when lookup is part of a live contract | `scripts/enhance_image.py` |
| Retired broad cleanup tools | `archive/scripts/legacy-organization/` | historical organizers only |

Root-level Python or shell scripts are not the default placement. If a script is
not a documented compatibility wrapper, explicitly grandfathered shell
entrypoint, or contract-bound root entrypoint, move it into the governed
topology or archive it with evidence.

## Environment Setup

Use repo-managed setup before running scripts:

```bash
make venv
make install-core
make check-environment
```

Optional runtimes use governed installers:

```bash
make install-ml-core
make install-ml-sam2
./scripts/setup/install_da3_runtime.sh --profile baseline
./scripts/setup/install_depth_pro_runtime.sh
./scripts/setup/install_raw_runtime.sh
./scripts/setup/install_fastvlm_runtime.sh
make check-fastvlm-runtime
```

Do not replace those paths with ad hoc optional dependency installs in current
operator guidance.

## Maintained CLI Entrypoints

Prefer installed console scripts for user-facing workflows:

| Workflow | Entrypoint |
|----------|------------|
| Lux Depth V3 | `.venv/bin/lux-depth-v3` |
| TIFF batch | `.venv/bin/luxury-tiff-batch` |
| Lux render | `.venv/bin/lux_render` |
| Video grading | `.venv/bin/luxury_video_grader` |
| Presence security | `.venv/bin/presence-security` |
| Depth-aware DOF | `.venv/bin/depth-aware-dof` |

Examples:

```bash
.venv/bin/lux-depth-v3 --help
.venv/bin/luxury-tiff-batch --help
.venv/bin/lux_render --help
.venv/bin/luxury_video_grader --help
```

## Setup Scripts

| Script | Purpose |
|--------|---------|
| `scripts/setup/install_da3_runtime.sh` | Install the isolated Depth Anything 3 runtime |
| `scripts/setup/install_depth_pro_runtime.sh` | Install the isolated Apple Depth Pro runtime |
| `scripts/setup/install_raw_runtime.sh` | Install the isolated RAW ingest runtime |
| `scripts/setup/install_fastvlm_runtime.sh` | Install the isolated advisory FastVLM runtime |
| `scripts/setup/ensure_node_version.sh` | Enforce the expected Node runtime for frontdoor workflows |
| `scripts/setup/run_repo_python.sh` | Run Python using repo-root path bootstrapping |
| `scripts/setup/pre-commit-check.sh` | Check root placement policy |
| `scripts/setup/download_samples.py` | Canonical sample fixture downloader; public wrapper remains `scripts/download_samples.py` |
| `scripts/setup/download_sam2_checkpoint.py` | Canonical SAM2 checkpoint downloader; public wrapper remains `scripts/download_sam2_checkpoint.py` |

Compatibility wrappers such as `scripts/install_models.py` and
`scripts/download_depth_models.py` exist only to preserve old public paths; the
canonical implementations live under `scripts/setup/`.

## Pipeline Scripts

Use pipeline scripts for production or project-specific processing flows that
are not exposed as console scripts:

| Script | Purpose |
|--------|---------|
| `scripts/pipelines/run_montecito_apex_full.sh` | Full Montecito APEX workflow |
| `scripts/pipelines/run_montecito_apex_lean.sh` | Lean Montecito APEX workflow |
| `scripts/pipelines/process_source_tiffs_apex.sh` | Source TIFF APEX processing |
| `scripts/pipelines/hdr_production_pipeline.sh` | HDR production processing |
| `scripts/pipelines/run_fixity_cycle.sh` | Archive fixity cycle |
| `scripts/pipelines/run_sealed_eval_72h.sh` | Sealed evaluation run |
| `scripts/pipelines/run_aerial_enhancement.py` | Aerial enhancement canonical implementation |
| `scripts/pipelines/lux_render_pipeline.py` | Compatibility script for the package Lux Render pipeline |
| `scripts/pipelines/depth_pro_export.py` | Canonical Apple Depth Pro export implementation; public wrapper remains `scripts/depth_pro_export.py` |
| `scripts/pipelines/run_depth_estimation.py` | Canonical APEX V2 depth-estimation wrapper; public wrapper remains `scripts/run_depth_estimation.py` |
| `scripts/pipelines/process_750_picacho_elite.sh` | Canonical 750 Picacho elite quick-start runner; public wrapper remains `scripts/process_750_picacho_elite.sh` |
| `scripts/pipelines/process_750_picacho_elite_batch.sh` | Canonical 750 Picacho elite batch runner; public wrapper remains `scripts/process_750_picacho_elite_batch.sh` |

For Lux Render operator use, prefer `.venv/bin/lux_render` and the `--input-glob`
flag.

## Utility Scripts

Utility scripts provide local inspection, conversion, visualization, and support
tasks. They should not own reusable domain logic if that logic belongs in
`src/`.

| Script | Purpose |
|--------|---------|
| `scripts/validation/check_image_processing_readiness.py` | Canonical image-processing readiness checker; public wrapper remains `scripts/check_image_processing_readiness.py` |
| `scripts/utilities/simple_image_processor.py` | Canonical minimal Pillow/numpy image operations; public wrapper remains `scripts/simple_image_processor.py` |
| `scripts/utilities/visualize_material_assignments.py` | Material-assignment visualization |
| `scripts/utilities/luxury_video_master_grader.py` | Compatibility wrapper for the video grader implementation |
| `scripts/utilities/create_board_textures.py` | Generate board texture assets |
| `scripts/utilities/depth_tools.py` | Depth utility operations |

If a utility becomes shared application behavior, move the reusable portion into
`src/transformation_portal/` and keep only a thin CLI wrapper under `scripts/`.

## Validation And Governance Scripts

| Script | Purpose |
|--------|---------|
| `scripts/governance/check_script_topology.py` | Enforce script placement and wrapper compatibility |
| `scripts/governance/check_docs_structure.py` | Enforce documentation structure |
| `scripts/governance/check_stale_docs_paths.py` | Detect stale documentation paths |
| `scripts/ci/lint_runner.sh` | Canonical shared flake8/pylint policy runner; public wrapper remains `scripts/lint_runner.sh` |
| `scripts/ci/local_ci_check.sh` | Canonical local CI simulation; public wrapper remains `scripts/local_ci_check.sh` |
| `scripts/ci/analyze_flakes.py` | Canonical test-flake ledger analyzer; public wrapper remains `scripts/analyze_flakes.py` |
| `scripts/ci/track_test_flakes.py` | Canonical pytest JSON flake tracker; public wrapper remains `scripts/track_test_flakes.py` |
| `scripts/ci/apex/` | Canonical APEX CI, dashboard, gate, ledger, policy, and contract verifier implementations; public wrappers remain `scripts/apex_*.py` |
| `scripts/validation/check_ml_test_isolation.sh` | Canonical ADR-031 ML test isolation check; public wrapper remains `scripts/check_ml_test_isolation.sh` |
| `scripts/validation/validate_dependency_constraints.sh` | Canonical ADR-032 dependency constraint validation; public wrapper remains `scripts/validate_dependency_constraints.sh` |
| `scripts/validation/test_v2_integration.sh` | Canonical Lux Depth V3 V2 integration validation; public wrapper remains `scripts/test_v2_integration.sh` |
| `scripts/validation/validate_ci_config.py` | Canonical implementation for CI workflow validation; public wrapper remains `scripts/validate_ci_config.py` |
| `scripts/validation/validate_depth_pro_checkpoint.py` | Canonical Depth Pro checkpoint validator; public wrapper remains `scripts/validate_depth_pro_checkpoint.py` |
| `scripts/validation/validate_ingest_contract.py` | Canonical implementation for ingest sidecar contract validation; public wrapper remains `scripts/validate_ingest_contract.py` |
| `scripts/validation/validate_path_filters.py` | Canonical implementation for CI path-filter validation; public wrapper remains `scripts/validate_path_filters.py` |
| `scripts/validation/validate_pbr_phase5d.py` | Canonical Phase 5D PBR material validation script; public wrapper remains `scripts/validate_pbr_phase5d.py` |
| `scripts/validation/validate_phase1_optimizations.py` | Canonical Phase 1 optimization validator; public wrapper remains `scripts/validate_phase1_optimizations.py` |
| `scripts/validation/validate_phase2.py` | Canonical Phase 2 implementation validator; public wrapper remains `scripts/validate_phase2.py` |
| `scripts/validation/parse_workflows.py` | Canonical workflow parser and validator; public wrapper remains `scripts/parse_workflows.py` |
| `scripts/validation/security_scan.sh` | Canonical CI-aligned Bandit security scan; public wrapper remains `scripts/security_scan.sh` |
| `scripts/validation/validate_phase6a.sh` | Canonical Phase 6A Gaussian rasterizer validation script; public wrapper remains `scripts/validate_phase6a.sh` |
| `scripts/validation/check_unicode_controls.py` | Detect bidirectional Unicode and format-control characters |
| `scripts/validation/check_portal_asset_budgets.py` | Validate portal asset budgets |
| `scripts/validation/validate_metadata_extraction.py` | Canonical implementation for the metadata extraction validation CLI; public wrapper remains `scripts/test_metadata_extraction.py` |
| `scripts/validation/validate_luxury_estate_pipeline.py` | Validate luxury-estate pipeline dependencies, optional runtimes, presets, and source fixtures |
| `scripts/validation/validate_luxury_estate_pipeline_fixes.py` | Validate shadow-clipping, AI padding, and depth-model accessibility fixes for the luxury-estate pipeline |
| `scripts/validation/validate_depth_phase1_complete.py` | Validate Depth Anything V2 phase-1 processing on the 750 Picacho fixture and write diagnostics under `/tmp` |
| `scripts/validation/validate_lux_depth_v3_16bit_output.py` | Validate Lux Depth V3 16-bit output and bit-depth manifests |
| `scripts/validation/verify_lux_depth_v3_16bit_handoff.py` | Verify Lux Depth V3 16-bit TIFF handoff format |
| `scripts/validation/validate_portal_lux_materials_live.py` | Live Lux materials validation |
| `scripts/ci/check_per_package_coverage.py` | Per-package coverage guard |
| `scripts/ci/check_cold_zone_touched_files.py` | Cold-zone coverage guard |
| `scripts/verification/verify_setup.py` | Canonical setup verification implementation; public wrapper remains `scripts/verify_setup.py` |
| `scripts/verification/verify_depth_pro.py` | Canonical Apple Depth Pro runtime verifier; public wrapper remains `scripts/verify_depth_pro.py` |
| `scripts/verification/verify_lux_depth_v3_surface.py` | Canonical Lux Depth V3 surface-contract verifier; public wrapper remains `scripts/verify_lux_depth_v3_surface.py` |
| `scripts/verification/verify_performance_ledger_fixes.py` | Canonical performance ledger fix verifier; public wrapper remains `scripts/verify_performance_ledger_fixes.py` |
| `scripts/verification/verify_run_card_integrity.py` | Canonical run-card integrity verifier; public wrapper remains `scripts/verify_run_card_integrity.py` |

## Analysis And Benchmark Scripts

| Script | Purpose |
|--------|---------|
| `scripts/analysis/architectural_context_extractor.py` | Canonical architectural PDF context extractor; public wrapper remains `scripts/architectural_context_extractor.py` |
| `scripts/analysis/extract_architectural_context.py` | Canonical lightweight architectural PDF context extractor; public wrapper remains `scripts/extract_architectural_context.py` |
| `scripts/analysis/benchmark_phase2.py` | Canonical Phase 2 optimization benchmark; public wrapper remains `scripts/benchmark_phase2.py` |
| `scripts/analysis/benchmark_phase3.py` | Canonical Phase 3 advanced optimization benchmark; public wrapper remains `scripts/benchmark_phase3.py` |
| `scripts/analysis/diagnose_pipeline_stage_color_balance.py` | Diagnose stage-level color balance on the 750 Picacho fixture and write visual diagnostics under `/tmp` |
| `src/transformation_portal/analyzers/codebase_philosophy_auditor.py` | Canonical codebase philosophy auditor; public wrapper remains `scripts/codebase_philosophy_auditor.py` |
| `src/transformation_portal/analyzers/decision_decay_dashboard.py` | Canonical decision-decay dashboard; public wrapper remains `scripts/decision_decay_dashboard.py` |
| `src/transformation_portal/analyzers/temporal_evolution.py` | Canonical temporal evolution roadmap helpers; public wrapper remains `scripts/temporal_evolution.py` |

`scripts/enhance_image.py` remains a direct root implementation because
`transformation_portal.lux_depth_v3.v2_runner.V2Runner` intentionally resolves
that exact subprocess path.

## Maintenance Scripts

| Script | Purpose |
|--------|---------|
| `scripts/maintenance/auto_fix_quality.py` | Canonical local quality auto-fixer; public wrapper remains `scripts/auto_fix_quality.py` |
| `scripts/maintenance/deprecate_docs.py` | Canonical duplicate-doc deprecation helper; public wrapper remains `scripts/deprecate_docs.py` |
| `scripts/maintenance/migrate_imports.py` | Canonical import migration helper; public wrapper remains `scripts/migrate_imports.py` |
| `scripts/maintenance/organize_docs.sh` | Canonical documentation organizer; public wrapper remains `scripts/organize_docs.sh` |
| `scripts/maintenance/pre_commit_hook.sh` | Canonical manual quality-gate hook runner; public wrapper remains `scripts/pre_commit_hook.sh` |

Recommended local placement proof:

```bash
python3 scripts/governance/check_script_topology.py --verbose
./.auto-organize.sh --check --verbose
```

## Compatibility Wrapper Rules

Compatibility wrappers must be thin and deterministic:

- Python wrappers import from the canonical implementation.
- Shell wrappers `exec` the canonical implementation.
- Bootstrap `src/` or the repo root when raw-checkout imports require it.
- Preserve exit codes with `raise SystemExit(main())` for Python CLI wrappers
  that return status integers, or `exec` for shell wrappers.
- Re-export canonical symbols only when tests or documented imports require it.
- Avoid duplicating implementation logic in the wrapper.

The topology checker enforces the current wrapper contracts.

## Current References

- [Quick Start Cheat Sheet](QUICKSTART_CHEATSHEET.md)
- [Pipeline Operations Guide](../pipeline_docs/PIPELINE_OPERATIONS_GUIDE.md)
- [Setup Guide](../guides/SETUP_GUIDE.md)
- [Lux Depth V3 CLI Guide](../cli/LUX_DEPTH_V3_CLI_GUIDE.md)
- [Documentation Map](../governance/DOCUMENTATION_MAP.md)
