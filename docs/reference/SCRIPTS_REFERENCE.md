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
| Pipeline runners and processors | `scripts/pipelines/` | `run_montecito_apex_full.sh`, `process_source_tiffs_apex.sh`, `lux_render_pipeline.py`, `run_aerial_enhancement.py` |
| Utilities and local tools | `scripts/utilities/` | `visualize_material_assignments.py`, `luxury_video_master_grader.py`, `create_board_textures.py` |
| Validation and governance gates | `scripts/validation/`, `scripts/governance/`, `scripts/ci/` | `check_script_topology.py`, `check_docs_structure.py`, `check_per_package_coverage.py` |
| Reusable domain code | `src/transformation_portal/` or another package under `src/` | `transformation_portal.pipelines.lux_render_pipeline`, `luxury_tiff_batch_processor.cli` |
| Legacy compatibility wrappers | `scripts/` root wrappers only when contract-preserving | `scripts/run_aerial_enhancement.py`, `scripts/visualize_material_assignments.py`, `scripts/synthetic_viewer.py` |
| Retired broad cleanup tools | `archive/scripts/legacy-organization/` | historical organizers only |

Root-level Python or shell scripts are not the default placement. If a script is
not a documented compatibility wrapper, move it into the governed topology or
archive it with evidence.

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

For Lux Render operator use, prefer `.venv/bin/lux_render` and the `--input-glob`
flag.

## Utility Scripts

Utility scripts provide local inspection, conversion, visualization, and support
tasks. They should not own reusable domain logic if that logic belongs in
`src/`.

| Script | Purpose |
|--------|---------|
| `scripts/check_image_processing_readiness.py` | Report current image-processing capability tier |
| `scripts/simple_image_processor.py` | Minimal Pillow/numpy image operations |
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
| `scripts/validation/check_portal_asset_budgets.py` | Validate portal asset budgets |
| `scripts/validation/validate_portal_lux_materials_live.py` | Live Lux materials validation |
| `scripts/ci/check_per_package_coverage.py` | Per-package coverage guard |
| `scripts/ci/check_cold_zone_touched_files.py` | Cold-zone coverage guard |

Recommended local placement proof:

```bash
python3 scripts/governance/check_script_topology.py --verbose
./.auto-organize.sh --check --verbose
```

## Compatibility Wrapper Rules

Compatibility wrappers must be thin and deterministic:

- Import from the canonical implementation.
- Bootstrap `src/` or the repo root when raw-checkout imports require it.
- Preserve exit codes with `raise SystemExit(main())` for CLI wrappers that
  return status integers.
- Re-export canonical symbols only when tests or documented imports require it.
- Avoid duplicating implementation logic in the wrapper.

The topology checker enforces the current wrapper contracts.

## Current References

- [Quick Start Cheat Sheet](QUICKSTART_CHEATSHEET.md)
- [Pipeline Operations Guide](../pipeline_docs/PIPELINE_OPERATIONS_GUIDE.md)
- [Setup Guide](../guides/SETUP_GUIDE.md)
- [Lux Depth V3 CLI Guide](../cli/LUX_DEPTH_V3_CLI_GUIDE.md)
- [Documentation Map](../governance/DOCUMENTATION_MAP.md)
