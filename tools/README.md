# Tools Directory

`tools/` contains repository-owned CLIs that are intentionally invoked by path
from docs, tests, workflows, or Make targets. Do not treat this directory as a
catch-all for one-off scripts: setup, pipeline runners, validation gates, and
reusable application code have governed homes under `scripts/` and `src/`.

For the broader script placement contract, see
[`docs/reference/SCRIPTS_REFERENCE.md`](../docs/reference/SCRIPTS_REFERENCE.md).

## Current Tool Groups

| Group | Representative files | Notes |
|-------|----------------------|-------|
| Phase 4 archive, provenance, and evidence CLIs | `build_machine_evidence.py`, `verify_phase4_chain.py`, `archive_hash_manifest.py` | Path-stable tools referenced by contracts, ADRs, and workflow/docs commands. |
| Cross-runtime and schema parity checks | `check_bundle_root_cross_runtime.py`, `check_governance_export_cross_runtime.py`, `validate_evalsuite_contract_schemas.py` | Keep under `tools/` while workflow and contract docs invoke these exact paths. |
| Portal evidence and telemetry helpers | `portal_rum_summary.py`, `portal_modernization_evidence.py`, `portal_telemetry_retention.py` | Operator evidence helpers documented by portal modernization and retention runbooks. |
| APEX evaluation helpers | `run_apex_eval.py`, `audit_apex_assets.py`, `characterize_apex_model_families.py` | Validation/evaluation commands referenced by APEX protocols. |
| Benchmarks | `benchmark_depth_backends.py`, `benchmark_unified_luxury_batch_io.py` | Evidence-producing benchmark CLIs; default behavior must remain opt-in and contract-stable. |
| AD editorial post-production | `ad_editorial_post_pipeline.py`, `sample_config.yml` | Maintained operator tool; guide lives in `docs/guides/AD_EDITORIAL_POST_PIPELINE.md`. |
| Deprecated AD snapshots | `deprecated/` | Historical only; not current operator entrypoints. |
| Investigation helpers | `investigations/` | Scoped diagnostics with local README context; promote or archive when no longer investigatory. |

## Placement Rules

- Keep a tool here only when its public path is already part of docs, tests,
  workflows, ADRs, or operator runbooks.
- Put setup and runtime installers in `scripts/setup/`.
- Put production pipeline runners in `scripts/pipelines/`.
- Put validation and governance gates in `scripts/validation/`,
  `scripts/governance/`, or `scripts/ci/`.
- Put reusable application or library logic under `src/`; keep `tools/` files
  thin when they wrap shared behavior.
- Move obsolete or point-in-time scripts to `archive/scripts/` or
  `tools/deprecated/` with evidence instead of leaving them as active tools.

## AD Editorial Pipeline

Current maintained entrypoint:

```bash
python tools/ad_editorial_post_pipeline.py run --config my_project.yml -vv
```

Start from the operator guide:
[`docs/guides/AD_EDITORIAL_POST_PIPELINE.md`](../docs/guides/AD_EDITORIAL_POST_PIPELINE.md)

The v2/v3 implementations and their tests are retained under
[`tools/deprecated/`](deprecated/) for historical audit only.
