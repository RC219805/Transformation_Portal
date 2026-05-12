# CLI Documentation

Last updated: 2026-05-12

This directory contains maintained command-line references for the current
Transformation Portal repository. Point-in-time CLI implementation reports live
under `docs/historical/cli/` and are retained as audit evidence only.

## Current Surfaces

| Document | Disposition | Scope |
| --- | --- | --- |
| [LUX_DEPTH_V3_CLI_GUIDE.md](LUX_DEPTH_V3_CLI_GUIDE.md) | canonical | Lux Depth V3, APEX, DA3/Depth Pro, Materials V3, PBR, run-card, and advisory captioning options. |
| [CLI_REFERENCE.md](CLI_REFERENCE.md) | current-support | Current entrypoint map for repo-local CLIs and validation commands. |
| [PBR_CLI_TESTING_GUIDE.md](PBR_CLI_TESTING_GUIDE.md) | current-support | Focused PBR CLI test and contract guidance. |
| [PBR_CLI_TESTING_QUICK_REF.md](PBR_CLI_TESTING_QUICK_REF.md) | current-support | Short command reference for the PBR CLI test lane. |

## Operator Baseline

Use the repository-managed environment instead of ad-hoc dependency installs:

```bash
source .venv/bin/activate
make install-core
make check-environment
```

When the virtual environment is not activated, invoke console scripts through
`.venv/bin/<command>` or module entrypoints through `.venv/bin/python -m ...`.

## Historical CLI Evidence

The following files were moved out of this live CLI reference directory because
they describe earlier point-in-time implementation or coverage states:

- [CHANGELOG_CLI_v1_3.md](../historical/cli/CHANGELOG_CLI_v1_3.md)
- [PBR_CLI_COVERAGE_REPORT.md](../historical/cli/PBR_CLI_COVERAGE_REPORT.md)
- [PBR_CLI_IMPLEMENTATION_CHECKLIST.md](../historical/cli/PBR_CLI_IMPLEMENTATION_CHECKLIST.md)
