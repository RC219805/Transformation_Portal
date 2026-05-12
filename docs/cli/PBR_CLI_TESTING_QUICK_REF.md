# PBR CLI Testing Quick Reference

Last updated: 2026-05-12

Run these commands from the repository root with the managed virtual environment
available.

## Required Setup

```bash
source .venv/bin/activate
make check-environment
```

## Fast Contract Lane

```bash
.venv/bin/python -m pytest -q tests/test_pbr_cli.py tests/test_pbr_cli_contract.py
```

Targeted examples:

```bash
.venv/bin/python -m pytest -q tests/test_pbr_cli.py::TestValidInvocations
.venv/bin/python -m pytest -q tests/test_pbr_cli_contract.py::TestCLIExitCodes
.venv/bin/python -m pytest -q tests/test_pbr_cli_contract.py::TestCLIManifest
```

## Stress Lane

```bash
.venv/bin/python -m pytest -q tests/stress/test_stress_large_batch.py -m stress
.venv/bin/python -m pytest tests/stress/test_stress_large_batch.py -m stress -s
```

Stress tests are marked `stress` and `slow`; run them on demand when batch scale
or resource behavior is part of the change.

## Debugging

```bash
.venv/bin/python -m pytest tests/test_pbr_cli.py -vv --tb=long -x
.venv/bin/python -m pytest tests/test_pbr_cli_contract.py -vv -s
```

## Docs Validation

```bash
git diff --check
make check-docs
make check-stale-docs
make check-doc-heading-links
python3 scripts/governance/check_docs_structure.py --all
```

## Current PBR CLI Shape

```bash
.venv/bin/python -m transformation_portal.lux_depth_v3.pbr_cli generate \
  --depth output/frame_depth.npy \
  --output output/pbr \
  --preset premium \
  --manifest output/pbr/manifest.json \
  --json
```

Batch mode:

```bash
.venv/bin/python -m transformation_portal.lux_depth_v3.pbr_cli generate \
  --depth-dir output/depth \
  --output output/pbr \
  --pattern "*_depth.*" \
  --preset standard \
  --max-files 25
```

Current presets: `standard`, `premium`, `draft`, `wood`, `metal`, `glass`,
`stone`, `fabric`.

Full guide: [PBR_CLI_TESTING_GUIDE.md](PBR_CLI_TESTING_GUIDE.md)
