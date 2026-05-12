# PBR CLI Testing Guide

Last updated: 2026-05-12

This guide covers the maintained PBR CLI test surface for
`transformation_portal.lux_depth_v3.pbr_cli`. Historical coverage and
implementation reports were moved to `docs/historical/cli/` and should not be
treated as current CI evidence.

## Test Lanes

| Lane | Command | Purpose |
| --- | --- | --- |
| Focused PBR CLI contract | `.venv/bin/python -m pytest -q tests/test_pbr_cli.py tests/test_pbr_cli_contract.py` | Fast behavior and contract coverage for the Typer CLI. |
| Stress/on-demand | `.venv/bin/python -m pytest -q tests/stress/test_stress_large_batch.py -m stress` | Large-batch, resource, and throughput checks. |
| Repo fast lane | `make test-fast` | Governed fast test lane used by local CI validation. |
| Docs governance | `make check-docs && make check-stale-docs && make check-doc-heading-links` | Ensures CLI documentation paths stay current. |

Run from the repository root with the managed virtual environment available:

```bash
source .venv/bin/activate
make check-environment
.venv/bin/python -m pytest -q tests/test_pbr_cli.py tests/test_pbr_cli_contract.py
```

## Focused CLI Coverage

The focused PBR CLI tests cover:

- Valid single-file and directory invocations.
- Supported presets and dynamic help output.
- Custom parameter overrides.
- Output directory creation and output naming.
- Case-insensitive depth file discovery.
- User-facing errors for missing, invalid, or conflicting inputs.
- Batch behavior when individual files fail.
- Exit-code contracts.
- JSON output and manifest generation.
- Dry-run and max-file guardrails.
- Overwrite and no-overwrite behavior.

Useful targeted commands:

```bash
.venv/bin/python -m pytest -q tests/test_pbr_cli.py::TestValidInvocations
.venv/bin/python -m pytest -q tests/test_pbr_cli_contract.py::TestCLIExitCodes
.venv/bin/python -m pytest -q tests/test_pbr_cli_contract.py::TestCLIManifest
```

For failure triage:

```bash
.venv/bin/python -m pytest tests/test_pbr_cli.py -vv --tb=long -x
.venv/bin/python -m pytest tests/test_pbr_cli_contract.py -vv -s
```

## Stress Tests

Stress tests are marked `stress` and `slow`; they are not a replacement for the
fast contract lane.

```bash
.venv/bin/python -m pytest -q tests/stress/test_stress_large_batch.py -m stress
.venv/bin/python -m pytest tests/stress/test_stress_large_batch.py -m stress -s
.venv/bin/python -m pytest -q tests/stress/test_stress_large_batch.py::TestLargeBatchProcessing::test_100_image_batch
```

Use stress results as operational evidence for batch behavior, resource growth,
and throughput trends. Do not promote old timing numbers from historical reports
as current performance guarantees without rerunning the lane on the target
machine.

## Fixture Patterns

The PBR CLI tests use Typer's test runner and synthetic depth assets:

```python
def test_example(cli_runner, sample_depth_npy, tmp_path):
    output_dir = tmp_path / "output"
    result = cli_runner.invoke(
        app,
        [
            "generate",
            "--depth",
            str(sample_depth_npy),
            "--preset",
            "premium",
            "--output",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0
    assert output_dir.exists()
```

Common fixtures include:

- `cli_runner`: Typer CLI test runner.
- `sample_depth_npy`: synthetic `.npy` depth file.
- `sample_depth_png`: synthetic `.png` depth file.
- `sample_depth_batch`: directory of mixed depth files.
- `empty_directory`: empty batch input directory.
- `corrupt_depth_file`: invalid file for error-path coverage.

## CLI Contract Under Test

The current PBR helper command is:

```bash
.venv/bin/python -m transformation_portal.lux_depth_v3.pbr_cli generate \
  --depth output/frame_depth.npy \
  --output output/pbr \
  --preset premium \
  --manifest output/pbr/manifest.json \
  --json
```

Batch mode uses `--depth-dir` and optional `--pattern`, `--recursive`,
`--max-files`, `--dry-run`, and `--fail-fast`.

Current presets:

- `standard`
- `premium`
- `draft`
- `wood`
- `metal`
- `glass`
- `stone`
- `fabric`

## Related References

- [CLI_REFERENCE.md](CLI_REFERENCE.md)
- [PBR_CLI_TESTING_QUICK_REF.md](PBR_CLI_TESTING_QUICK_REF.md)
- [LUX_DEPTH_V3_CLI_GUIDE.md](LUX_DEPTH_V3_CLI_GUIDE.md)
- [Historical PBR CLI coverage report](../historical/cli/PBR_CLI_COVERAGE_REPORT.md)
- [Historical PBR CLI implementation checklist](../historical/cli/PBR_CLI_IMPLEMENTATION_CHECKLIST.md)
