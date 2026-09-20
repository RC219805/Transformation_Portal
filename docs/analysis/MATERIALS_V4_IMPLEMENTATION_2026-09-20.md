# MaterialsV4 implementation validation and acceptance

**Date:** 2026-09-20

**Status:** Opt-in implementation candidate; production acceptance pending

**Baseline:** `211afde40d09046bb7dba4c3384b38850fd459db`

This report records the implementation and local evidence following the
[MaterialsV3 forensic analysis](MATERIALS_V3_FORENSIC_AND_V4_SUCCESSOR_2026-09-20.md).
It does not authorize a default cutover, calibrated automatic inference, or
production photographic use. The original implementation was recorded on
`codex/materials-v4`; review preparation uses a fresh
`codex/materials-v4-review-ready` branch from the baseline above. Results below
describe the recorded candidate runs, not a final immutable release.

## Implemented candidate

MaterialsV4 is a separate versioned evidence and photographic-response library.
It separates semantic confidence from mask geometry, preserves explicit unknown
and abstention states, binds supplied/inferred evidence to source bytes and
producer identities, and applies bounded linear-sRGB deltas through one
compositor. The planner protects uncertain and conflicting support, recomputes
remaining coverage, and preserves protected samples, alpha, and metadata.

LuxDepthV4 opts in through `--materials-manifest` and the additive
`tp.execution.plan.v3` contract. Complete policy and evidence identities are frozen
before execution; each output publishes the pre-material baseline and response
receipt for independent numerical verification. LuxDepthV3 remains available,
and ordinary LuxDepthV4 requests retain plan V2. No new executor or automatic
model switch is introduced. See the [operator guide](../guides/MATERIALS_V4.md)
and [plan reference](../reference/EXECUTION_PLAN_V3.md).

The local SAM2.1 Hiera Large/OpenCLIP ViT-B-32 experiment uses explicit local
weights and no inference cache. Its uncalibrated rankings cannot authorize
inferred edits. Calibration receipts require explicit policy admission and
matching recipe identities; a receipt alone is not trusted calibration.
The evaluation harness reports unavailable measurements for absent or inadequate
annotations and always leaves acceptance `not_assessed`.

## PR preparation review

The review branch starts from a clean checkout of the baseline above, with the
unrelated Desktop `AGENTS.md` edit excluded. Independent engine, artifact,
inference, integration, and compatibility review found one additional verifier
defect: NumPy reads a declared NPY header before checking its `max_header_size`.
The verifier now bounds the version-specific length prefix before invoking the
NumPy parser, with regression coverage for rejected oversized headers.

Thirty independent numerical engine probes and six malformed-evaluation probes
passed. The inference review confirmed a documented limitation: imported shadow
packages outside recorded distribution files are not bound by the partial
observed runtime digest. That digest does not authorize edits or cache reuse.
The operator guide now makes this limitation explicit, and forensic reproduction
commands accept caller-provided asset and evidence paths.

Fresh-branch validation logs are retained separately at
`/private/tmp/materials-v4-pr-evidence/`. The original implementation evidence
below remains a historical record; it is not hosted CI evidence.

The reviewed branch passed **605 contract tests**, the canonical `make ci`,
pre-commit, mypy across 28 modules, 15 structural/ingest tests, and wheel
verification against all 31 changed packaged source/schema files. The 25-test
receipt-verification suite includes both oversized-header regressions.
Fresh native MPS cold/warm runs independently verified 11 artifacts each,
reported a genuine warm cache hit, and preserved exact cold/warm arrays and
unmasked baseline pixels. They retained 1,280 float32 and 943 uint16 levels;
observed durations were 71.07/54.63 seconds. These remain synthetic engineering
measurements. The source/config/build-metadata inventory was unchanged across
both runs, with digest
`87f30bb056e5687d505c1f41e0f5519c98af16204ec1e56f93e67d287deddc5f`.
See `native-mps/results.json` and `native-mps/README.md` in that evidence directory.

## Proven local validation

Raw logs are retained outside Git under
`/private/tmp/materials-v4-implementation-evidence/`. Private photographs, model
weights, generated previews, and execution artifacts are not included here.

| Check | Recorded outcome | Evidence |
| --- | --- | --- |
| `make test-materials-v4-contract` | 603 passed | `materials-contract.log` |
| Canonical `make ci` | Passed on final executable source, including its configured contract/build checks | `make-ci-final-source.log` |
| `make pre-commit` | Passed | `pre-commit-final.log` |
| Mypy with repository configuration | No issues in 28 checked source modules | `mypy-final.log` |
| Structural and camera-native ingest regressions | 15 passed after version/snapshot update and existing extension build | `core-failure-rechecks.log` |
| Wheel build and packaged import | Version `0.5.0`, MaterialsV4 entrypoint, and packaged plan V3 schema verified | `build-wheel-final.log`, `verify-wheel-final.log` |
| Supplied-evidence CLI workflow | Import, inspect, policy, and V3 `--plan` succeeded without creating an output attempt; empty evaluation returned unavailable/exit 2 | `/private/tmp/materials-v4-docs-wcyoljpp/` |

The final successful canonical run is `make-ci-final-source.log`; earlier logs
retain both passing and failed attempts. The `make ci` target is distinct from
the broader core suite below. These results establish local engineering checks, not all acceptance
requirements or a clean full-suite result.

## Broader core-suite result and triage

The broader run recorded **14,228 passed, 90 skipped, 943 deselected, and three
failures** in `core-suite.log`. No full-suite green result is claimed.

| Failure | Classification and action | Final rerun |
| --- | --- | --- |
| `test_console_scripts_are_frozen_and_version_gated` | The new `materials-v4` entrypoint requires a SemVer minor increase and matching console snapshot. Updated to `0.5.0` and regenerated the console snapshot. | Passed in the 15-test structural/ingest rerun |
| `test_ingest_phase2_rejects_invalid_subprocess_tensor_shape` | The fresh worktree lacked the existing compiled `determinism._fpstate` extension. Built the extension with the repository interpreter; no ingest contract was changed. | Passed in the 15-test structural/ingest rerun |
| `TestBenchmark.test_benchmark_iterations_count` | Existing foundation product bug with an intermittent timing trigger: zero elapsed wall-clock durations cause division by zero when computing throughput. Unchanged by this candidate. | Unresolved baseline defect |

The benchmark module and its tests have no diff from baseline. The exact
baseline module was executed in an isolated external probe: 12 of 500 ordinary
CPU empty-operation benchmarks raised `ZeroDivisionError`, and equal controlled
clock readings reproduced it deterministically. A separate focused run also
failed for the fast nonempty operation `sum(range(100))`. The module SHA-256 was
`22d5114f7fcf97a3ce5a18994224b2c5c82532352f655ff40a54395b02e52de9`
for both baseline and candidate.

Evidence is retained in
`/private/tmp/materials-v4-forensic-engine/performance-monitor-baseline/`, including
`findings.md`, `probe.py`, `probe-result.json`, and `focused-tests.log`.
A separate repair should use monotonic high-resolution elapsed timing and handle
nonpositive measured totals explicitly. A passing retry does not repair this
existing defect. No unrelated foundation source change was made.

## Native and photographic evidence

The recorded native DA3 CPU cold/warm smoke verified **11 artifacts per run**.
The cold run reported one cache miss; the warm run reported one cache hit and no
misses, under the same frozen plan and runtime identity. Both preserved unmasked
float32 samples and verified the separate pre-material baseline. Evidence:
`native-lux/results.json`, `native-lux/native-cpu.log`, and the corresponding
completion manifests. This used a synthetic photographic ramp with supplied
engineering masks. It proves the exercised execution/cache path, not scene
quality, metric-depth accuracy, or performance acceptance.

The final native MPS cold/warm run used version `0.5.0` and observed the actual
`pytorch_mps` backend. It independently verified **11 artifacts per run** and
reported one cold miss followed by one genuine warm hit. Cold/warm master and
delivery arrays were bit identical, with 1,280 float32 levels and 943 uint16
delivery levels. Exactly 640 supplied-mask pixels changed; unmasked baseline
pixels remained exact. The maximum linear delta was `0.0004146695`, below `0.025`.
Observed cold/warm durations were 65.34/52.38 seconds; these tiny synthetic-run
timings are not performance acceptance. Final plan identity:
`974b6d4a4f9f2a9d3866dc4850d94deddb2ad4257f360ed99998110911119d84`.
Evidence and precise source/runtime identities are in
`native-lux/final-mps/results.json` and `native-lux/final-mps/README.md`.

The existing shared DA3 runtime correctly failed its governed dependency check.
It was preserved. Native validation used a separate disposable runtime built
with the supported baseline installer and current lock, without modifying or
forging authority markers. The offline SAM2/CLIP CPU smoke also completed;
its uncalibrated predictions remained nonauthorizing.

The photographic response probe covered **12 cases at three tile sizes each
(36 executions)**: five response families plus protected compositions on one
neutral synthetic fixture and one 512×768 crop from an audited real TIFF.
Tile outputs were bit identical, protected changed-pixel counts were zero,
alpha was preserved in the compositions, and the largest measured absolute
linear-RGB delta was approximately `0.012205`, below the `0.025` policy bound.
Evidence: `photography/validation-summary.json`, `photography/README.md`, receipts,
and retained numeric crop artifacts.

Every material mask and confidence in that probe was a synthetic supplied
engineering fixture, not a material annotation. The actual embedded TIFF ICC
identified Adobe RGB (1998), while evaluation metadata declared ProPhoto RGB
and a linear transfer. The probe honored the embedded profile through an
explicit LittleCMS float32 conversion before ingesting a small recognized-sRGB
crop. Source authoring intent remains unverified; this does not establish native
Adobe RGB/ProPhoto ingest support or photographic-quality acceptance. Display
previews are 8-bit; preservation and delta assertions use float32 linear pixels.
Full-resolution resource behavior was not tested by this crop probe.

## Outstanding acceptance and final reruns

| Requirement | Status at this report |
| --- | --- |
| Final native MPS rerun against the final candidate identities | Passed: cold/warm, 11 verified artifacts each, genuine warm cache hit |
| Final version/snapshot and compiled-extension regression checks | Passed: 15 focused tests |
| Final package/artifact verification after version `0.5.0` | Passed: wheel build and isolated packaged import |
| Final executable source checks | Passed: canonical `make ci`, wheel verification, native MPS and focused regression checks |
| Group-disjoint held-out semantic annotations and calibration | Pending; no calibrated automatic-edit acceptance |
| Representative photographic review and color-provenance resolution | Pending |
| Production-resolution latency, peak memory, and native reproducibility | Pending |
| Default activation or executor cutover | Not authorized by these results |

Numerical integrity, model availability, calibration quality, photographic
preference, and performance acceptance remain separate claims.

## Reproduction commands

Run from the candidate checkout using its governed repository environment and
the official Node 22 distribution on `PATH`. Build the existing native extension
in a fresh checkout before the core suite:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'from setuptools import setup; setup(script_args=["build_ext", "--inplace"])'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src make test-materials-v4-contract
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src make ci
PATH="$PWD/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src make pre-commit
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m mypy --config-file=mypy.ini src/transformation_portal/materials_v4 src/transformation_portal/core/execution_plan_v3.py src/transformation_portal/lux_depth_v4
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/structural/test_structural_integrity.py tests/spatial_ai/ingest/test_phase2_camera_native_linear.py -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/ -q -ra -m '(unit or security or regression or golden or integration) and not ml and not slow and not benchmark'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m build --wheel --no-isolation --outdir /private/tmp/materials-v4-wheel
```

The broader core command retains the known foundation failure above. Native
process-supervision and Metal checks require host execution where the process
and device APIs are available. Exact native and photographic scripts, commands,
inputs, and identities are retained in the external evidence directory.
