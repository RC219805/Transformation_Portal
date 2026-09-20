# LuxDepthV4 candidate interface and evaluation

LuxDepthV4 is an opt-in photographic successor candidate. LuxDepthV3 remains the
production baseline and rollback path. A successful candidate run, local unit
tests, synthetic benchmark, or hosted workflow badge does not establish visual
acceptance or production readiness. Representative independent paired runs and
photographic review remain required.

The shared-executor V3 boundary tests preserve existing preprocessing, depth-cache
identity-v3, and quantized-writer behavior with a declared synthetic inference
fixture. They do not prove the full V3 vertical slice: its orchestrator still
interleaves fallback, semantic gates, resizing, Materials, and manifest state.
Extracting and proving that complete slice remains a prerequisite for any V3
executor cutover. V4 explicitly selects its own versioned candidate graph.

## Public boundary

Python distribution version `0.4.0` introduces the additive `lux-depth-v4`
console entrypoint. The V3 entrypoint remains available.

The Python lifecycle is `prepare(LuxDepthV4Request) -> PreparedLuxExecutionV4`,
followed by `run(prepared)`. Preparation freezes source content, the governed
model selection, device, processing configuration, and bounded resource policy.
Execution does not accept a second mutable configuration. Physical output roots
are carried separately, so repeated output directories do not change plan bytes.

The core-owned plan family gains `tp.execution.plan.v2`; V3's
`tp.execution.plan.v1` continues to exist. V4 stage identity is
`tp.execution.identity.v4`, which binds every named input artifact plus source,
runtime and applicable model identity. These are versioned additions, not a
reinterpretation of historical V3 identities or cache entries.

The initial profile uses governed `da3_metric` without synthetic fallback.
Photographic ingest distinguishes the linear-sRGB floating-point master from the
encoded model proxy. Proxy resize/padding geometry and depth units are explicit;
DA3 metric output remains uncalibrated model inference, not measured scene truth.
Final delivery is 16-bit encoded sRGB TIFF with a corresponding profile. Unknown
input color/profile interpretation requires an explicit `srgb` or `linear_srgb`
correction. RAW input requires the governed isolated decoder. Preview maps, when
requested, are visualization artifacts rather than physically measured materials.
TIFF samples must declare the supported full 8-bit or 16-bit integer precision,
or 32-bit floating-point precision. Packed sample widths such as 12-bit require
an explicit precision-preserving conversion before ingest.
The optional `--companions-manifest` interface (Python request field
`companions_manifest`) binds supplied camera calibration and Materials masks to
individual source hashes. Omitted inputs retain explicit abstention and
unavailable meters. Supplied confidence is an advisory material score; it does
not become calibrated model confidence or evidence of physical material truth.

Run from the repository root with the repo-managed Python:

Execution requires the supported native Darwin arm64 parent ML-core profile
(`make install-ml-core`, including the locked `psutil` supervisor), plus the
isolated governed DA3 baseline runtime. Set
`TRANSFORMATION_PORTAL_DA3_PYTHON` or `--runtime-python` to that runtime's Python.
RAW additionally requires `--raw-python` or `TRANSFORMATION_PORTAL_RAW_PYTHON`.
The core-only environment supports resolver-only CPU planning; supervised
execution fails closed when its process monitor is unavailable.

```bash
PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth_v4 \
  --input-dir "$PRIVATE_INPUT_ROOT" --output-dir "$PRIVATE_OUTPUT_ROOT" \
  --input-color srgb --device cpu --plan

PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth_v4 \
  --input-dir "$PRIVATE_INPUT_ROOT" --output-dir "$PRIVATE_OUTPUT_ROOT" \
  --input-color srgb --device mps
```

Set those variables to distinct absolute directories. Choose the input color
from actual source provenance; the example is not permission to label every
TIFF sRGB. Use `auto` when the source has supported unambiguous color metadata.
Apple Silicon is the first evaluation target. Unsupported device/runtime or
input-color states must be reported as unavailable, not silently substituted.
The output directory must be new and its parent must already exist. Existing
attempts are never overwritten or resumed under different executor semantics.

## Optional calibration and Materials

Store the companion manifest and its masks in a private directory separate from
output and cache roots. Paths in the manifest are relative to the image input
root for `inputs[].path` and to the manifest directory for mask paths. Supply
actual SHA-256 values in place of the explanatory placeholders below:

```json
{
  "schema": "tp.lux.companions.v1",
  "inputs": [{
    "path": "photograph.tif",
    "source_sha256": "<source file SHA-256>",
    "calibration": {
      "coordinate_space": "canonical_master",
      "width": 6000, "height": 4000,
      "fx": 4200.0, "fy": 4200.0, "cx": 2999.5, "cy": 1999.5,
      "source": "measured camera calibration and image transform record"
    },
    "materials": {
      "coordinate_space": "canonical_master",
      "masks": {"water": {"path": "water.npy", "sha256": "<mask file SHA-256>"}},
      "confidences": {"water": 0.95}
    }
  }]
}
```

Either `calibration` or `materials` may be omitted for each selected source;
unlisted sources have neither. All geometry is explicitly the full-resolution,
orientation-normalized master frame. Intrinsics from sensor or unrotated EXIF
coordinates must be converted into that frame by the calibration provider.
Wrong dimensions fail before depth inference; V4 never guesses orientation or
focal calibration from a model name. The declared provenance must identify a
measured calibration, rather than an unavailable or estimated value.

Calibration scales each focal axis onto the resized proxy and transforms the
principal point using the pixel-center resize convention. Bottom/right padding
adds no translation. The pinned DA3 metric conversion is the mean effective
proxy focal length multiplied by native depth and divided by 300, following the
[official DA3 FAQ](https://github.com/ByteDance-Seed/Depth-Anything-3#-faq).
The native array remains unchanged. Calibrated runs additionally export
`metric-depth-m.npy` on the model grid and `aligned-metric-depth-m.npy` on the
master grid, recording source/effective intrinsics and the conversion method.
This is calibrated model inference, not independently measured scene distance.
Every run exports `aligned-depth-valid.npy` on the master grid and binds that
mask to the aligned derivatives in `photograph.json`. Remapping excludes invalid
samples from bilinear interpolation and maps validity with nearest-neighbor
sampling. Invalid derivative positions contain zero as a sentinel, never valid
near depth or meters. Depth-dependent finishing leaves these positions unchanged;
independently requested clarity or Materials operations retain their own scope.

Masks must be contiguous NPY v1/v2 arrays with bool or float32 HW samples in
[0,1], exactly matching master geometry. Headers, dimensions, aggregate mask
bytes, links, and content digests are checked before use. Object arrays,
unsupported formats, ambiguous coordinates, stale sources, changed manifests,
and changed masks fail closed. Missing per-material confidence, insufficient
confidence/coverage, and unsupported materials explicitly abstain. Eligible
Materials operations act through an encoded float32 view and apply their delta
to the linear master without 8-bit quantization.

Preparation embeds the complete optional records and exact manifest receipt in
the canonical plan. Execution revalidates them before backend initialization or
output creation. Named stage inputs and identity-v4 bind calibration and mask
semantics; cache hits cannot reuse depth admitted under another calibration.
The default graph remains unchanged when no companion manifest is selected.

## Completed outputs and managed publication

The final `execution-evidence.json` uses the closed
`tp.lux.execution.evidence.v2` schema. It binds the canonical plan, governed worker
runtime, ordered input coverage, cache outcomes, and each artifact's relative
path, byte count and SHA-256. A failed or interrupted run cannot publish a success
generation merely because some image files exist.

`verify_execution_evidence_v2(output_root, expected_plan_sha256=...)` checks the
complete inventory, required photographic artifacts, descriptor source bindings,
output budget, and canonical plan before returning immutable verified records.
It rejects extra files, links, incomplete evidence and changed bytes. This proves
integrity and provenance; it does not establish photographic acceptance.

`await publish_result(result, publisher=publisher, fence=fence)` is an opt-in
adapter to the existing `GenerationPublisher` and admitted `DispatchFence`.
Managed callers must pass their active publisher during preparation and execution:

```python
prepared = prepare(request, publisher=publisher)
# Admit prepared.canonical_plan_bytes and acquire the existing dispatch fence.
result = run(prepared, publisher=publisher)
await publish_result(result, publisher=publisher, fence=fence)
```

Preparation freezes the publisher's file-count, per-file-byte, total-byte and
manifest-byte limits in the canonical plan. It caps `max_output_bytes` at the
publisher's total-byte ceiling while preserving a lower requested budget.
Oversized inventories and per-file/manifest bounds fail before device probing,
backend initialization or output creation. Limits changing after preparation
require a new plan and admission; they cannot change inside an admitted attempt.

The resolver conservatively reserves two batch files and ten files per image,
including optional alpha and confidence that cannot be known without decoding.
Calibration reserves two more files for each calibrated input; preview maps
reserve three more per image. The default publisher's 200-file limit therefore
admits at most 19 images without calibration or previews. Raising the configured
file-count limit does not bypass the independently enforced manifest-size limit.
Per-file bounds use the declared master/proxy pixel limits, so a smaller active
publisher may require a lower `max_pixels` or output-byte budget. Split rejected
batches explicitly into separately prepared and admitted attempts.

Standalone `prepare(request)` / `run(prepared)` retain their 1,024-input and
64-GiB defaults. Their completed outputs cannot be retroactively treated as
managed-admitted results: managed publication requires the frozen publisher
binding, and managed execution requires its publisher before inference begins.
The output root and plan fingerprint must match the fence. The publisher's
optional `expected_file_integrity` mapping then checks its own staged snapshots
against the verified byte counts and hashes, closing the gap between verification
and staging. Existing callers that omit this argument retain their prior API.
Existing record-store lease and generation-visibility rules remain authoritative;
this adapter creates no admission route, queue or scheduler. Production activation
still requires the applicable service-backed fence and publication gates.

## Private corpus freeze

Set `PRIVATE_INPUT_ROOT` to the photographic source directory and
`PRIVATE_EVIDENCE_ROOT` to an absolute directory outside every Git worktree.
Freeze inputs before candidate tuning; the command does not load a model or
transform input pixels:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/validation/evaluate_lux_depth_v4.py freeze \
  --source-root "$PRIVATE_INPUT_ROOT" --root sample-set \
  --baseline-commit "$(git rev-parse HEAD)" \
  --output "$PRIVATE_EVIDENCE_ROOT/corpus.json"
```

Repeat `--root` for each relative directory selected for the evaluation. The
`tp.lux.evaluation.corpus.v1` manifest records content SHA-256, byte size and
available header metadata. Identical bytes have a `duplicate_of` reference;
instrumentation executes each unique content identity once per batch. Every
original file is rehashed before and after each observation, including duplicate
aliases. Same-stem files receive candidate scene groups for RAW/TIFF linkage;
human review must confirm those groups. Scene derivatives never count as
independent timing repetitions. Missing optional RAW header decoding is explicit
and does not constitute RAW ingest acceptance.

The manifest is exclusive-create and private. Changing sources, corpus selection,
or manifest bytes invalidates the experiment. Keep source names, hashes, outputs,
raw observations and photographic review artifacts private; do not add them to
the repository or upload them as ordinary CI artifacts.

## Independent paired evaluation

The harness is instrumentation infrastructure. The supplied runner must derive
receipts from actual execution evidence; it cannot infer success from a subprocess
exit code alone. No V3/V4 production wrapper is automatically selected.

`tp.lux.evaluation.run_spec.v1` requires:

| Field | Meaning |
| --- | --- |
| `corpus_sha256` | SHA-256 of the exact frozen corpus manifest file |
| `experiment` | One of `infrastructure`, `performance`, `model`, `photography` |
| `repeats` | Integer at least 20; number of independent pairs per scenario |
| `scenarios` | Distinct values from `cold`, `warm`, `cache_hit`, `cache_miss` |
| `variants` | Exactly `v3` and `v4`, each containing an `identity` object |
| `timeout_seconds` | Cold command timeout; positive finite seconds, default 3600 |

Each identity contains exactly `implementation_commit` (full Git commit), and
SHA-256 fields `source_sha256`, `plan_sha256`, `runtime_sha256`,
`interpreter_sha256`, `dependency_sha256`, `model_sha256`, `processing_sha256`.
Use measured source/runtime/dependency identities, including uncommitted source
content when applicable; a Git commit alone does not describe a dirty candidate.
Combined runtime identity may differ with implementation source. Interpreter and
dependency identities must match across variants. Model identity may differ only
in a `model` experiment; processing identity may differ only in a `photography`
experiment. Freeze each experiment before tuning and avoid combining model,
photographic and infrastructure changes in one result.

The CLI's cold adapter additionally requires `variants.v3.command` and
`variants.v4.command` as explicit argv arrays. Commands run without a shell, in
alternating V3/V4 then V4/V3 order:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/validation/evaluate_lux_depth_v4.py run \
  --corpus "$PRIVATE_EVIDENCE_ROOT/corpus.json" \
  --spec "$PRIVATE_EVIDENCE_ROOT/spec.json" \
  --output-dir "$PRIVATE_EVIDENCE_ROOT/paired-cold"

PYTHONPATH=src ./.venv/bin/python scripts/validation/evaluate_lux_depth_v4.py compare \
  --run "$PRIVATE_EVIDENCE_ROOT/paired-cold/run.json" \
  --output "$PRIVATE_EVIDENCE_ROOT/comparison.json"
```

The command adapter supplies `TP_LUX_EVAL_RECEIPT`, `TP_LUX_EVAL_OUTPUT_DIR`,
`TP_LUX_EVAL_CORPUS`, `TP_LUX_EVAL_VARIANT`, `TP_LUX_EVAL_SCENARIO`, and
`TP_LUX_EVAL_PAIR_INDEX`. A trusted wrapper prepares and executes the complete
frozen selection, publishes artifacts, then writes the requested receipt path.
The receipt is separate from pipeline-native execution evidence.

`tp.lux.evaluation.receipt.v1` requires `complete: true`, `synthetic: false`, a
non-synthetic `executed_backend`, matching `corpus_sha256` and `identity`, all
unique `input_sha256s`, and `artifacts`. Each artifact has a contained relative
`path`, `sha256` and `kind` (`image` or `evidence`). Each unique input needs an
image artifact carrying `input_sha256`; the batch also needs schema-tagged JSON
execution evidence. Missing, changed, synthetic or incomplete evidence fails.
The receipt also carries `host` with Python `platform.system()`,
`platform.machine()`, `platform.node()`, and `platform.platform()` values under
`system`, `machine`, `node`, and `platform`; both variants must match the
evaluation host. Stateful callbacks receive this frozen mapping as `request.host`.

Python callers use
`run_evaluation(corpus_path, spec, output_dir, runner)` from
`transformation_portal.lux_depth_v4.evaluation`. The callback accepts an immutable
`EvaluationRequest` and returns the receipt mapping. A warm callback must own a
real initialized backend outside the measured call, return `warmup_complete:
true` and retain the same nonempty `session_id` per variant across repetitions.
Cache callbacks must return `cache_authorized: true` and the actual matching
`cache_state` (`hit` or `miss`). The command adapter rejects warm/cache scenarios;
starting a new process repeatedly is not a warm measurement. Absence of a real
stateful runner is an explicit unsupported scenario, not a passed test.

One callback invocation is one complete-batch observation. Cold timing includes
process launch, preparation, initialization, execution and publication. Warm
timing covers the callback's completed batch, excluding the separately owned
warmup. Corpus verification and evidence rehashing are outside the timed call.
Never duplicate a batch duration across image capsules and count them as
independent observations. Retain the per-input pipeline timing separately if the
native execution evidence supplies it.

## Acceptance and limits

Comparison revalidates the exact specification, corpus, each raw observation and
all artifact hashes. It reports independent sample counts, raw seconds, median,
p95 and standard deviation. Existing real APEX relative bands provide descriptive
local verdicts: at most 10% p95 increase is `pass`, over 10% through 15% is `warn`,
and over 15% is `fail`. These are regression tolerances, not a promised speedup.
Exit 1 means incomplete/invalid evidence; comparison exit 2 means a measured
performance failure; exit 0 means valid comparison output, not production
acceptance. A deterministic 10,000-resample paired percentile bootstrap reports
a descriptive 95% interval for the p95 timing delta. Twenty pairs provide an
uncertain tail estimate; this interval does not establish the production
confidence policy or photographic acceptance.

`photographic_review` always remains `required`, `candidate_acceptance` remains
`not_established`, and `automatic_enforcement_eligible` is false. A reviewer must
check precision, color interpretation, orientation, dimensions, architectural
edges, gradients/banding, texture, halos, and shadow/highlight behavior on paired
outputs. Calibrate nondeterminism tolerances on repeated V3 baseline outputs
before tuning V4. A deliberate visual change needs explicit photographic review;
pixel difference alone is neither improvement nor failure.

The existing [APEX enforcement policy](../apex/policy/enforcement_policy.yaml)
requires real data, independent sample sufficiency, confidence, and 30-day baseline
history before automatic production enforcement. The history requirement does not
block immediate local candidate comparisons. Retain private raw evidence and
review references for at least 90 days. The legacy Nightly benchmark invocation,
synthetic APEX push lane, and tiny real APEX fixture lane remain separate evidence
surfaces; this candidate does not turn their badges into photographic acceptance.

Focused harness validation:

```bash
PYTHONPATH=src ./.venv/bin/pytest tests/lux_depth_v4/test_evaluation.py -q
PYTHONPATH=src ./.venv/bin/pytest tests/lux_depth_v4/test_lifecycle.py -q
PYTHONPATH=src ./.venv/bin/pytest tests/lux_depth_v4/test_publication.py tests/orchestrator/test_generation_publisher_unit.py -q
```
