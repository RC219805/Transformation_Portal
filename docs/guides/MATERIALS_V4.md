# MaterialsV4 candidate operator guide

MaterialsV4 is an opt-in, versioned material-evidence and photographic-response
library. It runs through the existing LuxDepthV4 executor. LuxDepthV3 remains
available, and ordinary LuxDepthV4 requests still use `tp.execution.plan.v2`.
Selecting `--materials-manifest` produces the additive
[Execution Plan V3 contract](../reference/EXECUTION_PLAN_V3.md).

The candidate separates region geometry from material classification, preserves
unknown and uncertain regions, and measures actual photographic changes. This
interface does not establish production readiness. Representative photographic
review, calibrated inference acceptance, and native Apple Silicon performance
acceptance remain pending. The
[forensic analysis](../analysis/MATERIALS_V3_FORENSIC_AND_V4_SUCCESSOR_2026-09-20.md)
records the reasons for this successor.

The additive `materials-v4` command ships with package version `0.5.0`. The
[implementation validation report](../analysis/MATERIALS_V4_IMPLEMENTATION_2026-09-20.md)
separates passing engineering checks from outstanding production acceptance.

## Prerequisites and private paths

Run commands from the repository root. The evidence tools use the repository
Python and require no model for supplied-mask import, policy generation,
inspection, or evaluation. Lux execution additionally requires the governed
parent environment and isolated DA3 runtime described in the
[LuxDepthV4 reference](../reference/LUX_DEPTH_V4.md).

Set these paths to your private photographic inputs and a working directory
outside the repository. The source-relative path must exactly match the selected
input's path under `PRIVATE_INPUT_ROOT`:

```bash
export PRIVATE_INPUT_ROOT=/absolute/path/to/photographs
export MATERIALS_WORK_ROOT=/absolute/path/to/materials-evidence
export PRIVATE_OUTPUT_ROOT=/absolute/path/to/new-lux-attempt
export SOURCE_RELATIVE_PATH=photograph.tif
export INPUT_COLOR=srgb
mkdir -p "$MATERIALS_WORK_ROOT"
```

Choose `INPUT_COLOR` from source provenance. `auto` requires supported,
unambiguous color metadata; use `srgb` or `linear_srgb` only when justified.
Masks use the full-resolution, orientation-normalized master frame, with shape
`[height, width]`. The evidence CLI uses the photographic raster decoder; it does
not invoke the isolated RAW decoder.

The output attempt and evidence bundle directories must be new, their parents
must exist, and evidence must remain separate from output and cache directories.
Keep private images, masks, annotations, model files, and generated receipts out
of Git.

## Import supplied evidence

Place your reviewed mask at `$MATERIALS_WORK_ROOT/water.npy`. Use a C-contiguous
NPY v1/v2 array with bool or float32 HW values in `[0,1]`, matching the canonical
master. The mask filename is illustrative; choose a label and score supported
by your actual supplied evidence. A numeric supplied score is explicitly caller
assertion, not calibrated model confidence.

The strict `tp.materials.supplied.v1` manifest contains exactly `schema`,
`source_sha256`, `shape`, and `regions`. Each region contains exactly
`region_id`, `label`, `mask_path`, `mask_sha256`, and `semantic_confidence`.
Mask paths are portable relative paths under the supplied-manifest directory;
linked, escaping, malformed, changed, or mismatched inputs fail closed.

Default evidence budgets admit up to 100 million pixels, 512 million decoded
mask bytes in aggregate, and 512 MiB for the encoded bundle. Numeric publication
uses the validated mask geometry, bounded NPY header, and remaining bundle
budget; the manifest has a separate 1 MiB limit. These are evidence admission
limits, not whole-process memory guarantees.

This command derives hashes and geometry from the actual files. Set `0.95` to
the confidence you explicitly intend to supply; it is not estimated by this
helper. Choose a new manifest path if the file already exists.

```bash
PYTHONPATH=src ./.venv/bin/python - \
  "$PRIVATE_INPUT_ROOT/$SOURCE_RELATIVE_PATH" \
  "$MATERIALS_WORK_ROOT/water.npy" \
  "$MATERIALS_WORK_ROOT/supplied.json" "$INPUT_COLOR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from transformation_portal.lux_depth_v4.photography import decode_master

source, mask_path, destination = map(Path, sys.argv[1:4])
source_bytes = source.read_bytes()
master = decode_master(
    source_bytes, source_name=source.name, input_color=sys.argv[4],
    max_pixels=100_000_000,
)
mask = np.load(mask_path, allow_pickle=False)
if mask.shape != master.shape or mask.dtype not in (np.dtype("bool"), np.dtype("float32")):
    raise ValueError("Mask must be bool/float32 HW in canonical-master geometry")
if not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
    raise ValueError("Mask must be finite in [0,1]")
record = {
    "schema": "tp.materials.supplied.v1",
    "source_sha256": master.source_sha256,
    "shape": list(master.shape),
    "regions": [{
        "region_id": "water-1", "label": "water",
        "mask_path": mask_path.relative_to(destination.parent).as_posix(),
        "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        "semantic_confidence": 0.95,
    }],
}
with destination.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, sort_keys=True)
PY

PYTHONPATH=src ./.venv/bin/python -m transformation_portal.materials_v4 import-supplied \
  --source "$PRIVATE_INPUT_ROOT/$SOURCE_RELATIVE_PATH" \
  --source-relative-path "$SOURCE_RELATIVE_PATH" \
  --input-color "$INPUT_COLOR" \
  --regions "$MATERIALS_WORK_ROOT/supplied.json" \
  --output-dir "$MATERIALS_WORK_ROOT/supplied-bundle"
```

Import publishes content-addressed numeric masks, `evidence.json`, and
`lux-materials.json`. Numeric files are written before the manifest. The bundle
binds mask bytes, scores, provenance, source bytes, and geometry; filenames and
execution timings do not become semantic authority. Canonical labels include
surface materials and scene labels such as sky and foliage. Exact aliases are
normalized; other labels become `unknown`, which cannot authorize an edit.

Inspect the bundle against its photographic source:

```bash
PYTHONPATH=src ./.venv/bin/python -m transformation_portal.materials_v4 inspect \
  --source "$PRIVATE_INPUT_ROOT/$SOURCE_RELATIVE_PATH" \
  --input-color "$INPUT_COLOR" \
  --evidence "$MATERIALS_WORK_ROOT/supplied-bundle/evidence.json"
```

## Freeze policy and run Lux

Generate the complete policy rather than constructing a partial JSON object:

```bash
PYTHONPATH=src ./.venv/bin/python -m transformation_portal.materials_v4 policy \
  > "$MATERIALS_WORK_ROOT/policy.json"

PYTHONPATH=src ./.venv/bin/lux-depth-v4 \
  --input-dir "$PRIVATE_INPUT_ROOT" --output-dir "$PRIVATE_OUTPUT_ROOT" \
  --input-color "$INPUT_COLOR" --device cpu \
  --materials-manifest "$MATERIALS_WORK_ROOT/supplied-bundle/lux-materials.json" \
  --materials-policy "$MATERIALS_WORK_ROOT/policy.json" --plan \
  > "$MATERIALS_WORK_ROOT/execution-plan.json"

PYTHONPATH=src ./.venv/bin/lux-depth-v4 \
  --input-dir "$PRIVATE_INPUT_ROOT" --output-dir "$PRIVATE_OUTPUT_ROOT" \
  --input-color "$INPUT_COLOR" --device cpu \
  --materials-manifest "$MATERIALS_WORK_ROOT/supplied-bundle/lux-materials.json" \
  --materials-policy "$MATERIALS_WORK_ROOT/policy.json"
```

`PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth_v4` is the
module equivalent of the console command. `--plan` loads no model and creates no
output attempt. A subsequent invocation prepares again: retain the printed plan
fingerprint and compare it with `execution-plan.json` in the completed attempt.
Python callers can pass the exact `PreparedLuxExecutionV4` returned by `prepare`
to `run` without a second preparation.

The default material policy requires explicit semantic confidence of at least
0.8 and at least 500 core mask pixels, both before and after conflict resolution.
It applies small linear gains to eligible glass/water, small luminance-detail
gains to stone/foliage, and luminance-detail attenuation to sky. Operation IDs,
strengths, confidence admission, conflict rules, working space, delta bound, tile
size, halo, and scratch budget are all frozen. These are neutral photographic
adjustments, not physical reflectance reconstruction or PBR material estimates.

Unknown, unsupported, or uncertain positive mask support protects those pixels
from other edits. Positive overlap between eligible regions also protects the
intersection; region ordering does not choose a winner. Masks weight the final
delta once. HDR samples outside unit RGB and fully transparent samples remain
unchanged, alpha and metadata are preserved, and output values are not clipped
to the display range. The default maximum absolute material delta is 0.025 in
linear sRGB. This bound is checked on final stored float32 samples.

Editing the policy requires a new execution plan. Omitting fields, adding
unknown fields, using Boolean numbers, or combining legacy companion material
masks with MaterialsV4 fails closed. Camera calibration companions can still be
selected separately. Unlisted sources in a MaterialsV4 batch retain an explicit
`no_evidence_for_source` abstention.

Each V3-plan photograph additionally publishes `materials-baseline.npy`, the
exact globally enhanced master immediately before material response. Its
`photograph.json` contains that baseline descriptor, the complete response plan,
and measured receipt. `source-master.npy` remains the original photographic
master; it is not the baseline for material-only differences. Completion
verification checks baseline/final hashes, policy/evidence bindings, actual
changed/max/mean deltas, and HDR/alpha protection. Unpublished original region
masks cannot be independently reconstructed by completion verification.

## Local inference experiment

The optional prototype uses local SAM2.1 Hiera Large geometric proposals and
local OpenCLIP ViT-B-32 classification. The SAM checkpoint must match the pinned
bytes in `materials_v4/proposal.py`; CLIP must be a nonempty, strictly compatible
local safetensors state dictionary. Safetensors content is checked regardless
of filename extension. No remote model identifier or automatic download is used.

Use an explicitly prepared experiment interpreter with the adapter's required
SAM2, OpenCLIP, Torch, torchvision, safetensors, and supporting dependencies.
The adapter observes model files, direct distribution bytes, import origins,
processing code, prompts, preprocessing, and the requested device. This partial
runtime observation is not a newly governed production runtime installer or a
DA3 lock. Import-origin paths do not bind source bytes in shadow packages outside
the recorded installed distributions, so this digest is not a complete replay
or runtime-trust guarantee.
Missing dependencies, changed identities, exhausted budgets, and unavailable
devices fail without heuristic or device fallback.

```bash
export MATERIALS_EXPERIMENT_PYTHON=/absolute/path/to/experiment/bin/python
export SAM2_CHECKPOINT=/absolute/path/to/sam2.1_hiera_large.pt
export CLIP_CHECKPOINT=/absolute/path/to/clip-vit-b-32.safetensors

PYTHONPATH=src "$MATERIALS_EXPERIMENT_PYTHON" \
  -m transformation_portal.materials_v4 infer \
  --source "$PRIVATE_INPUT_ROOT/$SOURCE_RELATIVE_PATH" \
  --source-relative-path "$SOURCE_RELATIVE_PATH" --input-color "$INPUT_COLOR" \
  --sam2-checkpoint "$SAM2_CHECKPOINT" --clip-checkpoint "$CLIP_CHECKPOINT" \
  --device cpu --proxy-longest-side 1024 --max-proposals 64 \
  --classifier-batch-size 8 \
  --output-dir "$MATERIALS_WORK_ROOT/inference-experiment"
```

Select `--device mps` explicitly for an Apple Silicon experiment; support checks
are not native performance acceptance. The CLI exposes CPU and MPS. Proposals
use bounded proxy geometry, no crop pyramid, and explicit proposal/mask budgets.
The default is one SAM prompt point per batch with a 512 MiB candidate budget.
Before generation and each batch, the adapter checks a conservative estimate
covering decoder mask scratch, worst-case proxy RLE construction, and retained
candidate logits/RLEs. It checks the cumulative pre-NMS candidate count and bytes
before the official generator accumulates the batch. Thus `--max-proposals`
also limits candidates before duplicate suppression; exceeding a limit fails
without truncation. Reduce `--proxy-longest-side` if candidate admission fails.
Python callers can also reduce `points_per_batch`. Decoded proxy and restored
master masks have separate byte limits. These checks are candidate-memory
admission estimates, not total process RSS bounds; model weights, native
allocator overhead, and measured production-resolution memory acceptance remain
separate.
Classifier crops exclude pixels outside each proposal and execute in bounded
batches. Rejection considers an unknown prompt, absolute cosine similarity,
relative top score, and top-two margin. Weak or ambiguous rankings emit unknown.

The emitted score type is `uncalibrated_clip_ranking_v1`. Its softmax is a
relative ranking over the fixed prompt vocabulary, not calibrated material
probability. SAM predicted IoU and stability remain geometric evidence. These
inferred regions do not authorize automatic photographic edits even when a
ranking is numerically high. The experiment emits no calibration receipt,
consults no inference cache, and never reuses a V3 segmentation cache.

Automatic inferred editing requires a separately reviewed calibration receipt,
explicit policy admission of its content digest, matching per-region binding,
`calibrated_material_probability_v1` score semantics, the photographic domain,
covered classes, and matching producer recipe identities. Receipt construction,
self-declared score strings, or a caller writing a digest do not create trusted
calibration. The default allowlist is empty; do not relabel prototype rankings
as supplied or calibrated evidence to bypass abstention.

## Evaluate declared annotations

The evaluator computes measurements and always reports `acceptance:
not_assessed`. It needs real annotations and declared source/property split
membership. Start with an explicitly empty input when no annotated corpus exists:

```bash
cat > "$MATERIALS_WORK_ROOT/evaluation-input.json" <<'JSON'
{
  "schema": "tp.materials.evaluation_input.v1",
  "records": [],
  "selection_sources": [],
  "evaluation_split": "test",
  "min_class_support": 5,
  "calibration_bins": 10,
  "boundary_tolerance": 1
}
JSON

PYTHONPATH=src ./.venv/bin/python -m transformation_portal.materials_v4 evaluate \
  --input "$MATERIALS_WORK_ROOT/evaluation-input.json" \
  --output "$MATERIALS_WORK_ROOT/evaluation-report.json"
```

That empty input returns `unavailable` and exit 2. Populate `records` only from
actual annotations. Every record has exactly these fields:

| Field | Meaning |
| --- | --- |
| `record_id` | Unique bounded record identifier |
| `source_sha256` | Actual photographic source digest |
| `property_group` | Group that keeps related property images in one split |
| `split` | `train`, `calibration`, `validation`, or `test` |
| `true_label` / `predicted_label` | Canonical taxonomy labels |
| `confidence` | Finite top-label score in `[0,1]`, or `null` when missing |
| `predicted_mask` / `target_mask` | Matching binary HW JSON arrays, or paired `null` values |

`selection_sources` explicitly lists sources used to select models, parameters,
or policies. Source or property-group leakage across splits, undeclared selection
sources, or use of a sealed-test source for selection makes the evaluation
unavailable. The split receipt proves consistency of these declarations; it
does not independently audit the producer's historical training or tuning data.

Per-class IoU pools record-instance pixels. Boundary F1 uses inner eight-connected
boundaries and an explicit Chebyshev pixel tolerance. Confusion and unknown
coverage are reported separately. Mask unknown coverage uses summed record
canvases, not unique scene area. Selective risk reports error among accepted
known predictions versus coverage over all evaluated records. Missing scores,
unknown predictions, and insufficient support do not become perfect metrics.
Calibration output contains equal-width-bin ECE and **top-label binary Brier**;
it is not multiclass Brier or proof that the model is calibrated.

The CLI exits 0 for measured output, 2 for unavailable evaluation evidence, and 1
for command/file failures. None of these establishes production acceptance.
Freeze group-disjoint development/calibration/test data before selection; retain
independent photographic review and production-resolution resource measurements.

## Migration and validation

MaterialsV4 does not replace V3 automatically or reuse V3 caches. The same
candidate change also repairs bounded V3 defects: unsupported classes abstain,
actual delta telemetry includes the writeback scope, resolved coverage and
bounding boxes are rechecked, priority ties have a stable order, and the
low-texture statistic no longer weights an already blended delta twice.
Missing semantic confidence is not replaced by SAM geometric IoU. The downstream
V2 mask handoff requires current executed material authorization and explicit
semantic confidence, while segmentation observations remain intact.

Restored V3 mask archives additionally have a 256 MiB aggregate decoded-byte
budget, checked before any ZIP member is opened and before each payload is
read. Stored and DEFLATE members are supported; mask payloads are read in bounded
chunks. Over-budget or unsupported cache observations are withheld from the V2
handoff. This limits restored-mask storage and decoder transients without
claiming a bound on total model or process memory.

The V3 segmentation cache schema becomes `materials-segmentation-cache.v2` so
older entries with the former confidence semantics are rejected and recomputed.
Do not rename old entries or alter their version to make them reusable. V3
photographic heuristics and its remaining acceptance boundaries are not promoted
to V4 guarantees by these fixes.

Run the canonical candidate contract gate from the repository root (no model
inference is required):

```bash
make test-materials-v4-contract
```

For a narrower diagnosis:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest tests/materials_v4 -q
PYTHONPATH=src ./.venv/bin/python -m pytest \
  tests/lux_depth_v4/test_materials_v4.py \
  tests/lux_depth_v4/test_materials_v4_verification.py \
  tests/materials/test_materials_v3_forensic_regressions.py -q
```

Local contracts, synthetic fixtures, and an isolated model smoke remain separate
from reviewed real-photograph quality, held-out calibration, MPS reproducibility,
and production-resolution latency/peak-memory acceptance.
