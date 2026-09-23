# LuxDepthV6 managed photography, grading, and depth maps

LuxDepthV6 is an opt-in photographic successor with two entry points. Managed
portal/API jobs accept original photographs, run governed V5 inference, and
finish the retained result with explicit grading, SDR rendering, and depth
products. The standalone CLI finishes an existing, independently verified V5
generation without new inference. Both reconstruct a conservative photographic
baseline from retained original pixels and native depth. LuxDepthV3 remains the
production default and rollback path.

V6 establishes reproducible processing and output verification. Representative
photographic quality, calibrated color accuracy, and production acceptance remain
unestablished. Its display transforms are local versioned recipes, not ACES,
AgX, camera profiling, or HDR delivery.

## Retained-source requirements

Keep the complete V5 output directory described in the
[V5 operator guide](LUX_DEPTH_V5.md). V6 verifies its canonical execution plan,
completion evidence, full artifact inventory, array geometry, and replayed image
products before admission. Verification binds retained bytes; it does not
independently authenticate those pixels against the original camera file.

A V5 generation with applied Materials responses is rejected: that generation
does not retain all masks needed to replay those operations independently.
Abstained Materials responses must also have zero reported changes. Calibration,
source ICC bytes, alpha, unavailable sky, and native-depth validity remain bound
to the V5 source evidence.

V6 uses the upstream strength, clarity, and precision settings. Guided
reconstruction additionally requires persistent native surfaces. A photograph
with any nonopaque alpha sample abstains from depth reconstruction because the
upstream model proxy does not account for transparency. Explicit grading still
applies to its fully opaque pixels; nonopaque pixels remain unchanged in the
unbounded graded master. Display rendering can change RGB for delivery while
preserving the alpha samples.

## Standalone plan, run, and verify

Run from the repository root with the supported core environment. V6 uses the
existing NumPy, Pillow, SciPy, tifffile, and imagecodecs dependencies. Retained
finishing requires neither a new model download nor an ML runtime; managed
fresh inference requires the governed DA3 runtime described below.

Choose an existing V5 generation and a new, disjoint V6 output directory whose
parent already exists:

```bash
TP_V6_PARENT=/absolute/output-v5
TP_V6_OUTPUT=/absolute/output-v6

PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth_v6 \
  --input-dir "$TP_V6_PARENT" --output-dir "$TP_V6_OUTPUT" \
  --exposure-stops 0.25 --saturation 1.05 --render perceptual_srgb --plan

PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth_v6 \
  --input-dir "$TP_V6_PARENT" --output-dir "$TP_V6_OUTPUT" \
  --exposure-stops 0.25 --saturation 1.05 --render perceptual_srgb

PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth_v6 \
  --input-dir "$TP_V6_PARENT" --output-dir "$TP_V6_OUTPUT" --verify
```

`--plan` reads and verifies the parent but creates no output. Running the same
controls against the same retained parent and processing implementation produces
the same canonical plan. The CLI prints that plan with a trailing newline; its
reported SHA-256 covers the canonical JSON bytes without that output newline.
Execution prints `plan_sha256`; supply that value as
`--expected-plan-sha256 <digest>` with `--verify` when verifying an exact intended
plan. The existing output is allowed only for verification. A failed attempt
keeps its partial directory for inspection and requires a new output directory
for a subsequent run.

Installing the current source also exposes `lux-depth-v6`. The module invocation
above remains available when an existing checkout environment has not refreshed
its console scripts. Python callers can prepare once and execute that exact
immutable carrier:

```python
from pathlib import Path

from transformation_portal.lux_depth_v6 import GradeRecipe, LuxDepthV6Request, prepare
from transformation_portal.lux_depth_v6.pipeline import run

prepared = prepare(
    LuxDepthV6Request(
        Path("/absolute/output-v5"),
        Path("/absolute/output-v6"),
        grade=GradeRecipe(exposure_stops=0.25, saturation=1.05),
    )
)
result = run(prepared)
```

## Managed portal and API execution

In **Build**, select **lux-depth-v6 (Grading and depth outputs)**, choose original
photograph input/output paths, and configure inference and V6 finishing on the
Outputs step. This input is a photograph directory, not a retained V5 output
bundle. Saved profiles keep V6 controls separate from V3 and V5. Build requires
a current successful configuration preview and server readiness. Those checks
are advisory: admission and workers revalidate authorized paths, physical
runtime bindings, resources, and the immutable plan.

Deploy matching API and worker code and the packaged plan schema together.
Apply `make db-upgrade` through `0007_photography_bindings`; V6 reuses that
migration's immutable physical bindings and needs no additional migration.
Before upgrades or rollback, stop admission and drain workers. An older worker
cannot consume `tp.execution.plan.v5`; disable V6 admission and drain those jobs
before returning to older worker code.

Load the existing managed service environment in each terminal. It must already
provide `TP_DATABASE_URL`, `TP_REDIS_URL`, backend/frontdoor authentication, a
protected shared `TP_ORCHESTRATOR_EXECUTION_ROOT`, authorized input/output roots,
and executable `TRANSFORMATION_PORTAL_DA3_PYTHON`. Configure the following on
both API and worker hosts:

```bash
export TP_LUX_V6_MANAGED_ENABLED=1
export TP_ORCHESTRATOR_STATE_BACKEND=postgres
export TP_ORCHESTRATOR_QUEUE_BACKEND=redis
export TP_ORCHESTRATOR_IN_PROCESS_WORKERS_ENABLED=0
```

The V6 flag is independent: `TP_LUX_V5_MANAGED_ENABLED` need not be enabled.
In pilot tenant mode, add `lux-depth-v6` to the existing
`TP_PILOT_ALLOWED_PIPELINES` list on API and workers. Keep
`TP_PILOT_CONTROL_PLANE_ENABLED=1` and the same dedicated
`TP_FRONTDOOR_IDENTITY_SECRET` on backend and frontdoor; use the authenticated
frontdoor and the actor's authorized tenant paths. Do not expose backend keys
in browser storage or request bodies. Runtime and optional cache selections
are server-owned. Optional RAW ingest uses `TRANSFORMATION_PORTAL_RAW_PYTHON`;
optional governed depth-cache reuse uses the shared
`TP_LUX_V5_CACHE_DIR/<tenant_id>` namespace. See the
[distributed Postgres runbook](../runtimes/orchestrator-postgres.md) for service
setup and execution-root protections.

With those settings loaded and Postgres/Redis running, start each long-running
process in its own terminal, from the repository root:

```bash
# Backend terminal; apply migrations before starting the API.
make db-upgrade
make run-backend-local-noreload
```

```bash
# Worker terminal; load the same managed environment and flags first.
make run-orchestrator-worker
```

```bash
# Frontdoor terminal; load managed auth and use the supported Node 22 runtime.
make run-frontdoor-local
```

`GET /v1/readiness` reports V6 separately. Submit the same strict JSON body to
`POST /v1/config-preview` and then `POST /v1/jobs` after correcting any reported
errors. The managed frontdoor preserves the existing session authentication and
response envelopes:

```json
{
  "pipeline": "lux-depth-v6",
  "args": {
    "input_dir": "/authorized/tenant/photos",
    "output_dir": "/authorized/tenant/output-v6",
    "model_key": "da3-metric",
    "input_color": "srgb",
    "device": "mps",
    "precision": "fp32",
    "target_size": 518,
    "refinement": "guided_bilinear",
    "strength": 0.25,
    "clarity": 0.0,
    "exposure_stops": 0.0,
    "white_balance": [1.0, 1.0, 1.0],
    "contrast": 1.0,
    "pivot": 0.18,
    "saturation": 1.0,
    "render": "perceptual_srgb",
    "shoulder": 0.8,
    "depth_refinement": "guided_bilinear_v4"
  }
}
```

The API also accepts the common bounded resource fields and optional
`companions_manifest` for source-bound calibration. It rejects unknown fields,
numeric strings, runtime/cache overrides, and non-null `materials_manifest` or
`materials_policy` before admission. Managed V6 always emits depth products;
there is no managed `depth_maps=false` option. The inherited `preview_maps`
option controls optional retained V5 numeric previews, not V6 depth delivery.
`refinement` freezes retained V5 reconstruction; `depth_refinement` selects the
V6 reconstruction shared by its photographic baseline and depth products.

The composite `tp.execution.plan.v5` freezes the embedded V5
`tp.execution.plan.v4` inference plan, every finishing recipe, processing
identity, and publication/resource limits before inference. Exact canonical
plan bytes and the separate `tp.job.photography.bindings.v1` carrier remain
immutable dispatch authority. The complete managed generation contains:

```text
execution-plan.json          # admitted composite plan
execution-evidence.json      # complete composite inventory after verification
source-v5/                  # complete retained V5 plan, evidence, and products
v6/plan.json                # source-bound tp.lux.grade.plan.v2
v6/evidence.json            # independently replayable finishing completion
v6/input-0000/delivery.tif   # 16-bit sRGB photographic delivery
v6/input-0000/preview.png    # bounded 8-bit sRGB photographic draft
v6/input-0000/depth-*.npy    # validity/support products; see inventory below
v6/input-0000/depth-relative.tif
v6/input-0000/depth-preview.png
```

Each additional input has its own `input-NNNN` directory. Publication verifies
both stages, their exact inventories, and the dispatch fence before exposing
any successful generation. It rejects completion changes after semantic replay.
Review prioritizes the photographic TIFF with its same-input PNG derivative;
depth previews and retained V5 evidence remain separately identified.

Managed `max_input_bytes` defaults to 1 GiB per original input, with a 2 GiB
ceiling. `max_pixels` is capped at 100 million. `max_output_bytes` defaults to
64 GiB for the **entire generation**, including retained V5, V6 finishing, and
outer records. The admitted total is the smaller of that request and the
publisher's total-byte limit. After reserving 17 MiB for the outer plan and
completion, each stage receives `floor((total - 17 MiB) / 2)` bytes. Finishing's
retained-source byte ceiling equals the inference-stage budget. Publisher
file-count, per-file, and manifest limits can reject a batch before inference;
actual image geometry is also checked against finishing memory/output
reservations. The execution deadline covers both stages together. Increasing a
byte ceiling does not establish photographic acceptance or add model detail.

## Optional depth reconstruction and maps

Pass `--depth-maps` when planning and running to select the versioned
`tp.lux.grade.plan.v2` contract. Existing commands without that flag retain the
V1 plan, historical reconstruction recipe, and photographic artifact set.
Verification reads the recorded recipe; omit the execution flags with `--verify`.

```bash
PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth_v6 \
  --input-dir "$TP_V6_PARENT" --output-dir "$TP_V6_OUTPUT" --depth-maps --plan

PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth_v6 \
  --input-dir "$TP_V6_PARENT" --output-dir "$TP_V6_OUTPUT" --depth-maps

PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth_v6 \
  --input-dir "$TP_V6_PARENT" --output-dir "$TP_V6_OUTPUT" --verify
```

In Python, set `depth_maps=DepthMapRecipe()` on `LuxDepthV6Request`; import
`DepthMapRecipe` from `transformation_portal.lux_depth_v6`. The optional
`--depth-refinement` selector requires `--depth-maps` and accepts `bilinear`,
`guided_bilinear_v3`, or the default `guided_bilinear_v4`.

V4 requires each selectable native sample to belong to a connected same-depth
surface in its complete 4-by-4 native neighborhood. This prevents disconnected
noise from passing the earlier sample-count test and being amplified by RGB
texture. The same reconstructed depth feeds both the photographic baseline and
the exported scalar products. Native evidence remains unchanged. Alpha
abstention and unknown sky produce no usable reconstructed surfaces.

The additional products are:

| Artifact | Meaning |
| --- | --- |
| `native-depth.npy` | Exact float32 native model API output, including invalid numeric sentinels |
| `native-numeric-valid.npy`, `native-support.npy`, `native-sky.npy` | Numeric validity, unpadded support, and sky evidence; unavailable sky is explicitly marked in the descriptor |
| `relative-depth.npy`, `depth-relative.tif` | Float32 depth at the original photograph's dimensions; normalized near 0, far 1 |
| `depth-valid.npy`, `depth-support.npy`, `depth-support-score.npy` | Usable-surface mask, image support, and interpolation support; the score is not an accuracy probability |
| `metric-depth-m.npy` | Optional reconstructed meters, only when the retained parent supplies calibration |
| `depth-preview.png`, `depth-preview-valid.png` | Bounded 16-bit grayscale visualization and its validity mask; nearest pixel-center sampling, maximum edge 1600 |
| `depth.json` | Geometry, source hashes, recipe, calibration, validity semantics, and product hashes |

Consult the validity mask: zero may mean either a valid near sample or an invalid
location. The preview has no photographic gamma/ICC transform. Original-size
reconstruction increases the sampling grid, not the model's inference resolution
or measured physical accuracy. Request a larger supported target during new
managed V6 or V5 inference to obtain a denser native grid; retained-only V6
finishing cannot recover absent detail.
See the [dated forensic evaluation](../analysis/LUX_DEPTH_V6_DEPTH_MAP_FORENSICS_2026-09-23.md)
for synthetic regression results and the remaining photographic acceptance work.

## Color controls

| CLI control | Default | Accepted range and meaning |
| --- | --- | --- |
| `--exposure-stops` | `0` | `-8` to `8`; multiplies linear RGB by `2**stops` |
| `--white-balance R G B` | `1 1 1` | Each gain `0.25` to `4`; explicit RGB multipliers, without hidden normalization or Kelvin inference |
| `--contrast` | `1` | `0.25` to `4`; signed luminance power around the pivot, preserving RGB ratios; zero luminance is unchanged |
| `--pivot` | `0.18` | `0.001` to `1`; linear-light contrast pivot |
| `--saturation` | `1` | `0` to `2`; scales Oklab chroma while retaining the named space's lightness and hue before rendering |
| `--render` | `perceptual_srgb` | `perceptual_srgb`, `soft_srgb`, or `clip_srgb`; selects the separate SDR transform |
| `--shoulder` | `0.8` | `0.1` to `0.95`; highlight shoulder threshold; inactive for explicit clipping |

Exposure and white balance run first, then luminance contrast, then chroma.
Default grading controls preserve reconstructed master pixel bytes exactly,
including signed zero. The float master retains negative and above-white values;
operations that exceed finite float32 representation fail instead of clipping
the master. Non-finite values, booleans masquerading as numbers, unknown recipe
fields, and unsupported recipe versions are rejected.

Oklab uses the [published 2021 reference transform](https://bottosson.github.io/posts/oklab/).
Numerical support for signed RGB does not establish perceptual accuracy for
negative values or near-cancelling cone responses. White-balance gains should
come from an intentional photographic decision or a suitable neutral reference;
V6 does not estimate an illuminant automatically.

The default `perceptual_srgb` renderer applies a smooth shoulder to Oklab
lightness expressed as a brightness-like cubic quantity, then contracts chroma
at fixed Oklab lightness and hue to fit the display gamut. Its bounded numerical
search is part of the recipe. This is a named-space numerical contract rather
than a universal perceptual guarantee. In-gamut values below the shoulder remain
unchanged. `soft_srgb` provides the earlier linear-luminance shoulder with chroma
contraction toward neutral; it does not promise perceptual hue constancy.
`clip_srgb` provides an explicit independent-channel clipping comparison.

The working space remains extended linear sRGB. Automatic input-color selection
fails closed on ambiguous pixels or unsupported ICC profiles. Explicit
`input_color="srgb"` or `"linear_srgb"` declares an interpretation; it is not an
ICC conversion. Convert Adobe RGB or another unsupported profile with a
color-managed tool before submitting sRGB pixels. RAW uses the governed decoder's
linear-sRGB output; do not assign encoded sRGB to RAW. Unsupported input profiles
and RAW highlight clipping in the original V5 ingest cannot be repaired by
labeling pixels differently or by increasing output bit depth. V6 does not add a
wide-gamut ingest transform or recover missing sensor or native-depth detail.

## Outputs and replay

Each input has separate artifacts:

| Artifact | Authority |
| --- | --- |
| `baseline.npy` | Float32 photographic baseline reconstructed from retained original pixels and native depth |
| `master.npy` | Unbounded float32 graded RGB; precision authority before display rendering |
| `display.npy` | Bounded float32 display-linear sRGB used by both delivery encoders |
| `alpha.npy` | Optional unchanged straight-alpha samples |
| `delivery.tif` | 16-bit encoded sRGB TIFF with embedded profile and normalized orientation |
| `preview.png` | Bounded 8-bit sRGB browser derivative, maximum edge 1600 pixels; reduction handles premultiplied alpha |
| `photograph.json` | Baseline/master/display descriptors and reconstruction, grade, render, and encoding receipts |

The standalone generation root (managed `v6/`) contains `plan.json` and, only after all products pass
semantic replay, `evidence.json`. The plan binds parent evidence, recipes,
resource ceilings, processing source hashes, and dependency versions.
Verification checks exact expected products and independently regenerates their
bytes from the retained V5 parent. Replacing an image and merely recomputing its
inventory hash cannot authorize the changed pixels. Processing-source or bound
dependency drift requires a new plan; verification under another implementation
is not silently treated as equivalent.

Verification securely snapshots the completion record again after replay. A
record that changes, disappears, or gains a link alias during verification is
rejected instead of returning the completion bytes cached at entry.

The processing identity covers the selected Python source files and dependency
version strings used by this recipe. It is a same-environment replay boundary,
not an attestation of every installed wheel byte, native library, operating
system component, or numerical kernel. Matching version strings alone do not
prove equivalence across different binary builds or hosts; independently verify
the products in the intended environment before claiming equivalence.

Keep the complete V5 parent alongside the V6 result. V6 is not a standalone
replacement for its upstream provenance bundle. An `ImageMaster` returned by
`render_master` is marked `color_domain="display_linear_srgb"`; passing it back
to grading or rendering is rejected to prevent a second display transform.
Low-level APIs are `grade_master(master, GradeRecipe(...))` and
`render_master(master, RenderRecipe(...))` in `lux_depth_v6.color`; each returns
an immutable master and a measured receipt.

## Resource and validation boundaries

Standalone defaults are 64 GiB retained input, 64 GiB output, 100 million pixels per image,
16 GiB admitted memory, and 3600 seconds execution time. The conservative V6
memory admission is `pixels * 256 + 256 MiB` per image; the upstream verification
also accounts for its proxy and ICC storage. Output admission reserves every
float product, TIFF, bounded PNG, metadata, plan, and completion before writing.
These ceilings are frozen and rechecked by plan parsing and source admission.
They are conservative numerical admission bounds, not hard operating-system
memory isolation. Cancellation and deadlines are observed at processing and
verification checkpoints.
These checks are cooperative: they do not preempt an individual NumPy/SciPy or
encoding operation in progress, and are not a hard process-isolation deadline.

Depth-enabled plans also bind the verified native dimensions. Their per-image
memory admission is `master_pixels * 320 + native_pixels * 128 + 256 MiB`.
Additional output admission reserves `native_pixels * 7 + master_pixels * 18 +
12 MiB`, including an enforced 8 MiB combined depth-preview ceiling, headers,
and metadata. Products are streamed before grade/display masters are allocated.

Use the focused validation gate:

```bash
make test-lux-depth-v6-contract
make test-lux-depth-v6-managed-contract

PYTHONPATH=src ./.venv/bin/python scripts/analysis/audit_lux_depth_v6_reconstruction.py \
  --output /private/tmp/lux-depth-v6-reconstruction.json
```

For the Postgres/Redis HTTP-to-worker lane, configure
`TP_DISPATCH_TEST_DATABASE_URL` (or `TP_PHOTOGRAPHY_TEST_DATABASE_URL`) with a
dedicated migrated `*_test` database and `TP_DISPATCH_TEST_REDIS_URL`, then run
`make test-lux-depth-v6-managed-services`. Missing services skip that lane;
skipped tests are not live-service acceptance. Its inference is controlled,
while dispatch, worker execution, verification, and artifact publication are real.

Synthetic contracts establish arithmetic, identity, failure handling, and
replay behavior. The [dated native smoke](../analysis/LUX_DEPTH_V6_DEPTH_MAP_FORENSICS_2026-09-23.md#fresh-native-managed-composite-smoke)
records one real-model raw-photograph-to-V6 generation; it does not establish
hosted portal acceptance or physical depth accuracy. Production promotion additionally needs representative interior
and exterior photographs, controlled color references, paired full-frame and
100% inspection, native runtime acceptance, and measured performance. Preserve
V3 and the retained V5 source until those independent acceptance gates pass.
