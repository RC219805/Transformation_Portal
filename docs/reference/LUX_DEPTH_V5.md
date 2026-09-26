# LuxDepthV5 candidate interface and evaluation

LuxDepthV5 is an opt-in successor candidate built on V4's governed execution,
runtime, image I/O, and publication machinery. It adds explicit depth validity,
controlled inference precision, conservative RGB-guided reconstruction, and
bounded photographic edits. DA3 Metric remains the initial model baseline; no
replacement model or production-quality claim is promoted by this interface.
V3 and V4 commands retain their existing behavior.

The [forensic analysis](../analysis/DEPTH_FORENSIC_AND_LUX_DEPTH_V5_SUCCESSOR_2026-09-20.md)
records the motivation. The [Execution Plan V4 contract](EXECUTION_PLAN_V4.md)
describes the additive plan, cache, and evidence versions.

## Run and plan

Use the existing supported parent environment and governed baseline DA3 runtime
described in the [V4 guide](LUX_DEPTH_V4.md). Native cache authority still requires
the lock-aligned Darwin arm64 DA3 runtime. V5 does not download models, substitute
a backend, or treat an ungoverned runtime as cache authority.

From the repository root:

```bash
PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth_v5 \
  --input-dir /absolute/photos --output-dir /absolute/output-v5 \
  --input-color srgb --device cpu --precision fp32 --plan

PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth_v5 \
  --input-dir /absolute/photos --output-dir /absolute/output-v5 \
  --input-color srgb --device mps --precision fp32 \
  --runtime-python "$PWD/.runtime/Depth-Anything-3/.venv-da3/bin/python" \
  --cache-dir /absolute/depth-cache --refinement guided_bilinear
```

Installing this source also exposes `lux-depth-v5`. `--plan` emits the exact
canonical bytes consumed by execution. CPU planning does not load a model or
write outputs; MPS/auto planning probes availability. The output directory must
be new, its parent must exist, and input/output/cache roots must be disjoint.
For an installed wheel used outside the repository, explicitly set
`TP_MODEL_LOCK_MANIFEST=/absolute/governed/config/model_lock_manifest.yaml`.
The external model policy remains required; missing policy fails closed.

The CLI retains V4's color, target-size, strength, clarity, raw-runtime, resource,
preview, calibration-companion, and MaterialsV4 options. Target size defaults to
518. Choose source color interpretation explicitly for untagged imagery;
`--input-color srgb` is an example, not a universal interpretation of TIFF/RAW.

For a higher-resolution comparison, add `--target-size 1008` to the execution
command and use a fresh output directory. The retained 29.998 MP photograph
passed governed FP32 MPS execution and independent output verification at both
518 and 1008. At 1008, several railing bars and fountain boundaries were more
distinct; fine ornament remained incomplete. This single-scene observation has
no measured depth reference and does not establish a better default or a
performance baseline. See the [resolution evidence](../analysis/DEPTH_ADAPTER_REMEDIATION_2026-09-20.md).

| Option | Default | Meaning |
| --- | --- | --- |
| `--precision` | `fp32` | Explicit full float32 network execution; `fp16` opts into mixed precision with the pinned upstream float32 depth head |
| `--refinement` | `guided_bilinear` | Bounded two-surface RGB selection; `bilinear` supplies the deterministic control |
| `--strength` | `0.25` | Bounds depth exposure magnitude in stops |
| `--clarity` | `0.0` | Bounds clarity addition to at most `0.05 * clarity` linear units per channel |

Both precision modes store float32 native output. The worker bypasses the pinned
upstream outer automatic autocast wrapper, validates the expected API/network
interfaces, and uses DA3's preprocessing and conversion. It accepts the governed
DA3 source revision `95a2adea1a8180104bf51937409034bdec70a244` only. Advancing that
revision requires auditing the recipe; this is not a generic DA3 compatibility
adapter. Explicit precision does not promise bitwise CPU/MPS equivalence.

## Managed job execution

In the portal **Build** view, select **lux-depth-v5 (Opt-in photography)**,
choose input/output paths, and configure the photographic controls on the
Outputs step. The default comparison is 518/FP32; 1008 remains explicit.
Color interpretation, device, refinement, bounded strength/clarity, and optional
MaterialsV4/calibration manifests use the closed V5 request contract. V3 presets
and V3-only flags are not applied to V5. Shared staged uploads support V5 and V6
when enabled for the authenticated portal cohort; the returned staged input path
remains subject to tenant authorization. Saved profiles preserve the separate
photographic configuration.

On the **Paths** step, **Choose files** or **Choose folder** stages source files
when the staged-upload feature is enabled for your cohort. A successful upload
replaces the Input Directory with the server staging directory and preserves
nested folder paths. The path field itself accepts a server-accessible directory,
not an individual image filename or a browser-local folder path. Uploading does
not change Output Directory or dispatch a job; choose a fresh authorized output
directory and obtain a new configuration preview. In pilot tenant mode, configure
staging under the intended tenant's workspace or CAS root. Both input and output
paths remain subject to tenant admission checks.

For explicit grading, separate SDR rendering, and reconstructed depth delivery,
see the independently enabled [managed V6 successor](LUX_DEPTH_V6.md#managed-portal-and-api-execution).
V6 freezes V5 inference within its composite plan and rejects Materials inputs;
it does not change the V5 request or V3 production default.

Dispatch requires a current successful configuration preview and server
readiness. Disabled or unavailable V5 servers remain blocked with their
prerequisite reasons. Preview checks are advisory; admission and the worker
revalidate paths, runtime authority, resources, and the exact frozen plan.
The [September 21 HTTP/portal audit evidence](../analysis/HTTP_PORTAL_LUX_SUCCESSOR_2026-09-21.md)
records the integration decision and validation boundaries.

`POST /v1/jobs` accepts the opt-in `lux-depth-v5` pipeline. The shared
`JobExecutionService` coordinates HTTP-admitted jobs and standalone workers:
subprocess lifetime, cancellation, lease-bound execution, and verified generation
publication use the same service. Existing routes, response envelopes, SSE event
names, and artifact retrieval remain unchanged. V3 remains the default pipeline.

Enable managed V5 only after applying `make db-upgrade` through
`0007_photography_bindings` and deploying matching API and worker code. Configure
both hosts with Postgres/Redis authority, the same protected shared execution
root, and the governed runtime selections:

```bash
export TP_LUX_V5_MANAGED_ENABLED=1
export TP_ORCHESTRATOR_STATE_BACKEND=postgres
export TP_ORCHESTRATOR_QUEUE_BACKEND=redis
export TP_DATABASE_URL='postgresql+asyncpg://user:password@db.example/service'
export TP_REDIS_URL='redis://redis.example:6379/0'
export TP_ORCHESTRATOR_EXECUTION_ROOT=/protected/shared/execution
export TRANSFORMATION_PORTAL_DA3_PYTHON=/governed/da3/.venv-da3/bin/python
# Optional, when RAW inputs or depth-cache reuse are required:
export TRANSFORMATION_PORTAL_RAW_PYTHON=/governed/raw/.venv-raw/bin/python
export TP_LUX_V5_CACHE_DIR=/protected/shared/photography-cache
```

Use the [distributed Postgres runbook](../runtimes/orchestrator-postgres.md) for
execution-root permissions and worker startup. Server-selected interpreters and
cache paths cannot be supplied in job arguments. The optional cache is isolated
under `TP_LUX_V5_CACHE_DIR/<tenant_id>`; leave the setting absent to disable it.
Worker policy rechecks tenant authorization, runtime selections, and data paths
before execution. In pilot tenant mode, include `lux-depth-v5` in
`TP_PILOT_ALLOWED_PIPELINES` and use the authenticated frontdoor as usual.

New plans also declare a photographic browser preview. Each input's final
master produces a PNG with a maximum edge of 1600 pixels, 8-bit sRGB encoding,
and preserved alpha. Reduction averages linear light with premultiplied alpha.
This is a Review derivative; the float master and 16-bit TIFF remain the
precision outputs. The PNG is reserved before execution and included in the
verified inventory and `tp.lux.photograph.v3` receipt. Depth `preview_maps`
remain separate numeric NPY derivatives.

Upgrade API and workers together before admitting these plans. Updated readers
continue to accept older plans and `tp.lux.photograph.v2`; execution of an older
plan retains its original artifact set. Older workers cannot consume the new
preview recipe. Drain new-plan jobs before rolling back the implementation, or
disable managed V5 and use V3. The existing output and plan envelopes retain
their versions; the new optional recipe and descriptor are described in the
[plan contract](EXECUTION_PLAN_V4.md#output-and-publication).

Submit a request using strict snake_case argument names and JSON types:

```json
{
  "pipeline": "lux-depth-v5",
  "args": {
    "input_dir": "/authorized/tenant/photos",
    "output_dir": "/authorized/tenant/output-v5",
    "input_color": "srgb",
    "device": "mps",
    "precision": "fp32",
    "refinement": "guided_bilinear",
    "preview_maps": false
  }
}
```

Paths must satisfy the existing server and tenant allowlists. Numeric strings,
unknown options, shell arguments, and runtime/cache overrides are rejected.
Optional `companions_manifest` and `materials_manifest` paths are admitted and
revalidated with the same tenant boundary. `GET /v1/readiness` reports the V5
prerequisites separately; readiness is not proof of native inference.

Admission retains exact `tp.execution.plan.v4` bytes and separately stores the
closed `tp.job.photography.bindings.v1` physical-path carrier on the immutable
dispatch attempt. `DispatchLocator.plan_digest` remains SHA256 of the complete
canonical plan bytes; the embedded semantic plan fingerprint is independently
verified. The bindings have their own digest recorded in admission evidence.
Neither queue messages nor mutable API projections can supply replacement
execution authority. Publication revalidates the exact plan digest, semantic
evidence, artifact hashes, active limits, and dispatch fence before visibility.
The normal job artifact `items` descriptors derive from verified inventory
records, without rescanning outputs through the legacy artifact scanner.
Blocking verification runs off the async event loop so lease heartbeats can
continue during large photographic reads and reconstruction.

The worker checks database claim authority again immediately before launch,
without extending its lease. Database checks and OS process creation are not one
atomic transaction; cancellation and heartbeat supervision remain active.
Abrupt worker `SIGKILL` or host failure can prevent the in-process reaper from
cleaning up children, so deployments still need host-level process supervision.
The publication fence remains authoritative even when cleanup is interrupted.

Migration 0007 leaves existing V1 attempts with null bindings. Stop new admission
and drain workers before deployment or rollback; API and worker releases must
understand the installed plan and binding schemas. The migration refuses
downgrade while any V5 attempt remains, including terminal tombstones. Do not
delete immutable attempts to bypass this guard; keep a compatible release or
repair forward. This integration does not promote native quality or performance
acceptance.

## Depth and finishing semantics

The native model raster is preserved, including padding and model sky
substitution. Numeric validity means finite and positive. In-frame support
excludes padding. Sky uses the upstream substitution threshold `>= 0.3`, rather
than the higher threshold on the exported convenience prediction. A usable
surface must satisfy numeric validity, in-frame support, and known non-sky.
Missing sky evidence disables surface use. The baseline supplies no contracted
accuracy-confidence raster.

Relative near=0/far=1 depth is a separate derivative normalized from usable
in-frame samples only. Calibration companions may additionally produce inferred
metric depth using supplied source-bound camera intrinsics and the DA3 focal
recipe; this is not measured scene distance. Native calibrated arrays retain
their numeric domain, while aligned metric arrays use zero outside usable
surface validity. Always consume the accompanying mask.

Guidance selects existing native surface values only when a complete valid 4×4
native neighborhood supports two locally stable depth groups and a sufficiently
strong RGB match. The `bounded_two_surface_rgb_selection_v2` recipe uses
unnormalized native values for this check, abstains on affine and gradual ramps,
and retains bilinear interpolation when neighboring evidence or color matches
are insufficient. Finite samples cannot establish a physical discontinuity;
a sufficiently steep smooth surface can resemble a step. Temporary guide
allocations are tiled, and native samples remain unchanged. These constraints
and synthetic checks do not prove recovery of missing real-world geometry.

Photographic exposure and clarity are bounded and attenuated by interpolation
support. Sky, invalid regions, explicitly protected pixels, and nonopaque pixels
remain unchanged by depth finishing. The support score describes interpolation
coverage, not probability of accurate depth. MaterialsV4 remains an explicit
subsequent response with its own evidence and baseline. Legacy material-mask
companions are unsupported in V5; use `--materials-manifest` and the
[MaterialsV4 guide](../guides/MATERIALS_V4.md).

`--preview-maps` emits nonphysical normal/roughness/AO previews. Each operator
requires a complete valid neighborhood and uses neutral values elsewhere;
unknown or invalid depth is never inferred to be physical material evidence.

## Stored evidence

Each image has a canonical source master, final master, 16-bit TIFF delivery,
native samples, numeric/support/sky/usable masks, aligned relative depth and
validity/support scores, depth-finishing baseline and receipt, and optional
calibrated metric arrays, alpha/ICC, previews, and MaterialsV4 baseline. The
batch includes its immutable plan, exact file inventory, runtime/model identity,
and cache hit/miss evidence. `production_acceptance` remains pending.
Newly prepared plans also include the bounded sRGB PNG browser preview; legacy
plans retain their original inventory without this derivative.

Use `transformation_portal.lux_depth_v5.evidence.verify_execution_evidence_v3`
to verify a completed directory. Managed publication independently verifies the
output and retains the generation publisher's dispatch/attempt fence. Semantic
verification reconstructs derivatives and finishing; it does not certify model
accuracy or the truth of operator-supplied calibration.

## Reference-bound quality evaluation

The evaluator reads already computed, frozen predictions; it does not run models:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/validation/evaluate_lux_depth_v5.py \
  --manifest /absolute/quality/manifest.json \
  --output /absolute/quality/report.json
```

The output must not exist. Array paths are relative to the manifest directory,
confined without symlink traversal, and bound by SHA-256 and exact byte size.
Depth arrays must be C-order float32/float64 NPY; masks must be boolean. Every
scene declares its canonical-master grid, source SHA-256, independence group,
and named baseline/candidate predictions. Hash the canonical grid object using
`transformation_portal.ingest.canonical_json.canonicalize_json`.

A manifest has this structure (replace placeholders with real hashes and array
records):

```json
{
  "schema": "tp.depth.quality.manifest.v1",
  "baseline": "bilinear",
  "candidate": "guided",
  "scenes": [{
    "scene_id": "scene-01",
    "independence_group": "property-01",
    "source_sha256": "SOURCE_SHA256",
    "grid": {"height": 512, "width": 768, "coordinate_space": "canonical_master"},
    "evaluation_mask": null,
    "predictions": {"bilinear": "PREDICTION_OBJECT", "guided": "PREDICTION_OBJECT"},
    "reference": "REFERENCE_OBJECT_OR_NULL"
  }]
}
```

Each prediction object contains `source_sha256`, `grid_sha256`, `semantics`,
`units`, `depth`, `valid_mask`, `recipe_sha256`, and a positive
`boundary_threshold`. Each array record is
`{"path":"depth.npy","sha256":"HASH","size_bytes":1234}`.
Allowed semantics/unit pairs are `metric_distance_m`/`m`,
`relative_distance`/`arbitrary`, and `relative_inverse_depth`/`arbitrary`.

The reference object has the same source/grid/semantics/units/depth/valid-mask
fields, plus `boundaries` (a boolean array record or null), `ordinal_pairs`
(possibly empty), and `provenance` with `kind`, descriptive `method`, and
`uncertainty_m`. Each ordinal pair is
`{"a":[row,column],"b":[row,column],"relation":"nearer"}`; `farther` and
`equal` are also accepted. Reference provenance must describe the actual
measurement or synthetic construction; an RGB edge map is not a depth-boundary
reference. The test fixtures in `tests/lux_depth_v5/test_evaluation.py` provide
complete executable examples.

Metric AbsRel/RMSE and delta accuracy use unaligned meters. Relative alignment
is reported separately in the declared distance or inverse-depth domain and
cannot establish metric accuracy. Boundary precision/recall/F1 use declared
depth-boundary references. Predicted and reference boundary pixels share a
complete valid 3×3 neighborhood within the reference/prediction overlap and
evaluation mask. Missing references or insufficient remaining boundary support
remain unavailable. The `reference_bound_depth_metrics_v2` evaluator recipe is
recorded in the report, and its recipe hash binds both that algorithm identity
and the settings. Manifest and report schemas remain v1. Ordinal
accuracy uses explicit reference relations. Differing prediction support is
reported and excluded from paired aggregate metrics instead of silently
rewarding discarded difficult pixels.

The optional `settings` object controls minimum support, boundary tolerance,
relative alignment, ordinal tolerance, and group bootstrap. Defaults require
five independently declared groups before descriptive 95% bootstrap intervals;
duplicate source images are rejected. Group independence and reference truth
are operator declarations, not independently certified facts. Source-image
bytes are not loaded by this evaluator. Reports explicitly leave production
acceptance and model promotion unestablished, including on perfect synthetic
fixtures.

## Validation and acceptance

```bash
make test-lux-depth-v5-contract
make test-lux-depth-v5-managed-contract
```

This checks controlled-worker contracts, evidence tampering, publication
admission, numerical regressions, and evaluator failure modes alongside V4 and
MaterialsV4 regression tests. Native cold/cache/optional-input checks, repeated
uncached precision comparisons, representative photography, ground-truth depth,
and performance measurements are separate acceptance evidence. Do not infer
those outcomes from a local contract-suite pass.

For the managed service gate, provision a dedicated disposable Postgres database
whose name ends in `_test`, migrate it with `make db-upgrade`, and
provide Redis. The service fixtures truncate that database:

```bash
export TP_DISPATCH_TEST_DATABASE_URL='postgresql+asyncpg://user:password@localhost/lux_v5_test'
export TP_DISPATCH_TEST_REDIS_URL='redis://localhost:6379/15'
TP_DATABASE_URL="$TP_DISPATCH_TEST_DATABASE_URL" make db-upgrade
make test-lux-depth-v5-managed-services
```

This gate exercises real immutable storage, migration guards, and the managed
job path. Controlled inference fixtures prove coordination and publication
contracts; they do not establish photographic quality or native performance.
