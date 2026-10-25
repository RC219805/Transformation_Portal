# Photographic review and approved delivery

The portal's Review workbench inspects photographic outputs, records explicit
human decisions, creates separate revisions, and packages an approved set.
It does not change a completed generation, approve photographs automatically,
or establish photographic production acceptance. V3 remains the production
baseline; [Unified](LUX_DEPTH.md), [V5](LUX_DEPTH_V5.md), and
[V6](LUX_DEPTH_V6.md) retain their independent admission and acceptance gates.
Keep private photographs, evaluation registrations, reports, ZIPs and runtime
outputs outside Git, for example in ignored `.runtime/` directories. Preserve
existing finalized deliveries when evaluating a new set. This workflow does
not require AWS or GPU validation.

## Managed access and deployment

Use the authenticated [managed front door](../guides/PORTAL_SECURE_FRONTDOOR_QUICKSTART.md).
The backend requires configured authentication, validates the current job's
tenant and ownership, and admits photographic review only for a `succeeded`
parent whose artifacts remain available. The UI's displayed state does not
authorize access. Reads, mutations, publication and downloads revalidate through
the existing artifact and tenant boundary; supplied filesystem paths grant no
access. Removed artifacts, stale source bytes and lost authorization fail closed.

Deploy matching API/worker code and apply the database migration before using
durable review:

```bash
make db-upgrade
```

Migration `0008_photographic_review`, after `0007_photography_bindings`, adds
job-bound review projections and canonical mutation events. Postgres mode uses
the configured orchestrator database and transactions. Memory mode is local
and loses decisions on process restart. A missing or incompatible durable
backend is an environment/deployment failure; do not switch storage modes to
make an acceptance claim. Readiness for a new inference remains the selected
pipeline's separate preview, license, runtime and dispatch contract.

The additive API prefix is `/v1/jobs/{job_id}/photographic-review`. JSON uses the
existing `ApiEnvelope` with schema `tp.photographic_review.v1`; image/file routes
return authenticated bytes with `no-store` and `nosniff`.

| Route | Purpose |
| --- | --- |
| `GET` at the prefix | Current scenes, variants, properties, decisions, presets and review version |
| `GET /render` | Bounded fit view, maximum edge 1,600 pixels |
| `GET /native-crop` | Exact source-grid crop, each dimension at most 2,048 pixels |
| `GET /quality` | Reference-relative advisory diagnostics |
| `POST /decisions` | Selected, Needs revision, or Approved decision |
| `POST /lighting-groups` | Explicit group membership and reference identities |
| `POST /revisions` | Preview or save a bounded explicit adjustment recipe |
| `POST /delivery` | Export the explicitly approved selection |
| `GET /files/{identity}/{filename}` | Job-bound saved revision/delivery retrieval |

Mutations carry `expected_version`. The store performs compare-and-swap against
the current review version and returns `409 REVIEW_CONFLICT` for stale writes.
Reload and reassess the current selection before retrying. Canonical event
bytes retain the authenticated actor, event kind, time and mutation details.
Changing a filename, a browser selection or a metric value cannot confer approval.

## Inspect the actual version

Choose the scene, primary variant and comparison variant. The workbench can
show an immutable-bound original, current final, previous revision and a retained
working master when the backend can bind all required native components.
Legacy jobs without immutable original bindings report that limitation rather
than infer an original from the live input directory. Inspect the properties
for actual dimensions, precision, encoding, artifact SHA and source identity.

Fit view is a downsampled navigation aid. Native view requests a 256, 512, 1,024
or 2,048-pixel crop from the original grid. One source pixel maps to one CSS
pixel only after source-identity headers and decoded PNG dimensions verify.
Synchronized scrolling, arrow keys and pan controls move the comparison views
together. Side-by-side, wipe and user-started blink help inspect gradients,
fixtures, edges and material texture. Wipe/blink require matching native
geometry; mismatched images need an aligned reference rather than a fabricated
pixel comparison.

Both browser views are **8-bit encoded-sRGB display derivatives**. The current
native display transform clips to the display range; it does not change the
master and cannot expose all extended/HDR values. A native crop preserves spatial
sampling, not master bit depth. Retain and inspect/download the precision
authority separately. Do not infer 16/32-bit precision from an 8-bit browser PNG.

Stable automation hooks include `data-ui="photographic-review"`,
`photographic-scene`, `photographic-primary`, `photographic-comparison`,
`photographic-comparison-mode`, `photographic-inspection`, `photographic-crop-size`
and the crop/pan controls. Existing portal routes, selectors and managed session
behavior remain in use.

## Source identity and retained float masters

An ordinary photographic source is bound to its physical file SHA-256. A native
V5/V6 `master.npy` also depends on `photograph.json`, optional `alpha.npy`, and
any selected original-profile bytes. Its `source_identity_sha256` is SHA-256 of
the canonical object below; a null component is explicit:

```json
{
  "schema": "tp.photo.source_identity.v1",
  "master_sha256": "<physical master.npy SHA-256>",
  "descriptor_sha256": "<physical photograph.json SHA-256>",
  "alpha_sha256": null,
  "source_icc_sha256": null
}
```

The domain accepts retained V5 `tp.lux.photograph.v2/v3` and V6
`tp.lux.graded_photograph.v1/v2` descriptors with the explicit linear-sRGB
`tp.image.master.v1` contract. Arrays must be finite, C-layout float32 NPY v1/v2,
match admitted geometry and have bounded headers. Straight alpha must have its
own exact hash and matching shape/range. Physical array SHA, pixel SHA, source
bundle and composite identity remain separate fields.

Current working precision is 32-bit extended linear sRGB, including negative
and above-one values. `original_source_bit_depth` describes the original input,
not current array storage. Original ICC metadata is provenance: V5's retained
byte-array carrier can supply verified raw profile bytes; V6 may retain only
the descriptor's original-profile digest. Missing profile bytes are never
manufactured to recreate an old content hash, and a camera profile is never
attached to the current linear-sRGB pixels. The observed working-master content
hash differs from the separately labeled descriptor-bound content hash.
Component integrity and descriptor validation do not replay the full generation
or independently authenticate camera-profile provenance.

For native sources, render/crop/quality query parameters and decision/revision
requests carry `source_identity_sha256` as well as physical `sha256`; reference
comparisons carry `reference_identity_sha256`. These fields remain optional in
the schema for ordinary-image compatibility, but omitting the composite for a
native source fails closed. Image responses echo both source hashes for browser
verification. Delivery should submit `approved_sources` entries containing
`artifact_path`, physical `sha256` and `source_identity_sha256`. The backend
requires each exact tuple to have a current Approved decision before materializing
it. This distinguishes native bundles with identical NPY bytes and different
descriptors. Legacy `approved_sha256` remains available for unambiguous physical
hash selection. At least one selection form is required; if both are supplied,
their physical-hash sets must match. Neither form bypasses composite approval
checks.

Decisions bind the physical artifact SHA **and** composite source identity.
Changing only descriptor or alpha bytes invalidates the old decision even when
`master.npy` bytes are unchanged. Lighting-group reference/member SHA fields use
the composite identity for native bundles and physical SHA for ordinary images.

## Processing profiles and explicit color

Build processing intent is distinct from Review delivery presets:

| Build profile | V3 requested output for enabled finishing stages | Precision boundary |
| --- | --- | --- |
| Photographic master | 16-bit encoded-sRGB TIFF | Bounded SDR; sources remain separate |
| Web delivery | Explicit 8-bit encoded-sRGB PNG | Cannot recover discarded source precision |
| Retouch / print | 16-bit encoded-sRGB TIFF | V3 does not produce a 32-bit HDR master |

New portal V3 requests use explicit `output_bit_depth=16`. An explicitly saved
8-bit draft/profile remains 8-bit. The existing CLI default stays 8-bit; use
`--output-bit-depth 16` deliberately. Unified/V5/V6 retain their float working
arrays and 16-bit delivery TIFFs; selecting an intent does not change their
canonical precision contract. Use the retained array for float retouch exports,
not a promoted 16-bit delivery TIFF. Container promotion restores neither lost
precision nor values already clipped during an earlier render.

V3 V2 finishing uses processing version `1.2.0`, contract `encoded_srgb_v1`.
`--v2-input-color` and managed `v2_input_color` accept `auto`,
`auto_assume_srgb`, `srgb`, or `linear_srgb`. Standalone
`scripts/enhance_image.py` uses `--input-color` with the same policies.
Automatic mode validates supported metadata and performs existing profile-aware
conversion before bounded encoded-sRGB finishing. It records the established
untagged-JPEG sRGB assumption and blocks ambiguous untagged PNG/TIFF. A human
who knows that an untagged ordinary image is sRGB may explicitly choose
`auto_assume_srgb`; that assumption is recorded. Invalid, conflicting or unknown
metadata still fails. Explicit encodings assert the source's actual encoding;
they do not relabel arbitrary ICC data. Finished pixels carry matching sRGB ICC
bytes, not an unrelated source profile.

Float32 input or a conversion requiring extended/out-of-range values fails
closed with guidance to Unified/V5/V6. The legacy exact-copy passthrough remains
byte-exact only when no processing/explicit encoding is requested. V2 plan
configuration schema `tp.stage.config.lux.v2.v2`, fingerprints and run-card
metadata bind color policy, contract and processing version. Stage registry ID
and plan V1 remain stable. Historical processing contracts require their matching
environment; do not weaken replay to silently reinterpret old plans.

## Decisions, quality signals and cohesive revisions

Record Selected while considering a version, Needs revision with a useful note,
and Approved only after human inspection. Originals are comparison references.
Every new revision starts unreviewed; its parent's approval does not transfer.
Creating or approving a new version does not overwrite the retained previous
version or a finalized delivery.

The `spatial_v2` diagnostics measure native-grid color differences, gradient
discontinuity/banding, edge-envelope overshoot and newly clipped samples.
Integer normalization uses dtype range, including near-black uint16 data;
opposite channel errors do not cancel. Signals are reference-relative and
composition/texture dependent. They are advisory rather than calibrated quality
gates. Full-frame summaries can hide local defects: inspect sky, ceiling coves,
chandeliers/fixtures, windows, railings, foliage and plaster at native resolution.
Managed quality comparisons use bounded encoded-sRGB samples on that native
grid; they do not measure the unbounded working domain. Inspect retained float
values separately when evaluating extended highlights or out-of-gamut content.

Create explicit lighting groups and choose one reference per group. Exposure
and chromaticity outliers suggest inspection; they do not silently match frames.
Keep twilight and intentional warm practical lights separate from neutral
daylight groups where appropriate. Per-image revisions record exposure
(-1 to +1 EV), RGB white-balance gains (0.8 to 1.25), saturation (0.8 to 1.2),
and optional labeled, feathered rectangular material regions. Local exposure
is bounded to ±0.5 EV, local RGB gains to 0.9–1.1, and overlapping adjustments
have a bounded combined gain. These are manual regions, not inferred semantic
masks. Preview the exact recipe before saving. Recipe/hash, reference identity,
input/output hashes, precision and preview render remain in the receipt.

For representative evaluation, register exact, aligned reference/candidate
TIFF/PNG/JPEG files outside Git using `tp.property.quality_evalset.v1`. Pairs
must already have normalized orientation, equal geometry, matching embedded
working profiles and an explicit working encoding; this evaluator does not
convert profiles or resize inputs. Float inputs currently require normalized
0..1 samples. Inspect extended float masters separately or prepare an explicitly
recorded bounded render for this comparison; never clip them silently.

An evalset records `evalset_id`, scene tags, lighting group, exact file hashes,
`working_color_space`, `working_transfer_function` and native crop boxes
`[left, top, right, bottom]` with exclusive right/bottom coordinates. Real images
require a real local rights record and hash-bound `reference_review`. Use
`decision: "pending_reference_review"`, the exact reference SHA and a reason
until a human reviews it; do not invent reviewer/time fields. Only an actual
`approved_reference` with actor/time/hash counts toward approved-reference
coverage. Synthetic fixtures validate the protocol only. Registered coverage
and approved-reference coverage are reported separately; neither establishes
representativeness or candidate approval.

After preparing `.runtime/property-quality-review/evalset.json` and its
registered images, run from the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m transformation_portal.evals.property_quality \
  --evalset .runtime/property-quality-review/evalset.json \
  --asset-root .runtime/property-quality-review \
  --output .runtime/property-quality-review/quality-report-001.json
```

The output parent must exist; existing reports are not overwritten. Retain
the full-frame/native-crop report with the registration. Target interior,
exterior, mixed-light and twilight scenes plus the listed architectural crop
concerns. Missing coverage remains visible and
`photographic_production_acceptance` remains `not_assessed`.

## Export an approved set

Choose a meaningful property name, per-scene delivery labels and approved
versions. The backend checks every current approved master and native composite
identity again before export. Filenames include property, scene, version,
identity prefix and destination, for example
`seaview-living-room-r002-<identity>-website.jpg`. Labels do not select files or
authorize arbitrary paths.

Before full export IO, the backend estimates scratch requirements from decoded
geometry and the complete selected source/component closure, then compares that
budget with current free space. This point-in-time disk preflight is not a
reservation; concurrent disk usage can still cause an export to fail safely.

| Review preset | Output | Intended use |
| --- | --- | --- |
| `photographic_master` | Original bytes; native NPY + exact descriptor/components and float TIFF companion | Preserve precision authority and explicit working color |
| `website` | sRGB JPEG; default long edge 2,560, quality 92 | Website delivery |
| `mls` | sRGB JPEG; default long edge 3,000, quality 95 | Operator-configurable provider export; no universal MLS-compliance claim |
| `social` | sRGB JPEG; default long edge 2,048, quality 92 | Aspect-preserving social export; no automatic platform crop |
| `print` | 16-bit SDR sRGB TIFF | Proof; printer-specific profile conversion remains external |
| `retouch_print` | Float32 extended encoded-sRGB TIFF | Retouch with preserved extended values and alpha; no tone mapping/clipping |

JPEG/SDR exports use an explicit V6 display render, supported shared editorial
resize/sharpen/JPEG operations, and one final quantization. The isolated editorial
command explicitly loads that shared source file without adding ambient import
roots or installing the full project into its governed dependency closure.
Managed callers import the supported module; RAW/PDF dependencies stay in the
separate editorial lane. Master/float-retouch
presets retain geometry without sharpening. Configurable JPEG quality is
50–100, long edge 1–16,384 or full size, and sharpening 0–1; downsizing never
upscales. SDR transparency requires an explicit external composite. Contact
sheets use a separately recorded white-background composite if necessary.
Only allowlisted, explicitly supplied descriptive metadata is exported; private
capture/GPS metadata is not copied automatically.

The inventory links each derivative to physical master SHA, source identity,
bundle, working content hash, recipe/hash, dimensions, bit depth, profile and
intended use. A contact sheet shows the requested SDR render, or a recorded
perceptual preview for master-only sets. Neither thumbnails nor byte verification
transfer human approval to derivatives.

The merged stdlib packager is the supported
`transformation_portal.core.verified_delivery` library. Its original
`scripts/utilities/package_verified_delivery.py` CLI/API is retained as a shim;
installed distributions need no `.git` directory or repository scripts. It
consumes the explicit `tp.delivery.package_spec.v1` selection, streams/hash-checks
every source, verifies every ZIP member and CRC, then publishes a new ZIP,
checksum and receipt without replacing existing names. Output/copy directories
must be outside the selected source root. Multi-file publication is not
crash-atomic; owned publication links are rolled back on detected failures.
The ZIP includes normalized manifest/checksum inventory and the managed
approval record. This proves selected-byte integrity, not semantic quality,
licensing or production acceptance. Native component preservation is not a
self-contained replay of the original generation.

Managed delivery publishes ZIP, checksum, receipt, inventory and contact sheet
under the parent job's authenticated artifact namespace. Downloads recheck
current approvals, ownership and retention. Withdrawing approval or changing an
authorizing component invalidates access to that saved approved-set delivery.
Private export failure cleanup removes only uncommitted request-owned artifacts;
uncertain state remains subject to parent retention and explicit reconciliation.
Request cancellation does not stop a running blocking operation. The handler
retains its semaphore and scratch paths until that owned worker is quiescent,
then propagates cancellation and cleans up; another image operation cannot
enter while the cancelled worker still owns those resources.
The artifact adapter also drains snapshot/companion reads, immutable writes,
readback and exact-key rollback before releasing its private scratch. A successful
new immutable write records ownership even if cancellation arrived during IO;
failed writes and collisions never authorize deleting a preexisting key.

## Validation and deployment limits

Focused source contracts from the repository root:

```bash
.venv/bin/python -m pytest tests/test_photographic_delivery.py \
  tests/test_photographic_native_master.py tests/test_package_verified_delivery.py \
  tests/test_v2_color_contract.py tests/test_ad_editorial_post_pipeline_paths.py -q
.venv/bin/python -m pytest tests/orchestrator/test_photographic_review_access.py \
  tests/orchestrator/test_photographic_review_http.py \
  tests/orchestrator/test_photographic_review_store.py \
  tests/orchestrator/test_photographic_review_thread_ownership.py \
  tests/evals/test_apex_metrics.py tests/evals/test_property_quality.py -q
make test-portal-contract
make test-frontdoor-contract
.venv/bin/python scripts/validation/check_portal_asset_budgets.py
```

The new UI stays deferred behind Review/Build surfaces. Use supported Node
`>=22.23.2 <23`, including the official distribution for compression-sensitive
asset/CSS evidence. The bounded Build feature exception is **15,500 raw bytes /
4,500 gzip bytes**, accommodating profile controls, explicit V2 input-color
selection, persistence/reconciliation and precision guidance. The local official
Node 22.23.3 measurement is 15,366/4,351 bytes. Eager `portal.js` remains capped at
421,000/104,500 (measured 420,950/104,287); existing Review JS remains capped at
55,000/12,500 (41,836/12,342). The photographic Review assets have separate
40,000/13,000 JS and 6,000/1,600 CSS limits, measured at 36,706/12,991 and
4,500/1,261 respectively. This exception does not raise eager
or existing Review budgets. Rebuild and validate current bytes rather than
treat these dated measurements as performance acceptance.

For the focused frontend units, from `web/secure-landing`:

```bash
node --test tests/photographic-review.test.mjs tests/portal-photographic-profiles.test.mjs
```

The real-HTTP browser fixture uses synthetic disposable photographs and actual
review/revision/export/ZIP handlers; it does not prove real managed admission,
representative photographic quality or live Postgres/S3 acceptance. In one
terminal at the repository root:

```bash
TP_RATE_LIMIT_PER_MINUTE=600 .venv/bin/python \
  tests/fixtures/photographic_review_backend.py --fixture-only --port 10008
```

In another terminal, using supported Node and installed managed Chromium:

```bash
cd web/secure-landing
PLAYWRIGHT_BASE_URL=http://127.0.0.1:3108 MOCK_FASTAPI_PORT=10008 \
  TP_NEXT_DIST_DIR=.next-smoke-photographic-real \
  TP_REVIEW_REAL_HTTP_JOB_ID=job_photographic_actual \
  npx playwright test ux-runtime.portal.spec.mjs --project=portal-browser \
  --grep 'real photographic API' --workers=1
```

Run the separate service-backed/managed gates appropriate to deployment.
Apply migration 0008 with matching application code; rolling back the application
does not authorize dropping review history. Its migration downgrade deletes
review projection/event tables and requires a deliberate preservation decision.
The source changes do not mutate old generations, advance model/runtime or
AWS/GPU acceptance, or promote source tests and synthetic browser evidence to
photographic production acceptance.
