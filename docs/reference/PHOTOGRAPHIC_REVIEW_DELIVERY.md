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

See the [photographic geometry and orientation audit](PHOTOGRAPHIC_GEOMETRY_AUDIT.md)
for canonical coordinate mappings, identity boundaries, synthetic regression
scope, historical V3/V2 blockers and the explicitly versioned corrected recipe.

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

The additive API prefixes are `/v1/jobs/{job_id}/photographic-review` and
`/v2/jobs/{job_id}/photographic-review`, with identical methods, authentication,
and review state. Returned review/download links use the canonical `/v1` prefix.
Both versions use the existing `ApiEnvelope` with schema
`tp.photographic_review.v1`; image/file routes return authenticated bytes with
`no-store` and `nosniff`.

| Route | Purpose |
| --- | --- |
| `GET` at the prefix | Current scenes, variants, properties, decisions, presets and review version |
| `GET /scenes` | Metadata-only scene pages, recorded review filters and related artifact summaries |
| `GET /scene` | Verified photographic detail for one selected scene |
| `GET /render` | Bounded fit view, maximum edge 1,600 pixels |
| `GET /native-crop` | Exact source-grid crop, each dimension at most 2,048 pixels |
| `GET /quality` | Reference-relative advisory diagnostics |
| `POST /decisions` | Selected, Needs revision, or Approved decision |
| `POST /lighting-groups` | Explicit group membership and reference identities |
| `POST /revisions` | Preview or save a bounded explicit adjustment recipe |
| `POST /delivery` | Export the explicitly approved selection |
| `GET /deliveries` | Bounded committed delivery metadata and request reconciliation, without image discovery |
| `GET /files/{identity}/{filename}` | Job-bound saved revision/delivery retrieval |

Mutations carry `expected_version`. The store performs compare-and-swap against
the current review version and returns `409 REVIEW_CONFLICT` for stale writes.
Reload and reassess the current selection before retrying. Canonical event
bytes retain the authenticated actor, event kind, time and mutation details.
Changing a filename, a browser selection or a metric value cannot confer approval.

## Managed native Materials revisions

Deploy the matching backend and browser assets before enabling
`TP_PHOTO_REVIEW_NATIVE_MATERIALS_ENABLED=1` on the backend. This independent,
default-off capability also requires `TP_PHOTO_REVIEW_ENABLED=1`; disabling
admission never enables an unreserved native fallback. No SQL migration or new
route is required. Both API prefixes retain `POST /revisions`, its envelope,
source selectors, presentation identity and `expected_version` compare-and-swap.
The browser advertises the mode only from `revision_constraints.native_materials`.

Select a retained JPEG/PNG/TIFF photograph with integer 8/16-bit SDR pixels.
The explicit display-SDR operation rejects a conflicting recorded working-stage
policy, native NPY masters, float/HDR sources and out-of-range results. For an
unrecorded raster, choosing this mode explicitly asserts display-SDR semantics;
filenames, pixel ranges and ICC profiles never infer processing stage. Existing
legacy revisions and their defaults are unchanged.

The closed recipe `tp.photo.materials_recipe.v1` contains `input_stage` equal to
`display_referred` and one to four `regions`. Each region has a unique `id`, a
`label` (`glass`, `stone`, `water`, `foliage` or `sky`), normalized
`region: [left, top, right, bottom]`, and `strength` from zero to 0.05. Repeated
labels must share a strength. These are human-specified rectangles, not inferred
semantic masks. Global grading and lighting-reference controls remain separate.
Normalized edges truncate to native pixel edges, with an integer snap only within
four floating-point ULPs. Genuine fractional edges retain truncation, and each
region retains at least one in-bounds pixel. Generated native-mask hashes bind
the affected pixels before execution.
Clients cannot select executable kernels, physical paths, calibration trust,
memory limits or an arbitrary policy. The server freezes the complete bounded
MaterialsV4 policy, operation identity and generated native-mask evidence before
applying the existing compositor to native floating-point pixels.

Preview and save use the same source/recipe authority. A saved result is a
separate TIFF16 with explicit display presentation and an unreviewed
`tp.photo.materials_revision.v1` receipt. The receipt records source and native
content identity, policy/evidence/operation hashes, actual execution deltas,
encoded-output and reopened-content identities, alpha, and diagnostic hashes.
Alpha and protected pixels are preserved; output is quantized only for TIFF16.
Recorded display pixels receive no second tone transform during preview,
reopened inspection or SDR export. New revisions never inherit approval.

The native result, bounded matching preview, diagnostic descriptor and mask
overlay publish through existing review ownership and CAS cleanup. Diagnostic
links use the authenticated `GET /files/{identity}/{filename}` route; they
revalidate the retained parent and committed product bytes. Diagnostics do not
authorize editing, approval or destination proofs. Existing inspection,
approval, delivery, response-loss recovery and revocation checks remain in force.

Native admission atomically reserves decoded source/mask arrays, renderer and
encoding intermediates, tile scratch, temporary files and publication copies.
It retains inspection headroom and existing process-local ceilings; unknown or
over-limit work fails with `413` before decoding. An earlier verified inspection
can supply exact geometry for costing. No automatic downsampling, deletion,
budget increase or recursive lease expansion makes an operation fit. Owned
workers drain before cancellation releases reservations.

Rollback disables `TP_PHOTO_REVIEW_NATIVE_MATERIALS_ENABLED` and restarts the
backend. Existing recipes, revisions, diagnostic products and approvals remain
stored and require compatible readers; rollback deletes nothing. The editor is
loaded with the already deferred photographic viewer, preserving existing
entry-asset budgets and selectors. Software contracts and fixture-driven HTTP
tests do not establish representative native-runtime/resource or photographic
acceptance. Retained-photo pilots, actual model qualification and Richard's
scoped photographic decision remain separate; AWS/GPU validation stays on hold.

## Offline native Materials CPU measurements

The additive CPU qualification command measures the existing compositor in fresh
processes using caller-authorized retained TIFFs. It does not perform inference,
modify managed feature flags, grant managed capacity, transfer approval, or
establish photographic improvement. The initial input contract is integer
8/16-bit SDR TIFF, at most 40 megapixels and 512 MiB, with explicit
`display_referred` recipe semantics. Use the existing governed CPU environment
with NumPy, Pillow, tifffile, SciPy and psutil; no model runtime reinstall is needed.

Create a private parent directory outside retained source directories, then run
from the repository root (substitute actual authorized paths and source hash):

```bash
PYTHONPATH=.:src ./.venv/bin/python -m transformation_portal.photographic_delivery.qualification \
  --source /absolute/retained/photo.tif \
  --source-sha256 <verified-source-sha256> \
  --recipe /absolute/private/materials-recipe.json \
  --output /absolute/private/qualification/new-run \
  --repeats 3 --preflight-only
```

The recipe is the same closed `tp.photo.materials_recipe.v1` payload described
above. Remove `--preflight-only` and choose another fresh output directory to
measure; preflight writes a planning report and never executes a trial. Existing
output directories are rejected. Inputs and output ancestors must not traverse
symlinks. Keep all reports, snapshots and generated products outside Git.

The independent offline ceiling is 12 GiB memory, 64 GiB scratch and 300 seconds
per trial, with one to five sequential trials. It cannot enlarge managed limits.
Planning uses the current conservative managed analytical memory estimate,
geometry-derived scratch including all retained trials, inspection reservations,
and physical free-space/headroom checks. The supervisor samples process RSS and
scratch every 50 ms and drains its owned worker on cancellation or a bound
violation. SIGINT/SIGTERM drain the owned worker, including repeated cancellation;
prior signal handlers are restored afterward. Surviving owned process groups are
escalated independently of leader exit; unexpected descendants fail the run.
Report publication is atomic, and interrupted publication records failure before
propagating cancellation. Run the command in the main thread.
These checks are process-local supervision, not OS-enforced limits or
multiprocess admission. The host must permit psutil to inspect its own worker;
a denied process inspection is an environment failure and stops the worker.

`qualification.json` records source, recipe, implementation and package identity;
per-process RSS high-water and incremental peak; sampled logical/allocated
scratch; latency; source revalidation; product and receipt hashes; reopened native
precision; and normalized-pixel repeatability. File-system caches are uncontrolled,
so fresh processes are not cold-cache evidence. Single-trial repeatability is
`null` (unassessed); two or more trials must agree. Zero-effect recipes verify
preservation within one 16-bit code; nonzero recipes provide distinct candidates
that still require Richard's scoped photographic decision.

The report compares the unchanged analytical reservation with measured
incremental peak plus 50%. An insufficient estimate fails qualification; the tool
never raises a limit or lowers a production profile. `managed_profile` compares
memory with the command environment only, not a running backend. Complete managed
eligibility, authenticated publication/download, representative model performance,
and photographic acceptance remain unestablished. Exit 0 means planned or
measured according to `status`; exit 2 means blocked or not qualified; exit 1 is
an invalid input, runtime, verification or worker failure. Retain failed private
namespaces for diagnosis; the command deletes only its successful trial's own
verified temporary source snapshot.

## Native allocation boundaries

Analytic sRGB ingest evaluates the unchanged pointwise float32 transfer in
two-axis tiles of at most 262,144 pixels, writing one final native RGB array.
Grayscale expansion is also tile-bounded; linear and ICC paths retain their
existing behavior. This avoids full-frame branch temporaries without changing
pixels, separate alpha, orientation or rendering semantics; see the
[color preparation implementation](../../src/transformation_portal/lux_depth_v4/color_preparation.py)
and [exact-sample allocation tests](../../tests/lux_depth_v4/test_analytic_color_allocation.py).

Native TIFF16 transfer validates the complete SDR master in bounded spans before
allocating its final native-resolution uint16 destination. Transfer and rounding
use the same pointwise float32 operations, with at most 65,536 pixels per span;
very wide or short images do not create full-width scratch. RGB and straight
alpha write directly into their final channels. TIFF metadata, ICC, orientation,
encoded bytes, receipts and diagnostic semantics are preserved. This changes
allocation lifetime, not resolution, grading, processing stage or approval.

Content hashing reads the existing buffer for exact, read-only C-contiguous
NumPy arrays whose bounded owner chain ends in immutable bytes. It preserves
the historical C-order digest, dtype, shape, names and semantic metadata.
Writable, strided, Fortran-layout, subclass and mutable-backed arrays retain
their historical snapshot path. This avoids a full native RGB or alpha hash
copy without changing processing or cache identities; see the
[hash compatibility tests](../../tests/core/test_image_artifact_hash_allocation.py).

The native revision helper freezes input/execution content hashes and scalar
metadata, then releases input and adjusted masters after TIFF encoding and before
reopening that TIFF for verified diagnostics. Zero-effect execution can alias its
input, so the released storage is one master; changed execution can hold two.
The reopened master and masks are released after their final verified use.

Let `P` be native pixels, `R` explicit region count and `A` equal one for alpha:

| Phase or retained object | Array-visible storage / ownership |
| --- | --- |
| Immutable master | `12P + 4AP` bytes; source and changed result can be distinct |
| Immutable masks | `4RP` bytes, plus temporary construction/hash copies |
| Compositor execution | Source, masks, mutable RGB output and bounded tile scratch; freezing a changed result adds an immutable master |
| TIFF transfer | Final `6P + 2AP` uint16 samples plus bounded transfer/quantization spans; encoded TIFF bytes remain separate |
| Output reopening | Encoded output, masks, decoder temporaries and reopened master; the managed `DeliveryImage` still owns its encoded input snapshot, while original/adjusted masters are no longer retained |
| Managed source loading | Private source file and verified immutable encoded snapshot, including transient chunk/join costs |
| Managed publication | Returned TIFF/diagnostic bytes remain owned across private-file writes, staged publication and streamed integrity checks |
| CAS and response | Large result buffers are dropped before mutation; receipt, bounded preview data URL, inventory and projection still remain |

These rows describe simultaneous ownership, not a summed process RSS limit.
Native source snapshotting, decoded-image/ICC intermediates, masks and hashes,
preview/overlay arrays, renderer scratch, publication/provider buffers, state and
response serialization still require complete managed qualification. Publication
also reserves geometry-derived private/staging/store copies on each filesystem.
Lease ownership must persist through worker and I/O quiescence, failure and
cancellation; releasing arrays early does not grant another operation that lease.

The CPU report adds `rss_high_water_checkpoints` after native revision helper
return and after independent reopened-output verification. The helper includes
decode, planning/execution, TIFF encoding/reopening, previews/overlays and receipt
validation. Both checkpoints are **cumulative process high-water
marks**, including earlier imports/snapshotting/work. Their difference is not a
phase-exclusive allocation or managed operation cost. Existing complete-worker
peak, incremental peak, resource guards and measured-margin decisions remain
unchanged. Historical reports without these optional checkpoints remain usable.

The generic native admission formula, historical measured floors, inspection
reserve and configured limits remain unchanged. Without the separately matched
managed profile below, the retained 30 MP four-region request still reserves about
10.35 GiB against 6 GiB effective default bulk capacity and remains ineligible.
Tiled transfer or an offline measured peak alone cannot authorize a lower profile,
larger budget or live feature enablement.

## Managed native resource qualification

The named `tp.photo.native_sdr_managed.v2` profile has a fixed **6 GiB** reservation;
it does not change the 12 GiB process-local ceiling, effective 6 GiB inspection
reserve, worker/I/O limits or scratch/free-space protections. It applies only to
HTTP-owned native revisions whose server-verified inspection, actual qualified
publisher, Postgres authorities and frozen runtime/source identity match.
The lower resource estimate also requires the live, nonclosed transport owner
installed by the deployed application. A manually mounted router or direct HTTP
handoff without that owner retains generic costing, even if its response is
marked HTTP-owned.
Retained Postgres `JobRecord` inventories qualify with the exact local artifact
publisher and durable job/review stores. A separate canonical path supports
Unified `process` V5/V4 plans, one to eight inputs, through real Postgres
operational records and a bounded loopback S3-compatible publisher. It validates
immutable plan/binding authority and pins the operational store alongside the
job/review stores and publisher. Other plan families retain conservative costing.
Changing queue settings never migrates or qualifies canonical inventory.

The canonical pilot additionally requires the server-owned
`TP_PHOTO_REVIEW_S3_PUBLICATION_ROOT` to identify the actual host bind mounted as
MinIO `/data`. This is an operator physical-storage mapping, not a client path.
Use the qualification command's container/bind/port/image receipt to verify it
before deployment. The runtime pins its canonical path and device/inode together
with the real initialized S3 client's endpoint, configuration and object identity.
No credentials enter resource identities or public responses. Missing, changed,
remote or unsupported storage bindings retain conservative costing; the setting
does not migrate a bucket or make a VM named volume a host directory. Requalify
after changing the service image, command or bind mapping; those are trusted
deployment settings outside per-request runtime inspection. Scratch
claims cover both temporary and actual publication filesystems, with all existing
remaining-commitment, inspection-reserve and free-space checks. Ordinary S3
storage remains supported without this optional qualification setting.
Unknown, historical or conflicting authority retains conservative fallback
costing. No filename, client option, environment label or supplied inspection can
select the lower profile. Requests whose complete reservation does not fit still
fail with typed `413`; no downsampling, deletion or automatic budget change occurs.
The fixed profile is selected only when it lowers the existing conservative
estimate; smaller operations keep that estimate and its existing admission behavior.

Eligible raster evidence is bound to exact immutable source bytes: integer
8/16-bit RGB or straight-alpha RGBA TIFF, at most 30,000,000 pixels, contiguous
uncompressed strips and canonical sRGB ICC. A small structural parser bounds
classic/BigTIFF headers before opaque decoder use: 64 KiB of header/tag data,
32 tags, 64 strips, a 1 KiB closed description and a 4 KiB canonical ICC. It rejects
extra directories, padding, overlapping/out-of-file offsets, unsupported metadata
and unbounded layouts from eligibility. Other codecs/layouts do not inherit this
qualification. Presentation and processing-stage authority remain independent.

The pure-data `native_resource_contract.py` freezes the profile's functional
source closure, current `pyproject.toml` and `requirements/base.in` declarations,
CPython executable, five native extensions, package versions and Darwin arm64
runtime. Qualification checks every active direct core requirement against the
actual installed versions, using the host's requirement markers. Stale installed
project metadata cannot authorize an older dependency floor. Missing, conflicting
or unsupported requirements select fallback. The recorded supported environment
includes Pillow 13.0.0, certifi 2026.10.19 and packaging 26.3.
The pilot is specific to CPython 3.12.14 and the recorded environment;
another platform/build selects fallback. Set `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and
`VECLIB_MAXIMUM_THREADS` to `1` before starting the backend. A stale or incomplete
contract cannot grant capacity. It is reviewed release data; never regenerate it
merely to make a mismatched installation qualify.

The analytical profile takes the maximum simultaneous native phase allocation,
including snapshots, masks, hashing, codecs, previews and publication, and adds
bounded metadata/response allowance. The worst 30 MP RGBA/four-region analytical
case is about 5.065 GiB. The separate 4 GiB incremental-RSS acceptance ceiling is
an acceptance requirement, not an observed measurement: measured incremental peak
plus 50% must fit the fixed 6 GiB reservation. Projection/job metadata must satisfy
the profile's 8 MiB heap, 65,536-node and depth-16 limits. Source/codec eligibility,
metadata limits and backing-store identity are revalidated; all actual
authorities are pinned for the operation so awaited I/O and uncertain cleanup use
the same authorities.

Native HTTP preview/save retain their complete memory reservation and outstanding
request ticket through CAS, envelope serialization and response completion,
failure or repeated cancellation. A request-owned ASGI finalizer wraps the complete
deployed application, outside buffering, security and error middleware. An inner
response returning after a middleware body handoff does not release capacity;
the downstream send and owned cleanup must first become quiescent. Decode and
photographic I/O slots are released when expensive work is quiescent, before CAS.
A slow response cannot release memory early or authorize a second bulk operation.
Direct service dictionaries, legacy revisions and existing download ownership
retain their prior contracts.
The selected source must first receive authoritative catalogue/scene inspection
in this backend process; a cold or expired inspection cache selects fallback.
TIFF decoding explicitly closes its owned in-memory file before color allocation,
so retained tifffile metadata cycles cannot keep encoded snapshots resident until
cyclic garbage collection.
Normalization reuses its independent float32 cast buffer, and decoded rasters and
mutable compositor output release before subsequent color/hash allocations;
historical pixel, master-identity and receipt comparisons remain exact.
Path-backed snapshots read at most their checked file size plus one growth-detection
byte, while retaining the configured maximum, confined descriptors and exact hash
checks. They do not allocate against the much larger generic snapshot ceiling.

### Private managed CPU qualification command

Use a fresh private namespace and a dedicated, already migrated Postgres database
whose name ends in `_test`. `TP_MANAGED_QUALIFICATION_DATABASE_URL` is local-only;
the tool does not reset a database, reinstall a runtime or dispatch inference.
Provision a separate CPU qualification environment from the current pinned
`requirements/base.txt`, `requirements/dev.txt` and `requirements/ci.txt`, using
`requirements/constraints.txt` and pip 26.2.2. Install the reviewed
project metadata and run `pip check` in that environment. Include `psutil==7.2.2`
as the explicit qualification dependency already pinned by the supported Darwin
ML lock; it does not require installing that ML environment or its umbrella.
The pilot preserves the canonical core environment and existing model runtimes.
Its closed manifest contains `schema: tp.photo.managed_materials_manifest.v1` and
one to eight ordered `sources`, each with an absolute `path`, verified `sha256`
and the closed native Materials `recipe`. It seeds only unique private succeeded
jobs and uses real signed authorization, the deployed application and middleware,
the shared router, `AppReviewAccess`, Postgres CAS and the selected publisher.
The transport barrier waits for both a blocked outer send and the inner native
response's actual return before checking its complete reservation and ticket.
This is managed backend qualification;
it does not claim a signed-in frontdoor/browser pilot or photographic acceptance.

```bash
PYTHONPATH=.:src /absolute/private/core-env/bin/python -m transformation_portal.photographic_delivery.managed_qualification \
  --manifest /absolute/private/retained-manifest.json \
  --expected-profile-identity <reviewed-profile-sha256> \
  --output /absolute/private/qualification/new-run \
  --repeats 3 --preflight-only
```

For canonical qualification, add `--provider canonical-s3`,
`--canonical-plan /absolute/private/retained-plan.json`,
`--canonical-bindings /absolute/private/unified-bindings.json`,
`--s3-container tp-native-canonical-minio-<session>` and
`--s3-publication-root /absolute/private/minio-data`. Supply explicit local-only
`TP_MANAGED_QUALIFICATION_REDIS_URL`, `TP_MANAGED_QUALIFICATION_S3_ENDPOINT`,
`TP_MANAGED_QUALIFICATION_S3_BUCKET`,
`TP_MANAGED_QUALIFICATION_S3_ACCESS_KEY_ID` and
`TP_MANAGED_QUALIFICATION_S3_SECRET_ACCESS_KEY` through the private environment.
These are isolated test-service credentials, not AWS account credentials. The
command rejects remote endpoints and records the owned running MinIO image,
version, loopback port and read-write `/data` host bind before launching work.
It copies unchanged original references, rebinds only their owned input root,
and imports retained TIFF fixtures through real Redis locator delivery, Postgres
claim/fence and immutable S3 generation publication. No model executes. The
report labels this fixture import; it does not claim that the imported bytes
were newly inferred from that plan. Plan input order and retained fixture order
must describe the same scenes. Verification uses fresh Postgres authority and
streamed S3 source/product bytes after the worker exits. Local and publication
scratch are included in campaign planning and physical free-space monitoring.
The canonical plan retains all source/object copies and every committed product
plus its later verifier copy. It separately reserves the maximum active native
snapshot, private/staging and MinIO write transients; per-operation staging is
not multiplied across already quiescent repeats. The report records every
term. This lifetime calculation changes no configured ceiling, inspection
reservation or physical free-space guard. The verifier starts after the managed
worker exits; arbitrary concurrent campaigns are outside this bound.
The default `retained-local` provider and its command remain compatible.
`--process-timeout-seconds` keeps the historical 300-second default and accepts
an explicit integer from 1 to 900 for each supervised worker and verifier. The
maximum canonical pilot requests 600 seconds explicitly; it does not retry or
extend deadlines automatically. This validation timer changes no production
admission wait, service CPU limit, memory reservation or scratch protection.
An unmet private observation barrier writes at most 64 KiB of phase/admission
metadata and task/thread code coordinates before cancellation. The exclusive
`failure-diagnostic.json` excludes locals, arguments, source lines and image
buffers; telemetry failure preserves the original timeout and owned drain.
Transport ownership barriers hold successful native responses only; error
responses pass through unchanged so their actual status and typed error surface.

Lower-cost S3 eligibility covers only HTTP loopback with the measured SDK
configuration. HTTPS, custom model loaders/search paths, altered service models,
or changed cached operation/shape semantics retain conservative costing. The
actual GET/HEAD/PUT/DELETE/list models and streaming behavior must match bounded
packaged models and SDK extras. Validation releases its private reference models
without relying on garbage collection; it does not change the live client.
Only the pinned SDK handler's exact before/after `requestUri` and `authPath`
states are normalized in private comparison copies. Ordinary S3 requests may
perform that rewrite without invalidating their resource scope; other HTTP,
service-model and cached-shape changes retain conservative costing.

The reviewed v2 profile identity is
`036f0468471ecc551049aa822805cd3e04368e50ad1bb289b58f119156d6466e`.
Its closure includes 69 source/declaration/schema files, 37 package versions,
eight compiled extensions, actual verified payload trees for 13 SDK/validator/
Redis distributions, and eleven standard-library transport/JSON/CSV source files.
Actual package bytes, tree membership and wheel RECORD hashes/sizes must agree;
version strings or RECORD metadata alone do not establish payload identity.
RECORD reads are bounded to 4 MiB and 32,768 total CSV rows before file-entry
materialization, including unrelated and missing-file rows. Noncanonical
metadata paths, malformed records and unsupported distribution adapters fail
closed without selecting the lower profile.

Set the five thread variables above and the dedicated database URL before running.
Inspect the planning report; remove `--preflight-only` and choose another fresh
output namespace to measure. The current source/runtime must match the expected
profile hash. The worker uses repeated cohorts in one process with its original
baseline; cumulative high-water checkpoints are not exclusive phase allocations.
The local campaign additionally requires 20 GiB of available host memory before
launch and protects 8 GiB while running. These physical-headroom guards do not
increase the admission controller's 12 GiB accounting ceiling or its 6 GiB native
reservation. Scratch planning includes source copies, all retained trial products
and publication copies; insufficient physical space blocks launch.
Independent TIFF/PNG/pixel/receipt checks and Postgres event/state reconciliation
run after that worker exits, in a separate process. Preserve failed namespaces
for diagnosis; retained sources and finalized deliveries are never cleanup targets.

Default qualification includes a same-tenant inspection while the native worker
still owns its complete lease. A single repeat or `--no-concurrent-inspection`
produces observations without full declared-cohort qualification. Controlled
`--scenario cancel_helper`, `cancel_publication` and `publication_failure` runs
require one source/one repeat and verify owned cleanup and unchanged review state.
`--scenario cancel_transport` also requires one source/one repeat. It cancels the
outer send after commit and inner response return, repeats caller cancellation
while send cleanup is held, then recovers and independently verifies the single
committed revision and its products. Transport cancellation preserves that
committed state; it does not undo publication or repeat the revision POST.
These fault observations alone do not qualify a photographic cohort. Host/provider
rollout and Richard's scoped native photographic decision remain separate gates;
AWS/GPU validation stays on hold.
Only all-zero-strength preservation recipes can establish declared-cohort
qualification. Nonzero runs retain resource observations and product/receipt
verification, but do not establish protected-delta or photographic acceptance.

### Reviewed pilot observations

The historical retained-inventory `tp.photo.native_sdr_local.v1` profile identity
was `005953dfe79ff58623d66e424452990732fdcae4e3928f3c9a75a96220671bab`.
It is preserved foundation evidence; it cannot authorize the current expanded
source/runtime/provider contract.
On its supported Darwin arm64/CPython runtime and retained-inventory providers,
three repeats of the eight retained 6708×4472 RGB16 photographs and an eight-source
30 MP RGBA8 synthetic cohort each completed 24 saves. Separate 30 MP RGB8,
RGBA8 and RGBA16 three-repeat probes, one bounded nonzero observation and the
20 MP Postgres success/cancellation/publication-failure modes brought the total
to 61 saves and 305 independently verified products. This includes one committed
revision and five verified products recovered after transport cancellation.
All 300 products from the 60 preservation saves verified; every preserved native
TIFF had zero 16-bit code changes. Sources remained unchanged and all reservation
counters drained. The nonzero observation is separate from
preservation and photographic acceptance.

The original alpha warm-cohort preflight blocked on physical scratch headroom
before worker launch and produced no saves or products. Its report remains
separate provenance; a fresh retry completed the measured 24 saves and 120
products under the unchanged runtime, resource ceilings and free-space guards.

The worst complete worker peak was 4,296,785,920 bytes; its measured incremental
peak plus 50% was 6,333,136,896 bytes, below the fixed 6,442,450,944-byte reservation
by 109,314,048 bytes.
Same-tenant inspection progressed while the complete native lease and encoded
output were held after helper return and after the inner response returned while
the deployed outer transport remained blocked. A distinct uncached crop verified
a fresh source read at that transport barrier. This is not an exclusive phase
measurement or a worst-case overlap measurement of two live raster computations.
Reports, source snapshots and products remain private, outside Git and finalized delivery;
these historical observations do not enable production flags or qualify canonical generations.

## Paged catalogue and review navigation

The additive `tp.photo.scene_page.v1` catalogue supports up to 64 scenes. Use
`GET /scenes?limit=8&status=all&sort=scene_id` under either API prefix. Page size
defaults to eight and cannot exceed 16. Filters are `all`, `unreviewed`,
`needs_revision` and `approved`; sorting is deterministic by `scene_id` or
`label`, with scene identity breaking equal-label ties. Follow `next_cursor`
with the same limit/filter/sort. The cursor binds the authenticated actor,
tenant, job, committed inventory, saved review projection and process lifetime.
It is signed, bounded and interchangeable across the v1/v2 aliases. A changed
snapshot or backend restart returns `409 CATALOGUE_CHANGED`; discard the old
pages and refresh. Invalid cursor bytes fail closed.

This index uses committed inventories and saved projections, without reading
image pixels, descriptors, ICC carriers or image headers. It stays outside
scarce photographic I/O admission. Each summary explicitly reports
`source_validation: not_checked`: its recorded decision is navigation metadata,
not a fresh source approval or delivery authorization. Public relative artifact
summaries distinguish masters, previews, depth and evidence; physical paths and
storage keys remain private. Every request reauthenticates and rechecks job
availability, including after a warm browser cache.

Load `GET /scene?scene_id=<id>&catalogue_identity_sha256=<page-identity>` for the
selected scene's `tp.photo.scene_detail.v1` response. This retains the rich
variants, properties, presentation policies and current verified decisions of
the original catalogue, restricted to one scene. It uses the existing atomic
resource lease and verifies the selected components. Mutations and renders
discover only their selected sources; approval, source identity, presentation
and publication validation remain authoritative. The existing unpaged `GET`
retains its 32-scene limit and response contract. Larger jobs return
`PAGED_CATALOGUE_REQUIRED` there, with no truncation or eager decode.

The workbench offers one card per scene, expandable related variants, filters,
previous/next scene, Next unreviewed and deterministic page navigation. A decision
shows its reviewer and date. Scene-card arrow keys move focus; native-view arrow
keys retain their pixel-pan meaning. Unsaved notes or revision adjustments block
navigation until explicitly discarded or saved. Selected scene/version,
comparison, presentation and ROI survive reload within bounded actor/job-scoped
session metadata. No images, API keys, storage locations or filesystem authority
are persisted. The artifact rail follows the selected published scene, while a
revision retains its own exact relative selector. A selected display or cached
image never grants approval, and no bulk approval shortcut is added.

The 64-scene contract also applies to approved-source selectors, labels and
lighting-group membership. Six export presets, 64 delivery records, 128 saved
revisions, the 2 MiB projection and 128 KiB event bounds remain unchanged. Delivery
history can describe at most 448 applied derivatives (64 times six presets plus
one native precision companion per source); this metadata ceiling does not grant memory or scratch.
Every chosen source is freshly validated before export. Existing calculated
memory, disk and publication-copy limits may still reject a large operation.

Deploy the backend capability first, then independently set
`TP_PHOTO_REVIEW_PAGED_UI_ENABLED=1` on the managed frontdoor (or backend for
standalone direct-debug bootstrap). The default is off; recorded presentation,
delivery history and admission retain their separate flags. Roll back the paged
workbench by disabling this flag and reloading. Keep the additive read endpoints
deployed for persisted jobs above 32 scenes; an older unpaged application cannot
serve their full catalogue. No migration, retained-file rewrite or approval
reset is required. No new backend pixel cache is introduced. The existing
identity/policy/view/ROI-bound browser cache keeps its 12-entry/40 MiB limit and
excludes mutable approval versions from pixel keys.

The design deliberately rejects merely raising the eager catalogue ceiling or
decoding every page: both retain the per-mutation full-set cost. Small selected
detail calls preserve the established verification boundary without creating a
second source authority. A 50-scene synthetic workflow proves navigation and
resource behavior; it does not establish photographic improvement or production
acceptance of a new photographic set.

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

Both browser views are **8-bit encoded-sRGB display derivatives**. Recorded
presentation uses the authority described below; the explicit clipping comparison
clips to the display range. Neither changes the master or exposes all extended/HDR values. A native crop preserves spatial
sampling, not master bit depth. Retain and inspect/download the precision
authority separately. Do not infer 16/32-bit precision from an 8-bit browser PNG.

Stable automation hooks include `data-ui="photographic-review"`,
`photographic-scene`, `photographic-primary`, `photographic-comparison`,
`photographic-comparison-mode`, `photographic-inspection`, `photographic-presentation`, `photographic-crop-size`
and the crop/pan controls. Existing portal routes, selectors and managed session
behavior remain in use.

Fit/native previews use an actor/job/source/presentation-scoped blob cache bounded to 12
entries and 40 MiB. Contact-sheet thumbnails load lazily and retain at most 64
entries of at most 1 MiB each. Scope changes abort outstanding requests and
release their object URLs; no image bytes or API keys enter browser storage.

## Recorded photographic presentation

`GET /render` and `GET /native-crop` accept `view=recorded` or
`view=clip_srgb`. Omitting `view` preserves the historical clipping comparison.
The workbench requests recorded presentation only when the catalogue advertises
it; otherwise it explicitly labels and requests the clipping fallback. A direct
recorded request without retained authority returns `409 PRESENTATION_UNAVAILABLE`.
Neither filenames, dtype, pixel ranges, ICC encoding nor a TIFF description
establish processing stage.

Catalogue variants expose `presentation.available_views`, the nullable validated
policy and its digest, and the separate `presentation_policy_sha256`. The immutable
`tp.photo.presentation_policy.v1` binds exact source identity, processing stage,
recipe and the retained descriptor or committed revision receipt. Its
`renderer_contract=tp.photo.presentation_renderer.v1` identifies the existing
versioned SDR numerical transform and render-before-reduction ordering. A change
to those numerical semantics requires a new renderer contract and matching
historical support; do not silently reinterpret an existing contract.

Verified V6 working masters use their recorded render recipe, including exports
whose requested destination recipe differs. Export receipts retain the requested
recipe separately from the actual applied recipe. Descriptor-bound
V6 delivery TIFFs are already display referred: inspection, SDR export and
contact sheets do not apply another tone transform. Destination resizing,
sharpening, encoding and quantization remain explicit export operations. An
unrecognized or legacy minimal descriptor has no recorded presentation authority;
its existing clipping view remains available. Conflicting retained recipe,
stage, path or content identities fail closed.

Recorded fit views render native working pixels before area reduction, retaining
premultiplied-alpha reduction. Native views crop at the exact source coordinates
before the pointwise transform. Responses preserve source and geometry headers
and add `X-TP-Presentation-View` and `X-TP-Presentation-Policy-SHA256` (`none` when
unavailable). The browser verifies those values and includes view and policy
identity in cache and in-flight keys. Policy changes do not reuse old pixels.

New `tp.photo.revision.v2` receipts explicitly record output presentation and
renderer contract. A revision from a display-referred delivery inherits that
stage even though its separate precision carrier is float32 TIFF; bounded edits
do not justify a second tone transform. Recognized historical
`tp.photo.revision.v1` receipts retain their original working-master/preview
render semantics. Unknown receipts receive no inferred recorded view. Receipt
hashing binds the completed receipt and never embeds its own policy digest.
When the input has no recorded authority, an explicit revision request retains
the existing legacy operation: save the unbounded adjusted master and render a
perceptual preview. Its new receipt marks
`input_presentation_authority=explicit_legacy_revision_operation`, with no input
policy identity. Only that actually executed output operation establishes the
new revision's recorded view; the parent's stage stays unknown and its recorded
view stays unavailable. Conflicting recognized authority cannot use this fallback.
New clients also send optional `presentation_policy_sha256` with revision
requests; a changed supplied policy fails before preview or publication.

Decisions may carry optional `inspection_presentation` containing view and policy
digest, validated against the current source. This is additive inspection audit
metadata. It does not invalidate or expand an existing source approval, approve
destination proofs, assert byte integrity, or establish production acceptance.
The managed review projection needs no database migration; preserve receipts and
decisions on rollback and deploy matching backend and workbench code when using
the additive request fields and V2 revision semantics.

Roll out the backend API before enabling the independent recorded-view UI flag:
`TP_PHOTO_REVIEW_RECORDED_UI_ENABLED=1` advertises
`photographicRecordedPresentation` through bootstrap. Configure the flag on the
managed frontdoor for authenticated sessions and on the backend for standalone
direct-debug sessions. The default is disabled. Set the corresponding flags to
`0`, restart the affected frontdoor/backend processes and reload the workbench to return
the UI to its explicitly labelled clipping view; API recorded-view support and
the no-second-render guard remain available. Delivery-history UI and resource
admission have separate rollout controls. This does not rewrite existing review
decisions or retained generations.
The UI flag is intent, not backend capability. An older catalogue without
presentation metadata retains the omitted-view clipping request, original
image-header checks and original decision/revision request shapes, even if the
UI rollout was requested. Advertised presentation capability enables strict new
header verification independently of the UI flag.

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

Default V3 V2 finishing uses processing version `1.2.0`, contract `encoded_srgb_v1`.
Opt-in `--geometry-recipe normalized_crop_support_v1` pairs with version
`1.3.0` without native Materials, normalizes orientation once, preserves crop support and requires a
source-bound depth/mask handoff. Corrected Materials V3 accepts only opaque
8-bit sources; RAW, PBR, reconstruction and spatial linear ingest fail closed.
Historical plan/cache identities and retained files remain unchanged. See the
[versioned geometry contract](PHOTOGRAPHIC_GEOMETRY_AUDIT.md#versioned-corrected-recipe)
for its precision, cache and acceptance limits.

The separate CLI native Materials selection binds explicit MaterialsV4 evidence
and a complete response policy under `native_sdr_v1` / V2 `1.4.0`. It edits the
original bounded 8/16-bit native master before encoded-domain finishing, with
no quantized image carrier. This does not change Review approval scopes,
enable managed native-Materials admission or establish photographic acceptance.
See the [native SDR Materials contract](PHOTOGRAPHIC_GEOMETRY_AUDIT.md#native-sdr-materials-finishing).

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

V3 retains WebP/BMP admission: its private lossless PNG carrier preserves source
ICC/EXIF evidence before shared color validation. Untagged WebP/BMP require the
same recorded sRGB assumption or explicit encoding assertion as untagged
PNG/TIFF; unsupported BMP calibration or malformed color data fails closed.
When Materials V3 is enabled, the source policy also governs its existing
proxy. The validated source transform converts that proxy to bounded encoded
sRGB using the selected geometry recipe. The default retains historical
depth/segmentation preprocessing; the corrected recipe records normalized
orientation and native crop support. Its internal
TIFF/PNG carries the resulting sRGB profile, and V2 validates that encoding with
`auto` rather than reapplying an original linear-source assertion. The source
preparation and proxy precision limits are recorded in segmentation metadata.
The established clipped `legacy_linear_srgb` RAW proxy is supported separately;
debug RAW preview is depth-only and cannot authorize a Materials color handoff.

Float32 input or a conversion requiring extended/out-of-range values fails
closed with guidance to Unified/V5/V6. The legacy exact-copy passthrough remains
byte-exact only when no processing/explicit encoding is requested. V2 plan
configuration schema `tp.stage.config.lux.v2.v2`, Materials configuration
`tp.stage.config.lux.materials_v3.v2`, fingerprints and run-card
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
Managed downloads use the authenticated frontdoor origin. In `direct_debug`,
thumbnails and delivery/companion downloads supply the page-local API key in
request headers, never in URLs. Large direct-debug downloads stream to the
operator-selected file through the browser's File System Access API. Where that
API is unavailable, the bounded fallback requires a known Content-Length of at
most 64 MiB and enforces that limit while reading. Use a supported browser or
the managed frontdoor for larger or unknown-length downloads.
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
Verified download scratch remains owned until response execution ends, including
invalid/malformed ranges, failed sends and repeated cancellation; cleanup does
not depend on a successful-response background task.

## Opt-in bounded resource admission

`TP_PHOTO_REVIEW_ENABLED=1` enables process-local photographic admission. It is
**off by default**. The initial pilot requires one backend process; these
accounting reservations do not reserve operating-system RAM or disk, coordinate
multiple processes, or control unrelated filesystem writers. Authenticate the
actor, tenant and parent job before entering a tenant queue; reference discovery
and photographic IO run only after admission. Delivery history and review-state
reconciliation remain lightweight control operations.

Defaults are 12 GiB photographic memory, a 4 GiB minimum inspection reserve and
an 8 GiB bulk class ceiling; two decode workers with at most one bulk operation;
and two IO slots with one strictly reserved for inspection. Each tenant may own
two operations, at most one noninteractive. Export and downloads serialize while
inspection can materialize concurrently. A bounded photographic thread pool
keeps admitted inspection separate from unrelated default-executor work.
The bulk memory ceiling applies to the aggregate of all noninteractive owners,
including failed work that has released worker slots but still retains buffers.

The measured inspection reserve takes precedence over its minimum. Baseline
CPU probes at `f79f49207ed2e4589b006017dec90edd51137cb2` measured a 40 MP float32
TIFF fit peak increment of 3,465,723,904 bytes. The 1.5-times measurement margin,
whole-GiB rounding and one-GiB allocator allowance protect **6 GiB**, leaving at
most 6 GiB for bulk work under the default process ceiling. Per-operation cost
is the greater of its conservative allocation estimate and 1.5 times measured
peak. Server-authored, hash-bound inspection geometry may reduce that cost;
verified source headers must remain within the admitted geometry before decode.
Verified catalogue inspections may supply that geometry before any approval.
Their metadata-only cache is bound to tenant, job, current inventory, path, SHA
and composite source identity, capped at 1,024 entries per service/256 per tenant
with least-recently-used eviction, and cleared at shutdown. It never stores
pixels or decisions. Missing or ambiguous entries retain conservative costs;
ordinary catalogue → preview → save does not require preliminary approval.
A native-alpha 40 MP revision exceeded the 8 GiB class ceiling after the margin
and is rejected rather than downsampled. Final recorded-path probes bind
presentation math `3f1f3a61da03855ff8b3a37dc75545e0c244da57` at report HEAD
`2f6faa712a6aea3e3b8cf82b1f32de26fe0928d5`, with eight source hashes verified:
24 cases completed and transparent SDR export was correctly rejected. The
largest recorded-fit increment was 3,506,323,456 bytes; the controller retains
higher peaks observed across all probes. These CPU allocation measurements do
not establish photographic or live-service acceptance. Review the limits before
enabling the pilot; never increase ceilings automatically to make work fit.

Outstanding requests are bounded to 32 per process/eight per tenant, including
coalesced render subscribers. Waiting work is bounded to 24 per process/six per
tenant, with five-second inspection and 30-second bulk/transfer deadlines.
Queues preserve FIFO within tenant/class and round-robin across tenants.
Equivalent in-flight renders share work only within the same tenant, job,
source/descriptor/receipt inventory, review version and render parameters; each
subscriber revalidates its own actor and source binding before receiving bytes.
If the shared owner's source read fails with 401/403, an independently authorized
subscriber may make one private attempt under its own complete resource lease.
That attempt never coalesces again or retries a source-integrity conflict;
ordinary shared reads and revocation after computation still compute once.
Mutation and export requests never coalesce. Disconnecting the final subscriber
cancels a queued render; a running worker retains its resources until it is
quiescent.

Scratch accounting is capped at 64 GiB, with 2 GiB protected for inspection and
an additional 2 GiB physical free-space floor. Complete export scratch includes
selected sources, decoded-geometry-derived products, ZIP and publication copies;
its preflight additionally preserves inspection headroom. Geometry-derived
claims cover the temporary and local publication filesystems separately. Every
physical check includes all owners' remaining commitments on that filesystem;
only this operation's tracked private allocation blocks reduce its outstanding
commitment. Existing retained directories never grant allocation credit.
Insufficient physical space never authorizes deletion of retained data. Downloads own their transfer
and scratch leases through response execution and cleanup, including failed
sends, invalid ranges and repeated cancellation. Publication uses its existing
operation lease and immutable-key cleanup fences; it never recursively acquires
capacity. Once workers and publication are quiescent, release decode/IO slots
before review compare-and-swap or rollback cleanup. Release memory only after
buffer-owning method and worker frames are quiescent and their retained locals
have been dropped; failed publication keeps memory charged through that unwind.
Private scratch remains owned until final cleanup. Those retained publication claims continue to
count against the bulk scratch ceiling and protect inspection headroom, without
holding released worker slots. Outstanding download requests remain charged
until their response finalizer completes. Shutdown stops admission and drains
owners before stopping the pool.
Legacy local-file discovery streams bounded chunks into a private verified
snapshot instead of collecting a whole file inside the transfer memory profile.

Configuration names use `TP_PHOTO_REVIEW_` plus the uppercase field names:
`MEMORY_BYTES`, `INTERACTIVE_MEMORY_BYTES`, `BULK_MEMORY_BYTES`, `DECODE_SLOTS`,
`BULK_DECODE_SLOTS`, `IO_SLOTS`, `INTERACTIVE_IO_SLOTS`, `NONINTERACTIVE_SLOTS`,
`TENANT_SLOTS`, `TENANT_NONINTERACTIVE_SLOTS`, `OUTSTANDING`,
`TENANT_OUTSTANDING`, `QUEUED`, `TENANT_QUEUED`, `INTERACTIVE_WAIT_SECONDS`,
`BULK_WAIT_SECONDS`, `SCRATCH_BYTES`, `INTERACTIVE_SCRATCH_BYTES`, and
`SCRATCH_FLOOR_BYTES`. Byte values are positive integer bytes, durations are
integer seconds, and `ENABLED` accepts only `0` or `1`. Invalid configurations
fail closed; limits cannot eliminate the measured inspection reservation.

Overload preserves the existing API envelope with `RESOURCE_EXHAUSTED`, a
bounded reason and `scope: process`. Tenant caps return 429; transient shared
capacity, queue deadlines and disk pressure return 503. Both are retriable with
`Retry-After: 1`, which is not an estimated completion time. Permanently
oversized operations return non-retriable 413. Publication preserves typed
overload and retry metadata after cleaning up only its owned immutable keys.
No overload changes precision, creates approval, raises a budget, or deletes
source/delivery data. Roll back by
setting `TP_PHOTO_REVIEW_ENABLED=0` and restarting matching API code; retained
review decisions, revisions, deliveries and bytes require no migration.

## Validation and deployment limits

Focused source contracts from the repository root:

```bash
.venv/bin/python -m pytest tests/test_photographic_delivery.py \
  tests/test_photographic_native_master.py tests/test_photographic_presentation.py \
  tests/test_package_verified_delivery.py \
  tests/test_v2_color_contract.py tests/test_ad_editorial_post_pipeline_paths.py -q
.venv/bin/python -m pytest tests/orchestrator/test_photographic_review_access.py \
  tests/orchestrator/test_photographic_admission.py \
  tests/orchestrator/test_photographic_memory_ownership.py \
  tests/orchestrator/test_photographic_review_http.py \
  tests/orchestrator/test_photographic_review_store.py \
  tests/orchestrator/test_photographic_review_thread_ownership.py \
  tests/orchestrator/test_photographic_review_response.py \
  tests/orchestrator/test_photographic_coalescing_authorization.py \
  tests/evals/test_apex_metrics.py tests/evals/test_property_quality.py -q
.venv/bin/python -m pytest tests/test_app_route_inventory.py -q
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
422,250/104,750; this is a scoped PR4 increase of 750/250 bytes for the
actor/job-bound scene-to-rail selector and independent bootstrap feature. The
selector preserves exact original/revision routes across normal job refreshes
while keeping non-photographic route fallback unchanged. Existing Review JS remains capped at
55,000/12,500. The photographic viewer retains its 54,400/18,400 JS and
6,000/1,600 CSS limits. Shared model/recovery/navigation support is a separate
deferred, manifest-backed asset capped at 20,000/7,500; the viewer import binds
its exact content fingerprint. Existing Review and photographic-viewer budgets
remain unchanged. The photographic allowance includes authenticated
thumbnail fetches and bounded/streamed direct-debug downloads. Rebuild and validate current bytes rather than
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

## Committed delivery history and uncertain responses

`GET /deliveries` uses the same authenticated job and tenant boundary as review,
without discovering sources, decoding photographs, or checking stored delivery
bytes. `limit` defaults to 20 and is bounded to 64. Follow the returned
`next_cursor` for older entries in reverse committed append order. Its snapshot
membership excludes later exports; current recorded approval observations may
change between pages. Invalid or mismatched cursors fail closed; refresh history
if its committed prefix changed. The existing 64-delivery creation cap and review
projection/event size limits remain in force.

History returns only explicit public identities, actor fields, creation time,
expanded destination recipes, archive/file identities, and authenticated file
URLs. Physical bindings, storage keys, raw receipts, inventories, and arbitrary
saved fields are excluded. Older entries retain null request metadata. Approval
at creation and current recorded approval are separate observations. Source and
file validation are `not_checked`: a history row does not prove present file
availability, intact source bytes, destination proof approval, or photographic
production acceptance. Downloads retain their existing live identity, approval,
authorization, and byte-integrity revalidation.

An optional `request_correlation_id` on `POST /delivery` is 1–64 ASCII letters,
digits, dots, underscores, colons, or hyphens. The server records it atomically
with a `tp.photo.delivery_intent.v1` fingerprint and the original approval
version. That fingerprint includes authenticated scope, original review version,
ordered exact source and recorded presentation identities, effective labels, and
ordered expanded export recipes. It does not make export admission idempotent.
Use `GET /deliveries?request_correlation_id=<token>` to reconcile an uncertain
response, never to authorize a retry.

The portal freezes each submitted request and its original review version. A
lost response enters `outcome_unknown`, then performs GET-only reconciliation.
A single matching saved intent recovers its existing delivery. Ambiguous matches
or an unchanged review version keep export blocked. A complete lookup with no
match and a strictly newer review version proves that the original version can
no longer commit; a new export still requires an explicit operator action after
review. No automatic retry advances the frozen version. Bounded actor/tenant/job
session metadata can recover the attempt across page reloads. History-enabled
export requires the frozen attempt to be written and read back in its exact
actor/job scope before POST; unavailable, full, or non-retaining session storage
blocks submission with an actionable diagnostic. Storage failure never turns an
unknown outcome into success or authorizes a retry.
History itself remains available after reload without browser metadata.

After resolving an original attempt, the portal keeps that exact frozen record
until an authoritative catalogue GET reaches at least the reconciled review
version. A failed or stale refresh enters `refresh_required` and keeps new exports
blocked; manual reconciliation repeats GET requests without changing the original
body or version. Unreadable or malformed recovery storage is distinct from an
absent record: it cannot be overwritten or treated as permission to submit. Once
storage recovers, reconcile its retained scoped attempt before another export.
A local rate-limit cooldown is checked before creating or saving an attempt, so
an unsent request creates no orphan recovery record. After the cooldown expires,
the operator may explicitly submit a fresh export.
A received HTTP 429 rejection compare-clears only the exact saved attempt and
retains its cooldown; a failed or conflicting clear keeps export blocked. HTTP
503 and transport failures remain uncertain and require GET-only reconciliation.
An approved historical download is bound to its authenticated actor, tenant and
job independently of the latest export selection; the server still revalidates
its approval, source identity and bytes when it is downloaded.

No additional database migration is required beyond
`0008_photographic_review`. New entry/event metadata is additive and older
readers preserve it. Deploy matching backend/frontdoor versions: an older strict
POST request model rejects the new correlation field rather than silently
stripping it. Postgres restart and object-store download acceptance remain
separate from memory-mode and synthetic contract tests.

History also exposes ordered public source summaries from historical approvals.
For new exports, `applied_derivatives` is the bounded, allowlisted execution
summary of actual per-source/per-derivative operations, distinct from the intent's
requested destination presets. It records byte preservation, analytic extended
sRGB transfer, an executed SDR render, or preservation of an already recorded
display, including its actual render mode and presentation identity. Missing
historical execution summaries remain null; history does not reconstruct them
from filenames or pixel ranges.

### Independent UI rollout and rollback

Deploy the backend history API before enabling its dependent UI. The API remains
available independently for authenticated inspection/recovery. Set
`TP_PHOTO_REVIEW_HISTORY_UI_ENABLED=1` in the serving bootstrap environment to
advertise `photographicDeliveryHistory`; it defaults to off. The recorded-view
UI has its separate `TP_PHOTO_REVIEW_RECORDED_UI_ENABLED=1` /
`photographicRecordedPresentation` switch, also off by default. The managed
frontdoor owns its bootstrap environment; the backend's direct-debug bootstrap
uses the same names. Admission controls remain independent of both UI switches.

When history UI is off, the portal does not call the history API or submit the
new correlation field, preserving the older export request contract. Turning
history off does not remove saved deliveries or metadata. Reconcile any known
unknown export before disabling it: an already retained unknown attempt stays
blocked and can be recovered through the authenticated API or by re-enabling
history. Rollback preserves all review/delivery state and uses matching
frontdoor/backend versions; no retained source, derivative, or finalized package
is deleted.
