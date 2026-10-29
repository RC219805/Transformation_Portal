# Photographic geometry and orientation audit

This ledger records the coordinate authority used by the canonical photographic
path and the separate limitations reproduced in legacy V3/V2. The original PR6
audit baseline is `0a374b0624069fe322d8cbd1e24c1480d9fdfc4d`. The additive
versioned correction described below starts from
`5e5ef3f09039bc31352b31ff16d6bdad54ace4aa`; the historical recipe remains the
default. The documentation catalog binds the exact reviewed source bytes.

No model inference, runtime reinstall, retained photograph modification or
AWS/GPU validation is needed for these contracts. Synthetic coordinates and
controlled model-boundary results establish spatial consistency; they do not
establish representative depth accuracy, photographic improvement or production
acceptance. Existing photographs, approvals and finalized deliveries remain
separate from generated test outputs.

## Canonical coordinates and identities

Shapes are `(height, width)`. Points are `(x, y)`, where `x` is the column and
`y` is the row. Integer points identify pixel centers. A raster of width `W`
and height `H` has half-open edge bounds
`[-0.5, W - 0.5) × [-0.5, H - 0.5)`. This continuous coordinate domain is
distinct from the integer sampling centers `0..W-1` and `0..H-1`.

| Boundary | Authoritative implementation and recorded evidence | Coordinate contract |
| --- | --- | --- |
| Encoded source → canonical native master | [`decode_master`](../../src/transformation_portal/lux_depth_v4/photography.py), [`ImageMaster`](../../src/transformation_portal/core/image_artifact.py) | Exact source SHA-256; orientation applied once to RGB and alpha; canonical native shape; extended linear-sRGB pixels and separate straight alpha |
| Canonical native → model proxy | [`create_proxy`](../../src/transformation_portal/lux_depth_v4/photography.py), `ProxyTransform` in [`image_artifact.py`](../../src/transformation_portal/core/image_artifact.py) | Aspect-preserving reduction with rounded dimensions; bilinear pixel centers; bottom/right edge padding to multiples of 14; exact master content hash |
| Model proxy → native model depth | [`DepthEvidence`](../../src/transformation_portal/core/depth_evidence.py), [`V5 pipeline`](../../src/transformation_portal/lux_depth_v5/pipeline.py) | Float32 depth and boolean validity/support/sky on the exact padded proxy grid; source, proxy and master identity bindings; padding excluded from photographic support |
| Native model depth → canonical photographic depth | [`align_depth`](../../src/transformation_portal/lux_depth_v5/photography.py), [`restoration helpers`](../../src/transformation_portal/lux_depth_v4/photography.py) | Remove padding before restoration; bilinear depth weighted by valid support; nearest validity mask; exact native photographic shape |
| Model masks → native Materials evidence | [`prepare_inference` / `infer_materials`](../../src/transformation_portal/materials_v4/inference.py), [`MaterialEvidence`](../../src/transformation_portal/materials_v4/contracts.py) | Frozen proxy identity and transform; remove padding, then nearest-neighbor mask restoration; canonical-master coordinate space and source identity |
| Materials evidence → native operation | [`plan_response` / `apply_response`](../../src/transformation_portal/materials_v4/engine.py) | Source, native geometry, evidence, operation and prepared-plan identities must agree; confidence and calibration authorization remain independent of correct geometry |
| Native master → retained working/output files | [`shared photographic pipeline`](../../src/transformation_portal/lux_depth_v4/pipeline.py), [`V5 evidence verification`](../../src/transformation_portal/lux_depth_v5/evidence.py), [`write_delivery`](../../src/transformation_portal/lux_depth_v4/photography.py) | Recorded master/depth/transform shapes and content identities; native array dimensions retained; delivery TIFF has normalized orientation `1`; straight alpha retained |
| Retained working master → managed Review | [`NativeWorkingMaster`](../../src/transformation_portal/photographic_delivery/native.py), [`review rendering`](../../src/transformation_portal/orchestrator/photographic_review_service.py) | Selected descriptor/array/alpha component hashes form the composite source identity; native crops use exact integer native coordinates and reject out-of-bounds rectangles |
| Review → destination derivative | [`export_delivery`](../../src/transformation_portal/photographic_delivery/exports.py), [`export primitives`](../../src/transformation_portal/core/photographic_export.py) | Source-copy preserves exact bytes; native float/16-bit paths retain dimensions; optional SDR long-edge reduction uses its recorded preset and Lanczos; encoded derivatives do not acquire destination-proof approval |

`ImageMaster.content_hash()` binds semantic metadata, sample shapes/dtypes and
pixel/alpha bytes. The proxy binds that exact master hash. Depth evidence binds
the source and exact proxy/master content, and aligned evidence binds the native
master before finishing. Existing photograph descriptors and execution evidence
already carry these identities and transforms. The audit does not add a second
geometry authority or change those preimages.

## Source orientation mapping

For an encoded source pixel center `(x, y)` with stored width `W` and height
`H`, the EXIF mapping into canonical native coordinates is:

| EXIF orientation | Canonical `(x, y)` | Canonical `(height, width)` |
| --- | --- | --- |
| 1 | `(x, y)` | `(H, W)` |
| 2 | `(W - 1 - x, y)` | `(H, W)` |
| 3 | `(W - 1 - x, H - 1 - y)` | `(H, W)` |
| 4 | `(x, H - 1 - y)` | `(H, W)` |
| 5 | `(y, x)` | `(W, H)` |
| 6 | `(H - 1 - y, x)` | `(W, H)` |
| 7 | `(H - 1 - y, W - 1 - x)` | `(W, H)` |
| 8 | `(y, W - 1 - x)` | `(W, H)` |

RGB and straight alpha undergo the same permutation. Unsupported orientation
values fail ingest; absent ordinary-image orientation uses the established
orientation-1 default. JPEG expectations are derived from the stored decoded
JPEG samples, not the pre-encoding array, because JPEG quantization is lossy.
The asymmetric coordinate fixtures distinguish mirrors and rotations even when
dimensions alone cannot expose a mismatch.

The governed [RAW decoder](../../src/transformation_portal/lux_depth_v4/raw.py)
records `orientation_normalized: true`,
`orientation_authority: "LibRaw default metadata orientation"` and
`source_orientation: null`. It records visible/raw and decoded dimensions, but
does not supply a complete original-raster orientation mapping. That mapping is
**unavailable**, not inferred from equal or swapped dimensions. The verified
canonical native → proxy contract can still be described independently. This
audit adds no RAW runtime execution or acceptance claim.

## Proxy forward, inverse and valid bounds

For native dimensions `(H, W)` and resized content dimensions `(h, w)`, define
`sx = w / W` and `sy = h / H`. Rounding can make these two effective ratios
slightly different even though proxy preparation selects one aspect-preserving
scale. Use the recorded dimensions, not a guessed common ratio.

`ProxyTransform.native_to_proxy(x, y)` maps pixel-center coordinates as:

```text
xp = (x + 0.5) * sx - 0.5
yp = (y + 0.5) * sy - 0.5
```

`ProxyTransform.proxy_to_native(xp, yp)` maps content coordinates back as:

```text
x = (xp + 0.5) / sx - 0.5
y = (yp + 0.5) / sy - 0.5
```

The `native_bounds`, `content_bounds` and `padded_bounds` properties describe
the respective half-open edge rectangles as `(left, top, right, bottom)`.
Bottom/right padding adds no coordinate offset. Padding extends model context
by copying edge samples; it is not additional photographic coverage. Forward
mapping requires native-domain coordinates, and inverse mapping requires
unpadded-content coordinates. A padded-only coordinate has no distinct native
photographic inverse. Malformed, non-finite or out-of-domain points fail rather
than being silently clamped. A valid point whose floating-point result rounds
onto an excluded edge is kept representable within the destination domain;
this numerical boundary handling never admits an invalid input point.

The inverse is a coordinate inverse. It does not undo interpolation,
quantization, padding duplication or lost source detail. These helpers do not
resample arrays or alter `tp.image.proxy_transform.v1` serialization, its
identity, existing output pixels or historical verification recipes.
The v1 payload permits dimensions beyond those generated by `create_proxy`;
mapping those recorded dimensions does not certify that an arbitrary transform
was produced by canonical proxy preparation.

For canonical typed depth, model output must already match the recorded padded
grid. Restoration removes padding before bilinear resampling to native
photographic geometry. Numeric depth is normalized by valid bilinear support;
nearest-neighbor validity retains holes, with unusable values carried as zero
sentinels alongside their masks. Materials labels use crop-then-nearest
restoration and cannot acquire new labels through interpolation. Generic legacy
restoration helpers also accept a differently sized model array; their initial
model-to-proxy resize is not permission to bypass the stricter typed depth
contract.

## Fail-closed boundaries and legacy blockers

Canonical V5 alignment rejects a different source, native shape, transform,
master content identity or proxy content identity before finishing. Materials
inference rejects a stale prepared master/proxy transform before proposal or
classification execution. Materials response rejects changed source, shape,
mask evidence or prepared-plan identity before calling the native operation.
Matching array dimensions alone do not supply these identities.

Legacy V3/V2 has a separate, reproduced coordinate contract:

| Reproduced limitation | Fixture evidence | Required follow-up |
| --- | --- | --- |
| Standard V3 preprocessing retains encoded EXIF orientation, while V2 independently normalizes it | All eight EXIF cases in [`test_geometry_legacy_boundary.py`](../../tests/lux_depth_v3/test_geometry_legacy_boundary.py); asymmetric mirrors demonstrate a mismatch even with equal dimensions | A separately versioned source → model → V2 mapping and coordinated execution/cache recipe migration |
| V3 preprocessing center-crops to multiples of 14; Stage A subsequently stretches that cropped depth over the full encoded source extent | Production Stage A with a controlled coordinate backend: stored `31×45` source → `28×42` crop → `31×45` depth; removed border samples are not recovered | Explicit crop/support transform, compatible output semantics, cache/plan identity changes and representative photographic validation |
| Historical V2 accepts bare depth/mask arrays | [`V2 enhance_image`](../../src/transformation_portal/lux_depth_v3/v2_enhance.py) hands arrays to [`EnhancementStage`](../../src/transformation_portal/stage_graph/stages/enhancement.py), which resizes mismatched depth and drops malformed/mismatched masks; the array carries no source/frame authority | An explicit verified geometry handoff before new native editing; matching dimensions are insufficient |

These are blockers to treating historical outputs as a verified geometry
handoff. Existing V2 processing version `1.2.0`, default execution-plan bytes
and historical cache semantics remain unchanged. The legacy witnesses preserve
that behavior independently of the opt-in correction below. Source review
approval does not waive photographic acceptance of a different recipe.

## Versioned corrected recipe

V3 adds explicit `--geometry-recipe normalized_crop_support_v1`. The default
`legacy_encoded_crop_stretch_v1` retains the historical `1.2.0` V2 processing
contract. The corrected recipe pairs with processing version `1.3.0` and is
bound consistently into preprocessing, depth, Materials and V2 execution-plan
nodes, run cards and depth-cache fingerprints. Omitted and explicitly selected
legacy recipes retain the same serialized identities. The outer execution-plan
V1 and stage registry identities remain stable; mixed or unknown recipes and
versions fail validation.

The corrected path normalizes all eight EXIF orientations once from the same
immutable source snapshot used for hashing and decode. RGB and straight alpha
receive the same permutation. Its closed `tp.lux.geometry.v1` transform records
the source SHA-256, encoded and normalized dimensions, target reduction,
resized/model dimensions, center-crop origin and edge padding. The corrected
recipe fixes `model_max_long_edge=504` in the preprocessing plan. It reduces
larger inputs with Lanczos before the multiple-of-14 crop/pad and never
upscales smaller photographs. Corrected DA3 inference explicitly uses the
recorded bounded grid with `process_res=max(model_grid)` and
`process_res_method=upper_bound_resize`; it rejects a changed output grid rather
than hiding an additional internal resize. This keeps inference bounded near
DA3's historical default resolution, rather than sending native multi-megapixel
frames to the model. Restoration does not invent coverage for removed borders.

Numeric restoration removes padding, places cropped samples in the resized
canvas, then uses support-weighted bilinear restoration and nearest-neighbor
native support. Discrete masks use nearest-neighbor restoration. Model output
must match the exact recorded model grid and contain finite float32 values.
Unsupported depth is zero with a separate boolean support mask. The published
uint16 preview normalizes only supported samples and encodes unsupported pixels
as zero, including when supported model values are negative. This preview is
relative depth, not metric evidence.

The corrected Stage A → V2 boundary requires a closed
`tp.lux.geometry_handoff.v1` receipt. It binds the original source, actual V2
input bytes, transform, plan fingerprint when prepared, depth file and array,
derived native support and ordered mask array identities. Source and produced
depth hashes are rechecked before receipt creation; serialized masks must
match the authorized producer arrays. An enabled Materials carrier must match
its recorded producer hash. Receipt fields contain no physical paths. V2
revalidates the exact snapshot and every bound array before editing, rejects
unknown fields or mismatched geometry, and echoes the full receipt and its
identity in its report. V2 receives a float32 derivative of the verified
published uint16 depth preview, with that normalization recorded explicitly;
the original model-depth cache is a separate authority.

Corrected enhancement cache keys bind the actual image, depth, support, named
masks and alpha arrays as well as the handoff identity and effective settings.
A caller-supplied identity cannot substitute for those array identities.
Corrected material edits execute in sorted name order; the cache records
`sorted_name_v1` semantics so earlier order-dependent experimental entries are
not reused. Historical material ordering and cache keys remain unchanged.
Missing or invalid geometry fails before a cache read. Programmatic V2 entry
points validate the closed recipe before path discovery, output creation or
processing; unknown values cannot silently select historical processing.

The same native support and opaque-alpha guard covers all V2 operations,
including global tone, clarity and material changes. Unsupported borders and
partially transparent pixels retain their source RGB; alpha remains separate.
Corrected depth percentiles exclude protected samples, and depth smoothing
normalizes by the valid support rather than bleeding unknown zero sentinels
into supported tonal decisions.
Corrected Materials V3 first restores its model-grid RGB delta and masks into
the normalized source frame. It accepts only opaque 8-bit sources, because the
legacy Materials carrier cannot preserve 16-bit or alpha precision. Corrected
depth-only and V2 paths support bounded 8/16-bit ordinary-image inputs. RAW,
PBR generation, reconstruction and spatial linear ingest are unsupported with
this recipe and fail closed before execution. These restrictions do not change
historical admission.

Corrected model preparation quantizes 16-bit samples with exact integer
arithmetic and retains 8-bit samples directly, avoiding native float64
conversion buffers. Corrected V2 prepares its photographic color pixels once
and does not materialize a discarded integer RGB image. Its emitted filename
retains the V2 label when the verified material-mask inventory is empty.

For an ordinary-image directory, inspect the corrected plan without inference:

```bash
./.venv/bin/lux-depth-v3 --input-dir /path/to/ordinary-images \
  --output-dir /path/to/new-geometry-corrected-output \
  --geometry-recipe normalized_crop_support_v1 --plan
```

Use a fresh output namespace for an explicitly authorized real run. Corrected
published-artifact resume is initially disabled: each invocation republishes
and verifies its handoff. A complete governed identity-v3 depth-cache hit may
reuse model output, with the new recipe in its identity; it still creates fresh
publication evidence. Legacy cache entries and retained generations are not
rewritten or admitted as corrected artifacts. Rollback selects the legacy
recipe for new runs and keeps already recorded recipes and their matching
verification environment.

Synthetic regressions include independent coordinate/interpolation oracles,
all eight orientations, 16-bit and alpha cases, source/depth/mask/carrier
replacement, real V2 subprocess execution and cache-hit republication. They
establish engineering geometry and handoff consistency. They do not establish
model accuracy, native Materials precision beyond the stated carrier contract,
photographic improvement or acceptance of corrected real-reference outputs.
Existing preservation acceptance applies to its recorded retained-output
comparison, not this new recipe. New inference, retained-data rewrites and
AWS/GPU validation are not part of this implementation.

## Regression and release gates

The focused synthetic evidence exercises real production decode, planning,
graph, serialization, semantic verification, Materials lifting and native
operation paths with a controlled model/runtime boundary:

- [`test_proxy_coordinates.py`](../../tests/core/test_proxy_coordinates.py):
  explicit forward/inverse landmarks, half-open bounds, non-finite/malformed
  point rejection, representable near-edge points, padding exclusion and
  unchanged historical payload/master/proxy identities.
- [`test_geometry_roundtrip.py`](../../tests/lux_depth_v5/test_geometry_roundtrip.py):
  all eight orientations; odd/asymmetric 16-bit RGBA TIFF through V5 output and
  evidence verification; JPEG and RGBA PNG ingress; independent scalar
  depth/validity interpolation; alpha/protected-sample preservation; rejection
  of missing or conflicting depth authority before editing/publication.
- [`test_geometry_alignment.py`](../../tests/materials_v4/test_geometry_alignment.py):
  independent crop/nearest mask oracle; padding-only masks remain outside the
  photograph; stale preparation and changed source/shape/mask fail before
  native operations; exact supplied-mask native edit support and alpha.
- [`test_geometry_legacy_boundary.py`](../../tests/lux_depth_v3/test_geometry_legacy_boundary.py):
  production legacy preprocessing, EXIF/V2 discrepancy and Stage A crop/stretch
  witnesses. Controlled depth scalars are not model accuracy evidence.
- [`test_geometry_corrected_boundary.py`](../../tests/lux_depth_v3/test_geometry_corrected_boundary.py),
  [`test_geometry_hardening.py`](../../tests/lux_depth_v3/test_geometry_hardening.py)
  and [`test_geometry_versioning.py`](../../tests/lux_depth_v3/test_geometry_versioning.py):
  independent corrected restoration, immutable snapshots, malformed authority,
  closed versions and byte-identical historical planning.
- [`test_geometry_orchestrator_handoff.py`](../../tests/lux_depth_v3/test_geometry_orchestrator_handoff.py)
  and [`test_v2_geometry_handoff.py`](../../tests/test_v2_geometry_handoff.py):
  real Stage A → V2 subprocess handoff, producer replacement rejection,
  support/alpha guards, cache-hit republication and strict receipt validation.
- [`test_da3_geometry_invocation.py`](../../tests/lux_depth_v3/test_da3_geometry_invocation.py):
  bounded no-upscale model preparation, exact DA3 resolution arguments and
  local/canonical/subprocess recipe propagation without model inference.
- [`test_geometry_proxy_allocation.py`](../../tests/lux_depth_v3/test_geometry_proxy_allocation.py):
  exhaustive 16-bit quantization equivalence, direct 8-bit reuse and unchanged
  orientation, alpha, source binding and model pixels without float64 proxies.
- [`test_v2_geometry_cache_and_entrypoint.py`](../../tests/test_v2_geometry_cache_and_entrypoint.py):
  actual shared-cache separation, validation before cache access or side
  effects, and unchanged historical keys and child output.

Run the focused regressions together with the existing geometry, color and
native-master contracts:

```bash
./.venv/bin/pytest \
  tests/core/test_proxy_coordinates.py \
  tests/lux_depth_v4/test_photography.py \
  tests/lux_depth_v5/test_geometry_roundtrip.py \
  tests/materials_v4/test_geometry_alignment.py \
  tests/lux_depth_v3/test_geometry_legacy_boundary.py \
  tests/lux_depth_v3/test_geometry_corrected_boundary.py \
  tests/lux_depth_v3/test_geometry_hardening.py \
  tests/lux_depth_v3/test_geometry_versioning.py \
  tests/lux_depth_v3/test_geometry_orchestrator_handoff.py \
  tests/lux_depth_v3/test_da3_geometry_invocation.py \
  tests/lux_depth_v3/test_geometry_proxy_allocation.py \
  tests/test_v2_geometry_handoff.py \
  tests/test_v2_geometry_cache_and_entrypoint.py \
  tests/test_v2_color_contract.py tests/test_photographic_native_master.py -q
```

Engineering closure requires these regressions, coordinate-helper boundary and
compatibility tests, documentation/catalog checks, `make ci-quick`,
`make test-fast`, pre-commit and the full local CI gate; hosted review and
required CI must pass on the exact PR head. Merge is followed by main CI and
its follow-on firewall. This document is not a record that any particular run
passed; retain command, commit and job evidence separately.

Operational validation and Richard's photographic acceptance remain separate.
The retained-eight and representative 50-photo pilots retain their own
resource budgets, output namespaces, review decisions and acceptance records.
No real-reference or destination-proof acceptance follows from synthetic
agreement, a successful model-boundary fixture or a source approval. Future
native-precision editing remains gated on verified frame authority and explicit
scoped photographic acceptance; AWS/GPU validation stays on hold.

No storage migration is required. No stored recipe, decision, approval or
retained file is rewritten, and historical processing and replay identities
remain valid in their matching environment.
