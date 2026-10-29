# Photographic geometry and orientation audit

This ledger records the coordinate authority used by the canonical photographic
path and the separate limitations reproduced in legacy V3/V2. It is the PR6
geometry audit, not a new processing recipe or native editing engine. The
review baseline is `0a374b0624069fe322d8cbd1e24c1480d9fdfc4d`, with the
additive coordinate helpers and synthetic regressions reviewed alongside this
document. The documentation catalog binds the exact reviewed source bytes.

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

These are blockers to treating legacy outputs as a fully verified canonical
geometry handoff. They are not silently corrected by this audit. Existing V2
processing version `1.2.0`, execution-plan configurations and historical cache
semantics remain unchanged. The witnesses intentionally preserve that behavior
until a migration supplies the required version identity and validation.
Source review approval does not waive that future migration or photographic
acceptance.

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

Run the focused regressions together with the existing geometry, color and
native-master contracts:

```bash
./.venv/bin/pytest \
  tests/core/test_proxy_coordinates.py \
  tests/lux_depth_v4/test_photography.py \
  tests/lux_depth_v5/test_geometry_roundtrip.py \
  tests/materials_v4/test_geometry_alignment.py \
  tests/lux_depth_v3/test_geometry_legacy_boundary.py \
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

Rollback removes the additive helpers and audit regressions without a storage
migration. No stored recipe, decision, approval or retained file is rewritten,
and the existing processing and replay identities remain valid.
