# Execution Plan V4 contract

`tp.execution.plan.v4` is the opt-in LuxDepthV5 contract. Its immutable carrier is
`transformation_portal.core.execution_plan_v4.ExecutionPlanV4`. V1, V2, and V3
remain independent contracts; no existing executor automatically selects V5.
See the [V5 operator guide](LUX_DEPTH_V5.md) for commands and acceptance limits.

## Preparation and authority

`prepare(LuxDepthV5Request)` freezes source hashes, model revision and license,
actual device, resources, precision, refinement, optional calibration, and
MaterialsV4 evidence. CPU planning neither loads a model nor creates outputs;
`auto`/`mps` planning may probe device availability in a subprocess. Physical
input/output/cache roots remain separate from portable canonical plan bytes.

The closed JSON schema is packaged at
`transformation_portal/schemas/execution/plan.v4.schema.json`. Validation also
reconstructs and compares the exact node graph and verifies the fingerprint.
An internal projection reuses older structural validators only; that projection
does not authorize execution. V4's public executor rejects the V5 carrier.

| Node | Stage | Role |
| --- | --- | --- |
| preprocess | `tp.stage.lux.preprocess.v2` | Canonical linear photographic master and bounded model proxy |
| depth | `tp.stage.lux.depth.v3` | Preserved DA3 native samples, explicit precision, and sky evidence |
| enhance | `tp.stage.lux.enhance.v3` | Validity-aware alignment, bounded finishing receipt, optional MaterialsV4 |
| output | `tp.stage.lux.output.v3` | Full-resolution delivery, native/derived arrays, and reconstructible evidence |

The additional configuration object is exactly:

```json
{"depth":{"precision":"fp32","refinement":"guided_bilinear"}}
```

`precision` allows `fp32` and `fp16`; `refinement` allows `bilinear` and
`guided_bilinear`. No implicit precision or synthetic fallback is admitted.
Only governed `da3_metric` is authorized. Legacy companion material masks are
rejected; explicit MaterialsV4 evidence and camera-calibration companions are
supported.

## Native inference identity

The cache namespace is `identity-v5`. Only the depth node is cacheable. Its
`tp.execution.identity.v5` preimage binds the native stage version/configuration,
source bytes, complete proxy content and geometry, parent/worker runtime, source
closure, governed model identity, actual device, explicit precision, inference
recipe, sky policy, and content-based seed policy. Camera calibration and
photographic strength/clarity/refinement are downstream derivatives, so changing
only those inputs can reuse native inference. The proxy itself remains bound;
changing decode/resize behavior cannot borrow another proxy's cache entry.

Other stage identities additionally bind the complete plan fingerprint and input
identifier. Every named node input is required. V4's identity/cache namespace
remains separate.

## Output and publication

Completion uses `tp.lux.execution.evidence.v3`, photograph descriptors use
`tp.lux.photograph.v2`, and native depth uses `tp.depth.artifact.v3`.
`tp.depth.aligned.v1` describes master-grid derivatives and
`tp.depth.response.v1` describes photographic finishing. Failure evidence is
`tp.lux.execution.failure.v3` and cannot authorize publication.

Native numeric validity, in-frame support, sky evidence, and usable surface
validity are separate masks. A zero derivative value is meaningful only with its
validity mask. Unknown sky evidence disables surface edits. Interpolation support
is explicitly not calibrated accuracy confidence.

V5 retains governed runtime checks, cancellation, resource ceilings, immutable
artifact inventory, and managed generation fences. Publisher admission reserves
the complete possible output inventory before device probing. The independent
V5 verifier reconstructs geometry, depth derivatives, finishing, and delivery
from the stored source master and native arrays; hash-only consistency does not
establish semantic validity. Verification does not certify physical scene depth
or recover original source-file bytes from a master array.

## Validation

```bash
make test-lux-depth-v5-contract
```

Native inference and reference-bound quality evaluation are additional evidence,
not implied by the controlled-worker contract suite.
