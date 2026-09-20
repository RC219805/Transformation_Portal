# Execution Plan V3 Contract

`tp.execution.plan.v3` is the additive core-owned photographic execution plan
for explicit MaterialsV4 use within LuxDepthV4. It preserves the existing Lux
executor, governed depth model, native depth authority, and identity-v4 mechanism.
It does not select a new executor or automatically switch material models.

LuxDepthV3 continues to consume [Execution Plan V1](EXECUTION_PLAN_V1.md).
LuxDepthV4 requests without `--materials-manifest` retain
`tp.execution.plan.v2`, including the existing optional companion interface.
See the [MaterialsV4 operator guide](../guides/MATERIALS_V4.md) for evidence
preparation and the [LuxDepthV4 reference](LUX_DEPTH_V4.md) for native runtime,
photographic ingest, cache, and publication prerequisites.

## Authority and carriers

`core.execution_plan_v3.ExecutionPlanV3` stores canonical immutable bytes.
`parse_photography_plan` accepts the closed V2 and V3 families; the general core
plan dispatcher also recognizes V3. Validation rejects unknown fields, invalid
resource budgets, noncanonical carriers, mismatched fingerprints, altered graphs,
wrong sources, and incomplete or non-normalized response policies.

V3 reuses the V2 cross-field invariants through an explicit structural
projection during validation. That projection is not executable and cannot
replace the V3 carrier. A prepared request holds physical input, output, runtime,
cache, companion, and material roots separately from the canonical logical plan.
Execution reconstructs and validates the exact core-owned carrier.

## Additive fields

The closed JSON schema is
[`plan.v3.schema.json`](../../src/transformation_portal/schemas/execution/plan.v3.schema.json).
The principal additions to V2 are:

| Field | Bound meaning |
| --- | --- |
| `schema` | Exactly `tp.execution.plan.v3` |
| `materials_manifest` | Portable path, SHA-256, and byte count of the source-to-evidence manifest |
| `inputs[].materials_v4` | Evidence-manifest path/hash/size, semantic content hash, source digest, and canonical HW shape |
| `configuration.materials_v4` | Complete normalized `ResponsePolicy`, including explicit operation versions/strengths and confidence, overlap, color, delta, and tile/resource policies |
| `nodes` | The closed photographic graph with MaterialsV4 evidence, response receipt, and pre-material baseline bindings |

At least one selected input must have a material evidence binding. Its source
digest must equal that selected input's digest. Geometry must match any supplied
camera-calibration geometry and the decoded canonical master. Unlisted images
remain selected photographs but receive explicit missing-evidence abstention.

The outer manifest uses `tp.lux.materials_manifest.v1`:

```json
{
  "schema": "tp.lux.materials_manifest.v1",
  "inputs": [{
    "path": "photograph.tif",
    "source_sha256": "<actual source SHA-256>",
    "evidence_path": "evidence.json",
    "evidence_sha256": "<actual evidence manifest SHA-256>",
    "shape": [4000, 6000]
  }]
}
```

The digests above are explanatory placeholders. Use the import/inference tools
to produce actual bindings. `path` is relative to the photographic input root;
`evidence_path` is relative to the materials-manifest directory. Material bundle
masks are relative to their own evidence manifest. Confined path validation,
regular-file reads, declared sizes, numeric-header budgets, aggregate bundle
budgets, and content checks apply before admission.

## Graph and execution sequence

The graph retains `preprocess`, `depth`, `enhance`, and `output`:

1. `preprocess` binds the source, optional camera calibration, and
   `$materials_v4` evidence identity, producing the immutable photographic master
   and explicit encoded model proxy.
2. `depth` executes or reuses governed native-depth evidence with the existing
   worker/runtime and identity-v4 rules.
3. `enhance` uses `tp.stage.lux.enhance.v2`. It computes the ordinary photographic
   enhancement, retains that exact master as `enhance.materials_baseline`, then
   plans and applies MaterialsV4 once. Outputs include the final master and a
   `tp.materials.execution.v1` receipt.
4. `output` consumes final master, depth, material receipt, and material baseline;
   it publishes the existing photographic outputs plus the V3-only baseline.

A request cannot combine legacy `companions[].materials` with MaterialsV4.
Camera-calibration companions remain permitted. There is no V3-to-V4 automatic
fallback and no second material compositor in the V3 graph.

Preparation freezes model, source, evidence, policy, graph, and resource
identities before model initialization or output creation. Execution rebinds the
outer manifest and every material bundle before inference, and rechecks the
individual source/evidence inputs during the batch. Changing masks, confidence,
provenance, manifest bytes, or policy requires new preparation.

Only native depth remains cacheable in this graph. Stage identities bind the
canonical plan, named evidence inputs, source/runtime identity, and applicable
model identity. Material inference experiments use their own explicit
`cache_policy: off`; V3 segmentation caches cannot authorize V4 evidence.

## Response plan and photographic authority

`tp.materials.response_plan.v1` is created inside the enhancement stage after its
actual pre-material master exists. It binds that master's content hash, source
hash, evidence hash, operation contract, complete response policy, and ordered
region decisions. The outer execution plan already freezes its evidence and
policy; `--plan` does not pretend to know post-depth pixel decisions before the
depth/photographic stages execute.

`plan_response(master, evidence, policy)` computes eligibility, protects uncertain
and overlapping support, and measures remaining support. `apply_response`
recomputes the bound plan and rejects stale or forged authority. The compositor
uses baseline-derived linear-sRGB deltas, one alpha weighting, fixed tile/halo
semantics, and final stored-sample bounds. It preserves untouched samples,
protected HDR/transparent samples, alpha, and metadata.

`RegionEvidence` carries material semantics separately from geometric quality.
Supplied scores authorize only when the frozen policy explicitly permits them.
Inferred edits require an explicitly admitted calibration digest, matching
region and producer recipe identities, supported calibrated-probability
semantics, photographic domain, and covered class. A calibration receipt is an
identity-bearing artifact, not self-authenticating trust. The local experimental
SAM2.1/OpenCLIP adapter emits uncalibrated rankings and cannot satisfy this
contract by changing a label or score string.

## Completion evidence and independent verification

Completion retains `tp.lux.execution.evidence.v2`; its `plan_schema` accepts V2
and V3 and must equal the parsed canonical plan's schema exactly. V3 completion
requires a complete material receipt for every selected source, including
explicit missing-evidence abstentions.

V3 adds `materials-baseline.npy` and `photograph.json.materials_baseline`, the
exact pre-material `ImageMaster` descriptor. `source-master.npy` remains the
original master before ordinary photographic enhancement. This distinction lets
verification measure material-only changes even when global enhancement is
nonzero.

The receipt contains its full response plan, plan/evidence/operation identities,
input/output master hashes, evidence status/reason, per-region decisions, changed
pixel counts, and actual maximum/mean absolute delta. Timing does not enter
those identities. Verification checks:

- Required baseline/final/alpha inventory and bounded float32 NPY headers,
  geometry, layout, payload lengths, finiteness, and hashes.
- Source/evidence/policy/operation bindings and response-plan fingerprint.
- Canonical region inventory, eligibility/abstention structure, support and
  changed-count invariants.
- Exact metadata, geometry, alpha, and source-color identity preservation.
- Independently measured material changed-pixel/max/mean statistics, the frozen
  final delta ceiling, and numerical HDR/zero-alpha protection.

The original semantic masks are not published as completion artifacts. This
verification cannot reconstruct class correctness or every protected semantic
boundary; those require the separately bound source evidence and evaluation.
It establishes execution integrity, not photographic or calibration acceptance.

Managed publication uses the existing `GenerationPublisher` and dispatch fence.
The V3 inventory reserves one additional baseline per image, including images
without material evidence; the publisher's file-count and byte limits still
apply before admission. No new route, queue, scheduler, or publication authority
is introduced.

## Validation and acceptance boundary

The canonical candidate contract gate is:

```bash
make test-materials-v4-contract
```

For completion-specific diagnosis:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest \
  tests/lux_depth_v4/test_materials_v4.py \
  tests/lux_depth_v4/test_materials_v4_verification.py \
  tests/lux_depth_v4/test_publication.py -q
```

These contracts do not establish representative photographic quality, empirical
calibration, native MPS performance, or production readiness. Keep those acceptance
records separate from local test, synthetic fixture, model-smoke, and cache
integrity evidence.
