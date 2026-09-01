# Execution Plan V1 Contract

`tp.execution.plan.v1` is the core-owned, closed semantic plan representation
designated by ADR-051. It describes intent only: parsing a plan does not build
a `StageGraph`, load a model, write an output, or invoke `CASDAGExecutor`.

The machine-readable schema ships in the installed package as
`transformation_portal.schemas.execution/plan.v1.schema.json`. Load and enforce
it through the bounded parser:

```python
from transformation_portal.core.execution_plan import parse_execution_plan_json

structural_plan = parse_execution_plan_json(plan_bytes)
print(structural_plan.configuration_completeness)
```

This parser establishes closed-world structure, bounds, graph semantics, and
fingerprint integrity only. It is deliberately not a Lux model/backend
authority check. Before any future execution-capable consumer accepts an
`execution_complete` plan, it must use the domain boundary:

```python
from transformation_portal.lux_depth_v3.execution_plan_adapter import (
    revalidate_lux_execution_plan_json,
)

authoritative_plan = revalidate_lux_execution_plan_json(plan_bytes)
```

That boundary rejects `structural_legacy` plans and independently revalidates
the complete candidate chain, enabled ensemble constituents, model registry or
checkpoint identity, locked revisions, licenses, and acknowledgements. It does
not re-resolve a selector or fill a missing identity.

## Contract contents

The v1 payload carries:

- `configuration_completeness`, distinguishing parse-only
  `structural_legacy` projections from declarative `execution_complete`
  carriers;
- the planned backend, bounded fallback chain, and a matching ordered
  `backend_candidates` array;
- complete per-candidate model contracts, including enabled ensemble
  constituents, configured weights/devices, repository revisions or pinned
  artifact paths plus digests, and license intent;
- a frozen, contained relative input selection plus decoded-pixel and
  decompression-ratio ceilings;
- deterministic configuration and canonical plan fingerprints;
- stable node IDs and allowlisted stage-registry identifiers;
- explicit dependency edges and typed, closed stage configurations;
- bounded per-node resource declarations;
- typed logical output declarations and requested-output intent.

Every object is closed to unknown fields. Unknown schema versions, registry
identifiers, configuration schemas, output kinds, edge endpoints, cycles,
unsafe input paths, duplicate IDs, and fingerprint drift fail closed. The JSON
entrypoint also bounds body bytes, nesting, string length, nodes, edges,
fanout, inputs, and requested outputs, and rejects duplicate object members
and non-finite numbers.

The plan fingerprint is SHA-256 over canonical `tp.canonical.json.v1` bytes of
the complete payload with `plan_fingerprint_sha256` omitted. It detects
payload drift; it is not a signature or authorization grant.

## Static registry boundary

`transformation_portal.stage_graph.registry` is a static semantic allowlist.
It intentionally contains no constructors, import paths, executable fields,
commands, or plugin hooks. The current entries describe the Lux v1 stages and
are closed for this version. New identifiers require a successor plan schema
and matching registry contract tests.

Schema validity alone grants neither model nor execution authority. A domain
adapter must revalidate carried model, revision, license, backend, tenant, and
authorization contracts at every execution-capable boundary.

V1 deliberately defines both carrier levels before schema freeze. Every stage
configuration is closed and discriminated by the same completeness value as
the plan. `execution_complete` configurations carry the resolved executor
inputs for preprocess/RAW ingest, depth/fallback/postprocessing/APEX policy,
Materials V3 segmentation, PBR, V2, reconstruction, and output/run-card
policy. A configuration fingerprint is drift evidence, not a substitute for
these typed values.

Requested evidence is independent from failure policy. In the current Lux
semantics PBR is optional, standard/premium Materials V3 is optional, APEX
Materials V3 is blocking, and run-card emission is nonblocking. A requested
output therefore may truthfully declare `required: false`; an optional node
must use `failure_policy: omit_outputs` and cannot declare required outputs.

## Lux v1 compatibility

The live CLI continues to emit `tp.lux.resolved_invocation.v1`. The read-only
adapter converts that payload into the canonical form without changing the
current executor:

```python
from transformation_portal.lux_depth_v3.execution_plan_adapter import (
    adapt_resolved_invocation_json,
)

canonical_plan = adapt_resolved_invocation_json(legacy_plan_bytes)
assert canonical_plan.configuration_completeness == "structural_legacy"
```

The provisional Lux v1 schema promised that readers ignore added unknown
fields. The compatibility adapter honors that promise only while producing a
new closed payload: unknown legacy fields never cross into the canonical
form. It rejects unknown versions/stages/artifacts, non-contained input paths,
and forged or drifted model/backend/license contracts.

Pre-#2068 Lux v1 payloads omitted `output_bit_depth` and represented a 16-bit
selection with the retired `bit_depth_16_intermediates` requested artifact (or
an additive deprecated `emit_master16` / `emit_upscaled16` alias). The adapter
normalizes a copy before validating it against the current Lux schema: a
truthy legacy marker selects `output_bit_depth: 16`, while no marker or only
false markers preserves the historical 8-bit default. Null aliases count as
omitted. An explicit 8-bit selection conflicts with a truthy 16-bit marker and
fails closed; an explicit 16-bit selection consumes the marker. Neither the
aliases nor the fictional artifact crosses into the canonical plan. The
canonical output-stage configuration carries only `output_bit_depth`, and one
`materials_v3_masks` declaration denotes the single per-input NPZ container.

Every result of this legacy adapter is `structural_legacy`, even when the old
payload happens to contain an authoritative DA3 contract. The adapter carries
known values and ordered backend IDs, but never invents missing candidate
identities or stage tuning and never upgrades the result to
`execution_complete`.

Promotion is deliberately partial where the old payload lacks authority. In
particular, the adapter rejects the current Apple Silicon Depth Pro fallback
shape when its chain permits DA3 but carries no DA3 model contract; it also
rejects `ensemble`, whose constituent model identities are absent, and
reconstruction without both required research acknowledgements. The live v1
executor remains the rollback path for those payloads. These are explicit A2
producer prerequisites: the direct producer must populate the already-defined
execution-complete v1 candidate identities and typed stage configurations. It
must not independently re-resolve or manufacture absent legacy identity. The
current unpinned/unimplemented DepthCrafter model path is representable but is
not Lux-authorizing until a trusted executable identity exists.

The legacy `lux_depth_v3.pipeline_coordinator.ExecutionPlan` import is also
preserved as the flat live-executor projection and is explicitly aliased as
`LegacyExecutionPlan`. It must not be confused with
`core.execution_plan.ExecutionPlan`.

## Non-activation boundary

This contract does not complete ExecutionIdentity v3, cache cutover, manifest
propagation, direct-Python lifecycle migration, or documented-workflow parity.
It is a partial, non-closing #2065 foundation and must not be compiled into
`StageGraph`. A2 must add the native execution-complete producer and consumer,
and the separate ADR-051 Phase C vertical slice must prove security, output,
cancellation, cache, publication, and performance parity before activation.
