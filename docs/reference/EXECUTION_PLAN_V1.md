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

## Prepared execution evidence

Native prepared runs publish a detached
`tp.lux.execution.evidence.v1` sidecar after their per-image manifests, batch
manifest, and optional run card. Existing manifests and run cards carry the
exact `plan_schema`, plan/config fingerprints, planned backend, permitted
fallback chain, per-input runtime backend outcomes, requested artifact kinds,
and the confined relative path to that sidecar. The detached form avoids a
self-hash cycle while allowing the sidecar to record the complete bytes of
those requested manifest artifacts.

`EnhanceOrchestrator.from_prepared(...)` must be entered through
`enhance_batch(prepared.input_root, input_files=list(prepared.input_files))`.
Prepared calls to `enhance_image(...)` or `enhance_batch_parallel(...)` fail
closed because neither public shortcut owns the complete batch lifecycle or
can publish authoritative final evidence. Unprepared compatibility execution
retains the historical single-image API when governed cache authority is not
required.

The binding uses the existing round-tripped extension maps so frozen artifact
formats remain readable by rollback tooling: `CombinedManifest.environment`,
`BatchManifest.config`, and the run card's `effective_config` each contain an
`execution_contract` object. That object carries the complete authoritative
plan payload, the runtime projection, and the evidence path. No new top-level
members were added to run-card v1/v2 or the legacy manifest dataclasses.

The sidecar is the prepared run's completion record. A reader must treat a
prepared manifest or run card as unconfirmed unless its pointer resolves to a
canonical sidecar that verifies against the carried plan and the final bytes
of every produced artifact. Capture and publication require descriptor-relative
no-follow traversal under one pinned output root. Publication retains and
revalidates every root ancestor plus the final parent and directory-entry
identity before and after the atomic rename; namespace drift fails the API and
leaves any uncertain temporary or final entry in place for explicit operator
reconciliation. On an ordinary write or durability failure, the publisher
revalidates the retained parent namespace and publisher-created inode before a
descriptor-relative cleanup and parent-directory sync. An already missing name
needs no action; a moved, linked, or replacement entry is retained rather than
removed. Portable POSIX has no atomic compare-and-unlink operation, so this
cleanup relies on the exclusive-writer boundary below. Platforms without the
required secure primitives fail closed. Windows drive-qualified pointers,
symlinks, hardlinks, duplicate inode aliases, and output-root escapes are not
valid evidence.

The process that owns this contract must also deny hostile rename access to the
output root and its ancestors. Portable descriptor APIs can detect a directory
that another actor moved during publication, but cannot prevent the temporary
file from being transiently relocated by an actor that already has authority to
rename those directories. Treat a retained orphan as untrusted until an operator
reconciles its inode and contents; publication failure never authorizes it.

Evidence starts from the exact requested output declarations in the carried
execution-complete plan. Each declaration retains its stage-registry owner,
scope, cardinality, and required bit, and is expanded across the frozen input
selection. Every expanded output has exactly one outcome:

- produced, with one or more output-root-confined relative paths, complete
  SHA-256 digests, byte sizes, media types, and file extensions;
- omitted, only for an optional declaration and with a typed reason; or
- failed, with a typed reason. Any failed required output fails the run after
  the evidence is durably published.

Observation count, per-outcome cardinality, individual file size, and aggregate
captured bytes are bounded before hashing. Known-oversize files are rejected
from metadata without reading their contents. Per-input output discovery reads
combined manifests only through that pinned, no-follow, bounded boundary.
Successful batch rows must cover the frozen input selection and point to each
input's verified combined manifest; combined-manifest backend selection must
match the runtime projection.
When a run card is requested, both its canonical payload digest and its final
run-card self-integrity sidecar must verify before the run-card output is
classified as produced.

The existing run-card `artifact_index` remains a compatibility catalog. It is
assembled through best-effort discovery and is not plan authority or proof of
complete requested-output accounting.

## Lux v1 compatibility

The live CLI and native prepared lifecycle emit `tp.execution.plan.v1`.
Historical `tp.lux.resolved_invocation.v1` payloads remain readable through a
bounded, read-only adapter that converts them into the canonical structural
form without granting execution authority:

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
reconstruction without both required research acknowledgements. The native
prepared producer populates the execution-complete candidate identities and
typed stage configurations directly; it does not independently re-resolve or
manufacture absent legacy identity. The current unpinned/unimplemented
DepthCrafter model path is representable but is not Lux-authorizing until a
trusted executable identity exists.

The legacy `lux_depth_v3.pipeline_coordinator.ExecutionPlan` import is also
preserved as the flat live-executor projection and is explicitly aliased as
`LegacyExecutionPlan`. It must not be confused with
`core.execution_plan.ExecutionPlan`.

## Non-activation boundary

The native prepared lifecycle, runtime evidence propagation, and maintained
documented-workflow parity complete the Lux #2065 contract surface. They do
not activate `StageGraph` or `CASDAGExecutor`, complete ExecutionIdentity v3,
or cut the verified depth cache over. The separate ADR-051 Phase C vertical
slice must still prove security, output, cancellation, cache, publication, and
performance parity before shared-executor activation.
