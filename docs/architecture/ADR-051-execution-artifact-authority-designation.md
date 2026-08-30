# ADR-051: Execution & Artifact Authority Designation

**Status:** Proposed
**Date:** 2026-08-28
**Decision Makers:** Architect (designation) + Specialist (migration)
**Replaces:** None
**Supersedes:** None — *provisional; see the conflict matrix below. If any disposition changes a
designation made by ADR-029, ADR-043, or ADR-045, this field and the affected ADR must be amended in
the accepting PR.*
**Related:** [ADR-029 execution-graph executor](ADR-029-execution-graph-abstraction.md), [ADR-043 Orchestrator Decomposition
Strategy](ADR-043-orchestrator-decomposition.md), [ADR-045 Monolith Decomposition
Residuals](ADR-045-monolith-decomposition-residuals.md), [Production Hardening Gap
2026-05-13](../governance/PRODUCTION_HARDENING_GAP_2026-05-13.md) §4 ("Existing primitives to reuse,
not reinvent"), "Repair, Designate, Prove" superseding recommendation (2026-08-28)

> Numbering note: ADR-050 is the active portal React migration decision
> (`docs/decisions/ADR-050-portal-react-migration.md`); this ADR takes the next free number, 051.

---

## Context

The repository carries multiple parallel implementations of execution scheduling, execution identity,
CAS/Merkle lineage, artifact storage, and event/audit recording. Verification (2026-08-27/28,
initially at `c6d620a` and refreshed below) established that no single repository authority spans
these concerns. Before any new plan, ledger, or artifact-store abstraction merges, this ADR proposes
designating one authority per concern and disposing of the competitors — including their test suites
and coverage floors.

### Inventory (initially verified at `c6d620a`; execution-surface corrections reverified at `61456c3`;
affected identity and evidence-binding claims refreshed at `81c9f15` — facts, not proposals)

**Execution surfaces (stage-level), plus a separate job-runner boundary:**

Seven stage-level implementation surfaces are inventoried below. `WorkerRunner` is listed as `J`
because its queue/lease responsibility is the job boundary, not an eighth stage executor.

| # | Implementation | Character | Verified consumers | Tests |
|---|---|---|---|---|
| 1 | `stage_graph/graph.py` wrapped by `core/cas_dag_executor.py` | ThreadPool DAG + CAS + MerkleDAG provenance, deterministic replay | No verified production construction of `StageGraph` or `CASDAGExecutor`; Lux V2 and DepthPro consume individual stage primitives and implementations, while execution-graph nodes consume `StageStatus` | `tests/` (stage_graph has a branch floor) |
| 2 | `spatial_ai/orchestration/graph/executor.py` (ADR-029) | Execution-graph executor with its own `ExecutionGraph`/`ExecutionPlan` | Package-local definitions and a default-off graph adapter, plus examples and tests; no verified production configuration enables construction | `tests/spatial_ai/` |
| 3 | `execution_graph/scheduler.py` (`PriorityDAGScheduler`) + `nodes/` | Priority DAG scheduler with in-process node execution and node library | `dashboard/execution_manager.py:379-405` constructs and consumes the scheduler inside the package-local `create_execution_router()` path; no in-repo application wiring invokes or mounts that router. `dashboard/dag_api.py` exposes a scheduler injection/visualization surface, although its import is type-only | own suites plus dashboard suites |
| 4 | `execution_graph/distributed_executor.py` (`DistributedDAGExecutor`) | Local/Ray wrapper around row 3, with its own Ray execution loop | Package re-export plus tests/doc examples only; no verified runtime construction or execution outside its own suite | `tests/execution_graph/test_distributed_executor.py` |
| 5 | `comfyui/executor.py` (+ `workflow_builder.py`, `custom_nodes.py`) | ComfyUI workflow executor | Package re-export, templates, documentation examples, and tests only; no verified external consumer or production construction | own suites |
| 6 | `runtime/engine.py` + `runtime/scheduler.py` (+ `process_executor.py`, `execution_manifest.py`, `ledger.py`, `replay.py`) | Runtime execution engine with Merkle-backed manifests and replay | Internal runtime references and package re-export, plus documentation examples and tests; no verified dashboard or other production construction | `tests/runtime/` (engine, execution_manifest, ledger, process_executor, …) |
| 7 | `lux_depth_v3` orchestrator + `pipeline_coordinator`/`execution_engine` | Bespoke per-image state machine; `ExecutionPlan` is a flat string list (`pipeline_coordinator.py:1051-1097`); facade classes production-dead | **Current Lux Depth production execution path** | `tests/lux_depth_v3/` |
| J | `orchestrator/worker.py` `WorkerRunner` + `orchestrator/worker_process.py` (standalone `python -m ...worker_process` entry point for multi-host workers) | Durable job-level queue/lease consumer — not a stage DAG | FastAPI lifespan (only in-process execution path since Phase 2.E); standalone process for multi-host deployment | `tests/orchestrator/` (incl. `test_worker_process_contract.py`) |

`execution_graph.patcher` has verified `rl`/`evals` consumers, but those consumers do not construct
either row 3 or row 4 and therefore are not evidence that either execution surface is live.

**Execution identity.** `core/execution_identity.py` (`ExecutionIdentity`, `compute_cas_id`,
`compute_code_hash`, `create_artifact_metadata`) implements model+config+code identity and is
**consumed within `core/`** (`core/execution_wrapper.py:49,314`, `core/cas_dag_executor.py:54`) but
**unwired from the Lux production path**. `config_resolver.compute_config_fingerprint` returns the
manifest `ConfigFingerprint`; it is not a separate identity implementation. The depth-cache
fingerprint is a separate projection (`config_resolver.py:637-659`), and the segmentation cache key
is another (`segmentation/_cache.py:251-320`). The manifest/config and depth-cache projections share
model identity through `resolved_model_identity_for_backend`, but these partial operational
projections do not capture complete code, dependency, and executed-path identity.
*Evidence updates (2026-08-28; facts, not designations):* PR #2070 (`e30bd58`) introduced the
provisional `ResolvedInvocation` serialization and contract-aware fingerprint surface. PR #2081
(`81c9f15`) then selected `da3_metric` as the commercial-safe default and repaired identity and
selector provenance across no-selector, explicit-selector, typed-preset, direct-Python, resolver,
invocation, backend/inference, depth-cache, manifest, and run-card fingerprint boundaries. With an
authoritative resolved contract present, the current plan, depth cache, manifests, and Stage-A reuse
derive model identity as `canonical_key:repo_id@revision`
(`resolved_model_identity_for_backend`). These landed behaviors feed the execution-identity
suitability matrix; the plan surface remains provisional and carries no designation weight.

**CAS / Merkle lineage.** `storage/merkle_dag.py` is **not orphaned**: consumed by
`runtime/{engine,execution_manifest,replay}.py`, `core/cas_dag_executor.py`, and
`dashboard/{time_travel,dag_api,artifact_api,execution_manager}.py`. `storage/cas_store.py` carries a
coverage floor. `lux_depth_v3/artifact_manager.compute_artifact_merkle_root` is live in the Lux
production path. `tp.merkle`/`tp.crypto` is the evidence plane — contract-bearing, out of scope for
consolidation (see Authority boundary).

**Event / audit / ledger recording — four distinct surfaces, not one.** (a) `orchestrator/storage/`
`JobEventStore` — live; job-event history backing SSE replay. (b) `OperationalAuditStore`
(`orchestrator/storage/base.py:315`) — live; paid-pilot operational audit. (c) `events/store.py`
`EventStore` — orphaned; gap-doc-designated for lifecycle events. (d) `runtime/ledger.py`
`ImmutableLedger` (line 102) — the literal append-only ledger implementation in the runtime plane.
These serve different purposes (SSE history, pilot audit, operation events, immutable record) and the
designation row below must dispose of each explicitly. **This ADR makes no presumption that the
synchronous `events/store.py` is the operational ledger** — that is a row to decide, with all four
surfaces as candidates for their respective roles.

**Artifact surfaces.** Three classes named `ArtifactStore` exist, and they are **not
interchangeable** — they serve different roles:

| Role | Current surface | Status |
|---|---|---|
| Immutable content-addressed bytes | `storage/cas_store.py:193` | live floor, few consumers |
| Spatial stage-result cache | `spatial_ai/orchestration/graph/artifact_store.py:223` | spatial_ai only |
| Job delivery / presigning / deletion | `orchestrator/artifact_store/base.py:85` (local + S3) | paid-pilot contract surface |
| Generation publication transaction | **not yet designated** | net-new role (repair 1.5-b) |

Plus `lux_depth_v3/artifact_manager.py` (index/hash/Merkle root, live) and `portal/job_artifacts.py`
(reused behind the Protocol per gap doc Phase 4.A).

### Current problems

1. Seven stage-level execution surfaces plus a separate job-runner boundary; none is both durable
   and used by Lux Depth.
2. Partial operational identity projections lack complete code, dependency, and executed-path
   identity; the code-aware `core/execution_identity.py` implementation remains unwired from Lux.
3. Four artifact roles across three same-named classes and two ad-hoc surfaces; the generation
   publication transaction has no owner.
4. Frozen or parallel implementations retain green test suites that no longer prove production
   behavior.

---

## Decision

Designate one authority per plane. Every `⟨DECIDE⟩` cell is a decision this ADR exists to make; no
cell may be filled without a completed **suitability matrix** (template below) — candidate language
in earlier drafts effectively pre-decided rows before the evidence comparison, and is removed.

| Plane | Designated authority | Migration path |
|---|---|---|
| Execution identity | ⟨DECIDE⟩ — one canonical implementation; cache fingerprints become derivations of it, with complete identity fields (code, model, config, dependency, executed path) | ⟨DECIDE⟩ |
| Plan/DAG representation | ⟨DECIDE⟩ — one schema + owner, chosen through the suitability matrix with no pre-selected candidate. The [#2065](https://github.com/RC219805/Transformation_Portal/issues/2065) `ResolvedInvocation` is hereby scoped as a **bounded, provisional pre-designation spike**: its serialization is marked `stability: provisional`, carries no designation weight, and is either adopted or migrated when this row is decided. *Status: PR [#2070](https://github.com/RC219805/Transformation_Portal/pull/2070) (`e30bd58`, 2026-08-28) introduced the merged spike, including revision pinning through the worker subprocess, plan/run input and backend-selection parity, and a wheel-shipped provisional schema. PR [#2081](https://github.com/RC219805/Transformation_Portal/pull/2081) (`81c9f15`, 2026-08-28) subsequently hardened default, selector, typed-preset, direct-Python, carrier, and execution-path model identity and licensing behavior. Both changes supply production evidence for this row's suitability matrix; neither designates the representation.* | spike (merged, provisional) → designated representation (Phase 3) |
| Stage executor | ⟨DECIDE⟩ — one production executor for local/runtime stage work, evaluated across inventory rows 1–7 | Phase 3 pilot |
| Job runner boundary | ⟨DECIDE⟩ — boundary statement with `WorkerRunner`: the designated stage executor runs inside a leased job; never a second queue | n/a |
| CAS/Merkle operational lineage | ⟨DECIDE⟩ — one identity + lineage authority for **operational** records (evidence plane excluded; see Authority boundary) | Phase 3 |
| Artifact roles & composition | ⟨DECIDE⟩ — designate how the four roles above **compose** (CAS bytes / stage cache / job delivery / generation publication); do not force one interface to absorb all four jobs. The publication-transaction owner defines per-backend semantics (filesystem rename vs object-store transaction) | [#2063](https://github.com/RC219805/Transformation_Portal/issues/2063) → 1.5-b → Phase 3 |
| Operational ledger & projections | ⟨DECIDE⟩ — canonical operational event/artifact record and projection rules, disposing of each of the four surfaces explicitly (`JobEventStore` SSE/job history, `OperationalAuditStore` pilot audit, `events/store.py` lifecycle events, `runtime/ledger.py` `ImmutableLedger`) — a designated composition, never a conflation | Phase 3 |
| Competing implementations | ⟨DECIDE per inventory row⟩: keep / adapt / freeze / remove — separately account for row 3's package-local dashboard scheduler consumer and absent application wiring, row 4's unconsumed distributed wrapper, the patcher/node consumers, and `runtime/`'s engine surfaces | staged with Phase 3 |
| Associated tests | Tests follow authority ownership: retained, migrated, consolidated, or **deleted with their implementation** | same PRs as dispositions |
| Coverage floors | Floors move to the owning package; a frozen implementation's floor is removed or frozen-and-annotated | `scripts/ci/check_per_package_*` updates in disposition PRs |

### Suitability matrix (required per candidate, per contested row)

| Criterion | What to record |
|---|---|
| Consumers | Verified production/tooling consumers, with citations |
| Verified capabilities | What it demonstrably does today (tests cited) |
| Missing capabilities | Gap to the row's requirement |
| Contract conflicts | Collisions with contract families, ADR designations, version planes |
| Security / trust boundaries | Entry-point and actor trust model; authentication, authorization, and tenant isolation; untrusted plan/config/node deserialization; arbitrary-code/plugin, subprocess, and remote-worker boundaries; filesystem/CAS path validation; digest and signature verification; secret handling; dependency/runtime provenance; resource/denial-of-service limits; and fail-closed test evidence. Record unsupported controls as missing capabilities, not assumptions |
| Migration cost / rollback | Effort to adopt; effort to back out |
| CI support | Which existing lanes exercise it; what new lanes it would need |
| Production-spike evidence | Result of a bounded spike on the Lux production path |

*Evidence standard for performance criteria (2026-08-28, repair 1.6-a merged as `6bf5c8b`):*
performance claims in a suitability matrix cite runs of the repaired Performance Monitor lane —
which now hard-fails unless `scripts/ci/check_performance_evidence.py` proves the benchmark tests
executed against valid committed baselines — or an equivalently evidence-gated measurement. Output
from the pre-repair lane (false-green on every nightly run) is not admissible evidence.

### Authority boundary: operational records vs cryptographic evidence

Operational records and cryptographic evidence are **separate authority planes**. The operational
record may be the generation-time source used to render contract artifacts, but it is not a
substitute evidence chain. Published evidence bytes, their schemas and canonicalization rules, their
contract-native commitments, and — where authenticity is asserted — their verified detached
signatures remain authoritative within their respective scopes. Operational records may reference
them by digest but may not reinterpret, repair, overwrite, or supersede them. **Divergence must fail
closed** — a binding REQUIREMENT on the designated authorities. Current verifier behavior cited here
is suitability evidence, not an authority designation. Review evidence (2026-08-28) found that
detached evidence-attestation binding (`bind_attestation_to_evidence`, reached by
`tools/verify_evidence_attestation.py`) had compared stored `evidence_sha256` values without
recomputing the canonical digest of `projected_envelope`. PR
[#2082](https://github.com/RC219805/Transformation_Portal/pull/2082) (`6e8e3a3`, 2026-08-28) closed
that specific gap on `main`: verification now recomputes the projected-envelope digest, binds the
file and bundle-root digests plus the attestation self-hash, and verifies exact native-GPG cleartext
and recorded signer identity, with adversarial library and CLI regression coverage. That landed
repair supplies evidence for this requirement and satisfies this specific evidence-binding
precondition. It does not select an operational-ledger/projections authority or change this ADR's
Proposed status; those decisions remain matrix-gated.

Additionally:

- Operational CAS roots, Phase 3 roots, Phase 4 provenance roots, Lux V1 commitments, and Lux V2 CT
  roots remain **distinct hash domains**; no designation merges them.
- Published evidence remains independently verifiable **without** the operational database.
- The publication authority owns safe atomic persistence; existing contract modules retain schema,
  serialization, commitment, and verifier ownership.
- A manifest or run card produced as a projection **retains its existing digest and attestation
  semantics** — projections change the producer, never the bytes' contract meaning, and not every
  evidence artifact is signed.

### Conflict matrix (to resolve before acceptance)

| Standing decision | What it currently designates | Potential conflict with this ADR | Required amendment if changed |
|---|---|---|---|
| ADR-029 | spatial_ai execution-graph executor for spatial pipelines | Stage-executor row may designate a different executor | Amend ADR-029 scope or record spatial_ai as an adapt-consumer; update Supersedes here |
| ADR-043 | Lux decomposition seams; facade extraction pattern; recorded finding that depth/materials stages resist extraction | Plan-representation and stage-executor rows change how the Lux loop executes | Record ADR-043's Phase-6 finding as an input constraint; amend its "completed" scope if the state machine is replanned |
| ADR-045 | Repo-wide extraction pattern; ~200 LOC/quarter orchestrator ratchet | Phase 3 removals must land as ADR-045-pattern PRs; freeze dispositions must not violate the ratchet commitment | Cite ADR-045 in each disposition PR; reconcile the ratchet if orchestrator code moves wholesale |
| PR #2070 / issue #2065 (`ResolvedInvocation`, `--plan`) — **merged to `main` as `e30bd58` (2026-08-28)** | Provisional Lux-owned plan serialization and contract-aware fingerprint identity, now live in production | Plan/DAG-representation and execution-identity rows may designate differently | Scoped above as a bounded provisional pre-designation spike (`stability: provisional`); adopted or migrated at designation; its identity algorithm and review-verified trust-boundary evidence are inputs to — not decisions of — the identity and plan rows |
| PR #2081 / issue #2066 — **merged to `main` as `81c9f15` (2026-08-28); issue closed** | Current Lux model-selection contract: commercial-safe `da3_metric` default; deprecated `da3` alias to acknowledgement-gated `da3_research`; explicit selector provenance; identity consistency across plan and execution carriers | Execution-identity and Plan/DAG rows may change representation, but do not implicitly change this model-selection and licensing contract | Preserve as landed contract input; any incompatible later decision requires explicit contract, version, test, and documentation handling. This row supplies evidence, not a designation |
| Production-hardening baseline (gap doc §4) | `orchestrator/*` Protocols, EventStore for lifecycle events, attestation chain authoritative, "reuse, don't reinvent" | Ledger and artifact-composition rows must extend, not parallel, these primitives | Any deviation from §4's designations is recorded here and in the gap doc's next refresh |

### Binding rules (effective on acceptance)

1. **No new scheduler, ledger, or artifact-store abstraction merges before this ADR is accepted.**
   Before acceptance this is a proposed convention, not an enforceable gate on `main`: it binds
   the proposers voluntarily from the proposal date, and any abstraction that merges during the
   proposal window must be added to the conflict matrix and disposed of at acceptance (the #2065
   provisional plan surface is the first such entry, introduced by #2070, hardened by #2081, and
   recorded above).
2. A frozen implementation cannot retain a permanently green suite that no longer proves production
   behavior — its tests are migrated, consolidated, or deleted in the disposition PR.
3. `plan` and `run` share one resolution path; `run` ultimately consumes the exact resolved plan
   object `plan` emits ([#2065](https://github.com/RC219805/Transformation_Portal/issues/2065)) —
   this invariant applies to every designated component. **Definition across process boundaries:**
   within one process, "exact" means object identity; across separate CLI invocations or the worker
   subprocess boundary, it means equivalence — byte-identical canonical serialization and equal
   config fingerprint. A deserialized plan carries no authority of its own: every carried-contract
   consumption boundary uses `validate_authoritative_model_contract` and fails closed on registry,
   specification, or revision mismatch. Current acknowledgement and licensing policy additionally
   fails closed at every execution-capable boundary. The explicitly metadata-only, idempotent
   `ConfigResolver` carrier revalidates contract integrity without independently enforcing licensing.
4. Structural write prohibition: manifest/run-card/artifact-path writes outside the designated
   authority fail CI, enforced by extending the existing raw-JSON governance gate
   (`check_raw_json_usage.py` pattern), not a new mechanism.
5. Contract families (ingest, machine-mode, evidence/attestation, run card, presets) are unchanged by
   designation; authorities produce them, never redefine them. Version planes stay separate.

---

## Migration table

Filed repairs feeding this ADR (Lux Depth V3 repair program, first waves):

| Repair | Issue | Relation to this ADR |
|---|---|---|
| 1.6-a Performance false-green | [#2062](https://github.com/RC219805/Transformation_Portal/issues/2062) — **closed by merged PR [#2071](https://github.com/RC219805/Transformation_Portal/pull/2071)** (`6bf5c8b`, 2026-08-28) | Restores trustworthy perf evidence for suitability spikes. *Landed:* the nightly lane is green only when `scripts/ci/check_performance_evidence.py` proves the four baseline-writer tests executed and passed, the four benchmark artifacts parsed, and the committed baselines were valid; the gate's exit code separates regression (failed writer comparison with complete evidence) from invalid evidence and non-writer suite failures. Performance rows in suitability matrices may now cite this lane's runs — validated end-to-end by dispatch run [33210317364](https://github.com/RC219805/Transformation_Portal/actions/runs/33210317364) (evidence, not designation) |
| 1.5-a Atomic evidence writes | [#2063](https://github.com/RC219805/Transformation_Portal/issues/2063) | Prerequisite primitives for the publication-transaction role |
| 1.4-a Depth cache identity | [#2064](https://github.com/RC219805/Transformation_Portal/issues/2064) | First derivation of the identity-authority pattern |
| P0-1 `--plan` / `ResolvedInvocation` | [#2065](https://github.com/RC219805/Transformation_Portal/issues/2065) — **open**; advanced by merged PR [#2070](https://github.com/RC219805/Transformation_Portal/pull/2070) (`e30bd58`) and direct-Python/default/preset/carrier hardening in merged PR [#2081](https://github.com/RC219805/Transformation_Portal/pull/2081) (`81c9f15`). The full documented-workflow parity harness and production-manifest plan fields remain issue-level closure items | Seed of the plan-representation row (spike merged, provisional) |
| 1.2 DA3 default (decision) | [#2066](https://github.com/RC219805/Transformation_Portal/issues/2066) — **closed by merged PR [#2081](https://github.com/RC219805/Transformation_Portal/pull/2081)** (`81c9f15`): Option A landed with `da3_metric` as the commercial-safe default and `da3` retained as a deprecated alias to acknowledgement-gated `da3_research` | Landed single-resolution contract input; not a designation |
| 1.3-a emit flags (decision) | [#2067](https://github.com/RC219805/Transformation_Portal/issues/2067) | Deliverable contract input to accounting |
| 1.3-b 16-bit flags (decision) | [#2068](https://github.com/RC219805/Transformation_Portal/issues/2068) | Deliverable contract input to accounting |

Held pending prerequisites (filed when unblocked): 1.1-a/1.1-b documentation gate, 1.3-c
requested-vs-produced accounting, 1.4-b weight identity, 1.4-c segmentation identity policy, 1.5-b
generation publication (**gated on this ADR's acceptance**), 1.6-b Lux branch floor.

---

## Consequences

- Phase 3 of the repair program (production-path pilot) cannot start ahead of the rows it depends on:
  stage executor, artifact composition/publication owner, execution identity.
- Disposition PRs carry real deletion work: superseded direct-write, skip/reuse, and shadow facade
  paths are removed as replacements land — completion is production use plus old-path removal, never
  the existence of a new class.
- The V4 go/no-go gate's first condition ("the designation ADR has selected real shared
  infrastructure") is satisfied by acceptance of this ADR **plus** at least one production vertical
  slice using the designated components.
- Coverage-floor moves and test migrations are visible in `scripts/ci/check_per_package_*` diffs and
  the cold-zone program tables; CLAUDE.md's coverage-program description is corrected in the same
  window (repair 1.1-b).

## Alternatives considered

- **Per-package status quo** (each subsystem keeps its own plane): rejected — reproduces the
  shadow-architecture failure mode already observed (production-dead facade classes, orphaned
  `EventStore`/`execution_identity`), and leaves the generation publication transaction ownerless.
- **New unified V4 kernel**: rejected — the "parallel abstraction" the gap doc §4 forbids; would be
  an eighth stage-execution surface.
- **Designation without disposition** (name authorities, leave competitors in place): rejected —
  binding rule 2 exists precisely because green-but-unproving suites and unwired "canonical" code are
  the documented failure mode here.

## Acceptance / merge checklist

This ADR merges to `main` as **Accepted** only when:

- [ ] No `⟨DECIDE⟩` cells remain; every contested row carries its completed suitability matrix.
- [ ] The conflict matrix rows are resolved and any amended ADRs updated in the same PR
      (including this ADR's Supersedes field).
- [ ] No verification placeholders or incomplete house-format sections remain.
- [ ] Decision makers named above have signed off per repo governance.

Until then it remains **Proposed** on its review branch; binding rule 1 is observed by convention
from the date of this proposal.
