# ADR-051: Execution and Artifact Authority Designation

**Status:** Proposed — design complete; exact-head owner sign-off pending
**Date:** 2026-08-28
**Last reviewed:** 2026-08-30
**Decision Makers:** Repository owner (acceptance) + Architect (designation) + Specialist (migration)
**Replaces:** None
**Supersedes:** ADR-029 only for the long-term stage-executor designation; ADR-043 only for its
permanent-retention finding for Lux depth/materials execution. ADR-029's shipped Spatial AI
contracts and ADR-043's completed decomposition seams remain migration constraints. ADR-045 is not
superseded.
**Related:** [ADR-029 execution-graph abstraction](ADR-029-execution-graph-abstraction.md),
[ADR-043 orchestrator decomposition](ADR-043-orchestrator-decomposition.md),
[ADR-045 decomposition governance pattern](ADR-045-monolith-decomposition-residuals.md), and
[Production Hardening Gap 2026-05-13](../governance/PRODUCTION_HARDENING_GAP_2026-05-13.md) section 4

> Numbering note: ADR-050 is the active portal React migration decision under `docs/decisions/`;
> this record therefore uses 051.

---

## Context

The repository has several independent schedulers, execution identities, cache stores, Merkle
producers, artifact stores, and event logs. Most have strong unit tests, but only the Lux Depth V3
loop, the publicly exported imperative `SpatialAIPipeline.process()` and `process_multiview()` paths,
and the orchestrator job boundary are current execution paths. Single-view graph mode is default-off;
multi-view reconstruction explicitly rejects graph mode. Green tests on an unmounted or unconstructed
implementation do not establish production authority.

This record designates a target authority for each plane. A designation is architectural direction,
not a claim that the target is already production-ready. Every missing control recorded below is a
binding migration prerequisite. The current Lux and Spatial imperative paths remain executor
rollback paths until their production vertical-slice gates pass; both adopt canonical publication
before any executor cutover.

The inventory and consumer searches were refreshed at `main` `ca47d4988` and PR head `085745a2d`
on 2026-08-30.

### Execution inventory

The orchestrator job-boundary composition is identified as `J` because queue, lease lifecycle,
durable job state, and reclaim reconciliation are job responsibilities, not a ninth stage executor.

| Row | Implementation | Verified current use |
| --- | --- | --- |
| 1 | `stage_graph/graph.py` + `core/cas_dag_executor.py` | Core and Depth/Lux primitives are reused, but no production construction of `StageGraph` or `CASDAGExecutor` was found. |
| 2 | `spatial_ai/orchestration/graph/` executor, graph, and cache | Default-off Spatial AI adapter plus examples and tests; no verified production configuration enables it. |
| 3 | `execution_graph/scheduler.py` + `nodes/` | Constructed by a package-local dashboard router that is not mounted by the application. Patcher and some nodes have independent consumers. |
| 4 | `execution_graph/distributed_executor.py` | Tests and documentation only; no verified runtime construction. |
| 5 | `comfyui/executor.py` | Domain workflow implementation with tests/templates; no verified production construction. |
| 6 | `runtime/engine.py` + runtime scheduler/process/GPU/sandbox/manifest/ledger/replay helpers | Package-local references, examples, and tests; no verified production construction. |
| 7 | Lux V3 orchestrator + `pipeline_coordinator`/`execution_engine` | Current Lux production execution path; its `ExecutionPlan` is a flat stage-name list. |
| 8 | `spatial_ai.orchestration.SpatialAIPipeline` imperative `process()` and `process_multiview()` paths | Publicly exported. Single-view imperative execution is selected whenever `use_execution_graph=False`, the default; multi-view always executes its own reconstruction/export loop and fails if graph mode is enabled. Repository deployment construction was not verified. |
| J | `QueueBroker` + `WorkerRunner` + `JobRepository` + app admission/reclaim coordinators, with standalone `worker_process.py`; target admission/fence transactions use `OperationalRecordStore` | Current leased-job composition in FastAPI lifespan and multi-host worker mode. The app currently checks global and optional per-tenant active-job caps under a process-local lock before enqueue; repository counts fail closed but admission is not atomic across API hosts. Broker, runner, repository, and reclaim roles remain split as detailed below. |

`execution_graph.patcher` consumers do not construct rows 3 or 4 and therefore do not make either
scheduler live. `tp.merkle`, `tp.crypto`, Phase 4 provenance, evidence envelopes, and attestations
are the cryptographic evidence plane; they are explicitly outside operational consolidation.

### Plane inventory

- `core/execution_identity.py` has the most complete code/config/environment/lock/platform identity,
  but it is not wired into Lux and permits a placeholder lock digest outside CI.
- `lux_depth_v3.ResolvedInvocation` is the only current canonical, schema-validated, license-enforced
  plan serialization. Its v1 payload is intentionally marked `stability: provisional`, and its stage
  representation lacks nodes, edges, resources, and typed stage configuration.
- `storage/cas_store.py` and `storage/merkle_dag.py` are shared operational byte/lineage primitives.
  Lux and Spatial AI maintain additional cache and Merkle projections.
- `orchestrator.artifact_store.ArtifactStore` is the live local/S3 job-delivery contract. It is not a
  CAS and does not provide an all-old-or-all-new generation publication transaction.
- `orchestrator.storage.JobRepository` is the live durable mutable job snapshot;
  `JobEventStore` is live job/SSE history; `OperationalAuditStore` is live managed-control-plane
  audit; `events/store.py::EventStore` is orphaned; `runtime/ledger.py::ImmutableLedger` is tested but
  has no verified production consumer.
- `app.py` owns current request admission orchestration: `MAX_CONCURRENT_JOBS` and an optional
  per-tenant cap are checked from repository snapshots under `JOB_ADMISSION_LOCK`. That lock is
  process-local, the tenant cap defaults to disabled, and count-then-enqueue is not atomic across API
  hosts; it is a migration seed, not durable admission authority.

---

## Decision

### Authority summary

| Plane | Designated authority | Owner | Current state and migration exit |
| --- | --- | --- | --- |
| Execution identity | `core.execution_identity.ExecutionIdentity`, evolved to schema v3 | `core` | Target authority. Wire authoritative model/license identity, executed path, materialized dependency/weight digests, and strict production completeness before any cache cutover. Lux fingerprints become projections. |
| Plan/DAG representation | The `ResolvedInvocation` contract family, promoted from `tp.lux.resolved_invocation.v1` to the single repository schema `tp.execution.plan.v1` | `core` contract owner; Lux supplies the first adapter | Adopt the landed single-resolution semantics, then add typed nodes, edges, resources, outputs, and stage-registry identifiers. The legacy Lux v1 reader remains a bounded compatibility adapter until issue #2065 closes. |
| Stage executor | Row 1, `core.cas_dag_executor.CASDAGExecutor` + `stage_graph.StageGraph` | `core` + `stage_graph` | Adapt and designate as the target. It must consume `tp.execution.plan.v1`, execute through an allowlisted stage registry, absorb the needed process/GPU primitives from row 6, and pass the vertical-slice gate before production activation. |
| Job runner boundary | Row J composition: `QueueBroker` + `WorkerRunner` + `JobRepository` + app admission/outbox/reclaim coordinators + `OperationalRecordStore` target transactions | `orchestrator` | Keep and designate with one owner per subresponsibility. The app owns authenticated policy/preflight coordination and idempotent outbox delivery; `OperationalRecordStore.admit_job` atomically reserves capacity and creates immutable attempt/dispatch IDs plus the outbox, while `claim_dispatch` is the once-only execution gate; broker owns queue/lease/cancel/reclaim primitives; runner owns polling, heartbeat, and lease lifecycle; repository owns durable job snapshots; the app reconciler makes `worker_lost` terminal. Each effective `JobLease` carries holder/epoch; the Postgres fence row carries its database-clock deadline. The executor runs inside one claimed lease; explicit retry is a new admitted job ID/attempt ID. |
| CAS/Merkle operational lineage | `storage.cas_store.ArtifactStore`, renamed `ContentAddressedStore` with a compatibility alias, plus hardened/versioned `storage.merkle_dag.MerkleDAG` v2 | `storage` | Keep and harden. Persist reader-visible generation roots only through the publisher's fenced operational-record commit. Contract-native evidence roots remain separate domains. |
| Artifact composition | CAS bytes: `ContentAddressedStore`; stage cache: package-specific identity index pointing to CAS; job delivery: `orchestrator.artifact_store.ArtifactStore`; publication preparation: `GenerationPublisher` in that existing package; visibility: fenced `OperationalRecordStore` generation commit | `storage` + `orchestrator.artifact_store` + `orchestrator.storage` | No fourth general store. Adapt the Spatial cache metadata/index; keep Lux/portal artifact catalogs as projections. S3/local stores hold immutable staged bytes only. A Postgres commit record is the sole reader-visible generation pointer; publication is inactive until issue #2063 and the gates below pass. |
| Operational record and projections | Evolve `OperationalAuditStore` in place into versioned `OperationalRecordStore`; keep `JobRepository`, the audit API, and `JobEventStore` as projections | `orchestrator.storage` | Postgres is the durable production backend, atomic admission/capacity owner, dispatch-outbox owner, and sole generation-visibility authority. Remove orphaned `events.EventStore`; freeze/remove `runtime.ImmutableLedger` after useful verification tests transfer. Operational records reference, but never replace, evidence. |
| Tests and coverage | Tests and floors follow retained authority or migrated behavior | Owning packages | No frozen implementation may keep a green suite that purports to prove production behavior. |

The target stage executor is not activated by accepting this ADR. Row 7 stays authoritative for live
Lux execution, and row 8 remains the public/default single-view and always-imperative multi-view
Spatial path, until separate feature-flagged row-1 slices prove contract, security, output, and
performance parity for each domain. Acceptance designates the direction; activation requires the
applicable evidence gate.

---

## Suitability Matrices

Each matrix records all eight required criteria. “None observed” is evidence of a gap, not an
assumption that the capability exists. The repaired Performance Monitor lane proves that benchmark
evidence is valid; it does not prove the performance of any executor candidate.

### Execution identity

| Candidate | Consumers | Verified capabilities | Missing capabilities | Contract conflicts |
| --- | --- | --- | --- | --- |
| `core.ExecutionIdentity` / `compute_cas_id` | `core.execution_wrapper`, row 1, and extensive core/determinism tests; not Lux | Immutable versioned identity over stage, inputs, code, config, environment, lockfile, and platform; deterministic hashing and cache-miss explanation | Authoritative model/license fields, planned/executed backend path, materialized weight digest, production-wide wiring; placeholder lock hash is allowed outside CI | Current schema `adr-032-v2` cannot be treated as complete Lux identity; projections must preserve existing manifest/run-card schemas |
| Lux `ResolvedInvocation` + config/depth/segmentation fingerprints | Live plan/run, backend, manifest, run-card, depth-cache, and Stage-A paths | Single license-enforced model resolution, canonical serialization, pinned model identity, plan/run parity | Code identity, complete dependency identity, materialized weights, executed path, one cache schema; v1 remains provisional | Making a cache projection the authority would redefine existing contracts and omit determinism inputs |

| Candidate | Security/trust boundary | Migration and rollback | CI support | Production-spike evidence |
| --- | --- | --- | --- | --- |
| `core.ExecutionIdentity` | Hash inputs are trusted Python values; source/lock fallback behavior is not sufficient for a hostile or incomplete production environment. Production use must fail closed on absent code, lock, model, or path identity. | Add schema v3 additively; dual-write v2/v3 without cache reuse, compare, then switch. Roll back by disabling v3 reads and invalidating the new cache namespace. | `tests/core/test_execution_identity.py`, `tests/determinism/test_execution_identity.py`, row-1 cache/failure/path tests, determinism gate | None on Lux; required before activation |
| Lux projections | Serialized untrusted payloads require JSON Schema validation and `validate_authoritative_model_contract`; licensing already fails closed at execution-capable boundaries | Feed the resolved contract into identity v3; retain existing serialized fields as projections. Roll back to current Lux keys by schema namespace, never by cross-reading incompatible entries. | `tests/lux_depth_v3/test_resolved_invocation.py`, config/depth/manifest/run-card suites | Current Lux path proves the projections run in production, not that they are complete identity |

**Selection:** adapt and designate `core.ExecutionIdentity` v3. `ResolvedInvocation` is a mandatory
plan-time input and the existing fingerprints are stable contract projections, not competing roots.

### Plan and DAG representation

| Candidate | Consumers | Verified capabilities | Missing capabilities | Contract conflicts |
| --- | --- | --- | --- | --- |
| Lux `ResolvedInvocation` v1 | Current Lux plan/run, config resolver, backend construction, worker subprocess, manifests | Frozen object, canonical JSON, shipped schema, exact resolved model/license contract, ordered stages, requested artifacts, inputs, backend/fallback intent | Explicit node IDs, edges, typed stage configs, resource limits, retry/cancel semantics, per-output identity; still `stability: provisional` | Lux namespace and flat stages are not yet a repository DAG; promotion must preserve #2081 license/model semantics |
| Row-1 `StageGraph` | Row-1 executor and tests only | Dependency graph, cycle/order validation, stage protocol | Stable external schema, canonical serialization, license/config intent, typed untrusted parser | Python stage objects cannot cross trust/process boundaries as plan authority |
| Row-2 Spatial `ExecutionGraph` / `ExecutionPlan` | Default-off Spatial adapter and tests | Explicit inputs/dependencies, topology checks, resource aggregation, optional/checkpoint metadata | External schema, canonical serialization, license and artifact intent, production consumer | ADR-029 domain types and cache semantics are Spatial-specific |
| Row-3 scheduler/node graph | Unmounted dashboard router and tests | Dependencies, priority, GPU summary, node results | Stable schema, canonical parser, identity/license/artifact contracts, production wiring | Node objects and dashboard request shape are not the Lux/public plan contract |
| Row-5 Comfy `Workflow` / `WorkflowBuilder` | Comfy executor, templates, export helpers, and tests | Serializable node IDs, enum node types, explicit connections, parameters, and metadata in a domain workflow | Canonical/versioned schema, strict parser, topology/resource limits, identity/license/artifact intent, output contracts, production consumer | Comfy class types, parameters, and file shape are an interchange/domain contract, not repository execution authority |
| Row-6 runtime sequence/manifest | Runtime examples/tests | Run-level Merkle manifest and sequential node description | A pre-execution DAG schema, license resolution, resources, trusted stage identifiers | A post-run manifest cannot be the plan it is meant to attest |
| Row-7 `PipelineCoordinator.ExecutionPlan` | Current Lux state machine | Honest current stage order and backend planning | DAG edges, serialization/schema, external validation, general stage identity | Flat Lux internal dataclass duplicates `ResolvedInvocation` intent |
| Row-8 imperative Spatial config/request | Public/default `SpatialAIPipeline.process()` plus always-imperative `process_multiview()` | Ordered single-view `PipelineConfig.stages`; fixed multi-view `reconstruction -> export`; typed camera/request contract; fail-closed research-tier validation; resource/error policy and direct sequencing | No shared plan object, node IDs, edges, canonical serialization, external schema, per-output identity, or broker boundary | Public config/request/result, camera policy, research-license gate, PLY/sidecar/provenance, and golden semantics must survive compilation; neither an ordered config list nor a fixed direct loop is shared DAG authority |

| Candidate | Security/trust boundary | Migration and rollback | CI support | Production-spike evidence |
| --- | --- | --- | --- | --- |
| Lux `ResolvedInvocation` v1 | Schema validation plus model-contract revalidation exist; arbitrary stage/class loading does not. Input path and resource policy remain outside the payload. | Promote semantics into `tp.execution.plan.v1`; ship a read-only v1 adapter; dual-render and compare. Roll back by continuing to execute v1 through row 7. | Strong ResolvedInvocation/license/fingerprint/carrier tests | Live single-resolution plan/run evidence; no DAG-executor spike |
| Row-1 `StageGraph` | Trusted in-process stage objects only; must never deserialize a class/module name from user payload | Make it the in-memory compiled form of `tp.execution.plan.v1`, not a second schema. Roll back to row-7 compilation. | Stage graph and CAS executor suites with branch floor | None on Lux |
| Row-2 Spatial graph | Trusted Python stages; no auth/tenant/untrusted parser | Reuse topology/resource test cases in the canonical compiler; preserve Spatial adapter while migrating. | Extensive Spatial graph/resource/cache tests | Default-off test spike only; no production configuration |
| Row-3 graph | Arbitrary node construction and synchronous execution are unsuitable for untrusted plans | Harvest only independently useful visualization/node behavior; remove scheduler representation. | Scheduler/dashboard suites | None; router unmounted |
| Row-5 Comfy workflow | JSON/file content and free-form parameters are untrusted; enum lookup is not a closed schema or authorization boundary | Keep as validated domain input/export. If needed, compile allowlisted Comfy operations into canonical stages; never promote the workflow format itself. | Workflow builder/template/executor suites; add schema, bounds, and path-negative tests before any adapter | None |
| Row-6 sequence/manifest | Trusted class objects are passed to subprocesses; manifest load is not a plan authorization boundary | Keep manifest as a projection; do not promote it to plan. | Runtime manifest/engine tests | None |
| Row-7 flat plan | Same trusted Lux config boundary as current production; licensing is carried by ResolvedInvocation instead | Compile current v1 into the canonical schema behind a flag; roll back to unchanged loop. | Lux contract suites | Current production path only |
| Row-8 imperative Spatial paths | Trusted local Python/config-file boundary; multi-view has typed request/camera and research-tier checks, but both direct paths write without a broker/tenant authorization layer | Compile single-view config and multi-view request into distinct allowlisted canonical stages behind opt-in flags. Preserve both imperative branches as executor rollback until their separate parity gates and ADR-045 windows pass. | Spatial single-view pipeline/preset/failure/golden/public-import suites plus multi-view camera/tier/license/request/result/PLY/sidecar/provenance suites | Both public paths are live when constructed; no canonical-plan production spike |

**Selection:** adopt the `ResolvedInvocation` contract family and promote it to
`tp.execution.plan.v1`. `StageGraph` is its trusted in-memory compiled form. The promotion is an
evolution of the landed contract, not a parallel plan abstraction.

### Stage executor, rows 1–8

| Row/candidate | Consumers | Verified capabilities | Missing capabilities | Contract conflicts |
| --- | --- | --- | --- | --- |
| 1 `CASDAGExecutor` + `StageGraph` | Tests and core primitives; no production construction | Dependency order, execution identity, cache locks, CAS/Merkle provenance, failure short-circuit | Lux adapters; process isolation; cancellation; resource budgets; scalar/list/path inputs in identity; configured parallelism is not used by the current sequential loop; output-byte determinism verification; current CAS reads are not rehashed before deserialization and `Stage.execute()` can use a second legacy JSON/NPY cache | Must compile the canonical plan and preserve Lux/Spatial contracts rather than expose its Python graph as a second public schema; the designated adapter cannot retain two cache authorities |
| 2 Spatial Executor | Default-off adapter and extensive tests | DAG validation, caching, provenance, optional stages, resource-limit planning | Production wiring, process isolation, canonical identity, cancellation/tenant controls | ADR-029 currently designates it for Spatial AI; its independent executor would remain duplicate authority |
| 3 `PriorityDAGScheduler` | Unmounted dashboard router; nodes/patcher partly independent | Priority/dependency ordering, GPU routing summary | Auth, canonical identity, cancellation, resource enforcement, durable artifacts; node failure result handling is incomplete | Dashboard/node request shape conflicts with the canonical plan and job boundary |
| 4 `DistributedDAGExecutor` | Tests/docs only | Local delegation and optional Ray wrapper | Production use, timeout/retry enforcement, identity, cancellation, tenant/artifact security | Premature distributed queue/execution plane before local authority is proven |
| 5 Comfy `WorkflowExecutor` | Templates/tests; no production construction | Domain node dispatch and topological ordering | Cycle-stack validation, authority identity, isolation, cancellation, resource/tenant controls; `output_dir` is unused | Comfy workflows are domain input, not the repository executor contract |
| 6 `runtime.ExecutionEngine` | Runtime tests/examples only | Spawned process execution, timeouts, GPU leases, CAS, Merkle lineage, filesystem guard helpers | DAG compiler, canonical identity, production wiring, real OS/network sandbox, workspace quota, complete path containment | Promoting the facade would duplicate row 1; its useful strategies can be composed underneath row 1 |
| 7 Lux loop | Current Lux production path | Full backend/fallback/APEX/cache/evidence behavior and broad contract coverage | Shared DAG, complete identity, process isolation, one artifact/record path | Bespoke scheduling perpetuates the shadow architecture |
| 8 imperative Spatial paths | Public/default single-view and mandatory multi-view paths when `SpatialAIPipeline` is constructed | Ordered single-view stages; fixed multi-view reconstruction/export; resources, retries/fallback, progress, camera/tier checks, PLY/sidecar provenance, and current Spatial outputs | Shared DAG/identity, process isolation, cancellation, tenant boundary, one artifact/record path | Public single-view and multi-view behavior is a migration constraint, but retaining either direct loop permanently would leave a second executor authority |

| Row/candidate | Security/trust boundary | Migration and rollback | CI support | Production-spike evidence |
| --- | --- | --- | --- | --- |
| 1 `CASDAGExecutor` + `StageGraph` | Trusted in-repo stages only. Stage names, roots, inputs, and output references need validation. No arbitrary import/class/plugin deserialization. Before designation, disable/remove `Stage.execute()`'s legacy JSON/NPY cache, use a closed cache-manifest schema, and rehash every CAS object before deserialization; an unconfined `__numpy__` reference or digest mismatch fails closed. | Add a static stage registry and plan compiler; compose row-6 process/GPU strategies; replace the second cache with one identity-index-to-CAS path; dual-run non-publishing comparison, then one publishing path. Roll back Lux by feature flag to row 7 and Spatial by feature flag to row 8. | Stage graph line/branch floors; CAS/cache/failure/path tests; add second-cache prohibition, manifest-schema, rehash/tamper, cancellation, quota, plan-parity, and real-process tests | None; bounded Lux and Spatial slices required |
| 2 Spatial Executor | Trusted stage objects; Spatial cache rejects traversal/object arrays and uses commit markers, but no request auth/tenant layer | Keep Spatial public adapters and its cache as a CAS-backed metadata index; compile stages to row 1; remove only its duplicate executor/plan after parity. | Strong Spatial graph/artifact suites | Test-only/default-off spike |
| 3 scheduler | Executes registered Python nodes in process; no independent auth or tenant boundary | Freeze; move any visualization-only consumer; retain independently consumed patcher/nodes; then remove scheduler/router. Roll back by restoring package-local router before deletion only. | Scheduler and dashboard suites | None |
| 4 distributed | Ray worker trust, remote code/dependency provenance, secrets, and tenant isolation are not established; configured retry is not enforced | Remove wrapper and its tests. A later distributed design requires a new accepted ADR after local activation. | Mocked/local package suite only | None |
| 5 Comfy | Treat workflow content as untrusted domain data; never let it choose Python handlers outside an allowlist | Keep templates/export; freeze executor as non-authoritative, or later adapt a validated Comfy stage. | Template contract and unsupported-node tests | None |
| 6 runtime engine | Process spawn gives fault isolation, not a security sandbox. Network disabling is not enforced; workspace is unbounded; some containment is lexical. | Harvest `ProcessExecutor`, GPU pool, and hardened sandbox strategy beneath row 1; freeze/remove the competing engine/scheduler facade after transfer. | Runtime engine/process/sandbox/manifest tests | None |
| 7 Lux loop | Existing CLI/API validation and licensing apply; direct writes and shared state remain migration risks | Before any executor migration, route unchanged row-7 outputs through the canonical publisher/record/projection path and prove parity. Then adapt one stage slice at a time under an opt-in flag. Executor rollback returns to row 7 while retaining the canonical publication path. | Broad Lux unit/contract suites, publisher-baseline contracts, and repaired benchmark evidence lane | Current production execution baseline and rollback reference |
| 8 imperative Spatial paths | Current local config/path validation applies; multi-view additionally fails closed on typed camera/request and research-tier rules; direct in-process stages and writes have no broker/tenant boundary | Before any Spatial executor migration, route unchanged single-view and multi-view outputs through the canonical publisher/record/projection path and prove parity. Then adapt one stage slice at a time, preserving every public contract. Executor rollback returns to the applicable row-8 loop while retaining canonical publication. | Spatial pipeline/preset/resource/error/golden/public-contract and multi-view camera/tier/license/result/PLY/sidecar/provenance suites plus publisher-baseline contracts | Current public/default single-view and mandatory multi-view construction paths and rollback references |

**Selection:** adapt and designate row 1. The designation is conditional on the hardening and
vertical-slice gates; rows 7 and 8 remain the active Lux and Spatial executors until their respective
gates pass.

### Job runner boundary

| Candidate | Consumers | Verified capabilities | Missing capabilities | Contract conflicts |
| --- | --- | --- | --- | --- |
| J orchestrator composition | FastAPI lifespan and standalone worker process | `QueueBroker`: queue/lease primitives; `WorkerRunner`: polling/heartbeat; `JobRepository`: durable snapshots; app: authenticated preflight, process-local global/optional-tenant admission checks, and reclaim coordination; audit store: current fail-closed managed-pilot audit | Multi-host atomic global/tenant admission with a mandatory positive production tenant cap; immutable attempt/dispatch identity, durable outbox/once-only claim/terminal tombstone; per-acquisition holder/epoch plus Postgres-fence deadline; locator-only execution; atomic canonical-record/projection updates | None if each named component retains only its subresponsibility and the stage executor remains inside one lease |
| A stage executor owning queue/retries/state | None | None observed | All durable queue, lease, cancellation, recovery, state, and audit behavior | Duplicates the J composition and risks double retry, acknowledgement, state, and publication ownership |

| Candidate | Security/trust boundary | Migration and rollback | CI support | Production-spike evidence |
| --- | --- | --- | --- | --- |
| J orchestrator composition | Broker and job-state data are untrusted for code selection. `OperationalRecordStore.admit_job` atomically reserves capacity and creates immutable job-attempt/dispatch IDs plus the dispatch outbox. Before spawn, `claim_dispatch` changes that exact dispatch from queued/dispatched to running once and binds holder/epoch; duplicate deliveries fail the claim. Raw executable/module/shell/`argv` fields are forbidden. Publication requires the matching unexpired fence. `worker_lost` and every other terminal record tombstone the dispatch and release capacity. The same job/dispatch never reacquires; explicit retry is a newly admitted job ID/attempt ID with a new slot and `retry_of` link. | Replace process-local count-then-enqueue with the Postgres admission/outbox/claim transaction family; keep current logic only as local-memory rollback before production cutover. Use a separately versioned/ACL-scoped locator queue, deploy new workers there, drain/revoke every raw-command worker/queue, then enable canonical producers. New workers reject legacy/raw fields; executor rollback retains new admission, queue, and job boundaries. | Existing admission/queue/worker/repository/reclaim/cancel suites plus concurrent multi-host quota, admission/outbox/claim crash-idempotency, duplicate delivery, terminal tombstone/slot release, explicit-retry admission, mixed-version isolation, raw-command rejection, tampered locator, wrong-holder/stale/expired fencing, and real Postgres/Redis external-worker cases | Live partial app admission and job composition; atomic admission, durable dispatch claims, locator namespace, and fencing not yet nested |
| Executor-owned queue | Unspecified | Rejected; no migration | None | None |

**Selection:** keep and designate the full J composition: the app admission/outbox/reclaim
coordinators; `OperationalRecordStore` admission, capacity, dispatch-claim, lease-fence, and
publication transactions; `QueueBroker` queue/lease primitives; `WorkerRunner` polling and lease
lifecycle; and `JobRepository` durable snapshots. Each retains only the subresponsibility named
above; none is the entire job boundary.

### Operational CAS and Merkle lineage

| Candidate | Consumers | Verified capabilities | Missing capabilities | Contract conflicts |
| --- | --- | --- | --- | --- |
| `storage.ArtifactStore` / target `ContentAddressedStore` | Rows 1 and 6 plus dashboard/runtime/core consumers; CAS has a coverage floor | SHA-256 validation, atomic/fsync writes, locking, corruption quarantine, content verification | Tenant namespace/authorization, remote backend, publisher-owned GC references; `materialize(..., verify=False)` can propagate corrupt bytes and destination confinement belongs to callers | It is bytes, not lineage, delivery, publication, or evidence; current name collides with other stores |
| `storage.MerkleDAG` v1 | Rows 1 and 6 plus dashboard/runtime/core consumers | Deterministic node hashes, traversal, export/load, missing-reference checks | Integrity verification does not recompute node keys or validate roots/types/cycles/content digests; imports are unbounded/trusting and export is non-atomic | SHA chaining is not authenticity and cannot be described as evidence authority |
| Lux `artifact_manager` Merkle root | Current Lux production outputs | Stable sorted artifact-index/root contract | General byte store, execution-node lineage, transaction; preimage omits path/type/size/domain/version | Root bytes and semantics are contract-bearing and cannot be silently replaced |
| Spatial cache/provenance | Spatial adapter/tests | Commit marker, pickle-disabled reads, corruption self-heal, provenance and eviction | Shared immutable-byte CAS, tenant/remote support, production consumer; marker does not bind payload/provenance digest to key | A second byte authority if payloads are retained locally |
| `tp.crypto` / Phase 4 / attestation | Contract and verifier consumers | Canonical evidence serialization, versioned roots, proofs, detached binding, fail-closed recomputation | Wrong role for operational scheduling/state | Evidence authority is excluded from consolidation |

| Candidate | Security/trust boundary | Migration and rollback | CI support | Production-spike evidence |
| --- | --- | --- | --- | --- |
| `ContentAddressedStore` | Validate digest and confined materialization destination; prohibit unverified reads in designated paths; tenant/job root and quota policy sits above the immutable store. | Rename with compatibility alias; dual-read immutable bytes; use publisher commits as the GC reference set. Roll back by leaving new objects unreferenced. | CAS atomic/hash/corruption/lifecycle suites; add power-loss, process-race, confinement, and GC-reference cases | None on Lux |
| `MerkleDAG` v2 | Untrusted imports require canonical hash recomputation, CAS-reference validation, cycle/root/type/count/depth limits, and tenant/run scope. | Add domain-separated schema v2; dual-read v1 as untrusted input and recompute. Persist only through the publisher. | Existing DAG tests plus adversarial import/tamper/bounds/CAS-binding/crash lanes | None on Lux |
| Lux root | Existing output validation applies; published evidence remains independently verifiable | Preserve exact algorithm/bytes as a projection referenced by operational lineage. A replacement requires a separately versioned contract. | Lux artifact/manifest/run-card suites | Live Lux baseline |
| Spatial cache | Safe-key/object-array/commit-marker tests exist; no tenant auth boundary | Move immutable payload to CAS and retain key-to-CAS/provenance metadata; dual-read old NPZ entries. Remove only if the Spatial executor/cache is retired. | Extensive Spatial cache suite; add payload/key binding | Default-off only |
| Evidence plane | Existing verifier-specific trust boundaries | No migration; publisher persists validated bytes and records digests only. | Existing evidence/attestation suites | Not applicable |

**Selection:** `ContentAddressedStore` and hardened `MerkleDAG` v2 own operational bytes and
lineage. Existing contract roots remain projections in distinct hash domains.

### Artifact roles and composition

| Candidate/surface | Consumers | Verified capabilities | Missing capabilities | Contract conflicts |
| --- | --- | --- | --- | --- |
| `ContentAddressedStore` | Core/runtime/dashboard tests and helpers | Immutable verified bytes, atomic local durability, quarantine | Job listing/presigning/deletion, tenant policy, generation commit | Must not absorb delivery or evidence-schema responsibilities |
| Spatial graph `ArtifactStore` | Spatial only | Stage serialization, commit marker, provenance, eviction/self-heal | Shared byte CAS, tenant/remote/delivery support, digest-bound marker | Must become an identity index, not another immutable-byte authority |
| Orchestrator local/S3 `ArtifactStore` | Live app artifact delivery | Validated job/path keys, local atomic writes, S3 streaming/presigning/readiness/deletion, backend contract tests | Expected-digest/conditional generation commit; current mirroring can expose partial generations; large hashes may be skipped | Must preserve authenticated URLs and lifecycle payloads and remain delivery-focused |
| Lux `ArtifactManager` + `portal.job_artifacts` | Live Lux indexing/Merkle and app helper use | Output naming/index/fingerprint/Merkle contract projections | General storage/transaction authority; bounded fingerprints are not publication digests | Replacing projections would change public artifact/run-card contracts |
| `GenerationPublisher` | No implementation exists | None observed | Immutable generation IDs, complete digests, idempotency/conflict, tenant/path/size/count policy, recovery, evidence validation, and a fenced handoff to the database visibility commit | Must compose with CAS/delivery and records, never become a universal store or an independent visibility authority |

| Candidate/surface | Security/trust boundary | Migration and rollback | CI support | Production-spike evidence |
| --- | --- | --- | --- | --- |
| `ContentAddressedStore` | Internal content API; validate digest, source, destination, namespace, and quota | Add identity metadata/references, not delivery methods. Rollback leaves CAS unreferenced. | CAS/path/corruption suites | None on Lux |
| Spatial cache | Safe keys/NumPy and commit markers are tested; no application auth | Store CAS references plus provenance; preserve old-entry read during migration. | Spatial artifact suite plus new digest-binding parity | Default-off only |
| Orchestrator store | Application performs auth/job/path checks; local symlink and S3-prefix traversal cases are covered. Credentials/endpoints remain deployment trust. | Keep delivery API/backends unchanged; compose with the publisher. Existing `local|s3` selection is the delivery rollback. | Parametrized local/S3 contract and paid-pilot gate | Live delivery, no generation transaction |
| Lux/portal projections | Current validation and evidence contracts apply | Generate unchanged catalogs/manifests from the committed generation; retain legacy scanner during dual-publish. | Lux and portal artifact suites | Live baseline |
| `GenerationPublisher` | Local/S3 are immutable staging planes, never visibility authorities. The publisher writes a closed manifest and all digest-verified bytes, then calls `OperationalRecordStore.commit_generation(job, tenant, generation, manifest_digest, holder, lease_epoch)`. That single Postgres transaction locks the job fence row; requires matching tenant/job/holder/epoch, running state, and `clock_timestamp() < lease_valid_until`; appends the canonical generation record; advances the reader-visible pointer; and writes the outbox. Readers never list a staging prefix; they resolve the database pointer and verify the referenced manifest/objects. | Implement after #2063 inside the existing artifact-store package; first route rows 7 and 8 through it without changing their execution semantics, then compare every byte/digest/projection before switching visibility. Rollback is an authorized, fenced **new** commit referencing a previous immutable generation, never pointer mutation or data deletion. | New shared backend contract; crash/concurrency/local, wrong-holder, stale-epoch, expired-before-reclaim, Postgres transaction/outbox, S3 immutability/versioning, and MinIO tests | None; bounded Lux slice required |

**Selection:** role composition, not one universal class. `GenerationPublisher` is a distinct
transaction protocol inside the existing orchestrator artifact-store package; this ADR authorizes
that bounded addition and forbids another general store. The Postgres operational-record commit is
the sole reader-visible generation authority for both local and S3 backends.

Publication has these mandatory crash-window semantics:

1. A crash before all staged bytes and the closed manifest verify creates no database record and is
   invisible; bounded garbage collection removes the partial staging prefix after its retention TTL.
2. A crash after complete staging but before the Postgres commit leaves an invisible orphan. A
   reconciler may verify and retry the same idempotent commit with a still-current fence, or garbage
   collect it after TTL; it may not expose the prefix directly.
3. The Postgres transaction may commit only after every manifest object and digest has been verified.
   Its generation row/pointer is the instant at which the complete generation becomes visible.
4. A crash after that transaction but before SSE delivery leaves the generation visible and correct;
   the transactional outbox eventually emits the projection. The database pointer, not SSE or an
   object-store marker, remains truth.

The Postgres fence row carries `holder_id`, monotonically increasing `lease_epoch`, and
`lease_valid_until` in the database clock. Acquisition establishes those fields only while the
immutable dispatch claim changes to running; heartbeat renews the deadline only after broker
extension succeeds. Publication always rejects an expired deadline, including the
`expiry < commit < reclaim` window. Reclaim quarantines rather than requeues the expired dispatch,
then atomically invalidates the fence, records terminal `worker_lost`, tombstones the claim, and
releases capacity. The same job/dispatch is never eligible for reacquisition. Failure to commit that
transition leaves it unavailable for reconciliation; it never falls back to requeue or unfenced
publish. These are required changes to the current protocol, whose broker-level requeue plus later
asynchronous cancellation/terminal guard is insufficient.

### Operational record and projections

| Candidate | Consumers | Verified capabilities | Missing capabilities | Contract conflicts |
| --- | --- | --- | --- | --- |
| `OperationalAuditStore` / target `OperationalRecordStore` | Live managed-pilot audit; Postgres-backed | Actor, tenant, request context, action, decision; app audit mutations fail closed | Versioned general record, read/query, global order, idempotency, generation transaction/outbox, artifact/evidence digest fields, retention | API append-only is not cryptographic authenticity; current audit schema is too narrow alone |
| `JobRepository` | Live API, worker/executor integration, restart recovery, cleanup, artifact projection, and reclaim reconciler; memory/Postgres backends | Durable mutable snapshots of request/effective request, state/progress/timestamps, bounded log tail, artifacts, run summary, errors, and recovery state | Append-only fact history, global order, record kind/version, actor and tenant on every fact, publication fencing/idempotency, evidence/manifest digests, and transactional outbox | Existing mutable snapshot and wire-projection semantics must be preserved, but overwriting fields loses the history required of the canonical record |
| `JobEventStore` | Live job events and SSE replay | Per-job monotonic sequence, memory/Postgres backends, payload snapshotting, restart replay | Global/canonical order, tenant in record, schema/version, digest binding; event publication deliberately continues if persistence fails | Cannot be canonical while failure is non-blocking; user-visible SSE must remain a projection |
| `events.EventStore` | No verified production consumer | Simple local type/time/correlation queries | Raw event ID can become a filename without validation; non-atomic/non-fsynced writes, overwrite/concurrency/import/tenant/retention gaps | Stale production-hardening guidance incorrectly names it as future JobEventStore |
| `runtime.ImmutableLedger` | Tests/examples only | Local hash chain, tamper/broken-chain tests, run queries | Hash omits entry ID/timestamp; signer is unused; startup trusts last line; no lock/fsync/crash/tenant/remote controls | Calling it authority would create a second evidence-like chain and operational ledger |

| Candidate | Security/trust boundary | Migration and rollback | CI support | Production-spike evidence |
| --- | --- | --- | --- | --- |
| `OperationalRecordStore` | Versioned bounded canonical payload; globally ordered ID; tenant/job/generation IDs; idempotency; event kind; atomic global/tenant capacity reservation; immutable attempt/dispatch ID with outbox, once-only execution claim, and terminal tombstone; current lease holder/epoch/database-clock deadline; fenced generation commit/pointer; CAS/lineage root and evidence-digest references; authorization, redaction, retention, and projection outbox. Hash chaining may detect corruption but only verified evidence signatures establish authenticity. | Evolve Postgres schema/API in place; retain `OperationalAuditStore` facade/event kind; dual-write before switching projections. Roll back code without deleting appended rows. | Factory/offline Postgres/paid-pilot suites plus migration, concurrent admission/capacity release, tenant, idempotency, dispatch redelivery/claim/tombstone, explicit retry, restart, wrong-holder/stale/expired fencing, generation visibility/outbox, and digest-divergence tests | Current audit path only; no Lux record spike |
| `JobRepository` | Request/effective-request content, log text, artifact paths, and mutable state are schema/size/redaction/path-sensitive; a snapshot is not authorization to execute raw commands or publish. | Rebuild/update it transactionally as the API/job-state projection of operational records while preserving `JobRecord` and wire shapes. Keep current direct mutations as the bounded rollback during dual-write; after cutover, projection divergence fails closed and is repairable from records. | Existing memory/Postgres/recovery/reclaim/artifact suites plus projection rebuild, transaction rollback, divergence, and compatibility tests | Live durable snapshot baseline; no canonical-record projection spike |
| `JobEventStore` | Callers authorize tenant/job; payload type/size/redaction allowlist. Memory history truncation is not durable evidence. | Project SSE/job history transactionally from committed operational records; preserve existing sequence and wire contract. Keep current store as rollback during dual-write. | Repository/Postgres/SSE contracts plus projection/rebuild/failure tests | Live SSE/job history |
| `events.EventStore` | Unsafe raw local JSON/path boundary | Freeze; add a one-release converter only if persisted consumers are discovered; then remove implementation and isolated tests. | Unit-only suite | None |
| `runtime.ImmutableLedger` | Local hash chain is unauthenticated and lacks tenant isolation | Freeze; transfer tamper-test ideas; remove with row-6 facade unless real ledgers are discovered and need a read-only converter. | Runtime ledger tests only | None |

**Selection:** adapt the live audit-store ownership into one canonical `OperationalRecordStore` and
retain role-specific projections. `JobRepository` stays the durable mutable job/API snapshot;
`JobEventStore` stays the ordered SSE/job-history projection; the audit API stays the paid-pilot
control-plane view. Artifact catalogs, manifests, run cards, and dashboards are also projections of
committed facts. None is canonical history, generation visibility, or evidence.

The audit-store family is the seed because it is already append-only and carries actor/tenant
context; the mutable `JobRepository` API is intentionally preserved as a snapshot projection. Both
need schema and transaction work, but promoting mutable field overwrites would discard the canonical
fact history this plane exists to provide.

---

## Implementation Dispositions

| Row | Disposition | Test and coverage disposition | Exit criterion |
| --- | --- | --- | --- |
| 1 | **Adapt + designate target** | Keep floors; add canonical-plan, single-cache, rehash/tamper, process, cancel, resource, identity completeness, output-byte verification, security, Lux parity, and Spatial parity coverage | Default production only after the applicable vertical-slice gate |
| 2 | **Keep domain stages/cache index; adapt consumer; remove duplicate executor/plan after parity** | Preserve public Spatial/golden/cache-safety tests; move executor behavior to row 1 before deletion | Spatial adapter compiles canonical plan and cache payloads reference CAS |
| 3 | **Freeze scheduler/router, then remove; retain only independently consumed patcher/nodes** | Migrate visualization/node contract tests to actual owners; delete scheduler-only tests with implementation | No mounted/runtime consumer and replacement tests prove retained behavior |
| 4 | **Remove** | Delete mocked Ray/local-wrapper suite with implementation | Local authority is active; any future distribution requires a new ADR |
| 5 | **Keep templates/export; freeze executor as non-authoritative** | Keep domain template and unsupported-node tests; add a canonical stage adapter only if production use is requested | Never exposed as general executor |
| 6 | **Harvest process/GPU/sandbox strategies; freeze then remove competing engine/scheduler facade** | Transfer behavior and harden tests under row 1; keep manifest tests only if manifest remains a projection | Row 1 owns execution; no facade consumers remain |
| 7 | **Adopt canonical publication first; then adapt execution by vertical slices; retain as executor rollback until cutover** | Keep all Lux contracts/evidence tests; add unchanged-row publisher parity; delete only duplicate scheduling tests after default cutover and rollback window | Row 7 publishes canonically before row-1 activation; canonical executor later becomes default and row 7 has a documented removal release |
| 8 | **Adopt canonical publication first; then adapt both public Spatial entrypoints by vertical slices; retain each as executor rollback until cutover** | Keep single-view public import/config/result, preset, resource/error, cache-safety, and golden-output tests plus multi-view camera/tier/license/request/result/PLY/sidecar/provenance tests; add unchanged-row publisher parity; delete duplicate imperative scheduling tests only after each cutover/window | Both row-8 paths publish canonically before activation; each canonical slice later becomes default and each imperative loop has a documented ADR-045 removal release |
| J | **Keep + designate the split job-boundary composition; replace process-local admission with record-store admission/outbox/claim transactions** | Keep admission/queue/repository/worker/cancel/reclaim suites; add multi-host quota, dispatch redelivery/claim, terminal tombstone, explicit-retry admission, locator-only, raw-command rejection, fencing, projection, and external-worker tests | Exactly one named owner for admission, dispatch, execution claim, queue, lease lifecycle, durable snapshot, terminal reclaim, explicit retry, capacity release, and publication |

ADR-045 remains binding for every extraction or removal: register the seam in
`MONOLITH_DECOMPOSITION_TARGETS.md` first, land module and tests before caller switch, preserve
compatibility re-exports and lazy imports, run its governance gates, and update the target row.

---

## Security and Trust-Boundary Requirements

These requirements are activation blockers, not future suggestions:

1. Only a versioned, closed-world, schema-validated `tp.execution.plan.v1` payload may cross an
   execution-process boundary. Unknown fields are rejected. Before parsing or allocation, enforce
   plan-body bytes, JSON nesting depth, string length, node/edge/fanout, input/artifact count, decoded
   pixel, and decompression-ratio limits. It contains stable stage-registry identifiers, never Python
   module/class names or executable fields. A broker carries only the locator in rule 3, not the plan.
2. Model selectors, revisions, licensing acknowledgements, and authoritative contracts preserve the
   #2081 fail-closed behavior at every execution-capable boundary. Schema validity alone grants no
   model or execution authority.
3. The queued request is locator-only: job ID, immutable attempt/dispatch IDs, immutable plan digest,
   API/schema version, and tenant ID. Each acquisition returns a scheduling lease; before spawn,
   `OperationalRecordStore.claim_dispatch` must atomically claim that exact dispatch once and bind
   holder ID plus a monotonically increasing epoch. The matching Postgres fence row adds
   database-clock `lease_valid_until`. The pre-acquisition request never stores those lease fields.
   Raw executable, module, shell, or `argv` fields are forbidden in broker payloads and authoritative
   job state even when signed. The worker fetches the digest-addressed plan, validates it, and
   reconstructs a fixed in-repo entrypoint plus allowlisted stage IDs. Tenant, actor, API version,
   plan schema/digest, job identity, holder, epoch, and unexpired deadline must match before spawn and
   publication. Duplicate delivery or a terminal/tombstoned dispatch fails before execution.
   Locator jobs use a separate versioned/ACL-scoped queue that legacy workers cannot consume; every
   raw-command queue and worker is drained and retired before canonical producers run.
4. Stage registration is a static in-repo allowlist. External plugins, Comfy workflows, and remote
   workers do not gain arbitrary-code authority through this decision.
5. Workspace, CAS, input, and output roots are tenant/job scoped; symlink and traversal checks happen
   after resolution. Materialization outside the scoped root fails closed.
6. CPU, GPU, memory, disk, output-size, stage-count, wall-time, per-tenant/global queued/running
   concurrency, and event/log amplification limits are explicit and bounded. Production rejects a
   disabled/unbounded tenant cap. Admission uses one Postgres transaction to reserve capacity and
   write the dispatch outbox; a process-local count/lock is not production authority. Process spawn
   is fault isolation, not an OS sandbox; network is denied by default through an enforced mechanism
   before untrusted workloads are considered.
7. Production cache reads and writes require complete identity v3 and one cache path. Placeholder
   code, lock, model, dependency, weight, or executed-path identity makes the operation non-cacheable
   and emits a bounded diagnostic. The row-1 adapter disables/removes the Stage-local JSON/NPY cache,
   validates a closed cache manifest, confines every reference, and rehashes each CAS object before
   deserialization.
8. A failed or concurrent publication exposes either the old committed generation or the complete
   new one, never a mixture. The Postgres generation record/pointer is the sole visibility authority;
   local/S3 prefixes and markers are never reader-visible authority. The same transaction requires
   the current tenant/job, holder, epoch, running state, and an unexpired database-clock deadline,
   then writes the record, pointer, and outbox. A stale or expired worker cannot publish, including
   before reclaim runs. Terminal events are emitted only from the committed outbox.
9. Operational records enforce payload schemas, size limits, redaction, retention, idempotency, and
   tenant/job authorization. They may reference evidence by digest but cannot repair, overwrite, or
   reinterpret evidence bytes.
10. Operational CAS roots, Phase 3/4 roots, Lux V1 commitments, Lux V2 CT roots, and attestation
    digests remain distinct hash domains. Any claimed binding mismatch fails closed.

---

## Conflict Dispositions

| Standing decision | Resolution |
| --- | --- |
| ADR-029 | Partially superseded only for long-term stage-executor authority. Spatial public APIs, presets, stage adapters, cache behavior, and golden outputs remain migration constraints. ADR-029 is amended in this PR and Spatial becomes an adapt-consumer. |
| ADR-043 | Its five completed decomposition seams and historical `COMPLETE` status stand. The 2026-03 finding that depth/materials logic “must stay” described the then-current extraction boundary, not a permanent ban on a contract-preserving shared-executor migration. ADR-043 is amended in this PR. |
| ADR-045 | No conflict and no supersession. Its eight-step extraction pattern is binding for every disposition PR. The approximately 200 LOC/quarter ratchet belongs to the Q2 roadmap, not ADR-045. |
| PR #2070 / issue #2065 | Adopt the landed `ResolvedInvocation` semantics as the canonical plan seed. Its current v1 remains provisional until `tp.execution.plan.v1`, compatibility parsing, manifest fields, and documented-workflow parity land. Issue #2065 remains open. |
| PR #2081 / issue #2066 | Preserve the commercial-safe `da3_metric` default, deprecated `da3` research alias, explicit selector provenance, pinned revision, and fail-closed license/model-contract behavior. No representation migration may weaken it. |
| Production Hardening Gap section 4 | Refresh the stale `events.EventStore` recommendation in this PR. Live `JobRepository`, `JobEventStore`, audit/record store, job-delivery `ArtifactStore`, operational CAS, and evidence are distinct roles. |

No additional scheduler, ledger, artifact-store, or plan abstraction was found to have landed during
the proposal window beyond the already-recorded `ResolvedInvocation` work.

---

## Binding Rules

1. No new scheduler, ledger, general artifact store, or plan abstraction may merge outside the
   authorities and migrations named here. A capability added to a designated family must reuse its
   contracts and tests.
2. Plan and run use one resolution path. In-process execution consumes the same object; across CLI or
   execution-process boundaries, “same” means a byte-identical canonical payload plus equal identity,
   followed by schema, registry, model-contract, license, tenant, and authorization revalidation. The
   broker carries only its immutable digest and the locator fields in security rule 3.
3. The stage executor never owns admission or queue/lease retry. The app owns authenticated
   admission-policy/preflight coordination plus idempotent outbox delivery; `OperationalRecordStore`
   owns atomic capacity reservation, dispatch-outbox records, terminal capacity release, and fenced
   publication; `QueueBroker` owns queue/lease primitives; `WorkerRunner` owns
   polling/heartbeat/lease-lifecycle delegation; `JobRepository` owns the mutable durable snapshot;
   the app reconciler makes reclaimed work terminal. Outbox delivery retry reuses one immutable
   dispatch ID and cannot execute after its durable claim or tombstone; a user/operator retry is a
   newly admitted job ID/attempt ID with a new capacity slot. None owns stage topology except the stage
   executor, and every retry has exactly one owner at its boundary.
4. A frozen implementation gets no new features. Its tests move to the designated owner or are
   deleted with it; coverage floors move with the behavior they protect.
5. Structural manifest/run-card/artifact-path writes migrate behind designated producers. Extend the
   existing raw-JSON governance pattern rather than add another checker.
6. Ingest, machine-mode, evidence/attestation, run-card, preset, route, selector, and CLI contracts do
   not change implicitly. Authorities produce those contracts; they do not redefine them.
7. Staged bytes, object-store markers, SSE delivery, and job snapshots never confer generation
   visibility. Only a fenced `OperationalRecordStore` commit may advance it; rollback appends a new
   authorized commit to a prior immutable generation.
8. A production cutover is forbidden until all activation gates below pass on the same candidate head.

---

## Migration and Activation Gates

### Phase A — contract and identity promotion

- Publish `tp.execution.plan.v1` and a read-only adapter for `tp.lux.resolved_invocation.v1`.
- Add identity v3 and prove plan/run, direct-Python, preset, worker, backend, cache, manifest, and
  run-card identity equality.
- Make incomplete production identity non-cacheable and close the scoped #2064 prerequisites.
- Keep #2067 and #2068 as separate governed product decisions; this ADR does not choose their
  deliverable behavior.

### Phase B — publication and records

- Complete #2063 atomic/durable writer primitives.
- Add local/S3 immutable staging through `GenerationPublisher` inside the existing orchestrator
  artifact-store package; do not add an object-store visibility pointer.
- Add the versioned operational-record schema, holder plus monotonically increasing lease epoch,
  database-clock `lease_valid_until`, sole Postgres generation pointer, and transactional outbox;
  project `JobRepository`, job/SSE, and audit views from committed records.
- Replace process-local count-then-enqueue with `OperationalRecordStore.admit_job`: atomically enforce
  positive production global/tenant limits, reserve capacity, allocate immutable attempt/dispatch
  IDs, append the admission fact, update the job projection, and write a Redis-dispatch outbox.
  Terminal/cancel/`worker_lost` records tombstone the dispatch and release capacity transactionally;
  crash reconciliation must neither leak nor double-release slots.
- Have the app-owned outbox projector enqueue locator records idempotently through `QueueBroker`;
  Postgres outbox state remains delivery truth. Redelivery reuses the same dispatch ID and does not
  reserve another slot; an explicit job retry calls `admit_job` for a new job ID/attempt ID and slot.
- Before spawn, atomically `claim_dispatch` once and establish holder/epoch/deadline. Renew the fence
  only after a successful broker heartbeat and check expiry on every publish. Reclaim quarantines the
  broker item, then atomically invalidates the fence, records terminal `worker_lost`, tombstones the
  claim, and releases capacity; the same dispatch is never requeued or reacquired. Coordination
  failure stops execution/transition and requires reconciliation.
- Introduce a separate versioned/ACL-scoped locator queue. Drain and retire every legacy raw-command
  queue and `argv`-capable worker before canonical producers can submit; new workers reject legacy
  payload fields permanently.
- Route unchanged row-7 and row-8 execution outputs through the canonical publisher, record, and
  projections; prove byte/digest/wire parity and rollback there before changing either executor.
- Prove every documented staging/DB/outbox crash window, including
  `enqueue succeeds -> projector crashes -> broker release -> outbox redelivery`; durable duplicate
  claim/tombstone rejection; explicit-retry admission; concurrent publisher; stale worker;
  `expiry < commit < reclaim`; wrong-holder; locator tampering; mixed-version isolation; raw-command
  rejection; path/symlink; tenant; and recovery behavior.

### Phase C — bounded production vertical slice

Run `preprocess -> depth -> output` through row 1 inside one J-composition lease under an opt-in
flag. Required evidence on one exact head:

- canonical plan and identity equality between plan and run;
- byte-for-byte or contract-approved semantic parity for existing outputs, manifests, run cards,
  selectors, routes, and CLI behavior;
- fail-closed tampered plan/model/broker/path/tenant cases;
- cancellation, timeout, holder/epoch/deadline lease loss, crash recovery, resource ceilings, and no
  double publication;
- cache miss/hit correctness with output-byte verification;
- the repaired Performance Monitor evidence gate with committed valid baselines and no regression;
- local and object-store publication contract suites;
- an explicit executor rollback drill to row 7 while retaining the canonical publisher, record,
  projection, and locator-only job path.

Only after this slice passes may the candidate become the default for those stages. Later stages move
individually under the same gate. Row 7 is removed only after the rollback window and ADR-045
disposition work complete.

Spatial follows the same gate separately for both public entrypoints. Compile the single-view config
and typed multi-view request into canonical plans; preserve public import/config/request/result,
camera validation, research-tier/license, golden output, PLY/sidecar, and provenance contracts;
verify the single-cache/CAS path; and drill executor rollback to the applicable row-8 loop while
retaining canonical publication. No Spatial stage changes default merely because the Lux slice
passes, and single-view parity does not authorize multi-view cutover.

---

## Repair Program Inputs

| Issue | Current state | Effect on this decision |
| --- | --- | --- |
| #2062 performance false-green | Closed by #2071 | Repaired evidence gate is admissible; it does not itself benchmark a candidate executor. |
| #2063 atomic/durable evidence writes | Open | Prerequisite to generation publication. |
| #2064 depth cache identity | Open | First production identity-v3/cache derivation migration. |
| #2065 `ResolvedInvocation` / `--plan` | Open, partially implemented by #2070/#2081 | Canonical plan seed; promotion, workflow parity, and manifest work remain. |
| #2066 DA3 default | Closed by #2081 | Landed contract must be preserved. |
| #2067 emit flags | Open governed product decision | Not decided by this ADR. |
| #2068 16-bit flags | Open governed product decision | Not decided by this ADR. |

---

## Consequences

### Positive

- One target executor, plan schema, execution identity, operational lineage family, job boundary, and
  canonical operational-record owner.
- Existing security, license, evidence, and public contracts become migration gates instead of being
  silently replaced.
- Useful process/GPU/cache/test behavior is harvested without promoting every dormant facade.
- Generation publication gains an owner without inventing a fourth general artifact store.

### Negative

- Acceptance does not immediately remove code. During migration the current Lux and imperative
  Spatial loops remain their executor rollback paths behind canonical publication, so temporary dual
  implementation is explicit and time-bounded.
- Identity and plan schema promotion require compatibility readers and cache namespace invalidation.
- Local and object-store staging need different durability mechanics before the same fenced Postgres
  visibility commit.

### Risks and mitigations

- **Target is production-dead today:** require the exact-head Phase C slice before activation.
- **Arbitrary code through plans or broker payloads:** locator-only dispatch, a fixed in-repo worker
  entrypoint, and allowlisted typed stages; no executable/module/shell/`argv` authority from stored or
  untrusted data, signed or otherwise.
- **Evidence-domain conflation:** preserve contract-native roots and independent verification.
- **Permanent shadow paths:** freeze rules, exit criteria, test/floor ownership, and ADR-045 removals
  make retention visible and reviewable.

The V4 go/no-go condition “designation selected real shared infrastructure” is met only by an
Accepted ADR plus at least one passing production vertical slice. ADR acceptance alone is not
activation evidence.

---

## Alternatives Considered

- **Keep per-package authorities:** rejected because it preserves incompatible identities, stores,
  schedulers, and green-but-production-dead suites.
- **Create a new V4 kernel:** rejected because it would be a ninth stage executor and another plan,
  ledger, and store family.
- **Designate row 2:** rejected as the shared authority because it is Spatial-specific and lacks the
  complete identity/process/job composition; its strong graph/cache tests are migrated.
- **Designate row 6 wholesale:** rejected because the facade lacks a DAG compiler, canonical identity,
  and production consumer. Its process/GPU/sandbox strategies are instead hardened under row 1.
- **Keep the Lux loop permanently:** rejected because it cannot become shared authority without
  retaining bespoke state, identity, and direct-write paths.
- **One universal artifact store or ledger:** rejected because immutable bytes, cache metadata,
  delivery, publication, operational projections, audit, and cryptographic evidence have different
  trust and lifecycle contracts.

---

## Acceptance and Sign-off

Acceptance uses two exact-head records so the status-changing commit cannot invalidate the design
approval that authorized it:

1. Publish a design-complete **Proposed** head and require its hosted checks to pass. The repository
   owner then records one PR comment that names that SHA and gives both role-qualified approvals:
   Architect approval of authority/conflict decisions and Specialist approval of migration,
   rollback, test-transfer, and coverage-floor dispositions.
2. After that comment only, create one acceptance-only commit. It may change ADR status/checklist and
   the architecture index/documentation-map statuses, and may cite the Proposed-head approval; it
   may not change a designation, requirement, matrix, disposition, gate, or conflict decision.
3. After hosted checks pass on the Accepted head, the owner records a second PR comment naming that
   final SHA and confirming the acceptance-only diff did not alter the approved design. No commit may
   follow that confirmation. Only then may PR #2069 leave Draft and merge.

The design-complete checklist is:

- [x] Every authority row is decided; all credible candidates and inventory rows 1–8 plus J have the
      eight-criterion evidence and an explicit disposition.
- [x] ADR-029 and ADR-043 amendments, ADR-045 preservation, the resolved conflict table, final
      `Supersedes` field, and Production Hardening Gap section-4 refresh are in this PR.
- [x] No decision markers or verification placeholders remain; missing capabilities are explicit
      activation prerequisites, and current open issues are not described as complete.
- [ ] Repository owner has recorded role-qualified Architect and Specialist approval of the exact
      Proposed head. The acceptance-only commit must replace this line with a checked item citing the
      approval comment and SHA.

Independent execution, storage/security, and governance audits inform these matrices but do not
replace owner approval. Because the PR author cannot self-approve a GitHub review, the two exact-head
PR comments are the durable sign-off mechanism. The first must state both:

- As Repository Architect, approval of the authority designations and conflict dispositions.
- As Specialist, approval of the migration, rollback, test-transfer, and coverage-floor dispositions.

The second must ratify the exact Accepted head and confirm that the only intervening commit was the
acceptance-only metadata/navigation diff. That final-head confirmation remains external because
editing this file to check another box would create a new, unconfirmed head.

Until both records exist in sequence and final-head checks pass, PR #2069 remains Draft and no merge
is allowed.
