# Architecture Remediation — 2026-09-12

**Classification:** Dated implementation and local validation evidence.
**Baseline:** `b4ce44942a7e611479a79b96bb07ccd8260a9fa3` (`origin/main`).
**Prior repair:** `1185040d3`, the first boundary-audit fixes.
**Branch:** `codex/architecture-integrity-audit-20260913`.

This follow-up addresses the remaining priorities in the
[initial audit](ARCHITECTURE_INTEGRITY_AUDIT_2026-09-12.md). It preserves the
native Lux and archive executors and does not activate the separately gated
StageGraph/CAS executor convergence. Implementation is local; hosted CI and
deployment acceptance require the published commit to pass their own gates.

## Implemented boundaries

```mermaid
flowchart LR
    Browser --> Frontdoor[Verified actor and request signature]
    Frontdoor --> API[Tenant policy and canonical plan]
    API --> DB[Postgres admission and immutable dispatch]
    DB --> Outbox[Transactional outbox]
    Outbox --> Redis[Versioned locator queue]
    Redis --> Worker[Lease holder and epoch]
    Worker --> Native[Existing Lux or archive executor]
    Native --> Staging[Immutable candidate generation]
    Staging --> Fence[Database-clock publication fence]
    Fence --> Pointer[Committed generation pointer]
    Pointer --> Delivery[Authorized artifact delivery]
```

| Priority | Result | Contract and operator guidance |
| --- | --- | --- |
| Tenant identity | The frontdoor signs its verified actor after Access/session/CSRF checks. The backend derives tenant membership from configured actor mappings, rejects unsigned/conflicting selectors, and confines data paths before preview and again at worker dispatch. Shared model/runtime selectors retain explicit trusted roots and provenance checks. | [Paid-pilot identity configuration](../../deployment/paid_pilot_services.md#pilot-tenant-admission-and-audit-mode), [assertion verifier](../../../src/transformation_portal/core/security/frontdoor_identity.py). |
| Distributed admission | One Postgres transaction reserves global/tenant capacity and stores the canonical plan, immutable tenant/attempt/dispatch identity, queued job, and outbox. Conflicting limits fail closed. Redis carries only a closed versioned locator. | [Distributed dispatch runbook](../../runtimes/orchestrator-postgres.md#distributed-dispatch-and-generation-authority). |
| Publication fencing | A database-clock lease, unique holder and epoch authorize the committed generation pointer. Conditional object creation and verified hashes precede that transaction. Duplicate delivery, cancellation, expiry and late publication cannot revive a terminal attempt or release capacity twice. | [Generation publisher](../../../src/transformation_portal/orchestrator/artifact_store/generation.py), [operational store](../../../src/transformation_portal/orchestrator/storage/operational.py). |
| Canonical execution | Lux consumes its existing prepared plan. All ten existing archive commands have closed typed configurations, input-file digests/private snapshots, and attempt-relative outputs. Queue data cannot choose an executable or arbitrary argv. | [Execution Plan V1 extension](../../reference/EXECUTION_PLAN_V1.md#distributed-archive-dispatch-extension). |
| Event retention | Transactional monotonic bigint cursors survive pruning; committed appends bound per-job replay history. Postgres SSE polls across hosts and observes committed terminal events, with an explicit replay-gap comment. | [Bounded event replay](../../deployment/paid_pilot_services.md#bounded-event-replay). |
| Security enforcement | Live `main` protection requires strict `CI Gate` and `Dependency Security`, both bound to GitHub Actions app ID 15368. The security workflow has read-only token permissions and a blocking checkpoint-loading source-policy scan. | [Required-check policy](../../ci/BRANCH_PROTECTION_SETUP.md). |
| Application typing | The fifteen existing `app.py` errors are corrected with precise annotations and narrowing. App, dispatch and security boundaries join the mirrored blocking mypy whitelist. | [Typing policy](../../ci/TYPE_CHECKING_POLICY.md). |

API route shapes, response envelopes, browser selectors and native CLI flags are
preserved. Enabling tenant mode intentionally requires signed identity on both
services. Enabling Redis dispatch intentionally requires the Postgres authority,
positive matching admission limits, a migrated database and a new queue
namespace. Stop and drain old producers/workers before cutover; old raw-command
messages cannot authorize the new worker. The memory runtime remains supported.

Migrations `0003_dispatch_authority`, `0004_bounded_event_replay`,
`0005_generation_cleanup` and `0006_periodic_generation_scrub` must be applied
together before using the new runtime.
Terminal dispatch tombstones and operational evidence remain durable; bounded
replay is a separate recovery surface. Artifact deletion revokes visibility
before storage cleanup. Failed cleanup remains retryable and cannot delete a
committed generation after an uncertain database response.
Descriptor-anchored source reads and cleanup reject substituted ancestors.
The final review reproduced output redirection before native execution, which
publication checks alone could not prevent. Both native executors now write
inside protected execution storage, then export bounded regular files through
pinned directory descriptors. `TP_ORCHESTRATOR_EXECUTION_ROOT` must identify
the same protected, service-owned shared directory on every distributed host.
Its deterministic per-attempt locator lets terminal cleanup and periodic scrub
remove private outputs after abrupt process exit or delayed file recreation.
Cancellation or terminal publication during startup also prevents child spawn.
A bounded, indexed hourly scrub revisits terminal private directories, including
files left after a delayed child exit. Worker hosts still need a supervisor
that reaps child process groups when the worker dies.

## Accelerate alerts: mitigation, not an upstream fix

Live alerts [425](https://github.com/RC219805/Transformation_Portal/security/dependabot/425)
and [428](https://github.com/RC219805/Transformation_Portal/security/dependabot/428)
both concern [GHSA-4j2p-28q2-5m79](https://github.com/advisories/GHSA-4j2p-28q2-5m79),
in the Darwin arm64 ML input and compiled lock. Both remain open. The reviewed
advisory lists no patched version, and inspecting the available 1.15.0 loader
did not establish a fix. No speculative lock update or alert dismissal was made.

The supported application paths do not call the two vulnerable Accelerate
checkpoint-loading entrypoints. Reviewed pinned Transformers 5.10.4 and
Diffusers 0.40.0 source also had no such call references. The new AST gate
rejects direct calls and statically resolvable aliases/imports to those APIs;
it is a source-policy guard, not a Python sandbox. Existing device mapping and
offload behavior remain available.

The shared checkpoint-index parser rejects duplicate JSON keys, non-finite
numbers, oversized maps, escaping/ambiguous shard paths and malformed indexes.
Owned HF and DA3 boundaries require contained regular files and bounded index
and shard inventories. FIFO rejection occurs before blocking reads. HF cache
blob links remain supported only inside their validated owned root. These
controls reduce repository exposure; they do not patch arbitrary external
Accelerate callers or make a concurrently hostile model cache trusted.
See [the current loading policy](../../../SECURITY.md).

## Representative production-image measurements

Three real 16-bit TIFFs were processed with commercial-safe DA3 Metric, premium
quality, CPU execution and 16-bit enhanced output. No fallback or cache hit was
used. The model was `depth-anything/DA3METRIC-LARGE`, Apache-2.0 revision
`4010e39f3634a45bc60553321fb49fb760bd594e`. The host was macOS 26.5.1 arm64;
the application used Python 3.12.13 and the existing isolated DA3 Python 3.11
runtime. Model weights were already cached, with network model access disabled.

| Input | Dimensions | File bytes |
| --- | --- | ---: |
| `V6750Picacho_Aerial.tiff` | 3600 × 6000, RGB uint16 | 123,809,195 |
| `V6750Picacho_Kitchen.tiff` | 3375 × 6000, RGB uint16 | 118,234,877 |
| `V6750Picacho_Pool.tiff` | 3375 × 6000, RGB uint16 | 117,957,893 |

The first comparison exposed existing nondeterminism in upstream DA3 sky
filling: two images changed despite identical plans and input bytes. The
repository-owned, one-shot DA3 worker now derives an image-content seed using
domain `tp.da3.rgb-seed.v1` and seeds Python, NumPy and Torch before inference.
The application process's random state is unchanged. Worker source participates
in runtime identity, preventing old cache authority from aliasing the new
policy. The seed does not promise bitwise agreement across different devices,
library builds or models.

After this fix, a fresh native CLI baseline and three fresh canonical-dispatch
processes using the final private-execution/export path produced
**exactly identical decoded depth and enhanced pixels** for
all three images. Actual model inference retained DA3's native downsampling;
the delivered depth arrays matched source dimensions.

| Fresh dispatch trial | Wall seconds | Images/minute | Peak process-tree RSS bytes |
| --- | ---: | ---: | ---: |
| 1 | 55.745 | 3.229 | 5,577,506,816 |
| 2 | 52.247 | 3.445 | 5,697,929,216 |
| 3 | 52.960 | 3.399 | 5,569,413,120 |

Median wall time was **52.960 seconds**, maximum observed RSS **5.31 GiB**.
RSS sums the measured parent and descendant processes and can count shared
resident pages more than once. These are local descriptive throughput/resource
measurements, not a before/after inference speedup claim, production capacity
commitment, GPU result or hosted APEX regression verdict.

Canonical plan SHA-256:
`fe40d943b0ec641f1fd63b20d6923ef92ffc30d7cc7dc2588b9d903c55adec44`.
Plan fingerprint:
`c69932163b59bfbdae73ae0f548b62717b602c74e9e5f0b472ea7b58ab990da2`.
The local `production-ml-performance-final.json` report records input hashes,
source hashes, per-artifact
pixel hashes, RSS sample counts and each native backend outcome.

## Validation evidence

The task uses disposable services on ports 15432 (Postgres), 16379 (Redis),
and 19000 (MinIO), separate from the user's running stack. Dedicated databases
and unique queue/object prefixes keep fixture resets isolated. The evidence
directory is `/private/tmp/tp-architecture-audit-evidence` and is not committed.

Confirmed checks:

- The complete core tier passed: **13,420 passed, 88 expected skips,
  943 deselected**, in 952.65 seconds. Global coverage was **70.31%** against
  the existing 30% floor. Every package line/branch floor and the touched-file
  cold-zone policy passed. Worker coverage was 100% line/branch; storage was
  76.63%/73.36%; `app.py` was 82.19%/75.00%.
- `make validate-ci check-doc-heading-links` and `make pre-commit` passed.
- `make ci` passed lint, governance/dependency checks, fast tests, orchestrator
  contracts (1,256 passed, 35 optional-service skips), 302 frontdoor Node tests
  and the production build.
- The complete mirrored mypy whitelist passed under CI-pinned mypy 1.20.1:
  128 source files, zero errors.
- Pylint also ran against every changed file eligible under the repository's
  policy, including uncommitted additions, with no blocking diagnostics. The
  Accelerate loading-policy gate scanned 1,787 Python files with zero violations.
- The documented execution-plan/cache authority ladder passed: 465 tests.
- Focused checkpoint, HF lock/loader, DA3 identity and security-workflow
  contracts passed: 162 tests.
- Core/archive dispatch and workflow-policy regressions passed: 144 tests.
- `make test-frontdoor-contract` passed Node contracts and the production build.
- `make test-paid-pilot-services-contract` passed its Postgres, app-authority,
  Redis queue, local/S3 artifact, Redis session and integrated native execution
  lanes. Component counts were 665/7/38/163 Python tests, six Node session
  tests and two integrated tests; two component cases skipped as documented.
- `make validate-frontdoor-browser` passed managed login, Build entry, logout
  and post-logout access denial. Chrome required the approved unsandboxed
  launch after its sandboxed DevTools endpoint could not start.
- The signed-actor service test completed a real archive operation in an
  independent worker and downloaded its hash-verified artifact. Tenant B
  received 404 on tenant A's detail, events, artifact, cancel and delete
  routes; unsigned impersonation received 401 and a conflicting selector 403.

The real Postgres/Redis/MinIO authority suite passed 17 tests: eight independent
admission processes respected the global/tenant limits; duplicate claims and
expired holders could not publish; killed-worker expiry released admission;
conditional storage writes preserved immutable bytes; lost commit responses
retained visible generations; late private-file recreation was scrubbed without
deleting committed objects. Fresh-session SQL tests exercised terminal guards,
including publication that lost its lease while waiting for a projection lock.

The separate live fault experiment paused Redis and then Postgres, keeping TCP
connections open. Before repair, neither signaled cancellation within two
seconds despite a one-second lease. After adding an independent monotonic
deadline, cancellation was observed in **0.894 seconds** and **0.943 seconds**
respectively. Both services were unpaused before cleanup. Real service restarts
then preserved Redis AOF delivery and the Postgres plan/outbox/claim path.
This experiment measures worker cancellation signaling; the separate real
subprocess regressions exercise process-group termination and reaping.

Exact service-specific entrypoints are:

```bash
PYTHONPATH=src:. .venv/bin/pytest tests/orchestrator/test_dispatch_authority_services.py -q
PYTHONPATH=src:. .venv/bin/pytest tests/orchestrator/test_tenant_dispatch_services.py -q
make test-paid-pilot-services-contract
```

These require the disposable service variables documented in their fixtures and
the [paid-pilot runbook](../../deployment/paid_pilot_services.md#validation-commands).
The complete fault experiment and production-image harness are retained in the
local evidence directory with their JSON results and logs.

The first broad core run exposed a missing in-place native `_fpstate` probe and
two stale fixtures: script loading through an unsupported package namespace and
a process-group test that omitted the matching session identity. The native
probe was built with the existing repository interpreter; the fixtures were
corrected without relaxing their contracts. The first coverage report also
exposed insufficient offline coverage for the new storage and worker paths.
Added transaction, cancellation, lease and cleanup failure tests restore the
existing floors; no coverage exclusions or thresholds were weakened.

The core validation command is:

```bash
PYTHONPATH=src:. .venv/bin/pytest -q tests/ -ra \
  -m '(unit or security or regression or golden or integration) and not ml and not slow and not benchmark' \
  --cov=src/transformation_portal --cov=src/tp --cov=lux_depth_v3 --cov=app \
  --cov-branch --cov-report=term \
  --cov-report=xml:/private/tmp/tp-architecture-audit-evidence/remediation-coverage-final.xml \
  --cov-fail-under=30 --cov-config=pyproject.toml
```

## Remaining limits

- The two upstream Accelerate alerts stay open until a verified patched
  dependency can replace the affected supported lock through its native lane.
- Local real-service interruption/restart evidence does not establish managed
  provider promotion, cross-region failover, backup recovery or fleet behavior.
- The CPU production-image sample does not cover GPU/Metal, APEX quality gates,
  materials, reconstruction, very large batches or fleet throughput.
- No deployment, executor-convergence activation, hosted exact-head CI result,
  PR publication or merge is implied by local validation.

## Publication review follow-up

The fresh PR branch `codex/tenant-dispatch-integrity-20260913` starts from the
same clean `origin/main` baseline. Its initial staged tree matched implementation
`563484139b5c92e97d39053a2c8f1bd4d44e849d` exactly. Independent publication review
then reproduced a live SSE stream that kept heartbeating after retention pruned
its terminal event. The follow-up refreshes committed terminal state and drains
retained events before closure, including a terminal commit between replay and
state reads. Authoritative job deletion closes the stream without synthesizing
a terminal event; repository read failures retain retry behavior. All five new
regression cases failed before their repairs. The final replay file passed
25 tests on real Postgres; 425 related runtime/identity tests and app typing
also passed.

The final branch receives canonical CI, typing, and focused replay validation.
The full core/coverage, service-fault, and production-image measurements above
precede this narrow SSE correction and are not reruns of the final PR commit.
Formatter changes in three migrations preserve their Python ASTs and SQL
strings. Hosted acceptance and deployment remain separate gates.
