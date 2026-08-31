# Repository Closeout Plan — 2026-08-31

**Purpose:** Verified execution plan for closing the complete open GitHub
backlog: issues #1814, #2063, #2064, #2065, #2067, #2068.

**Status:** Proposed execution plan. Dispositions for the two governed
decision issues (#2067, #2068) are proposals until the owner records them on
the issues themselves.

**Verified against:** `main` @ `3e494a8ed022b793c8eaf7c426689bdf6b1a8ca2`
(2026-08-31). Every file/line citation below was re-checked at this head; the
repair-program issues were filed against the older `c6d620a`, and drift is
called out where it exists.

**Relationship to standing authority:** This plan sequences work; it does not
override `docs/architecture/ADR-051-execution-artifact-authority-designation.md`
(Accepted 2026-08-28), the issues' own acceptance criteria, or CLAUDE.md
governance. Where this plan and those authorities could be read differently,
the deviations section near the end records the resolution explicitly.

---

## 1. Baseline and scope

- Protected `main` head `3e494a8` requires the **CI Gate** status check.
- **Six open issues, zero open PRs** (verified 2026-08-31). No open PR
  represents any of the remaining work.
- The five Lux issues are the open remainder of the "Repair, Designate,
  Prove" program (2026-08-28). That program exists **only as GitHub issues
  #2062–#2068 plus the "Repair Program Inputs" table in ADR-051** — there is
  no program document under `docs/`. Program state at this head: #2062
  (repair 1.6-a) closed by PR #2071; #2066 (repair 1.2) closed by PR #2081;
  #2063 (1.5-a), #2064 (1.4-a), #2065 (P0-1), #2067 (1.3-a), #2068 (1.3-b)
  open. Repairs 1.1-a, 1.1-b, 1.3-c, 1.4-b, and 1.5-b have **no dedicated
  issues** — they exist only as references inside sibling issue bodies, and
  their disposition is addressed in this plan.
- **GitHub is the authoritative issue tracker.** A connected-GitLab search
  found no matching project (one broader retry hit an upstream 502). If a
  GitLab mirror exists under another namespace it should be configured as a
  mirror, not allowed to grow a second divergent backlog. No repo-side change
  is needed for this.

## 2. Executive assessment

| Issue | Reality at `3e494a8` | Disposition |
| --- | --- | --- |
| #1814 GPU Docker runtime | Repo-side contract work green (`tests/validation/test_dockerfile_contract.py`, **6 tests** — the issue's "4 passed" comment predates two added tests). Closure blocked solely on NVIDIA-host evidence. | Run the exact validation matrix on an NVIDIA host, patch only if Python is prerelease or CUDA fails, paste evidence on the issue, close. |
| #2063 durable evidence writes | Unimplemented. `io_atomic.py` still probes the process-wide umask and has zero fsync; all four governed evidence writers truncate in place. Byte-identical to `c6d620a`. | Implement one durable atomic-write primitive (JSON-agnostic, bytes-in) and migrate all four writers. |
| #2064 depth-cache identity | Unimplemented in `depth_cache.py` (bare `np.load`, no sidecar, no checksum). One drift: post-#2070/#2081 the fingerprint's model identity on the contract-carrying path already embeds `canonical_key:repo_id@locked-revision`. | Rebuild cache entries around verified identity metadata on #2063's primitives; close on the issue's scoped criteria with the ADR-051 identity-v3 residue recorded explicitly. |
| #2065 ResolvedInvocation / `--plan` | **Partially implemented — do not reimplement.** PR #2070 landed the frozen single-resolution `ResolvedInvocation` (`tp.lux.resolved_invocation.v1`, explicitly `stability: provisional`), resolver-only `--plan`, and single-resolution consumption in `ConfigResolver`/DA3 backend. Outstanding: schema promotion to `tp.execution.plan.v1`, manifest plan fields (none exist yet), documented-workflow parity gate (no fixture exists yet). | Preserve the landed design; finish the residual contract work in two bounded PRs (#2065-A, #2065-B). |
| #2067 emit_marketing / emit_report | Decision not recorded. Both flags inert: no marketing writer exists; the combined report is unconditional (`orchestrator.py:3498`). Only `--plan` warns, and only for `emit_marketing`. | **Propose:** deprecate both flags; remove the fictional marketing deliverable from docs; document the combined report as unconditional. Owner records the disposition on the issue before implementation. |
| #2068 emit_master16 / emit_upscaled16 | Decision not recorded. Every behavior-gating read is the joint OR; no `master16/`/`upscaled16/` outputs exist; plan layer collapses both into one `bit_depth_16_intermediates` request and warns only when **both** are set. | **Propose:** Option B — one canonical `--output-bit-depth {8,16}` with deprecation shims for both legacy flags. Owner records the disposition (including the flag name) on the issue before implementation. |

## 3. Dependency and merge order

```
#2067 ─────────────┐
                   ├──> #2065-B: artifact accounting, docs truth, workflow parity
#2068 ─────────────┘
#2063 ────────────────> #2064: verified depth-cache entries (HARD edge: #2064's
                        sidecar protocol is defined in terms of the 1.5-a
                        atomic primitives — see issue #2064 Scope)
#2065-A ──────────────> #2064: identity projection source
        └─────────────> #2065-B: final contract closeout
#1814 runs independently whenever NVIDIA infrastructure is available
```

Merge sequence:

1. **#2067** decision record + implementation PR (deprecations).
2. **#2068** decision record + implementation PR (`--output-bit-depth`).
3. **#2063** durable evidence writes.
4. **#2065-A** plan-contract promotion (`tp.execution.plan.v1`), no executor
   activation.
5. **#2064** identity-verified depth cache on the #2063 + #2065-A
   foundations.
6. **#2065-B** manifest accounting, documentation truth, workflow parity
   gate; closes #2065 and discharges repairs 1.1-a and 1.3-c (record that on
   the issue at closure).
7. **#1814** in parallel on NVIDIA hardware at any point.

This ordering minimizes conflicts in `config.py`, `resolved_invocation.py`,
`manifest.py`, `config_resolver.py`, schemas, and the orchestrator, and it is
compatible with ADR-051's Phase A/B bundling (Phase B's first bullet is
"Complete #2063 atomic/durable writer primitives"; #2067/#2068 stay separate
governed product decisions per ADR-051).

---

## 4. Issue #2067 — deprecate `emit_marketing` and `emit_report`

### Decision to record on the issue (proposal)

- `emit_marketing`: **deprecate.** Do not implement an undefined marketing
  transformation to preserve a dead flag. (The issue offers implement-vs-
  deprecate with no recommendation on record for this flag — the owner call
  is required.)
- `emit_report`: **deprecate.** The combined report/manifest is governed
  evidence and remains unconditional; an off-state must not suppress it.
  (Matches the issue's own recommendation on record.)

### Current state (verified)

- Fields at `config.py:459-460`; CLI options at `__main__.py:516-525`.
- `emit_marketing` has exactly one consumer: the plan-warning at
  `resolved_invocation.py:238`. `emit_report` has zero readers; the combined
  manifest is written unconditionally (`orchestrator.py:3498`, filename built
  at `:3573`/`:4786`).
- `--plan` already excludes a marketing artifact from requested artifacts and
  lists `combined_manifest_json` unconditionally
  (`resolved_invocation.py:204-230`). Only `emit_marketing` gets a plan
  warning; the comment at `resolved_invocation.py:205-207` overstates by
  claiming both flags warn — fix the comment or the behavior in this PR.
- **No `DeprecatedOutputFlagWarning` exists.** Existing patterns to mirror:
  `DeprecatedModelSelectorWarning` (`model_resolution.py:42`, a
  `FutureWarning`, visible by default) and `compat/decorators.py`.

### Implementation PR — `fix(lux): deprecate inert marketing and report output flags`

1. Add a visible-by-default `DeprecatedOutputFlagWarning` (mirror the
   `FutureWarning` pattern of `DeprecatedModelSelectorWarning`).
2. Preserve both legacy CLI/config fields for one documented deprecation
   window; mark them deprecated in CLI help rather than hiding them.
3. Emit the warning on every path: CLI argument processing, programmatic
   `EnhanceConfig` construction/deserialization, and `--plan` (extend the
   existing plan-warning strings; add the missing `emit_report` warning).
4. Keep producing the combined report regardless of `emit_report`.
5. Documentation truth — the full inventory is larger than the issue lists:
   - `docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md:382,385` (conditional phrasing) plus
     flag docs/examples at 69-70, 129-130, 178-179, 272-273, 471-472, 489,
     506, 529;
   - `src/transformation_portal/lux_depth_v3/README.md:157,458-459` and flag
     examples at 150;
   - `docs/cli/CLI_REFERENCE.md:122-123`;
   - `docs/guides/LUX_DEPTH_V3_TROUBLESHOOTING.md:281,559`;
   - `scripts/pipelines/run_montecito_apex_full.sh:97,125` (prints a
     `marketing/` directory that is never created) and
     `run_montecito_apex_lean.sh:57`;
   - CLI docstring examples in `__main__.py:24-25,62-63,82-84`.
6. Portal blast radius (not in the issue, required for a truthful closeout):
   `app.py` allowed-args list (`app.py:5304-5311`), pipeline presets
   (`app.py:958-1046`, defaults at `:2010-2014`), generated CLI lines
   (`app.py:8621-8636`); browser toggles in
   `web/secure-landing/portal-src/portal.template.js:343-344` (rebuild
   `public/portal-assets/portal.js` via `npm run build:portal`;
   `make check-portal-asset-budgets` must pass).
7. Removal target: tie to the next major release or one formal deprecation
   window; use governed markers (`# TODO(#2067): ...`) — bare
   `TODO: remove` fails `check-todo-governance`.

### Required tests

- `emit_marketing=True` warns and produces no marketing artifact.
- `emit_report=True` warns and produces the normal report;
  `emit_report=False` warns and **still** produces the report.
- CLI and direct-Python paths produce equivalent warnings.
- `--plan` lists the unconditional combined manifest exactly once and warns
  for both flags.
- Docs contain no marketing deliverable or conditional-report claim
  (extend `tests/test_lux_depth_v3_doc_sync.py`-style checks).
- Update touched suites: `tests/lux_depth_v3/test_resolved_invocation.py`,
  `tests/validation/test_portal_smoke_scripts.py`,
  `tests/test_app_orchestrator_runtime.py`.

### Closure gate

Merge only when the issue contains the owner-recorded disposition and the PR
uses `Closes #2067`.

---

## 5. Issue #2068 — replace the two 16-bit aliases with one truthful setting

### Decision to record on the issue (proposal)

**Option B** with the name `--output-bit-depth {8,16}` (the issue offers
`--bit-depth` only as an example; the decided name must be recorded).
`--output-bit-depth` distinguishes output encoding from source bit depth and
internal precision. Note: `output_bit_depth` already exists as read-only
manifest metadata with fixed 8/16 semantics (`manifest.py:141,267`, populated
at `orchestrator.py:3298/3362/3423`) — the new option must stay consistent
with it, which is an argument for, not against, the name.

### Current state (verified)

- Every **behavior-gating** read is the joint OR (`execution_engine.py:633,850`;
  `orchestrator.py:2230,2529,3296,3413`; `resolved_invocation.py:226`). But
  the flags are recorded **individually** in non-branching surfaces: config
  fingerprint (`config_resolver.py:707-708`), Stage-A fingerprint contract
  (`artifact_manager.py:295-305` requires both field names present), manifest
  serialization (`manifest.py:322-375`), CLI parsing
  (`__main__.py:770-771,1073-1074`), and `app.py` plumbing.
- No code writes `*_master16.*`/`*_upscaled16.*` or creates `master16/` /
  `upscaled16/`; promises persist at `lux_depth_v3/README.md:155-156,456-457`
  and `run_montecito_apex_full.sh:95-96,123-124`.
- Plan layer already collapses either flag into one
  `bit_depth_16_intermediates` request (`resolved_invocation.py:226-229`) but
  warns only when **both** are set (`:240-244`) — the common single-flag case
  is currently unwarned.
- The Lux V3 stage sequence has no upscaling stage, and no deliverable is
  ever "upscaled" — but dormant machinery exists (`v2_upscaler_backend`
  forwarded to the V2 subprocess via `v2_runner.py:218-219`; an unwired
  `stage_graph/stages/upscaling.py`). The decision record should say the
  *deliverable* is fictional today, not that upscaling can never exist.
- `app.py` hard-defaults **both flags to True** (`app.py:2011-2012` and the
  argv builder at `:8606-8619`) — every portal-driven run requests both, so
  deprecation shims that warn on legacy-flag use will fire on every portal
  run unless `app.py` defaults/presets/argv-builder move to
  `--output-bit-depth 16` in the same PR.

### Implementation PR — `refactor(lux): replace legacy 16-bit emit aliases with output-bit-depth`

1. `output_bit_depth: Literal[8, 16] = 8` in config; `--output-bit-depth`
   CLI option.
2. Compatibility shims: either legacy flag maps to 16 **and warns**
   (including the single-flag case); both together warn once, consolidated.
3. Fail validation on contradictory input
   (`--output-bit-depth 8 --emit-master16`).
4. **Fingerprint interlock (decide in the PR, explicitly):** either keep the
   legacy field names in the Stage-A fingerprint and depth-cache payloads
   through the shim window (no invalidation), or accept that the collapse
   invalidates Stage-A manifest reuse and all depth-cache entries now and
   #2064 invalidates the depth cache again later. Silent breakage of
   `has_expanded_stage_a_fingerprint` is not acceptable.
5. Update plan payloads, JSON schemas, manifests/run cards,
   requested-vs-produced accounting, CLI examples, shell scripts.
6. Remove `master16/`, `upscaled16/`, `*_master16.*`, `*_upscaled16.*`
   references; also update the dead classification regexes at
   `portal/job_artifacts.py:202,234`, validation scripts
   (`scripts/validation/validate_lux_depth_v3_16bit_output.py`,
   `verify_lux_depth_v3_16bit_handoff.py`, portal live-validation scripts),
   portal UI toggles (`portal.html:993,997`, `portal.template.js` +
   `state.js` + rebuilt `portal.js`), and `app.py` surfaces.
7. No "upscaled" deliverable without a separately specified upscaling stage.

### Required test matrix

| Configuration | Expected |
| --- | --- |
| No argument | 8-bit output |
| `--output-bit-depth 8` | 8-bit output |
| `--output-bit-depth 16` | 16-bit output |
| `--emit-master16` | 16-bit + warning |
| `--emit-upscaled16` | 16-bit + warning |
| Both legacy flags | 16-bit + one consolidated warning |
| Explicit 8 + either legacy flag | validation failure |
| Explicit 16 + legacy flag | 16-bit + warning |

Tests must inspect the actual produced image encoding, not only config
values.

### Closure gate

Close #2068 once the new setting controls observable output, both aliases
warn, and no maintained documentation advertises nonexistent deliverables.

---

## 6. Issue #2063 — durable evidence-write foundation

Highest data-integrity priority, and a **hard prerequisite for #2064** (its
sidecar commit protocol is specified in terms of these primitives).

### Current state (verified)

- `io_atomic.py:35-39` umask probe (process-global, racy; called at `:91`
  and `:234`); zero fsync anywhere in the module; `os.replace` at
  `:96,117,237` with no file or directory fsync. Both branches (including
  `create_file=False` at `:104-126`) lose the rename on power loss.
- Truncating in-place writers: `BatchManifest.write` (`manifest.py:408`),
  `CombinedManifest.save` (`manifest.py:487`), run card + self-attestation
  sidecar (`orchestrator.py:6886,6889`), depth metadata JSON
  (`orchestrator.py:1982`).
- Known hazard to fix in passing: the failure cleanup at
  `orchestrator.py:6891-6903` unlinks a **completed** run card when the
  subsequent sidecar write fails.
- In-repo precedent is stronger than the issue text: `provenance.py:492-501`
  is temp + flush + **fsync** + rename (still no directory fsync; its
  `.with_suffix(".tmp")` temp name is collision-prone).
- **Reuse, do not reinvent:** durable-write primitives already exist at
  `spatial_ai/orchestration/json_io.py:37-61`, `storage/cas_store.py:79-105`
  (incl. cross-platform parent-dir fsync), `determinism/cas.py:14-31`,
  `core/_cas_helpers.py:52-60`, `ingest/sidecar.py:36-66`,
  `spatial_ai/orchestration/graph/artifact_store.py:328-365`. The Lux
  primitive should converge on these patterns (or extract a shared helper),
  not add a divergent eighth implementation.

### Implementation PR — `fix(lux): make governed evidence writes atomic and durable`

Contract: serialize once → unique same-directory temp → write → flush →
fsync(temp) → deterministic permissions → `os.replace` → fsync(directory).

Permission policy (removes the umask probe): preserve the existing
destination mode when replacing; fixed `0o644` for new files. Directory
fsync fails closed on real I/O errors; only explicitly recognized
unsupported-platform errors pass.

**Serialization routing (governance-critical):** the primitive must be
JSON-agnostic — canonical bytes/str are produced once at the already-
approved call sites (`manifest.py`, `orchestrator.py` are allowlisted in
`policy/json_raw_approved_modules.txt`; `io_atomic.py` is **not**) and passed
in. A primitive that itself calls `json.dump` fails `check-json-serialization`
until governance review allowlists it — avoid needing that. This also keeps
goldens byte-for-byte unchanged. (ADR-051 binding rule 5: extend the existing
raw-JSON governance pattern, don't add another checker.)

Writers to migrate: `BatchManifest.write`, `CombinedManifest.save`, depth
metadata JSON, run card + self-attestation sidecar.

### Required tests

Crash-injection pre/post-replace; concurrent writers (one intact winner, no
orphan temps); multithreaded permission test proving no process-umask
disturbance; mocked call-order test (file fsync before replace, directory
fsync after); golden parity for manifests/run cards/attestation;
serialization-failure and disk-failure cases leaving the prior file intact.

**CI lane placement (must be named in the PR):** crash-injection and
concurrent-writer tests spawn processes — under ADR-044 they likely carry
`slow`/`stress` markers, which the default `make test-fast` PR lane excludes.
State which lane executes them (e.g. `make test-full` in build.yml /
nightly), otherwise the acceptance criteria are never exercised in CI. All
new tests face `check_no_tautological_tests.py` and marker enforcement.

If the run-card writer extraction from `orchestrator.py` is done as a seam
extraction, register it first in
`docs/architecture/MONOLITH_DECOMPOSITION_TARGETS.md` (ADR-045; made binding
by ADR-051 rule for every extraction).

### Scope control

Generation-level publication (all-old-or-all-new artifact+evidence sets) is
repair 1.5-b, explicitly out of scope per the issue's non-goals; it needs no
issue until that functionality is activated under ADR-051.

### Closure gate

`Closes #2063` only when all four governed writers use the primitive and the
crash/concurrency tests run in a named CI lane under the required CI Gate.

---

## 7. Issue #2065-A — promote the plan contract without activating the new executor

### What is already landed (do not redo)

PR #2070 (`e30bd58`) + PR #2081 (`81c9f15`), verified at this head:

- Frozen `ResolvedInvocation` (`resolved_invocation.py:61-62`), schema
  `tp.lux.resolved_invocation.v1` (`:42`), canonical JSON via the approved
  serialization module (`:135-138`), license-enforcing single resolution
  (`:254-267`), plan fingerprint sharing the runtime `ConfigFingerprint`
  algorithm (`:336-342`), plan warnings (`:233-251`), schema shipped as
  package data (`schemas/lux/resolved_invocation.schema.json`;
  `pyproject.toml:190-192`).
- Resolver-only `--plan` (`__main__.py:705-716,1184-1188`): no model load, no
  output-dir creation, returns before orchestrator construction; covered by
  `tests/lux_depth_v3/test_cli_plan_mode.py` (incl. determinism and
  writes-nothing assertions).
- Single-resolution consumption: `config.resolved_invocation` carried into
  `ConfigResolver` (`config_resolver.py:942-966`) and `DA3Backend`
  (`depth/backends/da3.py:169-187`) via
  `validate_authoritative_model_contract` — the legacy compat-variant
  round-trip is defused. Independent resolution survives only as the
  documented fallback for direct callers with no carried invocation.
- Commercial-safe default `da3_metric` (`model_registry.py:68-71`), `da3`
  alias deprecated (still resolves the research model, warns).
- The payload is **explicitly `stability: provisional`**
  (`resolved_invocation.py:112-115`) pending exactly this promotion.

### What is genuinely outstanding (verified absent)

- No `tp.execution.plan.v1` anywhere under `src/`, `tests/`, `schemas/`.
- Manifests carry only the pre-existing
  `requested_backend`/`resolved_backend`/`resolution_status` trio
  (`manifest.py:179-181`); **no** `plan_fingerprint`, `planned_backend`,
  `candidate_fallback_chain`, or `executed_backend` manifest fields exist.
- No documented-workflow parity fixture exists in `tests/`.

### Preliminary status comment on #2065 (before implementation)

1. The historical baseline in acceptance criterion 3 (9 APEX failures /
   1 license failure / 1 pass, "before any other repair lands") is **already
   unsatisfiable at this head**: PR #2081 changed the default-model license
   outcome, and plan steps 1–2 change flag behavior before the fixture
   lands. Record a re-baseline: the parity gate asserts an explicitly
   declared current outcome for **each of the 11 documented workflows**
   (the count comes from the issue itself) at the head where the fixture
   lands, rather than reproducing the pre-repair snapshot.
2. Terminology: independent model re-resolution is forbidden; **fail-closed
   revalidation of a carried contract at a trust boundary is required and
   already implemented** (`validate_authoritative_model_contract`). Literal
   object identity across process boundaries is not the invariant.

### Implementation PR — `feat(core): promote ResolvedInvocation to canonical execution plan v1`

1. Introduce the core-owned schema `tp.execution.plan.v1` (ADR-051 assigns
   ownership to core; Lux supplies the first adapter). **A new schema
   package (e.g. `transformation_portal.schemas.execution`) needs its own
   `[tool.setuptools.package-data]` entry or the wheel silently omits it —
   both import surfaces are wheel-verified in CI.**
2. Promote the landed semantics (authoritative model+revision, license
   acknowledgements/evaluation, planned backend, candidate fallback chain,
   frozen input selection, config identity, requested outputs) and add the
   ADR-required structure: stable node IDs, typed stage-registry
   identifiers, dependency edges, typed stage config, resource
   requirements, output declarations, failure policy.
3. ADR-051 security rule 1 payload limits (byte/nesting/string/node/edge/
   count limits) are **part of promotion**, not later hardening. Compile
   external plans only through an allowlisted stage registry; never
   deserialize arbitrary module/class names.
4. Retain a read-only compatibility adapter for
   `tp.lux.resolved_invocation.v1` (must remain until #2065 closes, per
   ADR-051). Dual-render both forms in tests and prove equivalent model,
   license, input, stage-order, and artifact intent.
5. Migrate the direct `EnhanceOrchestrator` API to build/consume the same
   authoritative plan instead of the no-carried-invocation fallback path.
6. `executed_backend` **never** enters the plan payload — the landed design
   excludes it deliberately (`resolved_invocation.py:87-92`); it lands in
   manifests only, in #2065-B.
7. Do **not** activate `CASDAGExecutor` (exists at
   `core/cas_dag_executor.py`, zero production constructions). ADR-051 gates
   that cutover on a separate Phase C vertical slice; the live Lux executor
   remains the rollback path.

### Required tests

Canonical-JSON byte determinism; v1 payloads parse to the same authoritative
plan; unknown schema versions and unknown stage-registry identifiers fail
closed; a schema-valid but forged model contract fails lock revalidation;
plan/run resolve the same model key+revision; CLI and direct-Python compile
equivalent plans; worker-boundary round-trip with fingerprint equality;
`--plan` writes nothing (read-only output root); wheel-installed consumers
load and enforce both schemas.

This PR **advances but does not close** #2065.

---

## 8. Issue #2064 — identity-verified depth-cache entries

### Current state (verified)

- `depth_cache.py` is byte-untouched since `c6d620a`: `get()` at `:85-99`
  is `np.load(str(cache_path))` guarded only by `exists()`, blanket
  `except Exception → None`; `store()` writes bare `.npy`. Atomicity of the
  array write is already solved (temp+replace at `:151-165`) — the gap is
  strictly identity/integrity metadata.
- `allow_pickle=False` has been numpy's default since 1.16.3, so the missing
  flag is an explicitness/fail-closed hardening, not an active pickle hole
  under the pinned numpy. Pass it explicitly anyway.
- **Drift since the issue was filed:** on the contract-carrying path the
  fingerprint's `model_variant` now resolves to
  `canonical_key:repo_id@locked-revision`
  (`config_resolver.py:303-337`), so a model-lock change already invalidates
  keys on that path; the legacy/uncarried path (`:336-337`) is still
  revision-blind. The issue's "no revision identity" framing is now only
  true for the fallback path. Schema version, weights digest, and dependency
  identity remain absent everywhere, as the issue states.
- The segmentation cache remains the in-repo target pattern and exceeds the
  issue's description: schema in-key and re-checked on read, key-payload
  equality, `allow_pickle=False`, per-mask shape/dtype/sha256, atomic JSON
  sidecar (`segmentation/_cache.py:126,281,319,335-364`).
- `ExecutionIdentity` exists at schema `adr-032-v2`
  (`core/execution_identity.py:64,104`) with **no Lux wiring and no v3**.
  ADR-051 designates v3 as an additive target (dual-write v2/v3, no cache
  reuse; "Lux fingerprints become projections"). Derive/project — do not
  replace the fingerprint wholesale, which would contradict the accepted
  ADR.

### Implementation PR — `fix(lux): verify depth-cache entries against execution identity`

Cache identity: replace the two-string identity with a schema-versioned
projection carrying at least `cache_schema`, `image_sha256`,
`config_fingerprint_sha256`, `model_canonical_key`, `model_lock_revision`,
and — structured so identity-v3 fields (`execution_identity_sha256`,
`materialized_weights_sha256`, `dependency_lock_sha256`) can be added
**additively** when they exist. Key includes the cache schema and the
identity digest.

Sidecar (canonical JSON, committed **last** via the #2063 primitive):
schema, cache key, model canonical key, locked revision, `.npy` file
sha256 (of the exact serialized bytes), shape, dtype, byte length, plus the
identity-v3 fields when available.

Write path: array via #2063 primitive → hash published bytes → sidecar via
#2063 primitive. Failure before sidecar publication ⇒ incomplete entry,
never a hit.

Read path: require both files; parse/validate sidecar with size+schema
limits; compare requested identity and revision; open the `.npy` **once**,
hash through the open handle, seek(0), `np.load(handle,
allow_pickle=False)`; validate shape/dtype; hit only after every check
passes (same-handle hashing kills the path-swap TOCTOU).

Housekeeping: array+sidecar are one entry for LRU touch (today's
`cache_path.touch()` at `:95`), eviction, `clear()`, size accounting, and
statistics; legacy `.npy`-only entries are misses; malformed/orphaned
entries are lazily cleaned. Concurrent same-key writers may cause a miss
during interleaving but never a false hit. Decide whether validation
failures stay silent (current blanket-except behavior) or log at debug like
the segmentation cache — recommend the latter.

### Required tests

One per miss condition: no sidecar; sidecar without array; malformed
sidecar; cache-schema mismatch; revision mismatch; checksum mismatch; shape
mismatch; dtype mismatch; identity-field mismatch for any populated v3
field. Plus: pickled/object payload rejected without execution; crash
between array and sidecar ⇒ miss; concurrent writers never yield an
unverified hit; same image + same resolved identity ⇒ hit; model-lock
revision change ⇒ miss; eviction/clear leave no orphans.

### Closure gate — deviation from the external review, recorded

The external review proposed closing #2064 only when the cache is
"identity-v3 complete" including materialized-weight digests. That folds the
explicitly **held** repair 1.4-b into this issue and exceeds the issue's own
acceptance criteria. Resolution:

- Close #2064 on its issue-scoped acceptance criteria (all five, plus the
  additive-identity sidecar structure above), built on #2063 and consuming
  #2065-A's identity where available.
- At closure, record on the issue: (a) materialized-weight digest
  verification is repair **1.4-b — create that follow-up issue then** (it
  does not exist yet; scope includes extending
  `resolve_backend_model_artifact` beyond its depth_pro-only support,
  `pipeline_coordinator.py:426-430`); (b) ADR-051 security rule 7
  ("production cache reads/writes require complete identity v3") remains
  the gate for any *production* cache activation — closing #2064 does not
  discharge it, and no production cache hit may be enabled on model-lock
  revision alone while weights remain unidentified.

---

## 9. Issue #2065-B — manifest propagation, deliverable accounting, documentation truth

Final Lux closeout PR, after #2067, #2068, #2065-A (and ideally #2064, to
avoid touching the cache fingerprint surface twice).

### Implementation PR — `test(lux): prove plan-manifest identity and documented workflow parity`

Manifest propagation (all net-new — none of these fields exist today):
`plan_schema`, `plan_fingerprint`, `planned_backend`,
`candidate_fallback_chain`, `executed_backend`, `requested_artifacts`,
`produced_artifacts`, `omitted_artifacts`.

Rules:

- `planned_backend`/`candidate_fallback_chain` appear in plan **and**
  manifests; `executed_backend` appears in runtime evidence only (issue
  criterion 6; landed design already enforces the plan-side half).
- Every requested artifact ends `produced` (path/checksum/media metadata),
  `omitted` (typed, permitted reason), or `failed` (fails the run if
  required). This discharges repair **1.3-c** (requested-vs-produced
  accounting).
- The pre-existing `requested_backend`/`resolved_backend`/`resolution_status`
  metadata is preserved (CLAUDE.md manifest contract), not replaced.
- No manifest may claim a marketing, master16, or upscaled16 deliverable.
- Schema/validator/fixture/docs updates land in the same PR (contract-family
  rule), including run-card schema; note `executed_backend` already exists
  inside the run-card segmentation capability object — the new top-level
  field must not collide semantically.

The 11-workflow contract gate (discharges repair **1.1-a**): a
machine-readable fixture per maintained documented command — stable
workflow ID, exact argv, expected plan status, expected resolved
model/backend, expected stage set, expected requested artifacts, expected
warnings/validation error, source documentation anchor. CI runs every entry
through `--plan` (this runs in **core CI** by design — no ML stack needed —
but the tests must follow the ML-import isolation patterns since they import
the Lux CLI), proves zero model loads and zero writes, compares canonical
output to the contract, verifies doc/fixture argv identity, and fails on any
undeclared drift. Negative examples must be explicitly marked as such.
Expected outcomes are declared against the re-baselined reality recorded on
the issue (see #2065-A preliminary comment), not the pre-repair 9/1/1
snapshot.

### Final #2065 acceptance suite

Backend instance identity equals the carried contract; no compat
`model_variant` can select a different model; all 11 workflows have current
explicit outcomes; canonical plan determinism; `--plan` writes nothing;
planned/fallback fields in plan+manifests; executed backend in runtime
evidence only; CLI/direct-Python/worker paths consume equivalent plan
identity; v1 payloads readable through the bounded adapter; installed-wheel
schema validation passes.

Then `Closes #2065`, with a closing comment recording that repairs 1.1-a and
1.3-c are discharged here and 1.1-b (documentation truth) was consumed by the
#2067/#2068 PRs.

---

## 10. Issue #1814 — NVIDIA-host evidence

Repo-side state (verified): the `gpu` lane
(`Dockerfile:84-147`) uses explicit `python3.11` throughout;
`tests/validation/test_dockerfile_contract.py` now has **6** tests (the
issue's "4 passed" comment is stale); nothing rejects prerelease Python; the
`apple-silicon` target (`Dockerfile:149-158`) is a documented, contract-locked
CPU-only alias; no GPU validation evidence exists anywhere under `docs/`.

Evidence-first procedure on an NVIDIA host (or temporary self-hosted runner
with the NVIDIA Container Toolkit): record `nvidia-smi`, `docker info`,
`nvidia-container-cli info`, then run the issue's exact five commands, plus:

```bash
docker run --rm transformation-portal:gpu-runtime-maturity \
  python3.11 -c "import sys; assert sys.version_info[:2] == (3, 11); \
assert sys.version_info.releaselevel == 'final'; print(sys.version)"

docker run --rm --gpus all transformation-portal:gpu-runtime-maturity \
  python3.11 -c "import torch; x=torch.ones(8, device='cuda'); print((x * 2).sum().item())"
```

Outcomes:

- **All pass:** paste logs + environment info on #1814 and close — the
  owner's recorded closure criteria require evidence on the issue, not a
  `docs/` report; do not invent one.
- **Python is 3.11.0rc1:** one minimal Dockerfile PR pinning a final 3.11.x,
  plus a contract test rejecting prerelease Python; re-run the matrix; merge
  and close.
- **CUDA unavailable:** diagnose the host NVIDIA runtime separately from the
  image before changing repository code.

Record in the closing comment (known, accepted residue): the `gpu-build`
stage installs **unpinned** `torch`/`torchvision` from the cu121 index
(`Dockerfile:126`) outside the governed lock contract, and a governed CUDA
lock is not currently possible because the CUDA lock lanes are retired and
fail closed. Closure notes this; any pinning effort is a separately scoped
decision, as is any real (non-alias) Apple-container target. Do not hold
#1814 open for either.

---

## 11. Governance gates every implementation PR must clear

Program-wide merge gate: focused issue-specific tests; `make ci-quick`;
`make test-fast`; `make test-orchestrator-contract`; pre-commit governance
checks; the protected-branch **CI Gate**. Additionally, verified against the
actual gate inventory:

- **check-json-serialization:** `io_atomic.py` is not allowlisted; keep new
  primitives JSON-agnostic (bytes in), keep `json.dumps` at the approved
  call sites.
- **Coverage floors (the binding ones for this work):** `lux_depth_v3/` 72%
  line (validators 80%), `app.py` 79% line / 71% branch, `src/tp/` 75%,
  `hardening/` 95%. The diff-based cold-zone touched-file gate does **not**
  cover `lux_depth_v3/` or `app.py` — the floors are what bites.
- **Portal asset budgets + rebuild:** #2067/#2068 touch
  `portal.template.js` ⇒ `npm run build:portal` +
  `make check-portal-asset-budgets` in the same PR.
- **ADR-044 markers + ML isolation** for all new tests;
  `check_no_tautological_tests.py`; governed `TODO(#NNNN)` markers for
  deprecation shims.
- **ADR-045 seam registration** before any extraction from `orchestrator.py`
  or manifest-accounting helpers.
- **Package-data registration** for any new schema package (wheel-verified
  import surfaces).
- Contract-family rule: schemas, validators, fixtures, docs updated in the
  same change; docs placement per `DOCUMENTATION_POLICY.md`
  (`check_docs_structure.py` runs in CI).

For every closure: the PR body maps each acceptance criterion to a specific
test or evidence artifact; no criterion is checked merely because code
exists; issue comments record the merge SHA and validation output; the
closing keyword appears only in the final PR for that issue; no open review
threads or unresolved CI contexts remain.

## 12. Repository-management actions

- **Dependency comments on the issues** (done alongside this plan): each
  open issue carries a comment making merge order, prerequisites, and — for
  the two decision issues — the proposed disposition visible without this
  external document.
- **Labels/milestones — corrected from the external review:** the repository
  has no `labels.yml`, no milestone taxonomy, and the repair-program issues
  deliberately carry zero labels; existing labels are automation-owned
  (dependabot/nightly/perf bots, AI triage `type:`/`priority:`). Inventing
  an `area:`/`priority:` label scheme here would create an ungoverned
  parallel convention. The repair program is tracked by issue-title
  numbering and body cross-references; the dependency comments carry the
  ordering. If the owner wants a milestone ("Lux Depth V3 Repair Closeout"),
  that is a one-click owner action in the GitHub UI — nothing in this plan
  depends on it.
- **GitLab:** mirror-only unless a separately governed project is
  intentionally established; no repo-side action.

## 13. Verification record — corrections applied to the external review

All plan-critical claims of the external review were re-verified at
`3e494a8`; the following corrections are folded into the sections above:

1. `test_dockerfile_contract.py` has 6 tests, not 4; the GPU lane's torch
   install is unpinned/ungoverned (recorded as accepted residue in #1814's
   closure, not a new workstream).
2. Manifests carry **no** plan fields today; the entire #2065-B manifest
   surface is net-new. No workflow-parity fixture exists; the "11" count is
   sourced from issue #2065 criterion 3, whose 9/1/1 baseline is already
   unsatisfiable post-#2081 and must be re-baselined on the issue.
3. `emit_marketing`'s plan warning exists; `emit_report` has none; no
   deprecation-warning class for output flags exists; the #2067/#2068 doc
   and portal surface inventories are larger than the issues list (see
   sections 4–5).
4. The #2068 "only ever read jointly" claim holds for behavior-gating reads
   only; individual reads in fingerprint/manifest/plumbing surfaces create
   the compatibility interlock in section 5. `app.py` defaults both flags
   to True. Dormant upscaling machinery exists but produces no distinct
   deliverable.
5. The depth-cache fingerprint already embeds the locked model revision on
   the contract-carrying path (post-#2070/#2081); `ExecutionIdentity` v3
   does not exist yet and #2064 closes without claiming it does (section 8
   deviation record).
6. `io_atomic.py`'s narrow deficiencies are confirmed, but the repo already
   contains several durable-write implementations to converge on, and one
   Lux call site already bolts fsync onto the primitive
   (`orchestrator.py:2902-2939`).
7. The "Repair, Designate, Prove" program is issue-resident (no `docs/`
   document); its numbering includes 1.6-a (#2062, closed) and 1.2 (#2066,
   closed) beyond the list in the external review; 1.1-a/1.1-b/1.3-c/1.4-b/
   1.5-b have no dedicated issues and are dispositioned in sections 6–9.
8. The label/milestone cleanup was replaced with convention-consistent
   dependency comments (section 12).

## 14. Target final state

- #2063 durable evidence foundation merged; #2064 verified depth cache
  active (identity-v3 residue tracked in a new 1.4-b issue + ADR-051 rule 7
  gate); #2065 canonical plan contract with manifest accounting and the
  documented-workflow gate proven (discharging 1.1-a, 1.1-b, 1.3-c); #2067
  and #2068 dispositioned, deprecated flags warning, documentation truthful;
  #1814 closed on NVIDIA evidence.
- Open GitHub issues: 0 (plus one intentionally created 1.4-b follow-up).
  Open PRs: 0. GitHub authoritative; GitLab mirror-only.
- No duplication of work already merged in #2070/#2081, no weakening of
  fail-closed model governance, no premature activation of the ADR-051
  executor migration.
