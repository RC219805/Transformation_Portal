# Copilot Instructions - Transformation Portal (RC219805)

You are working in a governed production repository for **luxury real estate / ArchViz rendering, ingest, archive, and portal orchestration**.

Current baseline: repo-wide documentation refresh audit dated April 29, 2026,
building on `main` through PR #1562. Use the root `README.md`,
`docs/README.md`, `docs/governance/DOCUMENTATION_MAP.md`,
`docs/governance/DOCUMENTATION_REFRESH_AUDIT_2026-04-29.md`, and
`docs/governance/DOCUMENTATION_STATE_AUDIT_2026-04-27.md` for current
navigation. Historical project reports may retain old dates and facts; do not
treat them as live guidance unless the documentation map promotes them.

This codebase is broader than a single enhancement pipeline. It includes:

- **Lux Depth V3**: depth-aware orchestration, PBR map generation, Materials V3, optional V2 enhancement, and governed deliverables
- **Portal / orchestrator HTTP surfaces**: readiness and job-oriented endpoints for governed execution
- **Ingest + provenance + machine-mode contracts**: audit-grade metadata capture, typed JSON automation, evidence projection, and detached attestation flows
- **Archive / fixity / governance tooling**: manifests, Merkle roots, signatures, rights policy, and export utilities
- **Workflow assets**: ComfyUI examples/templates and operational scripts for repeatable execution

Recent current-state contract anchors:

- PR #1561/#1562 added the typed API v1 envelope/schema foundation and typed
  OpenAPI response models for `/healthz`, `/ready`, and `/v1/readiness` while
  preserving existing wire shapes.
- The managed front door in `web/secure-landing/` is Node 22.x only.
- Root `.env.example` is the Docker/FastAPI template; `web/secure-landing/.env.example`
  is the managed front-door template.
- `docs/ci/WORKFLOW_MATRIX.md` is the current 30-workflow GitHub Actions inventory.
- `docs/governance/audit/archive-gates-2026-04-27.md` is the current archive
  Gates A/B/C readiness audit evidence.

Optimize for:

- **Contract integrity**
- **Deterministic behavior**
- **Import-surface stability**
- **Performance under APEX / Quality Firewall rules**
- **Small, reviewable, low-risk changes**
- **Documentation and tests that stay in lockstep with behavior**

This repository is **not** a generic CRUD portal, government portal, or health IT system. Do not import assumptions, workflows, naming, or compliance language from those domains.

---

## Non-Negotiables

### 1) Preserve public contract surfaces

Treat these as binding until a versioned change explicitly says otherwise:

- `lux-depth-v3` CLI behavior and `python -m transformation_portal.lux_depth_v3`
- Portal / orchestrator HTTP behavior, including `/healthz`, `/ready`,
  `/v1/readiness`, job endpoints, typed OpenAPI response models, and typed
  `/v1/*` envelopes
- Import surfaces for both `transformation_portal` **and** `tp`
- Schema-backed automation and provenance contracts:
  - ingest contract (`v1.0.2`)
  - machine-mode JSON (`tp.meta.machine.v1`)
  - evidence / attestation artifacts
- Stable presets, quality-tier semantics, and deliverable naming where already governed

Do **not** silently change:
- default quality behavior
- preset meanings
- output file semantics
- machine JSON envelope keys
- portal request/response shapes
- import paths that CI or wheel installs rely on

### 2) Keep minimal environments working

Core CI and wheel-smoke paths must work without full ML stacks.

You must:

- avoid eager imports of heavy optional dependencies in `__init__.py`, CLI entrypoints, or help-path imports
- keep `transformation_portal` lazy-load friendly
- keep `transformation_portal.lux_depth_v3` lazy-load friendly
- preserve `tp` importability in both source-tree and wheel installs
- avoid making `torch`, `transformers`, `diffusers`, `rawpy`, or backend-specific packages mandatory for core import paths unless the change is explicitly scoped and documented

### 3) Do not collapse independent version planes

This repository has **multiple valid version planes**. Do **not** "clean them up" by forcing them into one number unless the change is intentionally version-governed.

Common planes include:

- repo/release baseline and stable contract era
- package metadata version
- runtime `transformation_portal.__version__`
- submodule-specific versions such as `lux_depth_v3.__version__`
- schema versions (for example ingest)
- machine/evidence/attestation schema identifiers
- feature/preset versioning in docs and changelog

Rule: **only bump the version plane you are actually changing**, and update all tests/docs/schemas/changelog entries that govern that plane.

### 4) Keep Lux Depth V3 behavior stable unless a contract says otherwise

Preserve:

- quality-tier semantics (`standard`, `premium`, `apex`)
- distinction between `--quality-tier` and `--preset`
- optional V2 stage behavior and backward-compatibility defaults
- input discovery hygiene that excludes derived artifacts / output directories
- governed deliverables such as manifests, run cards, depth and PBR artifacts
- backend-resolution transparency and license enforcement

Do not reintroduce older API shapes that have already changed. Example: if a PBR API now returns a dataclass, do not casually revert it to tuple unpacking.

### 5) Keep portal hardening intact

When touching `app.py`, request validation, or orchestration endpoints:

- preserve allowed-root path validation for input/output locations
- preserve API key and request-hardening behavior
- preserve trusted-host, request-size, concurrency, and rate-limit protections
- preserve pipeline allowlists and typed validation for archive-gate flows
- fail closed, not open

### 6) Never paper over failures

Do not fix CI by:

- weakening assertions without cause
- broad-skipping test families
- suppressing typed errors with generic fallbacks
- changing docs/comments to claim checks are skipped when CI still runs them

Fix root causes. If behavior changed intentionally, update tests and docs to match reality.

### 7) Respect the repository organization system

This repo uses an automated organization policy.

You must:

- keep the repository root minimal and operational
- avoid introducing new root files unless they clearly belong there
- place scripts in the correct `scripts/` subdirectory
- place documentation in the right `docs/` subtree
- keep generated artifacts out of source roots unless explicitly governed
- update organization docs/rules if you intentionally extend the file-placement policy

---

## Repository Reality Map

Place work in the right zone.

| Area | Purpose | Guidance |
| --- | --- | --- |
| `src/transformation_portal/` | Main package | Preferred home for production logic |
| `src/transformation_portal/lux_depth_v3/` | Governed orchestration pipeline | Keep public behavior stable; prefer decomposed modules over monolith growth |
| `src/tp/` | Contract / fixity / phase tooling import surface | Preserve import stability and cross-runtime correctness |
| `app.py` + `portal.html` | Portal / orchestrator surface | Preserve request validation and security defaults |
| `config/` | Presets and config | Keep preset taxonomy and compatibility stable |
| `schemas/` + `docs/schemas/` | Canonical schema artifacts | Update with validators/tests/docs in the same change |
| `tools/` | Governance, performance, metadata, archive, evidence utilities | Deterministic file IO; typed outputs; safe CLI behavior |
| `scripts/` | Operational runners, setup, validation, diagnostics | Thin orchestration and maintenance helpers; not the home for core domain logic |
| `workflows/` | ComfyUI examples/templates | Treat as workflow assets/examples, not hidden production logic |
| `tests/` | Contract, regression, ML, smoke, enforcement, security, performance coverage | Match marker policy and keep default lanes fast |
| `docs/` | Architecture, contracts, governance, troubleshooting, guides | Update whenever behavior or workflow changes |
| `.github/agents/` | Live custom-agent profiles and support docs | Keep aligned with `docs/architecture/agent_governance.md`, `docs/guides/CUSTOM_AGENT_GUIDE.md`, and `tests/test_custom_agent_config.py` |

Additional repo areas such as `assets/`, `textures/`, `archive/`, `artifacts/`, `data/`, `dashboard/`, and project-specific directories are first-class parts of the repository. Do not treat them as disposable clutter.

Custom-agent surface:

- `@transformation-portal-architect` governs contracts, dependency policy, CI/CD,
  security posture, docs topology, and architecture.
- `@portal-app-steward` owns managed browser-boundary execution for the
  frontdoor, portal shell, manifest-backed assets, selectors, and browser
  validation.
- `@transformation-portal-specialist` owns backend/orchestrator, Lux Depth,
  archive, ingest, machine-mode, and governed non-browser execution work inside
  existing contracts.
- `.github/agents/_archive/` and `.github/agents/rag_system/_archive/` are
  historical and must not define live agent behavior.

When agent guidance, ADRs, security policy, and CI enforcement conflict,
follow the precedence defined in `docs/architecture/agent_governance.md`
(mechanical enforcement → ADRs → security/dependency policy → Architect →
Steward / Specialist execution). Use the narrowest live profile that fits
the work and escalate per `docs/architecture/agent_governance.md`
§"Escalation Criteria".

---

## Architecture Rules

### Lux Depth V3 is now a decomposed orchestrator system

Do not keep stuffing behavior into one giant orchestrator file.

Prefer these seams:

- `config_resolver.py` for preset/config normalization
- `pipeline_coordinator.py` for backend/stage resolution
- `execution_engine.py` for stage execution logic
- `artifact_manager.py` for output hashing/indexing/provenance assembly
- `validators/` for schema/run-card validation
- `orchestrator.py` as the compatibility-facing orchestration surface, not the dumping ground

If you touch `EnhanceOrchestrator`, ask first:

1. Should this live in a focused helper/module instead?
2. Is this change preserving orchestrator re-export compatibility?
3. Will this reduce or increase coupling and test friction?

### Prefer explicit contracts over implicit branching

Use:

- Protocols / ABCs for backend and stage contracts
- registries / factories for backend selection
- adapters for backend normalization
- typed config objects and validated schemas
- pure-ish helpers for deterministic transforms and file projections

Avoid:

- backend/device selection scattered across many modules
- hidden global state
- config-dependent `if/elif` chains spread across unrelated files
- top-level imports with heavy side effects

### Preserve lazy loading

This repo intentionally uses lazy imports to keep:

- CLI help paths working
- core test lanes working
- wheel-smoke installs working
- optional ML dependencies optional

Do not move heavy imports into module top-level code for convenience.

---

## Public Surface Guidance

### CLI surfaces

`lux-depth-v3` is a primary public entrypoint.

Preserve:

- argument semantics
- default values unless version-scoped
- `--quality-tier` vs `--preset` distinction
- typed error behavior
- deterministic deliverable naming where governed
- machine-readable modes where supported

### Machine-mode JSON is a contract

For commands that support `--json`:

- preserve the envelope contract
- preserve stable top-level keys and typed error shape
- route automation by typed fields and exit codes, not human prose
- do not add shape drift casually
- update schemas, contract docs, validators, and tests together

### Portal/orchestrator HTTP surfaces

Treat request/response payloads and readiness behavior as contract-sensitive.

When changing portal behavior:

- validate inputs early
- preserve secure defaults
- preserve error typing and reason codes where already established
- keep path normalization and allowlist enforcement explicit
- avoid implicit filesystem trust

---

## Contract Families You Must Understand Before Editing

### 1) Repo / release stability
The stable release era begins at the governed repository baseline. Preserve backward-compatible behavior by default.

### 2) Ingest contract
The ingest layer has its own official schema version and quality-firewall semantics. Preserve:
- schema field meanings
- exit-code semantics
- deterministic file-derived fields
- audit-grade provenance guarantees

### 3) Machine-mode contract
`tp.meta.machine.v1` is an automation wire contract. Preserve:
- `schema`, `command`, `success`, `exit_code`, `data`, `error`
- typed errors
- stable per-command payload shape
- deterministic structure

Do **not** assume machine-mode bytes are the same thing as evidence canonicalization.

### 4) Evidence / attestation flows
Evidence projection, canonicalization, Merkle/signature flows, and detached attestations are governed separately from machine-mode wire output. Keep those layers distinct.

### 5) PBR and preset contracts
Stable presets and governed PBR surfaces are contract-bearing. Preserve stable preset behavior; treat canary/experimental paths as more flexible but still documented and validated.

---

## Backend, ML, and License Governance

### Depth backend policy

Assume:

- `da3` is the default commercial-safe production backend
- `depth_pro` is research-only and requires explicit acknowledgments
- research presets remain opt-in, not the default production path
- fallback behavior must stay transparent and traceable

If a backend is gated by license or research-only use:

- validate before processing starts
- raise clear typed exceptions
- do not silently downgrade rights requirements
- capture resolved backend/provenance where the current system does so

### Optional dependency discipline

- Core/import/test lanes must not require optional ML stacks
- ML code must degrade gracefully when optional packages are absent
- Tests must distinguish **offline** from **backend availability**
- Never add surprise model downloads to default tests

### Installation / dependency layering

Dependency policy is layered.

Rules:

- broad runtime ranges live in `pyproject.toml`
- operational dependency truth lives in layered `requirements/`
- edit the right `.in` source
- regenerate the matching lockfiles
- keep the core runtime lean
- use platform/capability layers instead of dragging heavy ML packages into the base path

Do not bypass the layered strategy because a one-off local install "worked on your machine".

### Banned / risky dependency changes

If the repo explicitly bans or constrains a dependency, respect that policy. Do not reintroduce blocked packages or broaden constraints casually.

---

## Performance and APEX Rules

Performance is a governed signal, not an afterthought.

### Quality Firewall mindset

Treat regressions as blocking when they affect:

- p95 latency budgets
- mean latency budgets
- failure rate
- disk/temp-artifact growth
- determinism or reproducibility metadata

### Performance engineering defaults

- avoid Python loops over pixels when vectorization is practical
- avoid repeated model loads per image/frame
- avoid needless tensor/NumPy/device ping-pong
- stream large media rather than loading entire videos eagerly
- use atomic writes for outputs, manifests, ledgers, and evidence artifacts
- preserve depth caching and artifact reuse when current flows depend on them

### Benchmark and real-pipeline separation

Keep lightweight PR lanes lightweight.

Do not:
- pull benchmark-only work into default gating lanes
- make PR CI depend on heavyweight real-pipeline runs unless explicitly intended
- claim a suite is "CI-excluded" unless marker selection and workflows actually enforce it

---

## Testing and CI

### Use the repo's actual test split

Prefer the repo's existing commands first:

```bash
make test-fast
make test-orchestrator-contract
make ci
```

If your change is ML-specific and the environment is provisioned for it, use the ML-targeted lane as well.

### Marker discipline matters

This repo is no longer just `ml` vs `slow`.

Respect the broader taxonomy in CI and tests, including families such as:

- `unit`
- `security`
- `regression`
- `golden`
- `integration`
- `ml`
- `slow`
- `benchmark`

Guidelines:

- keep default/core lanes fast and deterministic
- keep benchmark work explicitly marked and out of normal PR gates
- keep ML unit coverage offline and small-fixture based
- do not move slow/integration behavior into the wrong lane without updating CI

### CI guardrails are part of the contract

Preserve checks such as:

- docs structure / stale-doc-path validation
- raw JSON usage guardrails where restricted
- dependency-constraint validation
- `tp` import surface checks
- wheel install smoke
- relocatability / contained-output proof
- compliance schema validation

If you change packaging, imports, docs paths, workflow files, or output locations, expect to update the relevant tests and validators.

### Testing methodology

For non-trivial changes:

1. add or update a failing test
2. implement the narrowest fix
3. refactor only after behavior is locked

Favor:

- contract tests
- small regression tests
- deterministic golden/metric checks
- tiny fixtures
- typed error assertions
- explicit coverage of migration/backward-compat paths

Avoid over-mocking internal logic when a small real object/tensor/file will do.

---

## Lux Depth V3-Specific Guidance

### Preserve optional-V2 semantics

The V2 enhancement stage is optional, but backward-compatibility defaults matter. Do not silently flip defaults or remove fail-fast validation just because a local workflow doesn't need V2.

### Preserve input hygiene

Input discovery intentionally excludes derived artifacts and output directories. Do not weaken this and allow "depth of depth" or self-reprocessing loops back into the pipeline.

### Keep deliverables governed

Artifacts such as depth outputs, PBR maps, manifests, run cards, master/upscaled outputs, and marketing derivatives are part of the governed output surface in many workflows. Preserve naming and generation semantics unless the change is explicitly contract-scoped.

### Preserve separation of concerns

- preprocessing should stay preprocessing
- backend inference should stay backend inference
- postprocessing should stay postprocessing
- provenance/manifest logic should stay provenance/manifest logic
- scene/material decisions should stay explicit and testable

---

## Portal / Security / Filesystem Rules

### Filesystem safety

- use `pathlib.Path`
- normalize and validate untrusted paths
- enforce allowlisted roots
- create directories explicitly
- use atomic writes
- keep outputs inside the requested destination, not hardcoded repo paths
- avoid mutating the original workspace in scripts that claim relocatability

### Subprocess safety

- use `subprocess.run([...], check=True, capture_output=True, text=True)` unless streaming is necessary
- prefer explicit argument lists
- avoid `shell=True`
- apply timeouts when appropriate
- surface stderr/stdout meaningfully on failure

### Security posture

Inputs/outputs may be sensitive client or archive assets.

Do not:

- log unnecessarily sensitive paths or identifiers
- weaken API key or trusted-host defaults without explicit scope
- trust environment-derived paths without validation
- mix evidence generation with mutable in-place editing

---

## Documentation Rules

Docs are governance here, not optional garnish.

When behavior changes, update the right documentation layer:

- `README.md` for top-level user-facing behavior
- `AGENTS.md` for maintainer workflows and commands
- `docs/architecture/` for design and ADR-impacting changes
- `docs/api/` and `docs/schemas/` for contract surfaces
- `docs/apex/`, `docs/guides/`, `docs/cli/`, `docs/compliance/`, etc. for workflow-specific behavior
- `workflows/README.md` or related workflow docs if ComfyUI/workflow assets change
- changelog entries when the repo already treats the surface as release-notable

Do not update one doc while leaving the canonical contract doc stale.

---

## File Placement and Repo Hygiene

This repo enforces file-placement hygiene.

Before adding a file, ask:

1. Does it belong in `src/`, `tests/`, `docs/`, `schemas/`, `scripts/`, `tools/`, `workflows/`, `data/`, `archive/`, or `assets/`?
2. Is the repository root truly the correct place?
3. Does an existing helper / README / organization rule already define the destination?

Prefer:
- `scripts/validation/` for validation helpers
- `scripts/verification/` for verification entrypoints
- `scripts/bootstrap/` or `scripts/setup/` for install/bootstrap flows
- `scripts/pipelines/` for operational pipeline runners
- `tools/` for governed utilities and archive/performance/evidence CLIs
- `docs/` for analysis, plans, migration notes, and governance docs

---

## When Touching Workflows or ComfyUI Assets

- treat `workflows/` as examples/templates and generated assets, not as the home for hidden business logic
- keep reusable workflow-building logic in package code
- preserve example readability
- do not hardcode template output if a generator/builder already exists
- update the workflow README/examples when structure changes

---

## PR Hygiene

### Keep scope narrow
Prefer one clear feature/fix per PR. Avoid mixing:
- contract changes
- dependency overhauls
- broad formatting churn
- unrelated refactors

### Explain the "why"
PR notes and code comments should explain:
- what contract or invariant is being preserved
- what changed
- why the change is safe
- what tests/docs were updated

### Backward compatibility first
If a change is intentionally breaking:
- say so explicitly
- document the migration path
- update schemas/contracts/tests/docs/changelog in the same change
- do not hide the break behind silent behavioral drift

---

## Mistake-Proofing Checklist

Before you finish, check:

- [ ] Did I preserve both `transformation_portal` and `tp` import surfaces?
- [ ] Did I avoid eager heavy imports in package entrypoints?
- [ ] Did I keep the correct version plane(s) intact?
- [ ] Did I preserve CLI / HTTP / schema / machine-mode contract semantics?
- [ ] Did I keep Lux Depth V3 quality-tier, preset, V2, and input-hygiene behavior stable?
- [ ] Did I place new logic in decomposed modules instead of bloating a monolith?
- [ ] Did I keep tests offline where required and markers correctly assigned?
- [ ] Did I avoid benchmark/heavy-suite leakage into PR gates?
- [ ] Did I preserve atomic IO, path validation, and safe output placement?
- [ ] Did I update the right docs, schemas, and changelog entries?
- [ ] Did I avoid creating unnecessary root files?

---

## Default Decision Rules

When unsure:

- prefer existing contracts over local convenience
- prefer decomposition over monolith expansion
- prefer explicit typing/validation over inference-by-accident
- prefer lazy imports over eager heavy imports
- prefer offline deterministic tests over networked tests
- prefer additive, backward-compatible changes over silent semantic drift
- prefer small PRs with clear docs/tests over large "cleanup" rewrites

If a change touches contracts, schemas, portal behavior, archive governance, evidence/attestation, or performance thresholds, treat it as **governed work**, not routine refactoring.
