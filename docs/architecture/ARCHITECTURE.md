# Repository Architecture

**Status:** Maintained architecture overview
**Last updated:** 2026-05-12
**Canonical navigation:** [Documentation Map](../governance/DOCUMENTATION_MAP.md)

## Overview

Transformation Portal is a governed image and video processing platform for
luxury real estate rendering and architectural visualization. The current
architecture combines:

- A FastAPI portal/orchestrator backend in `app.py`.
- A managed Next frontdoor in `web/secure-landing`.
- Python package modules under `src/transformation_portal/`.
- Governed CLI and pipeline runtimes for Lux Depth V3, archive gates, APEX,
  Materials V3, spatial AI, ingest, attestation, and optional advisory VLM
  captioning.
- Layered dependency locks, deterministic validation, provenance, run-card,
  manifest, and documentation governance.

This document describes the current architecture at a system-boundary level.
Use more specific guides for operator procedures, API examples, and pipeline
commands.

## Runtime Topology

The browser-facing production path is intentionally frontdoor-first:

```text
Browser
  -> managed Next frontdoor (`web/secure-landing`)
  -> FastAPI backend (`app.py`)
  -> orchestrator jobs, pipeline CLIs, local runtimes, and artifacts
```

If a Cloudflare Worker is present, it is a frontdoor-only proxy. It must not
become a direct public bypass to the FastAPI backend.

Local development has two supported entrypoints:

- Backend-direct debugging through FastAPI routes such as `/portal`,
  `/portal/bootstrap`, `/healthz`, `/ready`, and versioned `/v1/*` API routes.
- Managed frontdoor debugging through `make run-frontdoor-local`,
  `make validate-frontdoor-browser`, and the Node 22.x guarded
  `web/secure-landing` runtime.

## Design Principles

### Contract Preservation

Routes, selectors, API envelopes, CLI flags, schema names, artifact paths, and
test semantics are treated as contracts. Behavior changes must update focused
tests and validation paths in the same patch.

### Fail-Closed Boundaries

Authentication, host validation, path resolution, artifact access, dependency
governance, and optional runtime discovery fail closed. Environment problems
should be classified as environment problems instead of weakening product
contracts.

### Deterministic Execution

Pipelines emit explicit metadata and avoid hidden ambient state where practical.
Run cards, manifests, provenance records, content-addressed storage, fixity
checks, and deterministic test fixtures are part of the architecture, not
incidental tooling.

### Layered Capabilities

Core Python dependencies, development dependencies, CI dependencies, and ML
capability layers are separated. Optional model runtimes such as DA3, Depth Pro,
RAW ingest, SAM2, and FastVLM are installed and validated through their
governed setup scripts and runtime manifests.

### Frontdoor And Backend Separation

The managed frontdoor owns browser authentication, session handling, access
posture, public route handling, and frontdoor health checks. The FastAPI backend
owns API contracts, pipeline readiness, job lifecycle, artifact access, and
direct-debug portal behavior.

## Repository Layout

```text
Transformation_Portal/
├── app.py                         # FastAPI portal/orchestrator backend
├── web/secure-landing/            # Managed Next frontdoor and portal bundle source
├── src/transformation_portal/      # Main Python package
├── src/luxury_tiff_batch_processor/# TIFF batch processor package
├── scripts/                       # Setup, validation, bootstrap, and pipeline scripts
├── tools/                         # Operator and governance tools
├── config/                        # Runtime presets, manifests, budgets, and model locks
├── requirements/                  # Layered dependency inputs, locks, and ownership metadata
├── schemas/                       # JSON schema contracts
├── tests/                         # Unit, contract, smoke, governance, and fixture tests
├── docs/                          # Maintained and historical documentation
├── workflows/                     # ComfyUI and workflow artifacts
└── cloudflare/                    # Frontdoor-only Worker package when used
```

## Core Backend And API

`app.py` is the FastAPI integration surface. It wires:

- Portal HTML and assets: `/`, `/portal`, `/portal/assets/*`,
  `/portal/bootstrap`, and portal video routes.
- Health and readiness: `/healthz`, `/ready`, and `/v1/readiness`.
- Operator configuration: `/v1/presets`, `/v1/config-metadata`, and
  `/v1/config-preview`.
- Portal event ingress: `/v1/portal/events`, `/v1/portal/rum`, and staged
  upload routes.
- Job lifecycle: `/v1/jobs`, `/v2/jobs`, job status, artifacts, cancellation,
  and server-sent events.

Typed response models live under `src/transformation_portal/api/v1/`. Versioned
API responses use the shared envelope model where applicable. The liveness and
readiness probes deliberately preserve raw response shapes:

| Route | Shape | Owner |
| --- | --- | --- |
| `/healthz` | Raw JSON with `ok` and `time` | Load balancers, frontdoors, probes |
| `/ready` | Raw JSON with `ok`, `time`, `version`, plus optional verbose fields | Backend readiness checks |
| `/v1/readiness` | `tp.orchestrator.readiness.v1` API envelope | Operator truth for pipeline dispatch readiness |

Do not wrap `/healthz` or `/ready` in the versioned API envelope unless the
probe contract is intentionally changed and contract tests are updated.

## Managed Frontdoor

`web/secure-landing` is the managed browser frontdoor. It handles:

- Login, logout, session cookies, Cloudflare Access posture, and local managed
  smoke credentials.
- Frontdoor `/healthz` readiness for backend connectivity, access config, user
  source, session store, and session-scaling checks.
- Proxying versioned backend requests while preserving auth and trace context.
- Portal bundle generation from modular `portal-src/` sources.
- RUM telemetry rollout controls, retry-after UX, deferred portal surfaces,
  asset manifest checks, CSS layer parity, and browser smoke selectors.

Node 22.x is the enforced runtime contract for frontdoor development,
validation, and builds.

## Package Modules

The main package is organized by responsibility:

| Package area | Responsibility |
| --- | --- |
| `api/v1` | Typed API envelopes and route response models |
| `core` | Configuration, security, storage, observability, device, validation, geometry, and processing primitives |
| `lux_depth_v3` | Main depth, materials, enhancement, artifact, run-card, and validation pipeline |
| `depth` | Depth backend interfaces, registry, Depth Pro/DA3 style backend integration, and depth utilities |
| `stage_graph`, `execution_graph`, `streaming`, `events`, `runtime`, `storage` | Execution primitives, scheduling, progress, event persistence, sandboxing, workers, and CAS/ledger storage |
| `spatial_ai` | Reconstruction, segmentation, materials, ingest, and orchestration research surfaces |
| `ingest`, `attestation`, `schemas` | Machine-mode metadata, provenance, archive attestation, and schema contracts |
| `vlm_captioning`, `vlm`, `models` | Optional advisory captioning and model manifest helpers |
| `pipelines`, `processors`, `enhancers`, `rendering`, `upscaling` | Image/video processing workflows and reusable processing engines |
| `metrics`, `reporting`, `evals`, `analyzers`, `dev` | Validation, reporting, evaluation, and development tooling |

Cross-module dependencies should remain directional. Shared primitives belong
in `core`, interface packages, or focused helpers. High-level pipelines should
not be imported by low-level utilities.

## Pipeline And Job Flow

The portal/orchestrator flow is:

```text
1. Frontdoor authenticates browser traffic and forwards allowed API requests.
2. Backend returns config metadata and preset catalogs for the selected pipeline.
3. `/v1/config-preview` normalizes arguments, validates paths, and returns
   expected commands, outputs, warnings, and readiness.
4. `/v1/jobs` enforces auth, rate limits, path policy, and readiness preflight.
5. The backend starts the selected runner or internal job path.
6. Events, status snapshots, and artifacts are exposed through job routes.
7. Pipeline outputs include governed manifests, provenance, run cards, fixity
   evidence, and advisory sidecars where enabled.
```

The governed pipeline set exposed through `/v1/readiness` currently covers
`lux-depth-v3` and archive gates A/B/C. Per-pipeline readiness can be `ready`,
`degraded`, or `blocked`; transport success is not the same as dispatch
readiness.

## CLI And Runtime Entry Points

Primary console scripts include:

- `lux-depth-v3` for depth, materials, PBR, enhancement, APEX, and run-card
  workflows.
- `depth-aware-dof` for single-image depth-aware depth-of-field packaging.

Operational runtime scripts live under `scripts/setup/`, `scripts/dev/`,
`scripts/validation/`, and `scripts/pipelines/`. Optional local runtimes belong
under `.runtime/` or dedicated venv paths managed by setup scripts, not ambient
global Python installations.

## Configuration And Presets

Runtime configuration is governed by structured loaders and validation:

- `config/` contains presets, runtime manifests, model locks, asset manifests,
  budgets, and environment-independent configuration.
- `config/presets/` separates stable, canary, and experimental presets.
- `requirements/lock_ownership.yml` and related validators define dependency
  lock ownership.
- API schemas, machine-mode schemas, run-card schemas, and contract snapshots
  are checked by focused tests and validation scripts.

Avoid ad hoc `yaml.safe_load` in new runtime code unless it is routed through
an approved loader boundary. The repo enforces YAML governance separately.

## Dependency And Runtime Governance

Dependency installation is not ad hoc. The governed paths are:

```bash
make install-core
make repair-core-venv
make check-environment
```

The layered lock model separates:

- Core runtime dependencies.
- Development and CI tooling.
- Security/tooling/archive layers.
- Target-owned Darwin arm64 ML locks.
- Optional local runtimes for DA3, Depth Pro, RAW ingest, SAM2, CoreML, and
  FastVLM.

Linux x86_64 and Darwin x86_64 ML lock lanes are retired fail-closed stubs.
Do not revive broad ML umbrella installs without a trusted checked-in lockfile
contract and matching validation updates.

## Security And Governance Boundaries

Security-sensitive flows are centralized and fail closed:

- API key enforcement on protected backend endpoints.
- Trusted host and proxy controls.
- Request size limits, rate limiting, and server-sent-event auth policy.
- Allowed input/output roots and artifact path validation.
- Frontdoor session cookies, Access posture, local user fixtures, and session
  scaling checks.
- RUM telemetry allow-lists, rollout controls, and sink policy tests.
- Dependency pinning, lock ownership, banned dependency checks, and workflow
  governance.

Any change to auth, path handling, telemetry, dependency policy, or artifact
serving must include focused tests and should preserve fail-closed behavior.

## Testing And Validation

Validation is layered by risk:

```bash
make test-fast
make test-orchestrator-contract
make test-frontdoor-contract
make validate-portal-browser
make validate-frontdoor-browser
make ci
```

Additional governance checks include:

```bash
make check-requirements-lock-contract
make check-dependency-pinning
make check-ci-sync
make check-portal-asset-budgets
make check-docs
make check-doc-heading-links
python3 scripts/governance/check_docs_structure.py --all
```

Use focused tests for small changes and broaden to contract/browser/live
validation when touching shared behavior, public routes, selectors, pipeline
dispatch, auth, dependency governance, or artifact handling.

## Extension Rules

When adding or changing architecture:

1. Choose the narrowest package boundary that owns the behavior.
2. Preserve public route shapes, API envelope names, CLI flags, selectors, and
   artifact contracts unless the change explicitly updates them.
3. Keep optional model/runtime dependencies out of the core layer.
4. Add structured validation instead of relying on ambient shell state.
5. Prefer existing loaders, serializers, storage helpers, path guards, and
   response models over one-off parsing or JSON/YAML writes.
6. Update tests, docs, and validation targets in the same patch as behavior
   changes.
7. Keep generated artifacts out of the worktree, or update ignore/governance
   rules when a new generated path becomes normal workflow output.

## Related Documents

- [Portal + Orchestrator Quickstart](../guides/PORTAL_ORCHESTRATOR_QUICKSTART.md)
- [Portal Secure Front Door Quickstart](../guides/PORTAL_SECURE_FRONTDOOR_QUICKSTART.md)
- [Lux Depth V3 CLI Guide](../cli/LUX_DEPTH_V3_CLI_GUIDE.md)
- [ADR-019 Depth Backend Unification](ADR-019-depth-backend-unification.md)
- [ADR-032 Dependency Pinning Strategy](ADR-032-dependency-pinning-strategy.md)
- [Documentation Map](../governance/DOCUMENTATION_MAP.md)
