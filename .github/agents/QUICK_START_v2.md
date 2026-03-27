# Transformation Portal Specialist Agent - Quick Start

**Status**: Ready to Use
**Scope**: Governed execution inside current repository contracts

---

## What The Specialist Is For

Use `@transformation-portal-specialist` for implementation and troubleshooting work across the repository's active execution surfaces:

- Lux Depth V3
- portal/orchestrator routes and runtime behavior
- archive-gate flows
- machine-mode and ingest tooling
- targeted docs, tests, and developer tooling updates

The Specialist is retrieval-first and contract-aware. It should ground answers in current repository files, tests, docs, and commands before recommending merge-ready changes.

## What The Specialist Is Not For

Do not use the Specialist as the final authority for:

- dependency policy changes
- CI/CD policy or branch-protection changes
- security posture changes
- public interface changes across CLI, HTTP, schemas, or packaging
- ADR interpretation conflicts

Those require escalation to the Architect.

## Fast Prompt Patterns

### Lux Depth V3

```text
@transformation-portal-specialist update Lux Depth V3 validation so run-card errors include the new field name
```

```text
@transformation-portal-specialist identify the smallest test set for changes in src/transformation_portal/lux_depth_v3/pipeline_coordinator.py
```

### Portal / Orchestrator

```text
@transformation-portal-specialist debug why /v1/jobs/{id}/events is missing the final done event in contract tests
```

```text
@transformation-portal-specialist update app.py validation without changing the /ready or typed /v1/* response contracts
```

### Archive Gates

```text
@transformation-portal-specialist trace why archive-gate-b is rejecting an allowlisted command payload
```

### Machine Mode / Ingest

```text
@transformation-portal-specialist fix this tp.meta.machine.v1 payload drift and update the matching tests
```

```text
@transformation-portal-specialist review this ingest contract change and list the docs, schemas, and tests that must move together
```

## Good Specialist Requests

Ask for:

- the smallest safe code change inside an existing contract
- exact files, tests, and commands relevant to a change
- debugging help grounded in repository evidence
- validation guidance that matches current repo entrypoints
- docs and tests that need to change with behavior

## Requests That Should Escalate

Escalate instead of implementing when the work touches:

- `pyproject.toml`, `requirements/`, lockfiles, dependency bans, or new models/runtimes
- `.github/workflows/*`, CI Gate composition, release automation, or deployment behavior
- security-sensitive input handling or trust-boundary changes
- public CLI/HTTP/schema/import-surface changes
- ADR ambiguity or architectural trade-offs

## Validation Defaults

Prefer the repo's existing entrypoints before inventing ad hoc command sets:

- `make test-fast`
- `make test-orchestrator-contract`
- `make check-test-markers`
- `make ci`
- `lux-depth-v3 --help`

Use the smallest command set that proves the change. If a broader command is needed, say why.

## Working Style

The best Specialist responses:

1. name the active repository surface
2. cite real files, tests, docs, or commands
3. explain contract or compatibility consequences explicitly
4. keep fixes narrow and reviewable
5. escalate cleanly when the task crosses Architect boundaries

## Related Documents

- `transformation-portal-specialist.md`
- `README.md`
- `RAG_IMPLEMENTATION_SUMMARY.md`
- `docs/guides/CUSTOM_AGENT_GUIDE.md`
