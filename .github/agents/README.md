# Transformation Portal Custom Agents

This directory contains specialized GitHub Copilot agents for the Transformation Portal repository.

## Available Agents

### Transformation Portal Specialist

**File**: `transformation-portal-specialist.md`

**Purpose**: Execution-focused implementation and troubleshooting agent for the repository's governed operational surfaces.

The Specialist is no longer scoped only to rendering pipelines. It now works across:

- Lux Depth V3 orchestration and governed media-processing workflows
- portal/orchestrator HTTP and UI surfaces
- archive-gate execution paths and command allowlists
- ingest, machine-mode, provenance, and evidence-adjacent tooling
- layered dependency and install flows
- targeted tests, docs, and developer tooling inside existing governance boundaries

The Specialist implements inside Architect-owned constraints. It should escalate when work touches dependency policy, CI/CD policy, security posture, public contract changes, or ambiguous ADR-bound trade-offs.

## How To Use The Specialist

Use `@transformation-portal-specialist` when you need repository-grounded execution help, for example:

```text
@transformation-portal-specialist update the orchestrator contract tests for a new /v1/jobs response field
```

```text
@transformation-portal-specialist debug why Lux Depth V3 run cards are failing validation after a config change
```

```text
@transformation-portal-specialist trace why machine-mode JSON is missing a required tp.meta.machine.v1 field
```

```text
@transformation-portal-specialist identify the smallest validation command set for this app.py change
```

The Specialist should ground answers in real repository files, tests, docs, and commands. It should prefer extending current modules and workflows over inventing parallel ones.

## Scope Boundaries

Use the Specialist for:

- Lux Depth V3 implementation and troubleshooting
- portal/orchestrator fixes within current `/ready` and `/v1/*` contracts
- archive-gate and machine-mode bug fixing
- ingest/provenance implementation work
- targeted docs and tests that need to stay in lockstep with current behavior

Escalate to the Architect for:

- `pyproject.toml`, `requirements/`, lockfile policy, banned dependencies, or new model/runtime choices
- `.github/workflows/*`, CI Gate composition, release automation, or deployment behavior
- security-sensitive trust-boundary changes
- public CLI/HTTP/schema compatibility changes
- ADR conflicts or architectural trade-offs that need explicit direction

## Supporting Documents

The Specialist brief is complemented by:

- `QUICK_START_v2.md` for prompt patterns and usage guidance
- `RAG_IMPLEMENTATION_SUMMARY.md` for the repository-retrieval support layer and its current role
- `docs/guides/CUSTOM_AGENT_GUIDE.md` for general custom-agent usage patterns in this repo

When these documents drift, the specialist brief itself is authoritative for role scope and escalation boundaries.

## Maintenance Notes

Update the Specialist materials when the repository changes in ways that affect:

- governed operational surfaces
- escalation boundaries
- canonical validation entrypoints
- machine-mode or ingest contract references
- portal/orchestrator or Lux Depth V3 workflow expectations

Keep the brief, these supporting docs, and `tests/test_custom_agent_config.py` aligned.
