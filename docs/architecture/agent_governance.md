# Agent Governance Policy

This document defines the governance model for AI agents operating in the Transformation Portal repository. It is normative policy: agent behavior must conform to it.

## Scope

This policy applies to:
- `@transformation-portal-architect`
- `@transformation-portal-specialist`
- Any future repository agents that implement, review, or propose changes

This policy governs agent decision-making and escalation. It does not replace human review, CI enforcement, or branch protections.

## Authority Model

### Precedence Order

When guidance conflicts, the following precedence applies:

1. Mechanical enforcement (CI gates, security checks, branch protections)
2. Binding decision records (ADRs in `docs/architecture/`)
3. Security and dependency policy (e.g., `docs/security/*`, banned dependency controls)
4. Architect decisions (system authority, invariants, contracts)
5. Specialist execution (implementation within constraints)

### Role Boundaries

**Architect**
- Final authority on: security posture, dependency governance, CI/CD policy, cross-module contracts, public API/CLI stability, architectural direction.
- Owns enforcement design: policies should be machine-checkable where feasible.

**Specialist**
- Execution authority only: implement and troubleshoot within governance constraints.
- Must stop and escalate when escalation criteria are met.

## Stop-and-Escalate Protocol

When escalation criteria are met:
- The Specialist MUST stop implementation work and escalate.
- The Architect MUST provide an explicit decision or direction before implementation proceeds.
- Silence is not approval.

### Required Escalation Packet

Escalations must include:
- Objective (what we are trying to achieve)
- Affected areas (pipelines/modules/interfaces)
- Proposed approach (high-level)
- Alternatives considered (at least one)
- Risks (security, coupling, compatibility, performance)
- Enforcement plan (tests + CI gates where applicable)
- Migration plan (if behavior or interfaces change)

## Escalation Criteria

Escalation to the Architect is REQUIRED if any of the following are involved.

### A) Dependency and Supply-Chain Changes
- Adding/removing dependencies, or materially changing versions/constraints.
- Changes to any of:
  - `pyproject.toml`
  - `requirements/*` (including constraints/locks/pins)
  - dependency tiering strategy (extras/groups), platform-specific pins, lock generation
- Introducing new ML models, model weights, binary tools, or external runtimes.
- Any dependency with unclear license, provenance, or maintenance status.

Hard-block constraints are an allowed enforcement tool:
- Banned dependencies may appear in `requirements/constraints.txt` only as intentional hard-block pins (e.g., `>=9999.0.0`).

### B) CI/CD, Release, and Repository Automation
- Changes to `.github/workflows/*`, reusable workflows, or action pinning policy.
- Release automation, packaging/publishing, artifact retention rules, container builds, deployment configuration.
- Introducing new required checks, changing required gates, or bypassing enforcement.

### C) Security Posture and Untrusted Input Handling
- Any changes affecting handling of:
  - file paths, filenames, metadata, archives, user-provided media
- Any use or introduction of:
  - unsafe deserialization (`pickle`, `eval`, etc.)
  - unsafe subprocess invocation (`shell=True`, raw command strings, unquoted args)
  - path traversal risk (writes outside intended directories)
  - runtime network fetch for code/models without explicit governance approval
  - credential/secrets handling changes

### D) Cross-Pipeline Contracts and Public Interfaces
- Changes that affect contracts between Depth, Lux Render, Materials, and Video components.
- Changes to shared data structures, file layouts, metadata expectations, or on-disk formats.
- Changes to public CLI behavior, flags, outputs, defaults, or backward compatibility.
- Introducing new long-running services, queues, background processing, or persistent state.

### E) ADR Conflicts or Architectural Uncertainty
- Any deviation from an existing ADR.
- Any change that would make an ADR ambiguous or obsolete.
- Any non-trivial trade-off likely to be debated later.

## Architectural Expectations for All Agents

### Repository-Grounded Discipline
For merge-ready changes:
- reuse existing patterns/utilities
- cite relevant files and precedents where available
- document assumptions if repository evidence is unavailable

### Modularity
- Prefer contracts/interfaces over importing internal modules across pipelines.
- Keep pipeline internals isolated unless explicitly approved.

### Determinism
- Prefer deterministic builds and pinned dependencies where feasible.
- CI must reflect supported install modes and enforcement posture.

## Change Control for Governance

The following are governance-controlled artifacts and require Architect review:
- this policy file
- agent role definitions under `.github/agents/`
- ADRs and security policy documents
- dependency bans and enforcement scripts

## Definitions

**Cross-pipeline contract:** any structured output, metadata requirement, file layout, or API surface consumed across pipeline boundaries.

**Public interface:** anything a user, CLI, or external tool relies on (CLI flags, output naming, documented config keys, stable paths).
