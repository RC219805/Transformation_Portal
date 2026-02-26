# ADR-039 Branch Staleness and Selective Integration Policy

## Status
Proposed

## Date
2026-02-26

## Executive Summary

This ADR defines repository policy for integrating long-lived branches after
major architectural phase changes. Such branches must not be merged wholesale
without a branch-delta audit against current `main`.

If audit stop conditions are triggered (for example, Phase 4/schema/contract
surface deletions), the branch is rejected as a merge candidate and treated as
reference-only input. Integration must proceed via selective extraction or a
clean reimplementation on top of current `main`.

This policy formalizes current governance discipline: preserve deterministic and
contract surfaces first, then admit forward-safe additive deltas.

## Context

The repository has accumulated hard governance boundaries across:

- Phase 4 provenance/capture surfaces,
- machine-mode schema contracts,
- ingest contract validation workflows,
- determinism and artifact-boundary enforcement,
- relocatability and editable-install proofing.

Long-lived branches created before these boundaries may appear to contain small
feature intent while actually carrying broad architectural rollback when diffed
against current `main`.

This creates repeated operational risk:

- accidental reintroduction of deleted/legacy behavior,
- contract and schema regressions hidden inside stale branches,
- CI destabilization and reduced bisectability.

## Decision

### D1. Baseline Tagging Is Required Before Phase-Boundary Audits

Before classifying a long-lived branch, create a reproducible baseline anchor:

1. sync `main` with fast-forward-only semantics,
2. create and push an annotated baseline tag.

This provides audit reproducibility and rollback clarity.

### D2. Whitespace-Insensitive File-Level Diff Classification Is Mandatory

Audit must include, at minimum:

- `git diff --stat main..<candidate_branch>`
- `git diff --name-status main..<candidate_branch>`

Classification is file-scope first, commit-scope second.

### D3. Stop Conditions Trigger Immediate Wholesale-Merge Rejection

If any of the following appear in branch delta, wholesale merge is prohibited:

- deletion/modification of `schemas/phase4/*` that weakens contract surfaces,
- deletion/modification of `schemas/tp.meta.machine.v1/*` or equivalent
  machine-mode contract schema surfaces,
- deletion/modification of `src/tp/phase4/*`,
- weakening/removal of determinism/contract enforcement workflows,
- broad test deletions outside the intended feature scope,
- dependency downgrades or policy-surface regressions.

When triggered, branch disposition is:

- `mergeable: false` (policy),
- `reference_only: true`,
- proceed with selective extraction or clean reimplementation.

### D4. Selective Integration Is the Default for Stale Branches

Accepted paths for stale branch intent:

1. selective extraction of forward-safe deltas via curated patch/cherry-pick,
2. clean feature reimplementation on top of current `main`.

Direct merge and blind cherry-pick are disallowed when stop conditions are hit.

### D5. Integration Must Preserve Governance Surfaces

Selective integration must preserve, without weakening:

- Phase 4 provenance/capture surfaces,
- machine-mode schema contracts,
- ingest contract validation workflow,
- determinism and artifact boundary enforcement,
- relocatability and editable-install proof checks.

### D6. Archive Namespace Policy for Stale Branches

Rejected stale branches must be archived to remove merge temptation while
preserving history:

- rename local branch under `archive/*`,
- remove legacy remote branch ref when present,
- push archived ref to origin.

## Enforcement

Policy enforcement is procedural plus CI-backed:

- diff classification and stop-condition checks in review workflow,
- baseline tagging before phase-boundary integration work,
- CI gates remain authoritative for determinism, contracts, and boundaries.

Any exception requires a superseding ADR with migration rationale.

## Consequences

Positive:

- prevents architectural rollback from stale branches,
- protects contract and determinism surfaces,
- improves bisectability and audit traceability,
- reduces accidental merge risk through archive namespace discipline.

Trade-offs:

- additional upfront audit time for long-lived branches,
- some feature intents require clean reimplementation instead of quick merge.

## References

- `pre-phase3.7-audit` (baseline tag)
- `archive/phase3.7-pre-phase4` (archived stale branch namespace)
- `docs/architecture/ADR-035-bundle-root-anchoring-invariants.md`
- `docs/architecture/ADR-037-repo-root-contract.md`
- `docs/architecture/ADR-038-operational-determinism-enforcement-layer.md`
