# ADR-037 Repo Root Contract

## Status
Proposed

## Date
2026-02-26

## Executive Summary

This ADR defines a deterministic repository root contract used by verification
scripts, shell tooling, and CI smoke gates. Root discovery is dual-anchored to
`pyproject.toml` and `.github/workflows/`, fails loudly when anchors are not
found, and does not fall back to `cwd`.

## Context

Recent relocatability hardening moved code to `src/`, relocated verification
scripts under `scripts/verification/`, and introduced `/tmp` absolute-path
execution checks in CI. These changes require a single root discovery contract
to avoid path entropy between local and CI invocation contexts.

Using `.git` as the only anchor is insufficient for this repository contract
because tooling must remain deterministic in checkout/export contexts where
`.git` may be absent while project governance artifacts remain present.

## Decision

### D1. Dual-Anchor Resolution Is Mandatory

Repository root candidates are valid only when both exist:

- `pyproject.toml` (file)
- `.github/workflows/` (directory)

### D2. Resolution Order Is Fixed

Resolver behavior is:

1. If `--repo` is provided, validate anchors at that path and return it.
2. Else walk upward from the provided start path (or module path) until a
   dual-anchor match is found.
3. If no match is found, fail with a non-zero exit and explicit error message.

### D3. Failure Is Loud and Deterministic

Resolver must not:

- fall back to `os.getcwd()`,
- silently pick a partial-anchor directory,
- continue with guessed roots.

### D4. Symlink Semantics

Input paths are canonicalized via `Path.resolve()`. When invoked through a
symlinked checkout path, the resolved physical repository path is authoritative.

### D5. Nested-Repo Semantics

In nested structures where multiple directories satisfy dual anchors, the first
matching directory encountered while walking upward from the start path is
authoritative.

## Consequences

- Python and shell tooling share a deterministic root contract.
- `/tmp` absolute-path invocation remains a stable CI invariant.
- Import surfaces for `src/`-based layouts are less likely to drift silently.
