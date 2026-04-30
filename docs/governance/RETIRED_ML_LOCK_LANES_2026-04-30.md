# Retired ML Lock Lanes - 2026-04-30

## Decision

The checked-in ML requirements contract now has one supported target-owned
lock lane: Apple Silicon (`requirements/ml-core-darwin-arm64.txt`).

The historical Linux x86_64 and macOS Intel ML lanes are retired unsupported
lanes. They must not exist as installable `requirements/*.in` or
`requirements/*.txt` manifests because GitHub dependency graph scanners treat
those files as live dependency metadata even when the repository policy marks
the lanes unsupported.

## Rationale

Dependabot alerts for PyTorch and Transformers were still attributed to
`requirements/all.txt` after the supported surfaces were remediated. Local
inspection showed `requirements/all.txt` did not contain PyTorch or
Transformers. The remaining graph entries came from static scanning of retired
ML lockfiles.

To close those alerts without promoting unsupported targets, the repository
keeps historical lane context here in governance documentation and removes the
retired manifests from scan-visible pip requirement files.

## Current Policy

- Apple Silicon is the only checked-in supported ML lock lane.
- Linux and macOS Intel ML install paths fail closed with explicit unsupported
  lane messaging.
- New Linux or macOS Intel ML support requires a separate governed lane
  decision, secure PyTorch baseline, lock ownership entry, tests, and CI
  validation.
- Runtime checkpoint hardening with `weights_only=True` remains mandatory
  defense in depth, but it is not the vulnerability fix for unsupported old
  PyTorch baselines.
