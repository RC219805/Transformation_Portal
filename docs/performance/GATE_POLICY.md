# Performance Gate Policy

**Status:** Current policy
**Created:** 2026-05-30
**Owner:** Repository Architect

This policy defines which performance signals are advisory, nightly-blocking,
or PR-blocking. It aligns the performance ledger with ADR-034: wall-clock
benchmarks stay out of fast PR gates unless they are deterministic enough to
run as bounded smoke checks.

## Enforcement Tiers

| Tier | Scope | Enforcement | Marker / command |
| --- | --- | --- | --- |
| PR-blocking | Deterministic contract and budget checks that do not depend on host speed | Required local/CI validation | `make check-portal-asset-budgets`, schema and contract tests |
| Nightly-blocking | Benchmarks with committed baselines and controlled variance | Scheduled or manual deep gate; failures require triage before baseline updates | `pytest -m benchmark`, `tools/performance_ledger.py` comparisons |
| Advisory | Exploratory profiling, hardware-specific ML timing, and one-off reports | Evidence only; never used alone to block a merge | `docs/performance/*` reports and local profiler output |

## Current PR-Blocking Signals

- Frontdoor and portal asset size budgets.
- Schema and manifest compatibility for emitted performance metadata.
- Deterministic logic tests for performance helpers, such as parser,
  percentile, and report-format contracts.

## Current Nightly-Blocking Signals

- SAM2 auto-mode latency and memory baselines when the checkpoint and matching
  hardware/device lane are available.
- Reconstruction and PBR performance budgets that carry `benchmark` or `slow`
  markers and compare against a committed baseline.
- Performance ledger baseline comparisons with explicit backend, device, and
  quality-tier metadata.
- `.github/workflows/performance-monitor.yml` is schedule/manual only and fails
  when pytest-benchmark reports a regression beyond its documented threshold
  or when benchmark execution fails and the run is not valid evidence.

## Advisory Signals

- Local developer timing on unpinned hardware.
- ML sampled coverage artifacts that include runtime measurements but are used
  as cold-zone evidence rather than a merge gate.
- Reports generated from partial image sets or experimental model variants.

## Baseline Update Rules

- Baselines are committed evidence, not auto-updated generated files.
- Baseline changes require a PR review that names the hardware/device lane,
  backend, model id, sample count, and reason for the change.
- A performance regression should be fixed or explicitly accepted before a
  baseline is raised.
- Environment failures such as missing checkpoints, unavailable accelerators,
  Docker unavailability, or service startup failures are reported as
  tooling/environment blockers, not product-green evidence.

## Workflow Alignment

- Fast PR workflows should select deterministic tests and skip `benchmark` and
  host-speed-sensitive `slow` tests.
- Nightly/deep workflows may run benchmark-marked tests and performance-ledger
  comparisons when their prerequisites are present.
- New performance tests must choose a tier at introduction time and document
  the exact command that owns the signal.
