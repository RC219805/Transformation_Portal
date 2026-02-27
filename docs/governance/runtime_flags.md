# Runtime Flags Governance

This document defines environment flags introduced for CI hardening and runtime safety.

## TimingContext Flags

`src/transformation_portal/metrics/timing.py` uses the following controls:

| Flag | Default | Scope | Effect |
| --- | --- | --- | --- |
| `TP_DISABLE_DEVICE_SYNC` | unset | global | Disables all device synchronization in `TimingContext` for both CUDA and MPS. |
| `TP_TIMING_SYNC_MPS` | unset | MPS only | Enables MPS synchronization when device is `mps`. |

Accepted truthy values are case-insensitive: `1`, `true`, `yes`.

### Precedence

Device sync behavior is resolved in this order:

1. If interpreter shutdown is in progress (`atexit` sentinel), do not synchronize.
2. If `TP_DISABLE_DEVICE_SYNC` is truthy, do not synchronize.
3. If device is not `cuda` or `mps`, do not synchronize.
4. If device is `mps`, synchronize only when `TP_TIMING_SYNC_MPS` is truthy.
5. If device is `cuda`, synchronize when CUDA is available.

`TP_DISABLE_DEVICE_SYNC` always overrides `TP_TIMING_SYNC_MPS`.

## Test Tier Flags

Reconstruction test budgets use the following controls:

| Flag | Default | Effect |
| --- | --- | --- |
| `CI` | unset | Runs contract-oriented bounded workloads in CI mode. |
| `TP_LONG_TESTS` | unset | Opts into heavy semantic runs intended for local/manual validation. |

`TP_LONG_TESTS` is an explicit opt-in and should be used only for intentional long-running validation.
