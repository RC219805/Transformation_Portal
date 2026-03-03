# Numeric Stability Policy for Reconstruction Tests

This document defines deterministic and numeric-comparison conventions for
`tests/spatial_ai/reconstruction/`.

## Why this exists

Reconstruction tests run across CPU, MPS, and CUDA with different floating-point
characteristics. A single hardcoded tolerance can produce flaky results on one
device and overly permissive checks on another.

To keep tests stable and meaningful:
- Seed all RNGs for deterministic setup.
- Use explicit dtype policy when precision matters.
- Use device-specific tolerance tables for floating-point assertions.

## Shared fixtures

Fixtures live in
`tests/spatial_ai/reconstruction/conftest.py`.

### `seed_all_rngs`

Seeds Python, NumPy, and torch RNGs.

Usage:

```python
def test_example(seed_all_rngs):
    seed_all_rngs(42)
    ...
```

### `torch_dtype_policy`

Provides centralized dtype choices:
- `default`: `torch.float32`
- `high_precision`: `torch.float64`

Usage:

```python
def test_example(torch_dtype_policy):
    dtype = torch_dtype_policy["high_precision"]
```

### `device_tolerance`

Provides device-specific tolerance budgets:
- CPU: `rtol=1e-7`, `atol=1e-9`
- MPS: `rtol=1e-5`, `atol=1e-7`
- CUDA: `rtol=1e-6`, `atol=1e-8`

Usage:

```python
def test_example(device_tolerance):
    tol = device_tolerance.get("cpu", device_tolerance["cpu"])
    assert np.allclose(a, b, rtol=tol["rtol"], atol=tol["atol"])
```

## Guidance

- Prefer fixture-driven tolerances over ad-hoc constants.
- Keep finite-difference checks on tiny fixtures and use higher precision
  (`float64`) where practical.
- If a test is device-sensitive, document the tolerance rationale in the test.
