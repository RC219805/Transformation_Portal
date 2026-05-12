# Test Run Summary

Last updated: 2026-05-12

## Proven Green

- `make test-fast`
  - Result: 77 passed in 2.91s
- `.venv/bin/python -m pytest --collect-only -q tests`
  - Result: 9971 tests collected in 2.95s

## Incomplete

- `make test-novideo`
  - Result: intentionally stopped during discovery because the broad non-video suite was too slow for this pass.
  - This is not counted as a green full-suite result.
