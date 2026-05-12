# Test Run Summary

Last updated: 2026-05-11 23:47:19 PDT

## Proven Green

- `make test-fast`
  - Result: 77 passed in 2.51s
- `.venv/bin/python -m pytest --collect-only -q tests`
  - Result: 10289 tests collected in 4.92s

## Incomplete

- `make test-novideo`
  - Result: intentionally stopped during discovery because the broad non-video suite was too slow for this pass.
  - This is not counted as a green full-suite result.
