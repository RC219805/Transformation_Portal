# Fast-Scope Coverage

Use this workflow when you need a fast, repeatable branch-coverage baseline for the audited paths identified in the config and streaming backlog.

## Scope

- `src/transformation_portal/core/config`
- `src/transformation_portal/streaming`

## Preferred Command

```bash
make coverage-fast-scope
```

## Equivalent Direct Invocation

```bash
pytest \
  --cov=src/transformation_portal/core/config \
  --cov=src/transformation_portal/streaming \
  --cov-branch \
  --cov-report=term-missing \
  tests/test_core_config_presets.py \
  tests/test_streaming_async_pipeline.py \
  tests/test_preset_health.py
```

## Notes

- This is intentionally narrower than repo-wide coverage.
- `term-missing` output is required so reviews can immediately see uncovered lines in the audited paths.
- Keep this workflow separate from PR 6 hotspot smoke coverage and from ML-heavy coverage runs.
