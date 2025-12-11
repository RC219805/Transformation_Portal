# Contributing

## Development principles
- Keep changes **additive** unless a breaking change is explicitly approved.
- Prefer **small PRs** with clear scope.
- Include tests for correctness changes.
- Keep security posture intact: do not introduce banned dependencies in requirements.

## Workspace cleanup

Before disk-intensive operations or when rotating benchmark outputs:

```bash
# Preview what will be cleaned
make clean-dry

# Clean all workspace artifacts
make clean
```

This removes:
- Test logs and benchmark outputs (`*.log`, `benchmarks_*/`, `output_*/`)
- Python cache files (`__pycache__`, `*.pyc`, `.pytest_cache`)
- Build artifacts (`build/`, `dist/`, `*.egg-info`)
- Temporary reports and system files

**Safety guarantee:** The cleanup tool protects tracked files and never touches `.venv`, `weights/`, or `.git`. Even if a file matches a cleanup pattern (e.g., `debug.log`), it will be skipped if it's tracked by git or in an excluded directory.

## PR checklist
- [ ] Code compiles/tests pass
- [ ] No new high-risk dependencies
- [ ] Docs updated if behavior changes
- [ ] Reproducibility metadata preserved where applicable

## Reporting security issues
Use `SECURITY.md` (if present) or open a private advisory if enabled.
