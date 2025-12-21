# Contributing

## Feature Freeze Period ❄️

**Active**: December 20, 2025 - January 10, 2026  
**Reason**: Golden Path consolidation and validation

### What's Allowed During Freeze

✅ **ALLOWED**:
- Bug fixes (correctness issues)
- Security fixes
- Documentation improvements
- Test improvements
- Performance optimizations (no behavior changes)

🚫 **BLOCKED**:
- New features
- Breaking changes
- Refactoring (non-critical)
- Experimental pipelines

📋 **PROCESS**: All changes require feature freeze check via issue template  
📚 **Policy**: See [docs/FEATURE_FREEZE_POLICY.md](docs/FEATURE_FREEZE_POLICY.md)

---

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
