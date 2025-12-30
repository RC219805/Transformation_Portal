# Contributing

## Core Stability Policy 🎯

### lux_depth_v2 Feature Freeze

**Status**: ❄️ **ACTIVE** (December 23, 2025 - March 1, 2026)
**Scope**: `lux_depth_v2/` module only
**Reason**: Golden Path consolidation and production stability

#### What's Allowed

✅ **ALLOWED**:
- Security fixes (CVE remediation, vulnerability patches)
- Bug fixes (correctness, crashes, memory leaks)
- Performance optimizations (no behavior changes)
- Documentation improvements
- Test coverage enhancements

🚫 **BLOCKED**:
- New features (presets, parameters, processing stages)
- Behavior changes (modified defaults, altered pipeline)
- Experimental integrations
- Breaking changes

#### Exception Process

For **critical production needs** that violate the freeze:

1. Open GitHub issue with `freeze-exception` label
2. Provide justification (business impact, risk assessment, rollback plan)
3. Require Architect review and approval
4. Document decision in issue

**Approval criteria**: Security vulnerability (CVSS ≥7.0), production blocker, data loss risk, regulatory compliance.

📚 **Full Policy**: [lux_depth_v2/FEATURE_FREEZE.md](lux_depth_v2/FEATURE_FREEZE.md)
📊 **Stability Metrics**: [docs/architecture/STABILITY_POLICY.md](docs/architecture/STABILITY_POLICY.md)

---

### Repository-Wide Guidelines

While **lux_depth_v2** is frozen, other modules follow standard development practices:

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
