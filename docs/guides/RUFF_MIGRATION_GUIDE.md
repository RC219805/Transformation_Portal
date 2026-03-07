# Ruff Migration Guide

## Goal: Consolidate Linting Toolchain

Currently using: `flake8`, `pylint`, `isort`, `black`, `autopep8`

**Target**: Migrate to `ruff` for unified, faster linting and formatting.

### Why Ruff?

- **10-100x faster** than existing tools (written in Rust)
- **Replaces multiple tools**: flake8, isort, black-compatible formatting
- **Better error messages** and autofix capabilities
- **Actively maintained** with strong community adoption

---

## Migration Plan

### Phase 1: Audit Current Rules (Week 1)

**Inventory current configuration**:
```bash
# Extract flake8 rules
grep -r "select\|ignore\|exclude" .flake8 setup.cfg pyproject.toml

# Extract pylint rules
cat .pylintrc | grep -A 20 "disable\|enable"

# Extract isort config
cat pyproject.toml | grep -A 10 "\[tool.isort\]"
```

**Document critical rules** that must be preserved (security, correctness, style).

### Phase 2: Install and Configure Ruff (Week 1)

**Add to requirements-lint.txt**:
```
ruff>=0.1.0
```

**Create ruff configuration** in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 127
target-version = "py310"

# Exclude directories
exclude = [
    ".git",
    ".venv",
    "venv",
    "venv_py311",
    "__pycache__",
    "build",
    "dist",
    "deprecated",
    "archive",
]

[tool.ruff.lint]
# Enable rule sets
select = [
    "E",      # pycodestyle errors
    "F",      # pyflakes
    "I",      # isort
    "N",      # pep8-naming
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "SIM",    # flake8-simplify
    "PIE",    # flake8-pie
    "PL",     # pylint
    "RUF",    # ruff-specific rules
    "S",      # bandit security checks (subset)
]

# Ignore specific rules
ignore = [
    "E501",   # line-too-long (handled by formatter)
    "PLR0913", # too-many-arguments (image processing needs it)
    "PLR0912", # too-many-branches
    "PLR2004", # magic-value-comparison
    "S301",   # pickle usage (we assess risk explicitly)
]

# Per-file ignores
[tool.ruff.lint.per-file-ignores]
"tests/*" = ["S101", "PLR2004"]  # Allow assert and magic values in tests

[tool.ruff.lint.isort]
known-first-party = ["transformation_portal", "luxury_tiff_batch_processor"]
section-order = ["future", "standard-library", "third-party", "first-party", "local-folder"]

[tool.ruff.lint.pylint]
max-args = 8
max-branches = 15
max-statements = 60

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "auto"
```

### Phase 3: Parallel Testing (Week 2)

**Run ruff alongside existing tools**:

```bash
# Existing workflow
black --check src/ tests/
isort --check-only src/ tests/
flake8 src/ tests/
pylint src/

# New ruff workflow
ruff check src/ tests/
ruff format --check src/ tests/
```

**Compare results**:
- Document any rules that behave differently
- Adjust ruff config to match critical existing rules
- Create exceptions for known false positives

### Phase 4: Update CI (Week 2)

**Update `.github/workflows/ci.yml`**:

```yaml
- name: Lint with ruff
  run: |
    ruff check src/ tests/ --output-format=github

- name: Check formatting with ruff
  run: |
    ruff format --check src/ tests/
```

**Keep parallel runs** for one sprint:
```yaml
- name: Lint (legacy) - will be removed
  continue-on-error: true
  run: |
    flake8 src/ tests/
    # ... existing checks
```

### Phase 5: Developer Adoption (Week 3)

**Update pre-commit hooks** (if using):

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
```

**Update CONTRIBUTING.md**:
```bash
# Old workflow
black src/ tests/
isort src/ tests/
flake8 src/ tests/

# New workflow
ruff check --fix src/ tests/
ruff format src/ tests/
```

### Phase 6: Remove Legacy Tools (Week 4)

Once ruff is stable:

1. **Remove from requirements-lint.txt**:
   ```diff
   - flake8>=7.0
   - pylint>=3.0
   - isort>=5.13
   - autopep8>=2.0
   + ruff>=0.1.0
   ```

2. **Remove configuration files**:
   ```bash
   rm .flake8 .pylintrc  # If standalone files
   # Remove [tool.pylint], [tool.isort] from pyproject.toml
   ```

3. **Update CI** to use only ruff

4. **Document in changelog**:
   ```markdown
   ### Changed
   - Migrated linting from flake8/pylint/isort to ruff for 10x faster checks
   ```

---

## Quick Reference: Command Mapping

| Old Command | Ruff Equivalent |
|-------------|-----------------|
| `black --check .` | `ruff format --check .` |
| `black .` | `ruff format .` |
| `isort --check .` | `ruff check . --select I` |
| `isort .` | `ruff check . --select I --fix` |
| `flake8 .` | `ruff check .` |
| `pylint src/` | `ruff check src/ --select PL` |
| All combined | `ruff check . && ruff format .` |

---

## Benefits After Migration

- **Faster CI**: Lint step runs in seconds instead of minutes
- **Simpler toolchain**: One tool instead of five
- **Better DX**: Clearer error messages and autofix
- **Easier maintenance**: Single configuration block
- **Modern rules**: Auto-updates to latest Python best practices

---

## Rollback Plan

If ruff causes issues:

1. **Revert CI changes**: Remove ruff, restore old tools
2. **Keep configuration**: Don't delete `[tool.ruff]` yet
3. **File issues**: Report problems to ruff project
4. **Re-evaluate**: Schedule retry in next quarter

---

## Success Criteria

- [ ] Ruff configuration matches critical rules from flake8/pylint
- [ ] All code passes `ruff check` and `ruff format --check`
- [ ] CI runs successfully with ruff
- [ ] Developers can run `ruff check --fix` locally
- [ ] Legacy tools removed from dependencies
- [ ] Documentation updated
- [ ] No regression in code quality enforcement

---

## Timeline

- **Week 1**: Audit + configure ruff
- **Week 2**: Parallel testing in CI
- **Week 3**: Developer adoption + training
- **Week 4**: Remove legacy tools

**Owner**: DevOps/Platform team
**Reviewer**: Architect (approval required for enforcement changes)

---

## References

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Ruff Rules Reference](https://docs.astral.sh/ruff/rules/)
- [Migration from flake8](https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-flake8)
- [Migration from pylint](https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-pylint)

---

**Note**: This is a proposed migration plan, not yet executed. Approval from Architect required before implementation.
