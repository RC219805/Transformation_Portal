# Quick Start: Quality Control System

Get up and running with the Transformation Portal quality control system in 2 minutes.

## One-Time Setup

```bash
# 1. Install git pre-commit hook
make install-hooks

# 2. Verify installation
ls -la .git/hooks/pre-commit

# 3. Test the system
make quality-check
```

That's it! The pre-commit hook will now run automatically before every commit.

## Daily Usage

### Before Committing

```bash
# Auto-fix common issues
make fix-quality

# Stage your changes
git add .

# Commit (pre-commit hook runs automatically)
git commit -m "your message"
```

### Before Pushing

```bash
# Run full CI simulation
make ci-full

# Or quick check (faster)
make ci

# Push when all checks pass
git push
```

## Common Commands

```bash
# Quick Checks
make ci              # Fast checks (30 seconds)
make pre-commit      # Pre-commit checks only
make lint            # Linting only

# Full Checks
make ci-full         # Full CI simulation (2-5 minutes)
make quality-check   # All quality validations

# Auto-Fix
make fix-quality     # Fix all issues automatically
make check-quality   # Preview what would be fixed

# Documentation
make check-docs      # Check if docs need organizing
make organize-docs   # Organize markdown files

# CI Configuration
make validate-ci     # Validate GitHub Actions workflows
```

## What Gets Checked

✅ **Flake8** - Critical errors (undefined variables, imports, syntax)
✅ **Pylint** - Code quality (warnings non-blocking)
✅ **Tests** - pytest suite
✅ **Documentation** - Markdown file organization
✅ **Whitespace** - Trailing whitespace (auto-fixes)
✅ **Syntax** - Python compilation
✅ **Debugging** - No debugging statements in commits

## Troubleshooting

### Pre-commit hook not running
```bash
# Reinstall
make install-hooks
chmod +x .git/hooks/pre-commit
```

### Tests failing
```bash
# Run locally
make test-fast

# Full test suite
make test-full
```

### Flake8 errors
```bash
# See exact errors
flake8 . --select=E9,F63,F7,F82 --show-source

# Most common: F82 (undefined name)
# Fix: Add missing import or fix typo
```

### Too many markdown files (> 10)
```bash
# Preview
make check-docs

# Organize automatically
make organize-docs
```

## Need More Help?

- **Full Documentation**: [scripts/README_QUALITY_CONTROL.md](README_QUALITY_CONTROL.md)
- **CI Configuration**: [.github/workflows/build.yml](../.github/workflows/build.yml)
- **Test Status**: [tests/TEST_STATUS.md](../tests/TEST_STATUS.md)

## Pro Tips

1. Run `make install-hooks` once after cloning
2. Use `make ci` frequently during development
3. Run `make ci-full` before creating a PR
4. Let `make fix-quality` handle common issues
5. Check `make help` for all available commands

---

**Questions?** Check the full documentation in [README_QUALITY_CONTROL.md](README_QUALITY_CONTROL.md)
