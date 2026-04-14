# Local Validation Quickstart

This guide covers the canonical workflow for validating the Transformation Portal locally before committing changes.

## Prerequisites

### Required Tools

| Tool | Version | Purpose |
|------|---------|---------|
| Python | ≥3.11 | Backend, tests, validation scripts |
| Node.js | 22.x | Managed frontdoor (secure-landing) |
| Chrome/Chromium | Latest | Browser smoke tests |

### Quick Environment Check

Run the pre-flight check to verify your environment:

```bash
make check-environment
```

This validates:
- Python version (≥3.11)
- Node.js version (22.x required for frontdoor)
- Chrome/Chromium availability
- Port availability (3000, 8000)
- Frontdoor npm dependencies

### Node.js Version (Important)

The managed frontdoor requires **Node.js 22.x** specifically. Native dependencies (`better-sqlite3`, `argon2`) are ABI-sensitive.

```bash
# Check current version
node --version

# Switch using nvm
nvm use 22

# Switch using fnm
fnm use 22

# Switch using volta
volta install node@22
```

See `web/secure-landing/.nvmrc` and `package.json` engines constraint.

## Validation Sequence

### Option 1: Full Validation Suite (Recommended)

Run all validations in the correct order:

```bash
./scripts/validation/run_full_validation_suite.sh
```

This runs:
1. Environment pre-flight checks
2. Fast Python tests
3. Orchestrator contract tests
4. Frontdoor contract tests
5. Portal browser smoke
6. Frontdoor browser smoke

### Option 2: Quick Validation

Skip browser smokes for faster iteration:

```bash
./scripts/validation/run_full_validation_suite.sh --quick
```

Or skip frontdoor validation entirely:

```bash
./scripts/validation/run_full_validation_suite.sh --skip-frontdoor
```

### Option 3: Individual Make Targets

Run specific validation steps:

```bash
# Python tests only
make test-fast

# Orchestrator contract tests
make test-orchestrator-contract

# Frontdoor contract tests (requires Node 22.x)
make test-frontdoor-contract

# Portal browser smoke
make validate-portal-browser

# Frontdoor browser smoke
make validate-frontdoor-browser
```

## Common Workflows

### Before Committing Changes

```bash
# Quick local CI check
make ci

# Or full validation
./scripts/validation/run_full_validation_suite.sh
```

### After Modifying Portal UI

```bash
# Test portal contracts
make test-portal-contract

# Run browser smoke
make validate-portal-browser
```

### After Modifying Frontdoor

```bash
# Ensure Node 22.x
./scripts/setup/ensure_node_version.sh

# Test frontdoor contracts
make test-frontdoor-contract

# Run browser smoke
make validate-frontdoor-browser
```

### After Modifying Backend APIs

```bash
# Test orchestrator HTTP contracts
make test-orchestrator-http-contract

# Test full orchestrator contracts
make test-orchestrator-contract
```

## Environment Variables

The validation scripts use sensible defaults, but you can override them:

| Variable | Default | Purpose |
|----------|---------|---------|
| `TP_API_KEY` | `contract-secret` | API key for protected endpoints |
| `TP_FRONTDOOR_USERS_FILE` | `/tmp/tp-frontdoor-users.json` | Seeded user credentials file |
| `TP_FRONTDOOR_USERNAME` | `smoke-admin` | Local smoke test username |
| `TP_FRONTDOOR_PASSWORD` | `correct horse battery staple` | Local smoke test password |
| `TP_PORTAL_BROWSER_BINARY` | auto-detect | Chrome/Chromium binary path |

### Seeding Frontdoor Users

For frontdoor browser validation, credentials must be seeded:

```bash
# Auto-seeds with defaults
make seed-frontdoor-user

# Or manually
cd web/secure-landing
node ./scripts/seed-frontdoor-user.mjs \
  --output /tmp/tp-frontdoor-users.json \
  --username smoke-admin \
  --password "correct horse battery staple"
```

## Troubleshooting

### Node Version Mismatch

```
Error: Node.js v20.x.x does not match required 22.x
```

**Solution:** Switch to Node 22.x using your version manager.

### Port Already in Use

```
Port 3000 is in use (managed frontdoor)
```

**Solution:** Stop the existing process:
```bash
# Find what's using the port
lsof -i :3000

# Or kill by port
kill $(lsof -t -i :3000)
```

### Missing npm Dependencies

```
node_modules not found in web/secure-landing/
```

**Solution:** Install dependencies:
```bash
cd web/secure-landing
npm install
```

### Chrome Not Found

```
Chrome or Chromium not found
```

**Solution:** Install Chrome or set the binary path:
```bash
export TP_PORTAL_BROWSER_BINARY="/path/to/chrome"
```

### Python Snippet Pasted Into zsh

If you need to run an ad hoc Python snippet, execute it through the repo interpreter instead of pasting raw Python into the shell:

```bash
.venv/bin/python - <<'PY'
print("run Python here")
PY
```

### Repo venv Drift or Wrong Interpreter

If pre-flight reports the wrong interpreter or `pip check` dependency conflicts, rebuild the core repo environment:

```bash
make repair-core-venv
make check-environment
```

If the conflict mentions `depth-anything-3`, keep DA3 isolated and reinstall it separately:

```bash
./scripts/setup/install_da3_runtime.sh
```

### Build Dirties Worktree

If `make test-frontdoor-contract` or other builds dirty the worktree:

```bash
# Check what changed
./scripts/validation/check_worktree_clean.sh --show-diff

# Ensure .next-build-verify is ignored
grep -q ".next-build-verify" .gitignore
```

The frontdoor build should use `TP_NEXT_DIST_DIR=.next-build-verify` for isolation.

## Quick Reference

```bash
# Pre-flight check
make check-environment

# Node version check
./scripts/setup/ensure_node_version.sh

# Full validation suite
./scripts/validation/run_full_validation_suite.sh

# Quick validation (no browser)
./scripts/validation/run_full_validation_suite.sh --quick

# Local CI
make ci

# Check worktree clean
./scripts/validation/check_worktree_clean.sh
```

## See Also

- [AGENTS.md](../../AGENTS.md) - Command reference
- [Portal Orchestrator Quickstart](PORTAL_ORCHESTRATOR_QUICKSTART.md) - Backend development
- [Portal Secure Frontdoor Quickstart](PORTAL_SECURE_FRONTDOOR_QUICKSTART.md) - Frontdoor development
