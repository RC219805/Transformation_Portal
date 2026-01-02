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

## Quick Start for Contributors

### 1. Set Up Development Environment

```bash
# Clone repository
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (automatic code quality)
pre-commit install

# Verify setup
make test-fast
```

### 2. Development Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, write tests
# ...

# Run quality checks
make lint              # Linting
make test-fast         # Quick tests
make security-quick    # Security scan

# Or run all checks at once
make ci                # Lint + test + security

# Check CI/CD health before committing
make workflow-health
```

### 3. Pre-Commit Hooks

Pre-commit hooks run automatically before each commit to ensure code quality:

- **Security checks**: Blocks sensitive files, large artifacts
- **Code formatting**: Ruff auto-formats Python code
- **Syntax validation**: Checks Python AST, YAML syntax
- **File hygiene**: Trailing whitespace, EOF newlines

**To run manually:**
```bash
pre-commit run --all-files
```

**To skip (emergency only):**
```bash
git commit --no-verify -m "Emergency fix"
```

### 4. CI/CD Monitoring

Monitor workflow health to catch issues early:

```bash
# Check all workflows
make workflow-health

# Check specific workflow
python scripts/workflow_health_check.py --workflow ci-consolidated.yml

# Get JSON output for automation
python scripts/workflow_health_check.py --json
```

**Healthy workflows**: Success rate ≥90%
**Action required**: Success rate <70%

See [CI/CD Monitoring Guide](docs/CI_CD_MONITORING.md) for details.

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
- [ ] Security checks pass (pre-commit hooks)
- [ ] No sensitive files committed (history, credentials, client data)
- [ ] Large files use Git LFS or are properly excluded

## Artifact Management

### What NOT to Commit

**NEVER commit these to the repository:**
- Shell history files (`.bash_history`, `.zsh_history`)
- Credentials (`.pem`, `.key`, `.env`, `id_rsa`)
- Build artifacts (`PKG-INFO`, `MANIFEST` at root)
- Client-specific data or identifiers
- Processing outputs (`output/`, `*_outputs/`)
- Large binaries >5MB (use Git LFS instead)
- Temporary files and logs

**Rationale**: Security, repository hygiene, and scalability.

### Proper Artifact Storage

**Local Processing Outputs**:
```bash
# Outputs go to ignored directories
lux-depth-v2 --input-dir data/ --output-dir output/

# .gitignore automatically excludes output/
git status  # Will not show output/ directory
```

**Test Fixtures**:
Small (<1MB) validation files can go in `tests/fixtures/`:
```bash
# Copy specific test cases
cp output/sample_small.png tests/fixtures/depth_validation/
```

**Large Assets**:
Use Git LFS for versioned large files:
```bash
# Install Git LFS
git lfs install

# Track large file types
git lfs track "*.pth"
git lfs track "*.safetensors"

# Commit .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for model weights"
```

**Client Deliverables**:
- Process in local workspace
- Store in directories excluded by `.gitignore`
- Deliver via secure channels (encrypted transfer)
- Document workflow (no client data) in `docs/client_workflows/`

### Pre-Commit Security Checks

Before committing, ensure pre-commit hooks are installed:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Manual check (if needed)
pre-commit run --all-files
```

The hooks will automatically block:
- Sensitive files
- Bidirectional Unicode characters
- Large files without LFS
- Output artifacts

### CI Security Gates

All PRs must pass security gates:
- No sensitive files in repository
- No bidirectional Unicode in code
- Proper `.gitignore` coverage
- No large uncommitted binaries

See `.github/workflows/security-gates.yml` for details.

## Repository Governance

For comprehensive security and governance policies, see:
- **[Repository Governance Guide](docs/REPOSITORY_GOVERNANCE.md)**: Complete governance policies
- **[Security Policy](SECURITY.md)**: Security reporting and practices
- **[License](LICENSE)**: Licensing terms and attribution requirements

## Reporting security issues
Use `SECURITY.md` (if present) or open a private advisory if enabled.
