# Repository Governance Guide

## Overview

This document outlines the governance policies and controls for the Transformation Portal repository to maintain security, maintainability, and production-grade quality.

## Security Controls

### Critical Security Gates

The repository enforces the following security controls at multiple layers:

#### 1. Pre-Commit Hooks

**Location**: `.pre-commit-config.yaml`

Automatically prevents commits containing:
- Shell history files (`.bash_history`, `.zsh_history`, etc.)
- Credential files (`.pem`, `.key`, `id_rsa`, etc.)
- Bidirectional Unicode characters in code files
- Large binary files >5MB (should use Git LFS)
- Output artifacts and temporary files

**Installation**:
```bash
pip install pre-commit
pre-commit install
```

**Manual check**:
```bash
pre-commit run --all-files
```

#### 2. CI/CD Security Gates

**Location**: `.github/workflows/security-gates.yml`

Runs on every PR and push to `main`:
- Verifies no sensitive files in repository
- Scans for bidirectional Unicode characters
- Validates `.gitignore` coverage
- Checks for large uncommitted files
- Secret scanning with TruffleHog

#### 3. Continuous Security Monitoring

**Location**: `scripts/security/`

Tools available:
- `pre_commit_security_check.py`: Standalone security checker
- `continuous_security.py`: Comprehensive security audit system

**Usage**:
```bash
# Quick security check
make security-quick

# Full security audit
make security-full

# Comprehensive health report
make security-audit
```

### Sensitive File Categories

The following file types are **NEVER** allowed in the repository:

#### Shell History
- `.bash_history`
- `.zsh_history`
- `.sh_history`
- `.history`

**Rationale**: May contain credentials, API keys, internal paths, and sensitive commands.

#### Credentials and Keys
- `*.pem`, `*.key`, `*.p12`, `*.pfx`
- `id_rsa`, `id_dsa`, `id_ecdsa`, `id_ed25519`
- `.env`, `.env.local`, `.env.production`
- `.aws/credentials`, `.ssh/config`

**Rationale**: Direct security risk. Use environment variables or secure vaults.

#### Build Artifacts
- `PKG-INFO` (at repository root)
- `MANIFEST`
- Binary distribution files

**Rationale**: Generated files, not source of truth. Belong in CI/CD artifacts.

#### Client-Specific Work
- `.local_backup/`
- `client_*/`
- Client identifier directories

**Rationale**: Client data and identifiers should not be in public repository history.

### Bidirectional Unicode Protection

**Threat**: Trojan Source attacks using Unicode control characters to hide malicious code.

**Protection**:
- Pre-commit hook scans all code and config files
- CI/CD gates verify no bidi characters present
- Blocked characters: LRE, RLE, PDF, LRO, RLO, LRI, RLI, FSI, PDI

**Allowed**: Unicode characters in markdown documentation for visual formatting.

## Repository Structure

### Production vs. Research Boundaries

```
Transformation_Portal/
├── lux_depth_v2/          # ✅ Feature-frozen production code
│   ├── SECURITY.md        # Security guidelines
│   └── README.md          # Production documentation
├── lux_depth_v3/          # ⚡ Active development
├── src/                   # 📦 Installable packages
├── tests/                 # ✓ Test suite
├── scripts/               # 🔧 Automation and utilities
├── docs/                  # 📚 Documentation
├── examples/              # 💡 Example usage
├── experimental/          # 🧪 Research (not production)
├── archive/               # 📁 Historical code
└── deprecated/            # ⚠️  Deprecated code
```

### Output and Artifact Management

**DO NOT commit**:
- `output/`, `output_*/` - Processing results
- `*_outputs/` - Task-specific outputs
- `sweep_runs/` - Benchmark runs
- `benchmarks_*/` - Benchmark results
- Large images (>5MB) - Use Git LFS or external storage
- Model weights (`.pth`, `.safetensors`) - Download on demand

**Exception**: Small validation fixtures (<1MB) in `tests/fixtures/`

**Proper Storage**:
1. **Git LFS**: For versioned binary assets (models, datasets)
2. **Cloud Storage**: For large outputs (S3, GCS, Azure Blob)
3. **Local Workspace**: For ephemeral outputs (`.gitignore`'d)
4. **CI/CD Artifacts**: For build outputs and reports

## Artifact Lifecycle

### Development Workflow

```bash
# 1. Process images (outputs go to local directory)
lux-depth-v2 --input-dir renders/ --output-dir output/

# 2. Review outputs locally
ls -lh output/

# 3. If needed for tests, copy to fixtures
cp output/sample.png tests/fixtures/validation/

# 4. Outputs are automatically ignored by .gitignore
git status  # Should not show output/ directory

# 5. Commit only source code changes
git add lux_depth_v2/pipeline.py
git commit -m "Improve depth processing"
```

### Client Deliverables

**NEVER commit client-specific data to public repository.**

**Proper Workflow**:
1. Process client data in local workspace
2. Store outputs in client-specific directory (ignored by `.gitignore`)
3. Deliver via secure channels (encrypted email, secure file transfer)
4. Document approach in `docs/client_workflows/` (no client data)

**Example**:
```bash
# Process client data (local only)
./scripts/process_client_batch.sh /path/to/client_data /path/to/output

# Deliverable goes to ignored directory
output_client_estate_20260101/

# Document the workflow (no sensitive data)
docs/client_workflows/luxury_estate_batch_workflow.md
```

## Branch Protection Rules

### Main Branch

**Required**:
- Status checks must pass
- Security gates must pass
- At least 1 approval required (recommended)
- Up-to-date with base branch

**Enforcement**:
- Direct pushes blocked
- Force pushes blocked
- Deletion blocked

### Feature Branches

**Naming Convention**:
- `feature/<feature-name>`
- `bugfix/<issue-number>-description`
- `security/<cve-or-issue>`
- `docs/<topic>`

**Best Practices**:
- Keep branches short-lived (<2 weeks)
- Rebase on main regularly
- Run `make ci` before pushing

## Code Review Process

### Security Review Checklist

Before approving any PR, reviewers must verify:

- [ ] No sensitive files added (`.bash_history`, credentials, etc.)
- [ ] No client identifiers or data in commit
- [ ] No bidirectional Unicode characters in code
- [ ] Large files use Git LFS or are excluded
- [ ] Output artifacts properly ignored
- [ ] Security gates pass in CI
- [ ] Dependencies have no known vulnerabilities
- [ ] Input validation for user-provided paths/data

### Quality Gates

All PRs must pass:
1. **Linting**: flake8, pylint
2. **Tests**: pytest suite
3. **Security**: Security gates workflow
4. **Documentation**: Updated for API changes

## Dependency Management

### Security Practices

1. **Version Pinning**: Use `requirements.lock.txt` for reproducible builds
2. **Vulnerability Scanning**: `make security-audit` before adding dependencies
3. **Supply Chain**: Verify package sources and maintainers
4. **Updates**: Regular security updates via `dependabot`

### Adding Dependencies

```bash
# 1. Check for vulnerabilities
make verify-security

# 2. Add to requirements.txt
echo "new-package>=1.0.0" >> requirements.txt

# 3. Update lockfile
make lock-prod

# 4. Test
pip install -r requirements.lock.txt
make test-full

# 5. Commit both files
git add requirements.txt requirements.lock.txt
git commit -m "Add new-package for XYZ feature"
```

## Incident Response

### Security Incident Procedure

If sensitive data is accidentally committed:

#### 1. Immediate Response (Within 1 hour)
```bash
# Stop any ongoing work
git checkout main

# Remove sensitive file
git rm --cached .bash_history  # or other sensitive file

# Commit removal
git commit -m "SECURITY: Remove accidentally committed sensitive file"

# Push immediately
git push origin main
```

#### 2. Assess Exposure (Within 4 hours)
- Check commit history: `git log --all -- <sensitive-file>`
- Identify exposed data (credentials, client info, etc.)
- Determine if data was pushed to remote
- Check if any forks or clones exist

#### 3. Remediation (Within 24 hours)
- **If credentials exposed**: Rotate immediately
- **If in recent commit**: Consider `git revert`
- **If in old history**: May need history rewrite (contact admin)
- **If client data exposed**: Notify client, legal review

#### 4. Prevention (Within 1 week)
- Update `.gitignore`
- Add pre-commit hook for pattern
- Document incident in `docs/security/incidents/`
- Team training if needed

### Contact Points

**Security Issues**: info@racluxe.com
**GitHub Admins**: @RC219805
**Urgent**: Create GitHub issue with `security` label

## Compliance and Auditing

### Regular Audits

**Monthly**:
- Run `make security-audit`
- Review dependency vulnerabilities
- Check `.gitignore` coverage

**Quarterly**:
- Full repository scan
- Secret scanning with multiple tools
- Access review (who has write access)

**Annually**:
- Security policy review
- Governance document updates
- Third-party security audit (if applicable)

### Documentation

**Required Documentation**:
- `SECURITY.md`: Security policy and contact
- `LICENSE`: Clear license terms
- `CONTRIBUTING.md`: Contribution guidelines
- This document: Repository governance

## Best Practices Summary

### ✅ DO
- Use pre-commit hooks
- Keep .gitignore up to date
- Review security gates output
- Store large files in Git LFS
- Document client workflows without client data
- Rotate credentials immediately if exposed
- Run security checks before committing

### ❌ DON'T
- Commit shell history files
- Store credentials in repository
- Commit client-specific data
- Push large binaries directly
- Disable security checks
- Force push to protected branches
- Commit build artifacts to root

## Additional Resources

- [Pre-commit Hooks Documentation](https://pre-commit.com/)
- [Git LFS Guide](https://git-lfs.github.com/)
- [OWASP Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)
- [GitHub Security Best Practices](https://docs.github.com/en/code-security)

---

**Last Updated**: January 1, 2026
**Version**: 1.0.0
**Owner**: Transformation Portal Architect Team
