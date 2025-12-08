# Security Hardening Checklist

**Version**: 1.0  
**Date**: 2025-12-08  
**Related**: PR-1 (Security + Repo Hygiene)

---

## Overview

This checklist ensures all security measures are implemented and verified before the repository is considered hardened. Each item must be checked off with evidence.

---

## Phase 1: Repository Hygiene (IMMEDIATE - Day 1)

### Artifact Purge

- [ ] **Remove `.bash_history`**
  - Evidence: `git log --all --full-history -- .bash_history` shows removal
  - Verification: File not in `git status`

- [ ] **Remove `.local_backup/`**
  - Evidence: Directory deleted
  - Verification: Not in working tree

- [ ] **Remove client-specific folders**
  - Files: `09_Client_Deliverables/`, `*_client_*/`
  - Evidence: Moved to separate private repository
  - Verification: Not in working tree

- [ ] **Remove temporary backups**
  - Files: `*.backup`, `*_BACKUP/`, `.branch_cleanup_backup/`
  - Evidence: Deleted
  - Verification: Not in working tree

### Gitignore Update

- [ ] **Add sensitive file patterns**
  ```gitignore
  # Security
  .bash_history
  .zsh_history
  .python_history
  *.pem
  *.key
  .env
  .env.*
  secrets/
  credentials/
  
  # Backups
  .local_backup/
  *.backup
  *_BACKUP/
  
  # Client data
  client_*/
  *_client_*/
  confidential/
  ```
  - Evidence: Patterns added to `.gitignore`
  - Verification: Test file creation fails to be staged

### Git History Purge

- [ ] **Install BFG Repo-Cleaner**
  ```bash
  brew install bfg  # macOS
  # or download from https://rtyley.github.io/bfg-repo-cleaner/
  ```
  - Evidence: `bfg --version` works

- [ ] **Create backup of repository**
  ```bash
  git clone --mirror https://github.com/RC219805/Transformation_Portal.git tp-backup.git
  ```
  - Evidence: Backup exists in safe location
  - Verification: `du -sh tp-backup.git` shows reasonable size

- [ ] **Purge sensitive files from history**
  ```bash
  bfg --delete-files .bash_history tp-backup.git
  bfg --delete-folders .local_backup tp-backup.git
  cd tp-backup.git
  git reflog expire --expire=now --all
  git gc --prune=now --aggressive
  ```
  - Evidence: `git log --all --full-history -- .bash_history` returns nothing
  - Verification: Repository size reduced

- [ ] **Notify contributors before force push**
  - Create GitHub issue: "Repository history will be rewritten on YYYY-MM-DD"
  - Evidence: Issue created with 48-hour notice
  - Verification: Wait for contributor acknowledgments

- [ ] **Force push cleaned history**
  ```bash
  git push --force origin main
  git push --force --all
  git push --force --tags
  ```
  - Evidence: Push succeeds
  - Verification: Fresh clone shows no sensitive files in history

### Secret Scanning

- [ ] **Install gitleaks**
  ```bash
  brew install gitleaks  # macOS
  # or download from https://github.com/gitleaks/gitleaks
  ```
  - Evidence: `gitleaks version` works

- [ ] **Scan full history**
  ```bash
  gitleaks detect --source . --verbose --report-path gitleaks-report.json
  ```
  - Evidence: Report generated
  - Verification: Review report, no secrets found or all false positives documented

- [ ] **Install trufflehog**
  ```bash
  pip install truffleHog3
  ```
  - Evidence: `trufflehog --version` works

- [ ] **Scan with trufflehog**
  ```bash
  trufflehog git file://. --only-verified --json > trufflehog-report.json
  ```
  - Evidence: Report generated
  - Verification: No verified secrets found

### Credential Rotation

- [ ] **Identify exposed credentials**
  - Review gitleaks and trufflehog reports
  - Evidence: List of credentials to rotate

- [ ] **Rotate API keys**
  - For each exposed key:
    - [ ] Regenerate in provider dashboard
    - [ ] Update in GitHub Secrets
    - [ ] Update in local `.env` files
    - [ ] Verify old key is revoked
  - Evidence: New keys working, old keys fail

- [ ] **Rotate SSH keys** (if exposed)
  - [ ] Generate new SSH key pair
  - [ ] Add to `~/.ssh/authorized_keys` on servers
  - [ ] Remove old key from authorized_keys
  - [ ] Test new key works
  - Evidence: `ssh -T git@github.com` succeeds with new key

- [ ] **Rotate database passwords** (if exposed)
  - [ ] Update in secret manager (AWS Secrets Manager, etc.)
  - [ ] Update application configs
  - [ ] Verify old password fails
  - Evidence: Application connects with new password

---

## Phase 2: CI Security Gates (Day 2)

### Script Implementation

- [ ] **Create `scripts/ci/enforce_safe_deps.py`**
  - Functionality:
    - Checks installed packages for banned packages
    - Scans code for banned imports
    - Fails CI if violations found
  - Evidence: Script exists and runs
  - Verification: `python scripts/ci/enforce_safe_deps.py` passes

- [ ] **Test script detects violations**
  ```bash
  # Install banned package
  pip install basicsr
  python scripts/ci/enforce_safe_deps.py
  # Should exit non-zero
  ```
  - Evidence: Script exits with code 1
  - Verification: Error message indicates banned package

- [ ] **Test script passes on clean environment**
  ```bash
  pip uninstall basicsr realesrgan gfpgan -y
  python scripts/ci/enforce_safe_deps.py
  # Should exit zero
  ```
  - Evidence: Script exits with code 0
  - Verification: "✅ Security gate passed" message

### Workflow Integration

- [ ] **Update `.github/workflows/security-scan.yml`**
  - Add `security-gate` job
  - Add `secret-scan` job
  - Add `vuln-scan` job
  - Evidence: Workflow file updated
  - Verification: Lint workflow file: `actionlint .github/workflows/security-scan.yml`

- [ ] **Test workflow on branch**
  ```bash
  git checkout -b test-security-gate
  git push origin test-security-gate
  ```
  - Evidence: Workflow runs in GitHub Actions
  - Verification: All jobs pass

- [ ] **Enable branch protection**
  - Go to Settings → Branches → Add rule for `main`
  - Require status checks:
    - `security-gate`
    - `secret-scan`
  - Evidence: Screenshot of branch protection settings
  - Verification: Try to merge PR without passing checks (should fail)

### Secret Scanning Setup

- [ ] **Enable GitHub secret scanning**
  - Go to Settings → Code security and analysis
  - Enable "Secret scanning"
  - Enable "Push protection"
  - Evidence: Features enabled
  - Verification: Try to push a test secret (should be blocked)

- [ ] **Add pre-commit hook**
  ```bash
  cp .github/pre-commit-hook.sh .git/hooks/pre-commit
  chmod +x .git/hooks/pre-commit
  ```
  - Evidence: Hook installed
  - Verification: Try to commit a test secret (should be blocked)

- [ ] **Configure gitleaks in CI**
  - Add gitleaks action to workflow
  - Evidence: Action added
  - Verification: Workflow runs gitleaks

### Dependency Scanning

- [ ] **Install safety**
  ```bash
  pip install safety
  ```
  - Evidence: `safety --version` works

- [ ] **Run safety check**
  ```bash
  safety check --file requirements.txt --json --output safety-report.json
  ```
  - Evidence: Report generated
  - Verification: No critical/high vulnerabilities or all documented

- [ ] **Install bandit**
  ```bash
  pip install bandit
  ```
  - Evidence: `bandit --version` works

- [ ] **Run bandit scan**
  ```bash
  bandit -r lux_depth_v2/ -ll -f json -o bandit-report.json
  ```
  - Evidence: Report generated
  - Verification: No high/medium issues or all documented

- [ ] **Add vulnerability scanning to CI**
  - Add `vuln-scan` job to workflow
  - Evidence: Job added
  - Verification: Workflow runs vulnerability scans

---

## Phase 3: Platform Core Security (Week 2)

### Input Validation

- [ ] **Implement `core/security/paths.py`**
  - `PathValidator` class
  - `sanitize_filename()` function
  - Evidence: File exists with full implementation
  - Verification: Unit tests pass

- [ ] **Test path validation**
  ```python
  validator = PathValidator(allowed_base="/data")
  # Should raise ValueError
  validator.validate("../../../etc/passwd")
  ```
  - Evidence: Test raises ValueError
  - Verification: Test suite passes

- [ ] **Implement `core/security/images.py`**
  - `validate_image_file()` function
  - Evidence: File exists with full implementation
  - Verification: Unit tests pass

- [ ] **Test image validation**
  ```python
  # Should raise ValueError for invalid images
  validate_image_file("malicious.exe")
  ```
  - Evidence: Test raises ValueError
  - Verification: Test suite passes

### Integration with Pipelines

- [ ] **Add path validation to Lux Depth V2 CLI**
  - Validate input paths before processing
  - Evidence: Code updated
  - Verification: CLI rejects invalid paths

- [ ] **Add image validation to Lux Depth V2**
  - Validate images before processing
  - Evidence: Code updated
  - Verification: Pipeline rejects invalid images

---

## Phase 4: Service Security (Lux Depth V2)

### Authentication

- [ ] **Implement API key authentication**
  - Add `verify_api_key()` function
  - Add `X-API-Key` header requirement
  - Evidence: Code updated in `lux_depth_v2/service.py`
  - Verification: Endpoint requires API key

- [ ] **Test API key enforcement**
  ```bash
  # Should fail (no key)
  curl -X POST http://localhost:8088/v2/process
  
  # Should succeed
  curl -X POST http://localhost:8088/v2/process \
    -H "X-API-Key: test-key"
  ```
  - Evidence: No key returns 403
  - Verification: Valid key returns 200

- [ ] **Document API key setup**
  - Update `lux_depth_v2/README.md`
  - Evidence: Documentation includes setup instructions
  - Verification: Instructions work for fresh user

### Rate Limiting

- [ ] **Install slowapi**
  ```bash
  pip install slowapi
  ```
  - Evidence: Package installed

- [ ] **Implement rate limiting**
  - Add `Limiter` to service
  - Set limit: 10 requests/minute per IP
  - Evidence: Code updated
  - Verification: Endpoint rate-limited

- [ ] **Test rate limiting**
  ```bash
  # Send 11 requests rapidly
  for i in {1..11}; do
    curl -X POST http://localhost:8088/v2/process \
      -H "X-API-Key: test-key"
  done
  # 11th request should return 429
  ```
  - Evidence: 11th request returns 429
  - Verification: Rate limiting works

### File Upload Limits

- [ ] **Implement request size limit middleware**
  - Add `RequestSizeLimitMiddleware`
  - Set limit: 100MB
  - Evidence: Middleware added
  - Verification: Large uploads rejected

- [ ] **Test file size limits**
  ```bash
  # Create 101MB file
  dd if=/dev/zero of=large.jpg bs=1M count=101
  
  # Should be rejected
  curl -X POST http://localhost:8088/v2/process \
    -H "X-API-Key: test-key" \
    -F "file=@large.jpg"
  ```
  - Evidence: Request returns 413
  - Verification: Size limit enforced

### HTTPS/TLS

- [ ] **Document HTTPS setup**
  - Add instructions for reverse proxy (nginx)
  - Add instructions for uvicorn with TLS (dev only)
  - Evidence: Documentation updated
  - Verification: Instructions tested

- [ ] **Test HTTPS in staging**
  - Deploy with HTTPS enabled
  - Evidence: Certificate valid
  - Verification: `curl https://staging.example.com/v2/health` succeeds

---

## Phase 5: Documentation (Day 3)

### README Updates

- [ ] **Update security section**
  - Reflect actual security posture
  - Document CVE-2024-27763 mitigation
  - Link to `lux_depth_v2/SECURITY.md`
  - Evidence: README updated
  - Verification: Security section accurate

- [ ] **Remove Real-ESRGAN references** (if removed)
  - Or document as "unsafe extras" if kept
  - Evidence: References removed/updated
  - Verification: No misleading claims

### Security Documentation

- [ ] **Update `lux_depth_v2/SECURITY.md`**
  - Reflect implemented controls
  - Evidence: File updated
  - Verification: Checklist items marked complete

- [ ] **Update root `SECURITY.md`**
  - Add vulnerability disclosure process
  - Add security contact
  - Evidence: File updated
  - Verification: Process clear

### Migration Guide

- [ ] **Create migration guide for contributors**
  - Instructions for re-cloning after history purge
  - Evidence: Guide exists
  - Verification: Contributor follows successfully

---

## Verification & Sign-Off

### Automated Checks

- [ ] **CI security gate passes**
  ```bash
  # All jobs pass
  - security-gate: ✅
  - secret-scan: ✅
  - vuln-scan: ✅ (or documented exceptions)
  ```
  - Evidence: GitHub Actions badge shows passing
  - Verification: Check recent CI runs

- [ ] **No secrets in repository**
  ```bash
  gitleaks detect --source . --verbose
  trufflehog git file://. --only-verified
  # Both should return no findings
  ```
  - Evidence: Clean reports
  - Verification: Reports reviewed

- [ ] **No banned packages**
  ```bash
  python scripts/ci/enforce_safe_deps.py
  # Should pass
  ```
  - Evidence: Script passes
  - Verification: Exit code 0

### Manual Review

- [ ] **Code review by security expert**
  - Review input validation
  - Review authentication implementation
  - Review rate limiting
  - Evidence: Review completed
  - Verification: Sign-off from reviewer

- [ ] **Penetration testing** (optional but recommended)
  - Test path traversal attacks
  - Test authentication bypass
  - Test rate limit bypass
  - Evidence: Pen test report
  - Verification: No critical findings

### Documentation Review

- [ ] **README accurate**
  - No security claims that aren't implemented
  - Evidence: README reviewed
  - Verification: Claims verified

- [ ] **SECURITY.md complete**
  - Vulnerability disclosure process documented
  - Security contact provided
  - Evidence: File complete
  - Verification: Process tested (dry run)

---

## Sign-Off

### Project Lead

- [ ] **Reviewed checklist**
- [ ] **Verified all items complete**
- [ ] **Approved for production**

**Signature**: ________________  
**Date**: ________

### Security Reviewer

- [ ] **Reviewed security controls**
- [ ] **Verified implementation**
- [ ] **No critical issues found**

**Signature**: ________________  
**Date**: ________

---

## Post-Hardening Maintenance

### Monthly Tasks

- [ ] Review security scan reports
- [ ] Update vulnerable dependencies
- [ ] Check for new CVEs affecting dependencies
- [ ] Rotate API keys (optional, every 90 days)

### Quarterly Tasks

- [ ] Full security audit
- [ ] Penetration testing
- [ ] Review and update security documentation
- [ ] Review access controls (GitHub team permissions)

### Annual Tasks

- [ ] External security audit
- [ ] Update security policy
- [ ] Review incident response plan
- [ ] Security training for contributors

---

**Version**: 1.0  
**Last Updated**: 2025-12-08  
**Next Review**: 2025-12-15 (post-implementation)
