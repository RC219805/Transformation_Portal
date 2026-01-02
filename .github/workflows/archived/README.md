# Archived Workflows

This directory contains GitHub Actions workflows that have been archived as part of workflow optimization initiatives.

## Archival Log

### Phase 1 Optimization (2026-01-02)

**Reason:** Workflow consolidation to reduce CI runtime and eliminate redundancy

#### Security Workflows → `security-unified.yml`
- **security-scan.yml** (archived 2026-01-02)
  - Replaced by: `security-unified.yml`
  - Reason: Consolidated into unified security workflow
  - Coverage: CodeQL + Bandit + Safety + CVE-2024-27763 checks

- **security-gates.yml** (archived 2026-01-02)
  - Replaced by: `security-unified.yml`
  - Reason: Consolidated into unified security workflow
  - Coverage: Sensitive file detection + TruffleHog + .gitignore validation

- **codeql.yml** (archived 2026-01-02)
  - Replaced by: `security-unified.yml`
  - Reason: Consolidated into unified security workflow
  - Coverage: CodeQL analysis for Python and GitHub Actions

**Benefits:**
- Single workflow for all security checks
- Parallel job execution (5-8 min/PR savings)
- Eliminates duplicate CodeQL scans
- Maintains 100% security coverage

---

## Restoration Instructions

If you need to restore any workflow:

1. Copy the workflow file from `archived/` back to `.github/workflows/`
2. Rename it to avoid conflicts (e.g., `security-scan-restored.yml`)
3. Update the workflow name in the YAML file
4. Test thoroughly before enabling on main branch

---

## Validation

All archived workflows have been validated to ensure:
- ✅ No security coverage lost
- ✅ No test coverage lost
- ✅ All functionality preserved in replacement workflows
- ✅ Documented replacement path

---

*Last Updated: 2026-01-02*
*Architect: Transformation Portal Architect*
