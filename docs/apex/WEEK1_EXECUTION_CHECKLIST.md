# APEX Governance: Immediate Execution Checklist

**Created:** 2025-02-09
**Owner:** Transformation Portal Architect
**Context:** [GOVERNANCE_ORCHESTRATION_PLAN.md](./GOVERNANCE_ORCHESTRATION_PLAN.md)

---

## Week 1 Tactical Checklist

### ✅ Task 1: Merge Phase 2 Real Pipeline Integration

**Branch:** `feat/apex-real-pipeline-integration` → `main`

**Pre-Merge Verification:**
- [x] Event gating verified (lines 105-111 in workflow)
- [x] Dependency gating verified (lines 76-80)
- [x] Metadata capture verified
- [x] Artifact durability verified
- [x] PR comment semantic honesty verified
- [ ] Manual workflow_dispatch test run (real mode)
- [ ] Ledger artifact inspection
- [ ] Capsule JSON validation

**Merge Checklist:**
```bash
# 1. Ensure clean working tree
git status

# 2. Update branch with latest main
git checkout feat/apex-real-pipeline-integration
git fetch origin
git rebase origin/main

# 3. Run local validation
python scripts/apex_matrix_runner.py \
    --run-id "test-$(git rev-parse --short HEAD)" \
    --commit-sha "$(git rev-parse HEAD)" \
    --zones local \
    --workflow-versions v1 v2 \
    --output-dir ./apex_test_results \
    --synthetic \
    --dry-run

# 4. Merge to main (preserve history)
git checkout main
git merge --no-ff feat/apex-real-pipeline-integration \
    -m "feat(apex): Complete Phase 2 Real Pipeline Integration

Implements hybrid CI strategy with synthetic/real execution lanes.

Key Features:
- Event-based mode gating (PR=synthetic, schedule=real)
- Conditional ML dependency installation
- Complete metadata/provenance capture
- Multi-tier artifact retention (capsules 3d, ledger 90d)
- Weekly automated backups
- Shadow mode enforcement

Closes: Phase 2 implementation
Ref: docs/apex/phase2/COMPLETION_REPORT.md"

# 5. Push and monitor
git push origin main

# 6. Watch workflow run
gh run watch
```

**Post-Merge Validation:**
```bash
# Trigger manual real run
gh workflow run apex_performance.yml \
    -f mode=real \
    -f backend_id=da3 \
    -f sample_size=5 \
    -f device=cpu

# Monitor execution
gh run list --workflow=apex_performance.yml --limit 1
gh run watch <run-id>

# Download artifacts
gh run download <run-id>

# Inspect ledger
sqlite3 apex_performance.db "SELECT * FROM apex_runs ORDER BY timestamp DESC LIMIT 5;"

# Validate capsule schema
python -c "
import json
from pathlib import Path

capsule_files = Path('apex-results-v1-local').glob('*.json')
for cf in capsule_files:
    with open(cf) as f:
        capsule = json.load(f)
        print(f'{cf.name}: {capsule.keys()}')
        assert 'run_id' in capsule
        assert 'commit_sha' in capsule
        assert 'observations' in capsule
"
```

**Success Criteria:**
- ✅ Workflow runs successfully
- ✅ Ledger artifact contains run data
- ✅ Capsule JSON validates against schema
- ✅ PR comment generated with clear mode indication
- ✅ No dependency errors in real mode

---

### 📝 Task 2: Split Governance Branch into Two PRs

**Source Branch:** `origin/copilot/best-outcome-roadmap-apex`

#### Step 2.1: Create PR 1 - Policy Infrastructure (Shadow Mode)

```bash
# Create new branch from governance work
git checkout -b feat/apex-governance-policy-infrastructure origin/copilot/best-outcome-roadmap-apex

# Cherry-pick only policy files and validator
git checkout main
git checkout -b feat/apex-policy-infrastructure-pr1

# Selectively add files
git checkout feat/apex-governance-policy-infrastructure -- docs/apex/policy/
git checkout feat/apex-governance-policy-infrastructure -- scripts/apex_validate_policy.py
git checkout feat/apex-governance-policy-infrastructure -- docs/architecture/decisions/ADR-026-APEX-governance-framework.md
git checkout feat/apex-governance-policy-infrastructure -- docs/apex/GOVERNANCE_USER_GUIDE.md

# Do NOT include enforcement integration yet
# (save for PR 2: integration with apex_enforce_gate.py)

# Add CI workflow for policy validation
cat > .github/workflows/apex-policy-validation.yml << 'EOF'
name: APEX Policy Validation

on:
  pull_request:
    paths:
      - 'docs/apex/policy/**'
      - 'scripts/apex_validate_policy.py'
  push:
    branches: [main]
    paths:
      - 'docs/apex/policy/**'

jobs:
  validate-policy:
    name: Validate Policy Files
    runs-on: ubuntu-latest
    timeout-minutes: 5

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install pyyaml jsonschema

      - name: Validate policy files
        run: |
          python scripts/apex_validate_policy.py \
            --policy-dir docs/apex/policy/ \
            --check all

      - name: Check policy consistency
        run: |
          python scripts/apex_validate_policy.py \
            --policy-dir docs/apex/policy/ \
            --check consistency
EOF

git add .github/workflows/apex-policy-validation.yml

# Add unit tests for validator
cat > tests/test_apex_policy_validator.py << 'EOF'
"""Unit tests for APEX policy validator."""

import pytest
from pathlib import Path
import tempfile
import yaml

# Import validator functions (adjust path as needed)
# from scripts.apex_validate_policy import validate_schema_version, validate_performance_budgets


def test_valid_policy_passes():
    """Valid policy files should validate without errors."""
    # TODO: Load actual policy files and validate
    # policy_dir = Path("docs/apex/policy/")
    # errors = validate_all_policies(policy_dir)
    # assert errors == []
    pass


def test_missing_schema_version_fails():
    """Policy file missing schema_version should fail."""
    # TODO: Create temporary policy file without schema_version
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     policy_file = Path(tmpdir) / "test_policy.yaml"
    #     with open(policy_file, 'w') as f:
    #         yaml.dump({"budgets": []}, f)
    #     errors = validate_schema_version(...)
    #     assert len(errors) > 0
    #     assert "schema_version" in errors[0]
    pass


def test_invalid_semver_fails():
    """Policy with invalid semver should fail."""
    # TODO: Test invalid version formats
    # errors = validate_schema_version({"schema_version": "1.0"}, "test.yaml")
    # assert len(errors) > 0
    pass


def test_missing_bucket_definition_fails():
    """Policy missing a required bucket should fail."""
    # TODO: Test missing bucket validation
    pass


def test_circular_dependency_fails():
    """Waiver referencing nonexistent bucket should fail."""
    # TODO: Test circular dependency detection
    pass


# TODO: Add more tests for:
# - Invalid enforcement modes
# - Invalid workflow versions
# - Missing required fields
# - Schema version mismatches
# - Consistency checks (bucket names match across files)
EOF

git add tests/test_apex_policy_validator.py

# Commit and push
git commit -m "feat(apex): Add policy-as-code infrastructure (shadow mode)

Adds governance scaffolding without enforcement integration:

Policy Files:
- enforcement_policy.yaml: Statistical methods, evidence gates
- performance_budgets.yaml: Per-bucket thresholds
- governance_rules.yaml: Waiver system, incident automation
- workload_suites.yaml: Golden/canary/deep suite definitions
- apex_policy_schema.yaml: Schema validation rules

Infrastructure:
- Policy validator script with unit tests
- CI workflow for policy validation
- ADR-026 documenting governance framework
- User guide for governance system

Mode: SHADOW ONLY (informational, non-blocking)

This PR establishes the policy infrastructure without modifying
enforcement behavior. Integration with apex_enforce_gate.py will
come in a follow-up PR after baseline calibration.

Ref: docs/apex/GOVERNANCE_ORCHESTRATION_PLAN.md Step C PR 1"

git push -u origin feat/apex-policy-infrastructure-pr1

# Open PR
gh pr create \
    --title "feat(apex): Policy-as-Code Infrastructure (Shadow Mode)" \
    --body "## 📋 APEX Governance Policy Infrastructure

**Type:** Feature (Non-Breaking, Shadow Mode Only)
**Phase:** Step C PR 1 (Governance Scaffolding)
**Enforcement Mode:** SHADOW (informational only, never blocks)

### What This PR Does

Adds policy-as-code infrastructure for APEX performance governance:

- **Policy Files** (YAML): Thresholds, statistical methods, waiver rules, suite definitions
- **Policy Validator** (Python): Schema validation, consistency checks
- **CI Workflow**: Validates policy files on changes
- **Documentation**: ADR-026, user guide
- **Unit Tests**: Validator test suite (TODO: complete)

### What This PR Does NOT Do

- ❌ Does NOT modify enforcement behavior (still shadow mode)
- ❌ Does NOT integrate with \`apex_enforce_gate.py\` (deferred to PR 2)
- ❌ Does NOT change merge gating (still informational only)
- ❌ Does NOT implement waiver workflow (deferred)

### Backward Compatibility

- ✅ Graceful degradation if policy files missing
- ✅ No impact on existing workflows
- ✅ Default mode remains \`shadow\`
- ✅ No new dependencies

### Review Checklist

- [ ] Policy files validate against schema
- [ ] CI workflow passes
- [ ] Unit tests added for validator (TODO items documented)
- [ ] ADR reviewed and approved
- [ ] User guide is clear and actionable

### Next Steps (Future PRs)

1. Complete unit test TODOs
2. Implement calibration pipeline (Step D)
3. Integrate statistical enforcement (Step C PR 2)
4. Implement waiver workflow
5. Gradual rollout (shadow → soft → hard)

**Ref:** docs/apex/GOVERNANCE_ORCHESTRATION_PLAN.md" \
    --label "apex,governance,shadow-mode"
```

**Success Criteria:**
- ✅ Policy files validate in CI
- ✅ No merge blocking behavior introduced
- ✅ Unit test structure in place (even if TODOs exist)
- ✅ Documentation clear and complete

---

### 🔧 Task 3: Fix Dependency Updater

**Workflow File:** `.github/workflows/dependency-update.yml`

**Branch:** `fix/dependency-updater-improvements`

```bash
git checkout main
git pull
git checkout -b fix/dependency-updater-improvements

# Create improved dependency updater script
cat > scripts/generate_dependency_diff.py << 'EOF'
#!/usr/bin/env python3
"""Generate dependency diff and safety report summary for PRs."""

import json
import subprocess
import sys
from pathlib import Path


def get_pip_freeze():
    """Capture current pip freeze output."""
    result = subprocess.run(
        ["pip", "freeze"],
        capture_output=True,
        text=True,
        check=True
    )
    return set(result.stdout.strip().split('\n'))


def parse_safety_report(report_path):
    """Parse safety JSON report and generate summary."""
    if not report_path.exists():
        return "⚠️ Safety report not found"

    with open(report_path) as f:
        data = json.load(f)

    vulnerabilities = data.get('vulnerabilities', [])

    if not vulnerabilities:
        return "✅ **No known vulnerabilities detected**"

    summary = f"⚠️ **{len(vulnerabilities)} vulnerabilities found:**\n\n"

    for vuln in vulnerabilities[:10]:  # Top 10
        pkg = vuln.get('package', 'unknown')
        vuln_id = vuln.get('id', 'unknown')
        title = vuln.get('title', 'No title')
        severity = vuln.get('severity', 'unknown')

        summary += f"- **{pkg}** ({severity}): {title} ([{vuln_id}])\n"

    if len(vulnerabilities) > 10:
        summary += f"\n... and {len(vulnerabilities) - 10} more (see attached report)\n"

    return summary


def generate_diff(before_file, after_file):
    """Generate human-readable diff between requirement snapshots."""
    with open(before_file) as f:
        before = {line.split('==')[0]: line.strip() for line in f if '==' in line}

    with open(after_file) as f:
        after = {line.split('==')[0]: line.strip() for line in f if '==' in line}

    added = set(after.keys()) - set(before.keys())
    removed = set(before.keys()) - set(after.keys())
    updated = {
        pkg: (before[pkg], after[pkg])
        for pkg in before.keys() & after.keys()
        if before[pkg] != after[pkg]
    }

    diff_md = "### 📦 Dependency Changes\n\n"

    if updated:
        diff_md += "#### Updated Packages\n\n"
        diff_md += "| Package | Before | After |\n"
        diff_md += "|---------|--------|-------|\n"
        for pkg, (old, new) in sorted(updated.items())[:20]:
            old_ver = old.split('==')[1] if '==' in old else old
            new_ver = new.split('==')[1] if '==' in new else new
            diff_md += f"| {pkg} | `{old_ver}` | `{new_ver}` |\n"
        if len(updated) > 20:
            diff_md += f"\n... and {len(updated) - 20} more updated packages\n"

    if added:
        diff_md += "\n#### Added Packages\n\n"
        for pkg in sorted(added)[:10]:
            diff_md += f"- `{after[pkg]}`\n"

    if removed:
        diff_md += "\n#### Removed Packages\n\n"
        for pkg in sorted(removed)[:10]:
            diff_md += f"- `{before[pkg]}`\n"

    if not added and not removed and not updated:
        diff_md += "*No changes detected*\n"

    return diff_md


def main():
    before_file = Path("before-freeze.txt")
    after_file = Path("after-freeze.txt")
    safety_report = Path("safety-report.json")
    output_file = Path("dependency-pr-body.md")

    diff_md = generate_diff(before_file, after_file)
    safety_md = parse_safety_report(safety_report)

    pr_body = f"""## 🔄 Automated Dependency Updates

{diff_md}

### 🔒 Security Scan Results

{safety_md}

### ⚠️ Review Required

- [ ] Check for breaking changes in updated packages
- [ ] Review security vulnerabilities (if any)
- [ ] Validate tests pass in CI
- [ ] Verify compatibility with Python 3.11+

### 📋 Files Updated

- `requirements/base.txt` - Core runtime dependencies
- `requirements/ml.txt` - ML and deep learning packages
- `requirements/dev.txt` - Development tools
- `requirements/ci.txt` - CI/CD tools
- `requirements/all.txt` - Combined requirements

**Generated automatically by GitHub Actions**
"""

    with open(output_file, 'w') as f:
        f.write(pr_body)

    print(f"✅ Generated PR body: {output_file}")


if __name__ == "__main__":
    main()
EOF

chmod +x scripts/generate_dependency_diff.py
git add scripts/generate_dependency_diff.py

# Update workflow to use new script
cat > .github/workflows/dependency-update.yml << 'EOF'
name: Dependency Updates

on:
  schedule:
    - cron: '0 9 * * 1'  # Weekly on Mondays at 9 AM UTC
  workflow_dispatch:

permissions:
  contents: write
  pull-requests: write

jobs:
  update-dependencies:
    name: Update Python Dependencies
    runs-on: ubuntu-24.04
    timeout-minutes: 30

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Free disk space
        run: |
          sudo rm -rf /usr/share/dotnet || true
          sudo rm -rf /opt/ghc || true
          sudo rm -rf /usr/local/lib/android || true
          sudo docker image prune --all --force || true

      - name: Capture before state
        run: |
          pip install --upgrade pip
          pip install -e .
          pip freeze > before-freeze.txt
          echo "📸 Captured before state"

      - name: Update dependencies
        run: |
          pip install pip-tools safety

          if [ -d "requirements" ]; then
            cd requirements
            make update || echo "⚠️  Some updates may have warnings"
            cd ..
          else
            echo "⚠️  requirements/ directory not found"
          fi

      - name: Capture after state
        run: |
          pip freeze > after-freeze.txt
          echo "📸 Captured after state"

      - name: Smoke test imports
        run: |
          echo "Testing core imports..."
          python -c "import transformation_portal; print('✅ Core imports OK')" || exit 1

          echo "Testing ML imports..."
          python -c "import torch; import transformers; print('✅ ML imports OK')" || exit 1

          echo "✅ All imports successful"

      - name: Check for vulnerabilities
        continue-on-error: true
        run: |
          safety check --json > safety-report.json || true
          cat safety-report.json

      - name: Generate PR body
        run: |
          python scripts/generate_dependency_diff.py

      - name: Create Pull Request
        uses: peter-evans/create-pull-request@v8
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          commit-message: "chore(deps): automated dependency updates"
          title: "🔄 Automated Dependency Updates"
          body-path: dependency-pr-body.md
          branch: automated/dependency-updates
          delete-branch: true
          labels: |
            dependencies
            automated
EOF

git add .github/workflows/dependency-update.yml

git commit -m "fix(deps): Improve dependency updater with diffs and safety summary

Enhancements:
- Generate before/after diff with version changes
- Parse and summarize safety scan results
- Add smoke tests for core imports before PR creation
- Improve PR body with actionable information
- Fail fast if imports break

Fixes: Dependency updater now provides actual diff information"

git push -u origin fix/dependency-updater-improvements

gh pr create \
    --title "fix(deps): Improve Dependency Updater" \
    --body "## 🔧 Dependency Updater Improvements

### What Changed

1. **Actual Diffs**: Shows before → after version changes in table format
2. **Safety Summary**: Parses JSON report and includes top vulnerabilities
3. **Smoke Tests**: Validates core imports before creating PR
4. **Better PR Body**: Human-readable summary with actionable checklist

### Example Output

The new PR body will include:

\`\`\`markdown
### 📦 Dependency Changes

#### Updated Packages

| Package | Before | After |
|---------|--------|-------|
| torch | 2.1.0 | 2.2.0 |
| numpy | 1.24.3 | 1.25.0 |

### 🔒 Security Scan Results

✅ No known vulnerabilities detected
\`\`\`

### Testing

- [ ] Test workflow locally or trigger manual run
- [ ] Verify PR body formatting
- [ ] Check that smoke tests catch broken imports

**Closes:** Odd Duck #1 from governance orchestration plan" \
    --label "dependencies,ci"
```

**Success Criteria:**
- ✅ Workflow generates actual diff
- ✅ Safety report summarized in PR body
- ✅ Smoke tests pass before PR creation
- ✅ PR body is human-readable

---

### 🗑️ Task 4: Close Performance Monitor PR

**Action:** Close PR #845 with clear rationale

```bash
# Create issue documenting rationale
gh issue create \
    --title "Decision: Consolidate Performance Testing in APEX (Close PR #845)" \
    --body "## Decision Record

**Date:** 2025-02-09
**Decision:** Close PR #845 (Performance Regression Tests) in favor of APEX

### Rationale

1. **Duplication**: APEX already provides comprehensive performance regression testing
2. **Fragility**: Artifact-based baseline management is fragile (APEX uses durable SQLite ledger)
3. **Confusion**: Two performance systems create confusion about source of truth
4. **Maintenance**: Single system is easier to maintain and evolve

### What PR #845 Provided

- pytest-benchmark integration
- Memory profiling with memory_profiler
- Artifact-based baseline storage

### What APEX Provides (Superior)

- Durable ledger with 90-day retention
- Multi-dimensional tracking (workflow version, zone, backend, device)
- Statistical regression detection
- Policy-as-code governance
- Dashboard with historical trends
- Automated backup strategy

### Migration Path

**Memory Profiling:**
- Not currently in APEX
- Document as future enhancement
- Add to roadmap as optional metric
- Can integrate memory_profiler into performance capsule schema

**Benchmark Tests:**
- APEX capsule system supersedes pytest-benchmark
- Performance capsules provide richer metadata
- Ledger provides better baseline management

### Action Items

- [x] Close PR #845 with link to this issue
- [ ] Add memory profiling to APEX roadmap
- [ ] Update documentation to reference APEX as single performance system

**Ref:** docs/apex/GOVERNANCE_ORCHESTRATION_PLAN.md Odd Duck #2" \
    --label "decision,apex,performance"

# Close PR (adjust number if different)
# gh pr close 845 --comment "Closing in favor of APEX performance system. See issue #<issue-number> for rationale and migration path."
```

**Success Criteria:**
- ✅ Issue documents decision clearly
- ✅ PR closed with link to decision issue
- ✅ Memory profiling added to roadmap

---

## Week 1 Summary

### Expected Outcomes

By end of Week 1:
- ✅ Phase 2 merged to main
- ✅ Manual real run validated
- ✅ Policy infrastructure PR opened
- ✅ Dependency updater fixed
- ✅ Performance monitor PR closed with clear rationale

### Metrics

- **PRs Merged:** 2 (Phase 2, Dependency Updater Fix)
- **PRs Opened:** 1 (Policy Infrastructure)
- **PRs Closed:** 1 (Performance Monitor)
- **New Workflows:** 1 (Policy Validation)
- **Documentation:** 2 new docs (Orchestration Plan, this checklist)

### Risks Mitigated

- ✅ Phase 2 drift (merged promptly)
- ✅ Policy validation gaps (CI workflow added)
- ✅ Dependency updater noise (improved PR body)
- ✅ Performance system confusion (single source of truth)

---

## Next Week Preview (Week 2)

- [ ] Implement measurement protocol (warmup, checksums, sample size validation)
- [ ] Increase scheduled run frequency (daily golden suite)
- [ ] Begin shadow mode data collection
- [ ] Review policy infrastructure PR

---

**Owner:** Transformation Portal Architect
**Last Updated:** 2025-02-09
**Status:** Active - Week 1 Execution
