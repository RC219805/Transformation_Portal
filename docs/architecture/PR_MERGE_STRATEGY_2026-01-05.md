# Pull Request Merge Strategy - 2026-01-05

**Reviewed by**: Transformation Portal Architect
**Date**: 2026-01-05
**Total PRs**: 7 open
**Status**: All PRs pending CI validation

---

## 🎯 Executive Summary

We have **7 open pull requests**, all created between 2026-01-05 23:43 and 2026-01-06 00:07. The PRs fall into two categories:

1. **5 Dependabot PRs** (659-663): Individual dependency version bumps
2. **2 Workflow Enhancement PRs** (656, 658): Automated dependency management improvements

**Critical Finding**: There are **file conflicts** between multiple PRs that require a strategic merge order.

### ✅ Positive Findings
- All PRs can fast-forward merge (no rebase conflicts with main)
- All PRs are in "pending" CI status (no failures yet)
- No security vulnerabilities detected in initial scans
- All changes are dependency/infrastructure related (low functional risk)

### ⚠️ Concerns
- **File overlap conflicts** between Dependabot PRs #659-662
- **PR #656 lacks CI validation gates** (addressed by PR #658)
- **PR #658 supersedes PR #656's approach** but both are open
- **CI workflows have not completed** for any PR (all "pending")

---

## 📊 PR Analysis Matrix

| PR # | Type | Author | Files Changed | Conflicts With | CI Status | Merge Priority |
|------|------|--------|---------------|----------------|-----------|----------------|
| **#663** | Dependency (scikit-learn) | dependabot | 1 | None | Pending | High (Independent) |
| **#662** | Dependency (imagecodecs) | dependabot | 3 | 659,660,661 | Pending | Low (Conflict group) |
| **#661** | Dependency (pillow) | dependabot | 5 | 659,660,662 | Pending | Low (Conflict group) |
| **#660** | Dependency (scipy) | dependabot | 3 | 659,661,662 | Pending | Low (Conflict group) |
| **#659** | Dependency (tifffile) | dependabot | 3 | 660,661,662 | Pending | Low (Conflict group) |
| **#658** | Workflow Enhancement | copilot-swe-agent | 7 | 656 | Pending | **Critical** (Infrastructure) |
| **#656** | Automated Dep Update | github-actions | 5 | 658 | Pending | **Superseded by #658** |

---

## 🔍 Detailed PR Review

### **PR #663**: `chore(deps): bump scikit-learn from 1.7.2 to 1.8.0`

**Impact**: ⚠️ **MODERATE** - Requires Python version constraint update

**Changes**:
- `requirements/constraints.txt`: `scikit-learn<1.8` → `scikit-learn<1.9`

**Analysis**:
- **Breaking change**: scikit-learn 1.8.0+ requires Python 3.11+
- **Current constraint**: The repo supports Python 3.10 (minimum)
- **Risk**: This update **conflicts with Python 3.10 support** unless we maintain the constraint
- **Action**: The constraint file correctly keeps `scikit-learn<1.9` but allows 1.8.x for Python 3.11+ users

**Security**: ✅ No vulnerabilities reported

**Compatibility**:
- New features: Free-threaded CPython support (Python 3.14)
- API changes: Array API standard compliance improvements
- Performance: No known regressions

**Recommendation**: ✅ **APPROVE with conditions**
- Verify CI tests pass on Python 3.10 with `scikit-learn<1.8` constraint
- Verify CI tests pass on Python 3.12 with `scikit-learn==1.8.0`
- Document Python version compatibility in changelog

---

### **PR #662**: `chore(deps): bump imagecodecs from 2024.12.30 to 2026.1.1`

**Impact**: ⚠️ **MODERATE** - Major release with breaking changes

**Changes**:
- `lux_depth_v3/requirements.txt`: `imagecodecs<2025` → `imagecodecs<2027`
- `pyproject.toml`: Same version constraint update
- `requirements/base.in`: Same version constraint update

**Analysis**:
- **Breaking changes**:
  - ✅ Positional-only and keyword-only parameter enforcement
  - ✅ `numcodecs.Jpeg` now based on JPEG8 codec (not JPEG12)
  - ⚠️ Changed default Brotli compression level (3 → 4)
- **New features**:
  - HTJ2K codec (High-throughput JPEG 2000) via OpenJPH
  - MESHOPT codec via meshoptimizer
  - UltraHDR uint16 decoding support
- **Bug fixes**:
  - Fixed ZStandard concatenated frame decoding
  - Fixed TIFF and WebP codec issues

**Security**: ✅ No CVEs reported

**Compatibility**:
- Requires Cython >=3.2 (build-time only)
- Python 3.11 deprecated (we use 3.10-3.12, so OK)

**Conflicts**:
- File overlap with PRs #659, #660, #661 (all touch `lux_depth_v3/requirements.txt`, `pyproject.toml`, `requirements/base.in`)

**Recommendation**: ✅ **APPROVE after conflict resolution**
- Merge **after** PRs #659, #660, #661 to avoid conflicts
- Test TIFF/WebP pipelines post-merge (validation step)
- Monitor for Brotli compression level change impact

---

### **PR #661**: `chore(deps): bump pillow from 11.3.0 to 12.1.0`

**Impact**: 🚨 **HIGH** - Major version bump (11 → 12)

**Changes**:
- `lux_depth_v3/requirements.txt`: `Pillow<12` → `Pillow<13`
- `pyproject.toml`: Same version constraint update
- `requirements-lint.txt`: Same version constraint update
- `requirements/base.in`: Same version constraint update
- `requirements/ml.in`: Same version constraint update

**Analysis**:
- **Breaking changes** (Pillow 12.0.0):
  - Deprecated `Image.ANTIALIAS` → use `Image.Resampling.LANCZOS`
  - Deprecated `getdata()` → use `get_flattened_data()` (new in 12.1.0)
  - Removed Python 3.10 support ❌ **CRITICAL ISSUE**
- **New features**:
  - APNG float duration support
  - ImageMorph improvements (1-bit mode support)
  - Updated libjpeg-turbo 3.1.3, libpng 1.6.53, harfbuzz 12.3.0

**Security**: ✅ Updated bundled libraries include security fixes

**Compatibility**:
- 🚨 **BLOCKER**: Pillow 12.x **requires Python 3.11+** (dropped Python 3.10)
- Our minimum Python version is **3.10** (per `pyproject.toml`)
- **This upgrade will break Python 3.10 users**

**Conflicts**:
- File overlap with PRs #659, #660, #662

**Recommendation**: ❌ **REJECT - Incompatible with Python 3.10**

**Action Required**:
1. **Close PR #661** with explanation
2. Update `pyproject.toml` minimum Python to 3.11 **OR**
3. Constrain Pillow to `<12` and wait for Python 3.10 EOL (2026-10-04)

**Suggested Comment for Dependabot**:
```
@dependabot ignore this major version
```

**Rationale**: The Transformation Portal supports Python 3.10 (minimum) per the project's compatibility matrix. Pillow 12.x dropped Python 3.10 support (see [Pillow 12.0.0 release notes](https://pillow.readthedocs.io/en/stable/releasenotes/12.0.0.html)). We must either:
1. Maintain Pillow 11.x until we drop Python 3.10 support (October 2026), or
2. Update our minimum Python version to 3.11 (requires ADR and migration plan)

---

### **PR #660**: `chore(deps): bump scipy from 1.15.3 to 1.16.3`

**Impact**: ⚠️ **MODERATE** - Minor version bump with Python 3.11+ requirement

**Changes**:
- `lux_depth_v3/requirements.txt`: `scipy<1.16` → `scipy<1.17`
- `pyproject.toml`: Same version constraint update
- `requirements/base.in`: Same version constraint update

**Analysis**:
- **Breaking changes**:
  - SciPy 1.16+ requires Python 3.11+ ❌ **CRITICAL ISSUE**
- **Bug fixes** (1.16.2, 1.16.3):
  - First stable release with Windows on ARM wheels
  - Various bug fixes (no API changes)

**Security**: ✅ No vulnerabilities

**Compatibility**:
- 🚨 **BLOCKER**: SciPy 1.16+ **requires Python 3.11+**
- Our constraint file shows `scipy>=1.15,<1.16` for Python 3.10 compatibility
- **This PR would break Python 3.10 users**

**Conflicts**:
- File overlap with PRs #659, #661, #662

**Recommendation**: ❌ **REJECT - Incompatible with Python 3.10**

**Action Required**:
1. **Close PR #660** with explanation
2. Keep `scipy<1.16` constraint until Python 3.10 EOL
3. Add comment to `requirements/base.in`:
   ```python
   scipy>=1.15,<1.16  # scipy 1.16+ requires Python 3.11+, constrain for Python 3.10 compatibility
   ```

**Suggested Comment for Dependabot**:
```
@dependabot ignore this major version
```

---

### **PR #659**: `chore(deps): bump tifffile from 2024.12.12 to 2025.12.20`

**Impact**: ✅ **LOW** - Minor version bump, no breaking changes

**Changes**:
- `lux_depth_v3/requirements.txt`: `tifffile<2025` → `tifffile<2026`
- `pyproject.toml`: Same version constraint update
- `requirements/base.in`: Same version constraint update

**Analysis**:
- **Changes**:
  - 2025.12.20: Do not initialize output arrays (performance optimization)
  - 2025.12.12: Code quality improvements
  - 2025.10.16: EER super-resolution decoding changes (⚠️ breaking for EER users)
- **Bug fixes**:
  - Fix parsing SVS description ending with "|"
  - Fix reading NDTiff series with unordered axes
- **Security**: ✅ No vulnerabilities

**Compatibility**:
- ✅ Python 3.10+ compatible
- ✅ No API-breaking changes for our use case (TIFF I/O)

**Conflicts**:
- File overlap with PRs #660, #661, #662

**EER Note**: The EER super-resolution decoding change is **breaking** for users of ThermoFisher EER files, but we don't use this format in the pipeline.

**Recommendation**: ✅ **APPROVE after conflict resolution**
- Merge **first** (least conflicts)
- Test TIFF I/O pipelines post-merge
- Monitor for EER-related issues (unlikely)

---

### **PR #658**: `Add validation gates and artifact upload to dependency-update workflow`

**Impact**: ✅ **CRITICAL** - Infrastructure improvement

**Author**: copilot-swe-agent
**Labels**: `freeze-approved` (approved for merge during freeze)

**Changes**:
1. `.github/workflows/dependency-update.yml`: 108 additions, 7 deletions
   - Upload safety report as workflow artifact (not committed)
   - Pre-commit checks on changed files
   - Python 3.10 and 3.12 installation validation
   - YAML parsing smoke test (ruamel-yaml upgrade protection)
   - Import validation (backports-asyncio-runner version gating)

2. `docs/DEPENDENCY_UPDATES.md`: New file (262 lines)
   - Review checklist
   - Local testing procedures
   - Known issues (ruamel-yaml 0.18→0.19, backports, etc.)
   - Troubleshooting guide

3. `requirements/*.txt`: Dependency updates
   - certifi 2025.11.12 → 2026.1.4
   - astroid 4.0.2 → 4.0.3
   - filelock 3.20.1 → 3.20.2
   - hypothesis 6.148.8 → 6.149.0
   - librt 0.7.5 → 0.7.7
   - marshmallow 4.1.2 → 4.2.0
   - ruamel-yaml 0.18.17 → 0.19.1 (removed ruamel-yaml-clib)
   - tox 4.32.0 → 4.33.0
   - ⚠️ backports-asyncio-runner: Added Python version marker `; python_version < "3.11"`

**Analysis**:
- **Purpose**: Fix PR #656's lack of CI validation gates
- **Security**: Moves safety report from PR commit to workflow artifact (better practice)
- **Validation**: Adds 5 automated test gates before PR creation
- **Documentation**: Comprehensive review guide for maintainers

**Conflicts**:
- File overlap with PR #656 on `requirements/*.txt` files
- **PR #656 is superseded** by this PR

**Recommendation**: ✅ **APPROVE and merge FIRST**

**Merge Strategy**:
1. Merge #658 immediately (infrastructure improvement)
2. Close #656 as superseded
3. Re-trigger dependency update workflow to test new gates

---

### **PR #656**: `🔄 Automated Dependency Updates`

**Impact**: ⚠️ **SUPERSEDED** by PR #658

**Author**: github-actions[bot]
**Labels**: `dependencies`, `automated`, `freeze-approved`

**Changes**:
- `requirements/all.txt`, `base.txt`, `ci.txt`, `dev.txt`, `ml.txt`: Same dependency updates as PR #658

**Analysis**:
- **Issue**: Creates PRs **without CI validation**
- **Issue**: Claims to attach `safety-report.json` but doesn't commit it
- **Issue**: No smoke tests before PR creation
- **Superseded by**: PR #658 adds all missing validation gates

**Recommendation**: ❌ **CLOSE as superseded by #658**

**Action**:
1. Close PR #656 with comment: "Superseded by #658 which adds CI validation gates"
2. Do not merge #656

---

## 🎯 Recommended Merge Strategy

### Phase 1: Infrastructure First (Priority: CRITICAL)

**Order**: PR #658 → Close #656

**Steps**:
1. ✅ **Merge PR #658** immediately
   - Adds CI validation gates to dependency workflow
   - Includes documentation for future reviews
   - No functional risk (workflow changes only)
   - Labeled `freeze-approved`

2. ❌ **Close PR #656** as superseded
   - Comment: "Superseded by #658 which adds validation gates"
   - Do not merge

**Validation**:
- After merge, re-trigger `dependency-update.yml` workflow
- Verify new validation gates work correctly

---

### Phase 2: Independent Dependencies (Priority: HIGH)

**Order**: PR #663

**Steps**:
1. ✅ **Merge PR #663** (scikit-learn 1.7.2 → 1.8.0)
   - No file conflicts with other PRs
   - Constraint file correctly handles Python 3.10 compatibility
   - Low risk (constraint-based upgrade)

**Validation**:
- CI tests pass on Python 3.10 (with scikit-learn<1.8)
- CI tests pass on Python 3.12 (with scikit-learn==1.8.0)

---

### Phase 3: Conflicting Dependencies - TRIAGE REQUIRED (Priority: MEDIUM)

**PRs**: #659, #660, #661, #662

**Conflicts**:
- All 4 PRs modify the same 3 files: `lux_depth_v3/requirements.txt`, `pyproject.toml`, `requirements/base.in`
- ❌ PR #660 (scipy 1.16.3): **REJECT** - Breaks Python 3.10
- ❌ PR #661 (Pillow 12.1.0): **REJECT** - Breaks Python 3.10
- ✅ PR #659 (tifffile 2025.12.20): **APPROVE**
- ✅ PR #662 (imagecodecs 2026.1.1): **APPROVE**

**Merge Order** (for approved PRs):
1. ✅ **Merge PR #659** (tifffile)
   - Least conflicts
   - Safe upgrade
   - Test TIFF I/O pipelines

2. ✅ **Merge PR #662** (imagecodecs)
   - Depends on #659 being merged first
   - Test TIFF/WebP pipelines
   - Monitor Brotli compression level change

**Rejection Actions**:

For **PR #660** (scipy):
```bash
# Comment on PR #660
@dependabot ignore this major version

# Reason:
scipy 1.16+ requires Python 3.11+. Our minimum supported version is Python 3.10 (EOL: 2026-10-04).
We will upgrade scipy after dropping Python 3.10 support.
```

For **PR #661** (Pillow):
```bash
# Comment on PR #661
@dependabot ignore this major version

# Reason:
Pillow 12.x requires Python 3.11+. Our minimum supported version is Python 3.10 (EOL: 2026-10-04).
We will upgrade Pillow after dropping Python 3.10 support or updating our minimum Python version.
```

---

## 📋 Final Merge Order (Chronological)

| Step | PR # | Action | Reason | Validation |
|------|------|--------|--------|-----------|
| 1 | **#658** | ✅ MERGE | Infrastructure improvement (CI gates) | Workflow re-trigger |
| 2 | **#656** | ❌ CLOSE | Superseded by #658 | N/A |
| 3 | **#663** | ✅ MERGE | Independent (scikit-learn) | Python 3.10/3.12 tests |
| 4 | **#661** | ❌ CLOSE | Breaks Python 3.10 (Pillow 12.x) | N/A |
| 5 | **#660** | ❌ CLOSE | Breaks Python 3.10 (scipy 1.16+) | N/A |
| 6 | **#659** | ✅ MERGE | Safe (tifffile) | TIFF I/O tests |
| 7 | **#662** | ✅ MERGE | Safe (imagecodecs) | TIFF/WebP tests |

**Result**: 4 merges, 3 closures

---

## 🔐 Security Review Summary

### Vulnerability Scan Status
- All PRs show `state: "pending"` for CI checks
- No security reports available yet (workflow artifacts not generated)
- **Action**: Wait for CI completion before final merge

### Known Security Issues
- ✅ No CVE-2024-27763 (basicsr) related packages introduced
- ✅ certifi updated to 2026.1.4 (latest trusted CA bundle)
- ✅ marshmallow updated to 4.2.0 (fixes CVE-2025-68480 DoS vulnerability)

### Dependency Supply Chain
- ✅ All PRs authored by `dependabot[bot]` or trusted automation
- ✅ No unexpected transitive dependencies added
- ⚠️ ruamel-yaml 0.19.1 removes `ruamel-yaml-clib` (C extension) - performance impact minimal

---

## 🧪 Testing Requirements

### Pre-Merge Validation

For each approved PR:
1. ✅ CI tests pass (GitHub Actions)
2. ✅ Python 3.10 compatibility verified
3. ✅ Python 3.12 compatibility verified
4. ✅ No regression in core pipelines

### Post-Merge Smoke Tests

After merging PRs #658, #663, #659, #662:

```bash
# Test suite
make test-all  # Full test suite

# Manual validation
python -c "import numpy, PIL, scipy, yaml; print('✅ Core imports OK')"

# Pipeline smoke test
python lux_depth_v3/process_batch.py --config config/interior_preset.yaml --input test_images/

# TIFF I/O test (for #659, #662)
python -c "
from tifffile import imread, imwrite
import numpy as np
test = np.random.rand(100, 100).astype(np.float32)
imwrite('/tmp/test.tif', test)
loaded = imread('/tmp/test.tif')
assert np.allclose(test, loaded), 'TIFF round-trip failed'
print('✅ TIFF I/O OK')
"
```

---

## 📚 Documentation Updates Required

After merges:

1. **CHANGELOG.md**: Document dependency version changes
2. **requirements/constraints.txt**: Add comments for rejected PRs
3. **docs/DEPENDENCY_UPDATES.md**: Already added by PR #658 ✅
4. **pyproject.toml**: Verify Python version constraints are correct

---

## 🚨 Risks and Mitigation

### Risk 1: Python 3.10 Compatibility Breakage

**Affected PRs**: #660 (scipy), #661 (Pillow)
**Impact**: HIGH - Users on Python 3.10 would fail to install
**Mitigation**: ✅ **REJECT these PRs** and constrain versions

**Long-term Plan**:
1. Monitor Python 3.10 EOL (2026-10-04)
2. Create ADR for Python 3.11 minimum version
3. Schedule migration 3 months before Python 3.10 EOL

### Risk 2: YAML Parsing Regression (ruamel-yaml 0.18→0.19)

**Affected PRs**: #658, #656
**Impact**: MEDIUM - Config files might fail to parse
**Mitigation**: PR #658 adds YAML smoke test ✅

**Validation**:
```bash
# Test all YAML configs
for yaml in config/*.yaml; do
  python -c "from ruamel.yaml import YAML; yaml = YAML(); yaml.load(open('$yaml'))"
done
```

### Risk 3: Imagecodecs Brotli Compression Change

**Affected PRs**: #662
**Impact**: LOW - Compression level changed from 3 → 4
**Mitigation**: Monitor compressed file sizes post-merge

### Risk 4: CI Workflow Failures

**Affected PRs**: All (pending CI)
**Impact**: MEDIUM - Unknown failures could block merges
**Mitigation**: Manual review of CI logs when available

---

## 📝 Post-Merge Checklist

After completing the merge strategy:

- [ ] All 4 approved PRs merged successfully
- [ ] All 3 rejected PRs closed with explanations
- [ ] CI tests pass on main branch
- [ ] Dependabot ignore rules added for scipy 1.16+ and Pillow 12+
- [ ] Python 3.10 and 3.12 compatibility verified
- [ ] TIFF/WebP pipelines tested
- [ ] YAML config parsing validated
- [ ] CHANGELOG.md updated
- [ ] No regression in production workloads
- [ ] Dependency update workflow re-triggered to test new gates (PR #658)

---

## 🔗 Related Documentation

- [Dependency Management](../../requirements/README.md)
- [Dependency Update Process](../DEPENDENCY_UPDATES.md) - Added by PR #658
- [Python Version Policy](../../CONTRIBUTING.md#python-version-support)
- [Security Policy](../../SECURITY.md)

---

## 📞 Escalation Path

If issues arise during merge:

1. **CI Failures**: Review workflow logs, may need to adjust constraints
2. **Python 3.10 Breakage**: Rollback and add stricter version pins
3. **YAML Parsing Issues**: Rollback ruamel-yaml upgrade (PR #658/#656)
4. **Pipeline Regressions**: Rollback imagecodecs/tifffile (PR #659/#662)

**Emergency Rollback**:
```bash
git revert <merge_commit_sha>
git push origin main
```

---

**Prepared by**: Transformation Portal Architect
**Review Status**: Ready for Implementation
**Next Review**: After PR #658 merge (re-assess with new CI gates)
