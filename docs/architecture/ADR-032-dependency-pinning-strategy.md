# ADR-032: Dependency Pinning Strategy

**Status:** Accepted
**Date:** 2026-02-16
**Scope:** Repository-wide dependency management
**Stakeholders:** Platform Engineering, Security, CI/CD
**Related ADRs:** ADR-031 (Test Dependency Isolation)

---

## Executive Summary

This ADR defines the **Dependency Pinning Strategy** for the Transformation Portal repository to ensure deterministic builds, security compliance, and maintainability. It establishes three constraint styles (strict pins, range pins, lower-bound-only), documents banned packages, and implements automated enforcement via pre-commit hooks and CI validation.

---

## Context

### Current State

The repository uses `pip-compile` workflow with abstract requirements (`.in` files) compiled to pinned concrete requirements (`.txt` files). However:

1. **No enforcement of constraint styles**: Developers can commit unpinned dependencies
2. **No validation of compiled artifacts**: `.txt` files may be stale or manually edited
3. **No banned package scanning**: Unmaintained/vulnerable packages can slip in
4. **No documented pinning philosophy**: Inconsistent decisions across files

### Analysis of Current Dependencies

**Pattern Distribution (as of 2026-02-16):**

| Constraint Style | Count | Percentage | Files               |
|------------------|-------|------------|---------------------|
| Range pin        | 30    | 65%        | All files           |
| Lower-bound-only | 9     | 20%        | dev.in, ml.in       |
| Strict pin       | 1     | 2%         | ml.in (rawpy)       |
| Complex/Other    | 1     | 2%         | ci.in (pypdf)       |
| Unpinned         | 0     | 0%         | None (good!)        |

**Key Observations:**

- **Range pins dominate** (65%): good practice for production dependencies
- **Lower-bound-only** (20%): concentrated in development tools (linters, type checkers)
- **Strict pin** (2%): only `rawpy==0.26.0` for deterministic RAW processing
- **No unpinned dependencies**: excellent baseline

### Risks Without Enforcement

1. **Non-deterministic builds**: Unpinned deps cause "works on my machine" failures
2. **Security vulnerabilities**: Outdated deps without minimum version enforcement
3. **Compatibility regressions**: Missing upper bounds break on new major releases
4. **Maintenance burden**: Manual detection of violations during code review
5. **Compilation drift**: `.txt` files manually edited instead of compiled from `.in`

---

## Decision

### 1. Constraint Style Taxonomy

Define three constraint styles with clear usage criteria:

#### A) **Strict Pin** (`package==X.Y.Z`)

**Use when:**
- Deterministic behavior is critical (e.g., ML model weights, raw file processing)
- Known incompatibilities between versions
- Security-sensitive package with specific patched version
- Reproducible research results required

**Examples:**
```
rawpy==0.26.0               # Deterministic RAW demosaic (LibRaw wrapper)
```

**Constraints:**
- Must include inline comment explaining rationale
- Requires periodic review (quarterly) to prevent stagnation
- Security patches override strict pins (document in git commit)

#### B) **Range Pin** (`package>=X.Y,<Z`)

**Use when (DEFAULT for production dependencies):**
- Package follows semantic versioning
- API stability expected within major version
- Need security patches without manual intervention
- Standard production dependency

**Examples:**
```
numpy>=1.24,<2.5.0         # Upper bound for opencv compatibility
Pillow>=10.0.0,<13         # Standard range for stable API
pydantic>=2.0,<3           # Major version boundary
```

**Constraints:**
- Lower bound: minimum tested version (not arbitrary)
- Upper bound: next major version (for semver) or known incompatibility
- Must test against both bounds before committing
- Security minimum overrides developer preference
- The enforced `.in` constraint is the policy source of truth; ADR examples must mirror it.

#### C) **Lower-Bound-Only** (`package>=X.Y`)

**Use when (EXCEPTIONS ONLY):**
- Development tool with stable CLI (linters, formatters, type checkers)
- Package has strong backward compatibility guarantee
- Upper bound is impractical or counterproductive
- Not a production runtime dependency

**Examples:**
```
mypy>=1.10                 # Type checker: CLI stable, benefits from latest rules
pylint>=3.0                # Linter: CLI stable, benefits from latest diagnostics
flake8>=7.0                # Linter: new rules are improvements
```

**Constraints:**
- **ONLY allowed in `dev.in` and `ci.in`** (not `base.in` or `ml.in`)
- Must have strong rationale (document in ADR-032 or inline comment)
- Regular monitoring for breaking changes (quarterly review)

#### D) **Unpinned** (BANNED)

**Never allowed** in any `.in` file. Unpinned dependencies cause:
- Non-deterministic CI failures
- "Works on my machine" debugging nightmares
- Security audit gaps

**Enforcement:** Automated validation fails on unpinned dependencies.

---

### 2. Banned Packages Registry

Maintain a registry of banned packages with migration guidance.

| Package      | Reason                     | Migration Path                          |
|--------------|----------------------------|-----------------------------------------|
| `realesrgan` | Unmaintained (no updates since 2022) | Use local implementation in `src/spatial_ai/reconstruction/` |

**Process for adding to banned list:**
1. Document reason (unmaintained, security, license, architecture)
2. Provide migration path or alternative
3. Update validation script with detection rule
4. Add to this ADR table

---

### 3. Security-Sensitive Dependency Policy

**Minimum Version Enforcement:**

Certain packages require minimum versions due to CVEs or security patches:

| Package                 | Minimum Version | Reason                                  |
|-------------------------|-----------------|-----------------------------------------|
| `sentence-transformers` | >=3.1.0         | CVE-73169 (arbitrary code execution)    |
| `Pillow`                | >=10.0.0        | Multiple CVEs in 9.x series             |

**Update Cadence:**

- **Critical CVEs** (CVSS ≥9.0): Immediate update within 48 hours
- **High CVEs** (CVSS 7.0-8.9): Update within 1 week
- **Medium/Low CVEs**: Include in quarterly review cycle
- **Quarterly dependency audit**: Review all deps for updates, CVEs, deprecations

**Security Monitoring Tools:**

- `pip-audit` (in `requirements/security.in`): **Sole governed blocking dependency scanner.** Run in `.github/workflows/security-unified.yml` (`dependency-scan` job) on every PR and on the nightly + Tuesday-staggered schedules. **Scan target:** the installed Python 3.11 environment built from `requirements/constraints.txt` + `requirements-ci.txt` + editable project install + `requirements/security.txt`; the scanner bootstrap additionally pins the non-vulnerable toolchain `pip==26.1.2` and `setuptools==83.0.0`. The workflow invokes `pip-audit` with no `-r` flag, so it audits resolved packages rather than a single lockfile spec. **Block policy:** every reported advisory fails the job and blocks merge; any audit execution failure, missing report, or malformed report also fails closed. `pip-audit` 2.10.1 does not include severity in its JSON report, so the workflow does not attempt severity filtering; it counts the records under `dependencies[].vulns[]` and propagates the scanner's nonzero exit. **Where to find details:** the full pip-audit JSON is printed to the job log and uploaded as the `security-reports-<sha>` `audit-report.json` artifact (30-day retention); the GitHub step summary records the nested vulnerability count. **Exception process:** time-bound suppressions use `pip-audit --ignore-vuln <CVE-ID>` with an inline comment in the workflow naming the CVE, the upstream tracking link, and the removal trigger. Suppressions are reviewed every quarter; un-removable ones escalate to the next Approved Exceptions table entry.
- CI bootstrap tools are part of the security baseline. Determinism workflows pin `pip==26.1.2`, `setuptools==83.0.0`, and `wheel==0.46.2`; lock-generation workflows pair pip 26.1.2 with `pip-tools==7.5.3`, the first compatible 7.5.x release. These pins must move atomically when a bootstrap-tool advisory or compatibility boundary changes.
- `bandit[toml]` (in `requirements/security.in`): Static security analysis for Python source.
- Dependabot: GitHub-native automated PR creation for security patches.

> `safety` was removed from this toolchain in March 2026 — see the Amendments section below for rationale. Re-introduction requires a fresh ADR amendment.

---

### 4. Exception Criteria

Exceptions to pinning rules require explicit approval via one of:

**A) Documented in ADR-032** (this document)
- Add to "Approved Exceptions" table below
- Include rationale, risk assessment, monitoring plan

**B) Inline Comment in `.in` File**
- For one-off cases not warranting ADR update
- Must include: reason, reviewer, date

**C) Emergency Override**
- Security patches may temporarily violate rules
- Document in git commit message
- Create follow-up issue for permanent resolution

**Approved Exceptions Table:**

| Package       | File    | Constraint      | Rationale                                  |
|---------------|---------|-----------------|------------------------------------------- |
| `mypy`        | dev.in  | `>=1.10`        | Type checker: benefits from latest rules   |
| `black`       | dev.in  | `>=26.3.1`      | Formatter: deterministic, auto-updates OK  |
| `flake8`      | dev.in  | `>=7.0`         | Linter: new rules are improvements         |
| `pylint`      | dev.in  | `>=3.0`         | Linter: CLI stable across minor versions   |
| `PyYAML`      | ml.in   | `>=6.0`         | Config parser: strong backward compat      |
| `colour-science` | ml.in | `>=0.4.2`    | Color math library: stable API             |
| `coremltools` | ml.in   | `>=7.0`         | Apple ML tools: platform-specific updates  |
| `psutil`      | ml.in   | `>=5.9.0`       | System utilities: OS compatibility layer   |
| `memory-profiler` | ml.in | `>=0.61.0`  | Dev/profiling tool in optional deps        |
| `types-PyYAML` | dev.in | `>=6.0.12`     | Type stubs: must track PyYAML version      |
| `pypdf`       | ci.in   | `>=6.13.3`      | PDF utilities: backward-compatible 6.x security floor |

---

### 5. Validation Script Design

Create `scripts/validate_dependency_constraints.sh` with:

**Validations:**

1. **Constraint Style Check**
   - Detect unpinned dependencies (fail)
   - Detect lower-bound-only in `base.in`/`ml.in` (fail unless approved)
   - Validate strict pins have inline comments (warn)

2. **Compilation Freshness Check**
   - Verify `.txt` files are newer than `.in` sources
   - Detect manually edited `.txt` files (check for pip-compile header)
   - Suggest `make compile` if stale

3. **Banned Package Scan**
   - Check for packages in banned registry
   - Provide migration guidance in error message

4. **Security Minimum Version Check**
   - Enforce minimum versions for CVE-patched packages
   - Fail if version below security threshold

**Output:**

- Exit 0: All validations pass
- Exit 1: Blocking violations found (unpinned, banned, security)
- Exit 2: Non-blocking warnings (missing comments, suggested improvements)

**Example Output:**

```
🔍 Validating dependency constraints...

✅ requirements/base.in: All constraints valid
✅ requirements/dev.in: All constraints valid
❌ requirements/ml.in: Violations found

  Line 15: realesrgan>=0.3.0,<1
  └─ ERROR: Package 'realesrgan' is BANNED (unmaintained)
     Migration: Use local implementation in src/spatial_ai/reconstruction/

  Line 23: some-package
  └─ ERROR: Unpinned dependency (no version constraint)
     Fix: Add version constraint (>=X.Y,<Z for production, >=X.Y for dev tools)

  Line 8: critical-lib>=1.0.0,<2
  └─ ERROR: Security minimum not met (need >=1.5.0 for CVE-2024-12345)
     Fix: Update constraint to >=1.5.0,<2

⚠️  3 violations found (3 errors, 0 warnings)
Run 'make compile' after fixing .in files to update .txt files.
```

---

### 6. Enforcement Strategy

Implement 4-layer defense-in-depth (mirrors ADR-031 test isolation):

**Layer 1: Documentation** (this ADR)
- Canonical reference for pinning decisions
- Onboarding for new contributors
- Rationale for current state

**Layer 2: Pre-commit Hook**
- Local validation before commit
- Fast feedback (<10s)
- Blocks unpinned deps, banned packages, stale `.txt` files
- Provides fix guidance

**Layer 3: CI Validation**
- Automated gate in Quality Firewall workflow
- Runs on every PR and push to main
- Fast-fail (early in pipeline)
- Generates validation report artifact

**Layer 4: Developer Guide** (CONTRIBUTING.md)
- Workflow documentation (edit `.in`, run `make compile`)
- Constraint style decision tree
- Exception request process

---

## Implementation Plan

### Phase 1: Validation Script (Completed)

**File:** `scripts/validate_dependency_constraints.sh`

**Features:**
- ✅ Constraint style validation (unpinned, lower-bound-only rules)
- ✅ Compilation freshness check (`.txt` vs `.in` timestamps)
- ✅ Banned package scanning (registry-based)
- ✅ Security minimum version enforcement
- ✅ Colorized output with fix guidance
- ✅ Exit codes (0=pass, 1=fail, 2=warn)

### Phase 2: CI Integration (Completed)

**File:** `.github/workflows/ci-quality-firewall.yml`

**Jobs:** `validate-dependency-constraints`, `validate-python-compatibility`

**Behavior:**
- Runs after checkout, before tests
- Fast-fail on violations
- Simulates dependency resolution for supported Python versions (3.11, 3.12)
- Uploads validation report as artifact
- Blocks PR merge on failure

### Phase 3: Pre-commit Hook (Completed)

**File:** `.pre-commit-config.yaml`

**Hook:** `validate-dependency-constraints`

**Behavior:**
- Runs on `git commit` for `.in` or `.txt` file changes
- Blocks commit on violations
- Suggests `make compile` if needed
- Provides inline fix guidance

### Phase 4: Documentation (Completed)

**Files:**
- `docs/architecture/ADR-032-dependency-pinning-strategy.md` (this file)
- `CONTRIBUTING.md` (new "Dependency Management" section)

**Content:**
- Constraint style decision tree
- Update workflow (edit `.in` → `make compile`)
- Exception request process
- Security patch workflow

---

## Migration Guidance

### Addressing Current Violations

**Step 1: Identify Violations**

Run validation script:
```bash
./scripts/validate_dependency_constraints.sh
```

**Step 2: Fix Violations**

For each violation type:

**Unpinned dependency:**
```diff
- my-package
+ my-package>=1.0.0,<2  # Production dep: use range pin
```

**Lower-bound-only in production file:**
```diff
# In base.in or ml.in:
- my-lib>=1.5
+ my-lib>=1.5,<2        # Add upper bound for determinism
```

**Banned package:**
```diff
- realesrgan>=0.3.0,<1
+ # Use local implementation in src/spatial_ai/reconstruction/
```

**Security minimum not met:**
```diff
- sentence-transformers>=3.0.0,<6
+ sentence-transformers>=3.1.0,<6  # CVE-73169 remediation
```

**Step 3: Recompile**

```bash
cd requirements && make compile
```

**Step 4: Validate**

```bash
./scripts/validate_dependency_constraints.sh
```

---

## Alternatives Considered

### Alternative 1: Strict Pins for Everything

**Approach:** Pin all dependencies with `==X.Y.Z`

**Pros:**
- Maximum determinism
- Zero surprises

**Cons:**
- Security patches require manual intervention
- Maintenance burden (every patch = PR)
- Dependency hell (transitive deps conflict)
- Breaks `pip install -e .` workflow (over-constrained)

**Decision:** Rejected. Range pins with upper bounds provide determinism AND security flexibility.

---

### Alternative 2: Unpinned with Lockfiles Only

**Approach:** Use unpinned `.in` files, rely on `requirements.txt` as lockfile

**Pros:**
- Simple `.in` files
- Familiar to JavaScript developers (package.json + lockfile)

**Cons:**
- `.in` files provide no guidance to developers
- Security minimums not enforceable
- Banned packages must be detected in `.txt` (late)
- No constraint documentation (rationale lost)

**Decision:** Rejected. Abstract constraints in `.in` + concrete lockfile in `.txt` is best of both worlds.

---

### Alternative 3: Poetry/Pipenv Instead of pip-compile

**Approach:** Use modern packaging tool (Poetry, Pipenv)

**Pros:**
- Integrated dependency resolution
- Better UX for developers
- Built-in lockfile management

**Cons:**
- Migration effort (rewrite all `.in` files)
- Different workflow (breaking change for contributors)
- Tool lock-in (pip-compile is standard)
- No enforcement of constraint styles (same validation needed)

**Decision:** Rejected for now. Keep pip-compile workflow; revisit in 2027 dependency audit.

---

## Consequences

### Positive Consequences

1. **Deterministic builds**: All envs (dev, CI, prod) use same dependency versions
2. **Security compliance**: Minimum versions enforced, CVE patches automated
3. **Developer clarity**: Clear rules for "which constraint style to use?"
4. **Reduced review burden**: Automated validation catches violations before review
5. **Maintainability**: Quarterly review process prevents stagnation
6. **Auditability**: Git history + inline comments document every pinning decision

### Negative Consequences

1. **Upfront learning curve**: Developers must learn taxonomy and rules
2. **Process overhead**: Exception requests require justification
3. **Quarterly reviews**: Ongoing maintenance burden (mitigated by automation)
4. **Potential for over-constraining**: Range pins may exclude compatible versions (monitor via CI failures)

### Mitigations

- **Onboarding docs** (CONTRIBUTING.md): Decision tree, examples, FAQ
- **Fast validation** (<10s pre-commit): Minimize friction
- **Actionable errors**: Fix guidance in every error message
- **Grace period**: Warnings before errors for new rules (30-day baseline)

---

## Metrics and Success Criteria

### Leading Indicators (Process Compliance)

- **Constraint coverage**: 100% of dependencies have version constraints
- **Validation pass rate**: 100% of PRs pass dependency validation
- **Exception rate**: <5% of dependencies require exceptions
- **Quarterly review completion**: 100% on-time

### Lagging Indicators (Outcome Quality)

- **Security patch latency**: Median time from CVE disclosure to deployment <48h for critical
- **Dependency staleness**: Median age of dependencies <6 months
- **Build reproducibility**: 100% of CI runs use identical dependency versions (for same git SHA)
- **Dependency-related failures**: <1% of CI failures due to dependency issues

### Baseline (2026-02-16)

- Total dependencies: 46 (across all `.in` files)
- Constraint coverage: 100% (0 unpinned)
- Range pins: 30 (65%)
- Lower-bound-only: 9 (20%, all in dev/ci)
- Strict pins: 1 (2%, rawpy)
- Banned packages: 1 (realesrgan, commented out)
- Security minimums enforced: 2 (sentence-transformers, Pillow)

---

## Enforcement Checklist

Track implementation progress:

- [x] **ADR-032 created** (this document)
- [x] **Validation script** (`scripts/validate_dependency_constraints.sh`)
  - [x] Constraint style validation
  - [x] Compilation freshness check
  - [x] Banned package scanning
  - [x] Security minimum version enforcement
  - [x] Colorized output with fix guidance
- [x] **CI integration** (`.github/workflows/ci-quality-firewall.yml`)
  - [x] `validate-dependency-constraints` job added
  - [x] `validate-python-compatibility` matrix job added (3.11, 3.12)
  - [x] Runs on PR and push to main
  - [x] Blocks merge on failure
  - [x] Uploads validation report artifact
- [x] **Pre-commit hook** (`.pre-commit-config.yaml`)
  - [x] Hook entry added
  - [x] Runs on `.in` and `.txt` changes
  - [x] Blocks commit on violations
- [x] **Documentation** (`CONTRIBUTING.md`)
  - [x] Dependency Management section added
  - [x] Constraint style decision tree
  - [x] Update workflow documented
  - [x] Exception process documented
- [x] **Baseline validation** (all current `.in` files pass)
- [ ] **Quarterly review scheduled** (first review: 2026-05-16)

---

## References

### Internal

- **ADR-031**: Test Dependency Isolation Contract (enforcement pattern template)
- **Issue #796**: CI Health & Stability (parent initiative)
- **CONTRIBUTING.md**: Developer guide (workflow documentation)

### External

- **PEP 440**: Version Identification and Dependency Specification
  https://peps.python.org/pep-0440/
- **Semantic Versioning 2.0.0**: https://semver.org/
- **pip-tools Documentation**: https://github.com/jazzband/pip-tools
- **NIST NVD**: National Vulnerability Database (CVE tracking)
  https://nvd.nist.gov/

### Security Resources

- **PyPI Advisory Database**: https://github.com/pypa/advisory-database (data source backing `pip-audit`)
- **pip-audit**: https://github.com/pypa/pip-audit
- **Snyk Vulnerability Database**: https://security.snyk.io/

---

## Appendix A: Decision Tree for Constraint Styles

```
┌─────────────────────────────────────┐
│ Need to add dependency to .in file │
└────────────────┬────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │ Which file?   │
         └───┬───────┬───┘
             │       │
    ┌────────┘       └────────┐
    ▼                         ▼
┌───────┐                 ┌───────┐
│base.in│                 │dev.in │
│ml.in  │                 │ci.in  │
└───┬───┘                 └───┬───┘
    │                         │
    ▼                         ▼
┌─────────────────────┐   ┌──────────────────────┐
│ Production runtime  │   │ Dev/CI tool only     │
└──────────┬──────────┘   └──────────┬───────────┘
           │                         │
           ▼                         ▼
    ┌──────────────┐         ┌─────────────────┐
    │ Determinism  │         │ CLI stable?     │
    │ critical?    │         │ (linter, etc.)  │
    └───┬────┬─────┘         └────┬─────┬──────┘
        │    │                    │     │
    Yes │    │ No             Yes │     │ No
        ▼    ▼                    ▼     ▼
    ┌────┐ ┌────────────┐   ┌────────┐ ┌────────┐
    │==  │ │>=X.Y,<Z    │   │>=X.Y   │ │>=X.Y,<Z│
    │X.Y.Z│ │(preferred) │   │(OK)    │ │(safer) │
    └────┘ └────────────┘   └────────┘ └────────┘
```

---

## Appendix B: Quarterly Review Checklist

Use this checklist every quarter (Feb, May, Aug, Nov):

### 1. Security Audit
- [ ] Reproduce the `security-unified.yml` `dependency-scan` job locally so quarterly results are comparable to merge-gate results: in a clean Python 3.11 venv, `pip install -c requirements/constraints.txt -r requirements-ci.txt`, then `pip install -c requirements/constraints.txt -e .`, then `pip install -r requirements/security.txt`, then `pip-audit --ignore-vuln <each-currently-active-suppression> --format json --output audit-report.json` against the installed environment (no `-r`, matching the workflow's scan target)
- [ ] Cross-check the local audit JSON against the latest scheduled run's `security-reports-<sha>` artifact (30-day retention) — any delta is itself an audit finding
- [ ] Review every active `--ignore-vuln` suppression in `.github/workflows/security-unified.yml`; remove any whose upstream fix has shipped
- [ ] Check Dependabot alerts
- [ ] Review NIST NVD for new CVEs
- [ ] Update security minimums table in ADR-032

### 2. Staleness Check
- [ ] List packages >6 months old: `pip list --outdated`
- [ ] Check for new major versions with breaking changes
- [ ] Update range pin upper bounds if safe
- [ ] Document compatibility testing in PR

### 3. Banned Package Review
- [ ] Check for newly unmaintained packages (no commits in 12+ months)
- [ ] Verify replacements for existing banned packages
- [ ] Update banned registry table

### 4. Exception Review
- [ ] Review approved exceptions table
- [ ] Validate rationale still applies
- [ ] Remove exceptions if no longer needed
- [ ] Add new exceptions if requested

### 5. Tooling Updates
- [ ] Update `pip`, `pip-tools`, `setuptools` to latest stable
- [ ] Test `make compile` workflow still works
- [ ] Update validation script if new patterns emerge

### 6. Documentation Sync
- [ ] Update baseline metrics in ADR-032
- [ ] Update CONTRIBUTING.md if workflow changed
- [ ] Archive old quarterly review notes

---

**Document Version:** 1.2
**Last Updated:** 2026-08-06 (Amendment A2)
**Next Review:** 2026-08-16 (Q3 2026; quarterly cadence per Appendix B)
**Approvers:** Platform Engineering Team

---

## Amendments

### A1 — 2026-05-18: Safety removal, pip-audit promoted to sole blocking scanner

**Scope:** §3 "Security Monitoring Tools", Appendix B Quarterly Review §1, References → Security Resources.

**Change:**
- Removed `safety` from the documented security-monitoring toolchain.
- Promoted `pip-audit` to **sole governed blocking dependency scanner**, with explicit block policy (HIGH/CRITICAL → fail) and `--ignore-vuln` exception process.
- Corrected `bandit`'s home file (`requirements/security.in`, not `ci.in`).
- Replaced the dead Safety DB external link with the upstream `pip-audit` repo.
- Updated quarterly review §1 to invoke `pip-audit` and to require a sweep of active `--ignore-vuln` suppressions.

**Rationale:** Safety was removed from `requirements/security.in` in March 2026 because (a) it pulls `nltk` as a transitive dependency, which generated ungoverned advisory noise, and (b) `pip-audit` provides equivalent CVE coverage via the PyPI Advisory Database without the transitive baggage. The change shipped in the workflow and the `security.in` source-of-truth then, but ADR-032 still cited Safety as a live tool — finding #8 in `docs/governance/PORTAL_AUDIT_REPO_WIDE_2026-05-18.md` flagged the drift, and this amendment closes it (backlog item I-4 in `docs/governance/audit/PORTAL_AUDIT_2026-05-18_backlog.md`).

**Sources of truth for the new posture:**
- `requirements/security.in` — toolchain composition (pip-audit + bandit; Safety removal recorded inline).
- `.github/workflows/security-unified.yml` — invocation, blocking semantics, current `--ignore-vuln` suppressions.

**Re-introduction policy:** Re-adding Safety (or any additional dependency scanner) requires a follow-up ADR amendment that explains how the transitive-dependency noise problem is mitigated and why pip-audit alone is insufficient.

---

### A2 — 2026-08-06: pip-audit aligned with its JSON schema and fail-closed policy

**Scope:** §3 "Security Monitoring Tools" and `.github/workflows/security-unified.yml`.

**Change:**
- Changed the governed block policy from severity filtering to blocking every reported advisory.
- Counted pip-audit 2.10.1 findings from `dependencies[].vulns[]` in the gate and job summary.
- Required missing or malformed reports and every nonzero scanner exit to fail closed.

**Rationale:** pip-audit 2.10.1 reports advisory records without severity metadata. The previous workflow queried nonexistent top-level `vulnerabilities` and `severity` fields, then ignored the captured scanner exit, so a report with known vulnerabilities could incorrectly pass. Blocking every advisory matches the direct pip-audit invocations in `.github/workflows/ci.yml` and `.github/workflows/ci-quality-firewall.yml` and keeps the governed scanner deterministic.

---

**END OF ADR-032**
