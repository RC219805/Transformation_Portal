# CI Workflow Matrix

**Purpose**: Canonical reference for all GitHub Actions workflows. Tracks the full inventory and the consolidation roadmap.
**Owner**: Transformation Portal Architect
**Last Updated**: 2026-09-21 UTC

---

## Status Snapshot

- **Workflow files**: 31 (`.github/workflows/*.yml`), 8,298 lines total
- **Required PR check snapshot (2026-09-18 UTC)**: GitHub main protection requires `CI Gate` and `Dependency Security`, both bound to GitHub Actions app `15368`, with strict up-to-date checking. Re-read remote settings before merge.
- **Additional remote rules**: active default-branch ruleset `9331244` configures Copilot review, code quality (`all`), and CodeQL scanning (both alert thresholds `all`); status contexts alone are not the entire merge policy.
- **Historical consolidation target**: 31 → ~19 workflows; the [roadmap](#consolidation-roadmap) is a proposal, not current authorization to remove gates.
- **Prior matrix doc** (2026-03-25) listed 12 workflows — this revision corrects the omission of 18 that exist on disk.

---

## Complete Inventory

Every `.github/workflows/*.yml` file, current as of the timestamp above. The **Recommendation** column is the proposal — *not yet executed*. Discuss before acting. **LOC** values count `Path.read_bytes().splitlines()` for each tracked workflow; blank and comment lines are included. The forensic baseline was `6e6c204456df404881c77f9db03585da3fc7f829`; LOC values below reflect the updated source. See the [forensic audit](GITHUB_FORENSIC_AUDIT_2026-09-18.md) for evidence, implemented changes, and residual work.

| # | File | Name | Triggers | Source behavior | LOC | Recommendation |
|---|------|------|----------|-----------|-------------|----------------|
| 1 | `build.yml` | CI (Lint, Tests & Manifest) | push, PR, manual | Aggregated by CI Gate | 1363 | **Keep** — primary PR gate; aggregated `CI Gate` check; lightweight documentation catalog, maintained-link and editorial lock validation |
| 2 | `ci.yml` | CI Quality Firewall (push) | push (main, develop) | Post-merge | 614 | **Investigate → Selective port into `build.yml`** — overlaps with `build.yml` on `lint`, `typecheck`, `test-core`, `test-ml`, but has **unique jobs** that build.yml does not currently provide: `security` (bandit + pip-audit on the push commit range), `coverage-gate`, `build` (packaging artifact), `repo-hygiene`, `quality-summary`. **Before retiring, port each unique job into `build.yml`** (or confirm it's shadowed by `security-unified.yml` / `enforcement.yml`) and **expand `build.yml`'s push branches to include `develop`** so post-merge coverage on `develop` isn't dropped. Naive deletion would lose real signal. |
| 3 | `ci-quality-firewall.yml` | CI Quality Firewall (post-CI) | workflow_run | Post-CI trusted push/manual | 971 | **Keep pending equivalence proof** — trusted `workflow_run` verification includes resolution, isolation, security, coverage, and flake-analysis evidence. A passing upstream `CI Gate` alone does not prove this workflow redundant. |
| 4 | `enforcement.yml` | Enforcement | push, PR, schedule | ⚠️ Partial | 231 | **Keep** — owns action-pin, banned-deps, HF-revision, artifact-boundary, layer-1/2 tests, golden-regression. Distinct from `build.yml` test surface. |
| 5 | `quality-gate.yml` | Quality Gate | PR, push | ⚠️ Advisory | 41 | **Investigate → Replace with pre-commit** — runs `scripts/lint_runner.sh advisory` and `scripts/setup/pre-commit-check.sh --all`. If those advisory lint and pre-commit checks are run by devs locally and by `build.yml`'s lint job, this duplicates. |
| 6 | `codeql.yml` | CodeQL Advanced | push, PR, schedule | Failing checks; see remote snapshot | 114 | **Keep** — GitHub semantic SAST for Actions, Python, and JavaScript/TypeScript; includes frontdoor and Worker source. |
| 7 | `security-unified.yml` | Security Unified | schedule, push, PR, manual | Dependency Security required on main | 279 | **Keep** — pip-audit + checkpoint-loading source gates; distinct from CodeQL. Unfiltered main PR trigger, read-only dependency job, no secret dependency or soft failure. |
| 8 | `dependency-review.yml` | Dependency Review | PR | ⚠️ Advisory (warn-only) | 30 | **Keep** — GitHub-native PR dependency check; minimal cost. The workflow sets `warn-only: true` and the job name is "Dependency Review (advisory)", so findings never fail a PR — they only post warnings. |
| 9 | `dependency-submission.yml` | Dependency Submission | push, PR, manual | ❌ No | 244 | **Keep** — feeds GitHub dependency graph; distinct concern. |
| 10 | `dependency-update.yml` | Dependency Updates | schedule, manual | ❌ No | 228 | **Keep** — Dependabot supplement; scheduled. |
| 11 | `dependency-pinning-check.yml` | Dependency Pinning Check | push, PR | Job fails on drift | 46 | **Keep** — focused exact-pin drift guard for requirements and the dependency-pinning validator; the same check also runs through `make ci`, but this workflow surfaces supply-chain drift quickly on scoped changes. |
| 12 | `secure-install-pilot.yml` | Secure Install Pilot | PR | ⚠️ Advisory | 63 | **Investigate** — pilot/experimental; check whether the pilot has graduated or should be retired. |
| 13 | `nightly.yml` | Nightly Deep Checks | schedule (2 AM UTC), manual | ❌ No | 457 | **Keep** — owns stress, benchmarks, memory leak, deep dep audit, full integration. |
| 14 | `ml-slow-suite.yml` | ML Slow Suite | schedule (3:30 AM UTC), manual | ❌ No | 155 | **Merge → `nightly.yml`** — overlapping concern (long-running ML coverage). Note schedule differs (`nightly.yml` runs `0 2 * * *`, `ml-slow-suite.yml` runs `30 3 * * *`); the merger needs to either preserve both crons (two `cron:` entries with a job-level guard on schedule) or pick one — choosing 3:30 AM UTC keeps the ML cache warm from the 2 AM nightly run. Decide before consolidation. |
| 15 | `performance-monitor.yml` | Performance Monitor | schedule (3:30 AM UTC), manual | ❌ No | 219 | **Keep** — schedule-only by design; runs the Lux perf smoke harness against committed baselines (`tests/benchmarks/baselines/`) and is green only when `scripts/ci/check_performance_evidence.py` proves the required tests executed and the four artifacts were produced (repair 1.6-a, #2062); the gate's exit code also classifies regression (valid evidence, failed baseline-writer comparison) vs failed (invalid evidence, or a non-writer suite failure); distinct from nightly benchmarks. |
| 16 | `apex_performance.yml` | APEX Performance Matrix | PR, push, manual, schedule | ⚠️ Advisory | 657 | **Keep** — APEX-specific matrix runner with synthetic PR/push evidence and real scheduled/manual backend runs. Standalone domain. |
| 17 | `apex_policy_validation.yml` | APEX Policy Validation | PR, push | Failing checks; see remote snapshot | 125 | **Merge → `contract-validations.yml`** (proposed new) |
| 18 | `evalsuite_contract_validation.yml` | Eval Suite Contract Validation | push, PR, manual | Failing checks; see remote snapshot | 95 | **Merge → `contract-validations.yml`** |
| 19 | `ingest_contract_validation.yml` | Ingest Contract Validation | PR, push, manual | Failing checks; see remote snapshot | 229 | **Merge → `contract-validations.yml`** |
| 20 | `machine_mode_contract_validation.yml` | Machine Mode Contract Validation | push, PR, manual | Failing checks; see remote snapshot | 159 | **Merge → `contract-validations.yml`** |
| 21 | `determinism-gate.yml` | determinism-gate | PR | Failing checks; see remote snapshot | 205 | **Merge → `determinism.yml`** (proposed) — combine with cross-ISA |
| 22 | `determinism-cross-isa.yml` | Determinism Harness / Cross-ISA Parity | PR | ⚠️ Advisory | 195 | **Merge → `determinism.yml`** |
| 23 | `docs.yml` | Documentation | PR, push, manual | ⚠️ Advisory | 133 | **Keep** — multi-job (build, validate-markdown, navigation); coherent. |
| 24 | `frontdoor-deployment-gate.yml` | Frontdoor Deployment Gate | manual only | Manual | 131 | **Keep** — operational gate for frontdoor deploys. |
| 25 | `diagnostic-trial.yml` | diagnostic-trial | PR, manual | ⚠️ Advisory | 188 | **Keep / revisit naming** — advisory determinism/provenance diagnostic gate running `scripts/diagnostics/full_chain_determinism_trial.sh` plus bundle-root and evidence-manifest checks. Concrete behavior, just badly named. |
| 26 | `submit-pypi.yml` | Submit to PyPI | push (tag), manual | Release | 119 | **Keep** — tag pushes publish production; manual dispatch can publish only Test PyPI. |
| 27 | `tag_retention_prune_preservation_tags.yml` | Tag Retention (Preservation Tags) | schedule, manual | ❌ No | 40 | **Keep** — periodic cleanup. |
| 28 | `ai-code-review.yml` | AI Code Review | PR | ❌ No | 321 | **Measure advisory usefulness** — missing credentials skip checkout/install; service diagnostics remain log-only. PR #1558 rate-limit observations are historical, not current telemetry. |
| 29 | `summary.yml` | Issue Summarizer | issue_comment, PR, issues | ❌ No | 275 | **Measure advisory usefulness** — expected service failures are advisory; inspect current run evidence before changing policy. |
| 30 | `smart-issue-management.yml` | Smart Issue Management | issues, pull_request_target | ❌ No | 347 | **Investigate → Consolidate or retire** — same. |
| 31 | `issue_printer.yml` | Print Issue Info | issues | ❌ No | 24 | **Retirement candidate** — prints title/body to logs without enforcement or artifacts; retained pending ownership decision. |

**Interpretation:** A workflow can fail even when its status is not required
by branch protection. Only the dated GitHub snapshot above establishes remote
required checks. `continue-on-error`, `warn-only`, conditional jobs, and gate
aggregation must be inspected separately; the source-behavior column is not a
claim that every listed workflow blocks merge.

The post-CI firewall remains `workflow_run`-only and verifies the exact upstream
SHA after a successful same-repository push/manual run. It is not redundant
merely because `CI Gate` passed. The consolidation text below is retained as a
historical proposal; no retirement or policy change is authorized by this doc.

Post-CI core tests use the push firewall's `pytest -n auto` parallelism within
the existing 20-minute job limit. Both retain the same positive marker selection
and coverage reports; post-CI also retains its per-Python JSON test evidence.

---

## Recommendations Grouped by Tier

> **Historical proposal, retained for context (2026-04-27).** Reassess current
> workflow responsibilities and trusted-event boundaries before any change.
> These estimates and retirement suggestions are not current implementation
> status or approval to remove gates.

### Tier A — Definitely retire / merge (low risk, high confidence)

These are clear wins. Aggregated reduction: **3 fewer workflows**.

1. **Retire `issue_printer.yml`** — 24 lines, prints to logs, zero downstream consumers. Search the repo for "Print Issue Info" or `issue_printer` to confirm no documentation references it. Then delete.

2. **Selectively port `ci.yml` into `build.yml`, then retire `ci.yml`** — `ci.yml` shares `lint`, `typecheck`, `test-core`, `test-ml` with `build.yml`, but **owns unique jobs**: `security` (bandit + pip-audit on the push commit range), `coverage-gate`, `build` (packaging artifact), `repo-hygiene`, `quality-summary`. Before retiring:
   - Port each unique job into `build.yml` guarded by `if: github.event_name == 'push'` (or confirm it's already shadowed — e.g., `security-unified.yml` runs pip-audit; `enforcement.yml` runs banned-deps).
   - **Expand `build.yml`'s push branches to include `develop`** (currently only `main`); otherwise retiring `ci.yml` silently drops post-merge coverage on the `develop` branch.
   - Update branch-protection required-checks list if `ci.yml` jobs are listed there.

3. **Retire `ci-quality-firewall.yml`** — 959 lines (largest secondary workflow file after `build.yml`) of `workflow_run` secondary gating. If `build.yml` is required at branch protection, this re-litigates the same checks. Confirm by listing what jobs `ci-quality-firewall.yml` runs that `build.yml` doesn't, then delete.

### Tier B — Consolidate domains (medium effort, clear benefit)

Aggregated reduction: **5 fewer workflows** (4 contract validators → 1 = net −3; 2 determinism → 1 = net −1; ml-slow → nightly = net −1).

4. **Create `contract-validations.yml`** combining four single-purpose contract validators:
   - `apex_policy_validation.yml` (125 lines)
   - `evalsuite_contract_validation.yml` (95 lines)
   - `ingest_contract_validation.yml` (229 lines)
   - `machine_mode_contract_validation.yml` (159 lines)

   Each becomes a job in the unified workflow. Triggers: `pull_request, push, workflow_dispatch`. Use a change-classification job and job-level `if:` conditions so only affected validators run; GitHub does not support a job-level `paths:` trigger. Estimated combined size after dedupe of setup steps: ~400 lines (vs. 608 today).

5. **Create `determinism.yml`** combining the two determinism workflows (`determinism-gate.yml` + `determinism-cross-isa.yml`). 400 lines combined → likely ~300 after dedupe.

6. **Merge `ml-slow-suite.yml` into `nightly.yml`** as a `ml-slow` job. **Different cron schedules** (`nightly.yml` = 2 AM UTC; `ml-slow-suite.yml` = 3:30 AM UTC) — the consolidation must consciously choose one schedule (or define two `cron:` entries on the unified workflow with a job-level guard). 3:30 AM UTC keeps the ML cache warm from the 2 AM nightly run, but either choice is a real operational change vs. today.

### Tier C — Investigate before acting

These need maintainer judgment, not mechanical merging.

7. **Three AI advisory workflows** (`ai-code-review.yml`, `summary.yml`, `smart-issue-management.yml`, total 937 lines). They share a hardening baseline (per `AI_WORKFLOWS_HARDENING_STATUS.md`) and currently fail with rate-limit errors on most PRs. Options:
   - **Consolidate** into one `ai-advisory.yml` with three jobs sharing setup (saves ~200 LOC)
   - **Retire** if the value-to-noise ratio is unfavorable (recent PR #1558: 4 of 4 AI calls errored)
   - **Keep separate** if isolation is desirable for independent ownership

8. **`quality-gate.yml`** — likely redundant with pre-commit + `build.yml`'s lint job. Confirm what it adds.

9. **`secure-install-pilot.yml`** — explicitly a pilot. Check whether it has graduated, in which case fold its jobs into `security-unified.yml`; if dormant, retire.

10. **`diagnostic-trial.yml`** — concrete behavior is now clear (advisory determinism/provenance diagnostic gate; runs `scripts/diagnostics/full_chain_determinism_trial.sh` plus bundle-root and evidence-manifest checks), but the name doesn't reflect that. Either rename to `determinism-trial.yml` or fold the jobs into a future `determinism.yml` (see Tier B item 5). No retirement candidate.

### Tier D — Keep as-is

`build.yml`, `enforcement.yml`, `codeql.yml`, `security-unified.yml`, `dependency-review.yml`, `dependency-submission.yml`, `dependency-update.yml`, `dependency-pinning-check.yml`, `nightly.yml`, `performance-monitor.yml`, `apex_performance.yml`, `docs.yml`, `frontdoor-deployment-gate.yml`, `submit-pypi.yml`, `tag_retention_prune_preservation_tags.yml`. Each owns a distinct concern.

---

## Consolidation Roadmap

Sequence the work so each PR is independently revertible.

| PR | Tier | Scope | Net workflow count |
|----|------|-------|---------------------|
| 1 | A | Retire `issue_printer.yml` | 31 → 30 |
| 2 | A | Audit `ci.yml`'s unique jobs; port into `build.yml`; delete `ci.yml` | 30 → 29 |
| 3 | A | Audit `ci-quality-firewall.yml`'s unique jobs; retire | 29 → 28 |
| 4 | B | Create `contract-validations.yml`; delete the four single-purpose validators | 28 → 25 |
| 5 | B | Create `determinism.yml`; delete the two single-purpose determinism workflows | 25 → 24 |
| 6 | B | Merge `ml-slow-suite.yml` into `nightly.yml` | 24 → 23 |
| 7 | C | Discuss & resolve AI advisory consolidation | 23 → 21 (if consolidated) |
| 8 | C | Resolve `quality-gate.yml`, `secure-install-pilot.yml`, `diagnostic-trial.yml` | 21 → ~19 |

After Tier A + B alone: 31 → 23. After Tier C: ~19 (depending on calls). The older "~18" target remains achievable only if the dedicated dependency-pinning signal is folded into a broader security/dependency workflow after its contract stabilizes. The "~10" target from the architectural review is achievable but requires consolidating things like the `dependency-*` family into `security.yml`, which trades clarity for count.

**Each PR should:**
- Verify on a feature branch that the consolidated workflow runs all the original checks
- Update branch-protection required-checks list (if any retired workflow was required)
- Re-run `make validate-ci` to confirm the workflow contract validators still pass
- Add an entry to the [Change Log](#change-log) below

---

## Workflow Design Principles

Current principles, checked against source and the remote protection snapshot:

1. **Preserve both required checks** — `build.yml` owns `CI Gate`; `security-unified.yml` owns the separately required `Dependency Security`. Branch protection is remote state; verify the dated snapshot before merging.
2. **Scheduled workflows own their domain** — `nightly.yml`, `performance-monitor.yml` are non-blocking validation.
3. **Actions are SHA-pinned** — All third-party actions reference commit SHAs for supply-chain security. Enforcement: `enforcement.yml` → `action-pins` job.
4. **Issue creation is deduplicated** — Automated workflows check for existing open issues before creating new ones.
5. **Concurrency follows side effects** — Cancel superseded validation within its event/ref group. Serialize dependency branch writers without cancelling a publication transaction. Release jobs, scheduled evidence, and issue/PR automation need distinct policies; a universal cancellation rule is unsafe.

---

## Governance Notes

### Action Pinning

```yaml
# ✅ Good — SHA pinned
- uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6.0.2

# ❌ Bad — floating tag
- uses: actions/checkout@v6
```

Enforcement: `enforcement.yml` → `action-pins` job.

### Issue Deduplication

Automated failure notifications must check for existing open issues:

```javascript
const { data: issues } = await github.rest.issues.listForRepo({
  owner: context.repo.owner,
  repo: context.repo.repo,
  state: 'open',
  labels: 'relevant,labels',
  per_page: 10
});

const existingIssue = issues.find(i => i.title.includes('Expected Title'));
if (existingIssue) {
  // Update existing issue
} else {
  // Create new issue
}
```

### PR Change Detection

For conditional job execution on PRs, use `dorny/paths-filter` instead of unreliable `github.event.head_commit.modified`:

```yaml
- uses: dorny/paths-filter@de90cc6fb38fc0963ad72b210f1f284cd68cea36  # v3.0.2
  with:
    filters: |
      ml:
        - 'src/transformation_portal/ml/**'
```

---

## Related Documentation

- [`.github/workflows/AI_WORKFLOWS_HARDENING_STATUS.md`](../../.github/workflows/AI_WORKFLOWS_HARDENING_STATUS.md) — PR #1028 AI workflow hardening report (specific to the three AI advisory workflows)
- [`.github/workflows/AI_WORKFLOW_PATTERN.md`](../../.github/workflows/AI_WORKFLOW_PATTERN.md) — Pattern for advisory workflows
- [`docs/governance/DEPENDABOT_PR_GOVERNANCE.md`](../governance/DEPENDABOT_PR_GOVERNANCE.md) — Dependabot triage policy

---

## Change Log

| Date | Change | Rationale |
|------|--------|-----------|
| 2026-09-18 | Forensic review: repaired PR classification, serialized dependency writers, reduced advisory setup, separated release destinations, extended CodeQL language coverage, and refreshed source counts. | Preserve check identities and trusted-event boundaries while removing proven waste and stale guidance. |
| 2026-09-12 | Recounted 31 workflows (8,225 lines), distinguished workflow failure from remote required checks, and labeled historical consolidation proposals. | Source review at the recorded baseline; no workflow policy changes. |
| 2026-09-04 | Refreshed all 31 workflow line counts from the frozen live `.github/workflows/*.yml` inventory (8,225 total). | Keep the maintained matrix synchronized with the governed workflow files. |
| 2026-06-11 | Current inventory refreshed to 31 workflows after `dependency-pinning-check.yml` landed; status snapshot, table counts, keep-list, and roadmap math updated from live `.github/workflows/*.yml` line counts. | Remove drift between the maintained matrix, live workflow files, and top-level documentation references. |
| 2026-04-27 | Second review pass: `quality-gate.yml` description corrected (runs `lint_runner.sh advisory` + `pre-commit-check.sh --all`, not `pre-commit run --all-files`); `dependency-review.yml` reclassified as ⚠️ Advisory (warn-only) since the workflow uses `warn-only: true` and self-identifies as "(advisory)"; `diagnostic-trial.yml` row + Tier C item 10 now describe its actual behavior (determinism/provenance diagnostic gate running `scripts/diagnostics/full_chain_determinism_trial.sh` + bundle-root and evidence-manifest checks); legend expanded to disambiguate Required / Advisory / Partial / Post-merge / Manual / Release / No. | Address review feedback on row accuracy and column-semantics ambiguity. |
| 2026-04-27 | Review-driven corrections: `ci.yml` row now lists actually-unique jobs (security/coverage-gate/build/repo-hygiene/quality-summary) and flags the `develop` branch coverage risk; `ml-slow-suite` row corrects the conflated schedule (nightly = 2 AM, ml-slow = 3:30 AM); LOC column re-labeled as approximate; Tier B count math corrected (5, not 6); status-snapshot phasing rephrased to match the 8-PR roadmap. | Address review feedback on the inventory accuracy. |
| 2026-04-27 | Complete inventory rebuild (30 workflows) + consolidation roadmap | Prior revision listed 12; the missing 18 went undocumented. Phase 1.4 of remediation plan from PR #1558. |
| 2026-03-26 | Added link to Dependabot PR governance documentation | Cross-reference dependency update policy |
| 2026-03-25 | Major update: Fixed enforcement.yml PR detection, performance-monitor.yml baseline handling, nightly.yml deduplication, pinned all actions | Address workflow correctness bugs and governance hygiene |
| 2026-02-04 | Initial creation | Baseline documentation |

---

**Maintained by**: Transformation Portal Architect
**Review Frequency**: Monthly (or after any workflow change)
