# GitHub Actions Workflows

Current workflow inventory and source-derived counts are in
[CI Workflow Matrix](../../docs/ci/WORKFLOW_MATRIX.md). The workflow YAML files
are the execution authority; a listed workflow is not automatically a required
GitHub status check.

## Main CI and conditional execution

[`build.yml`](build.yml) is **CI (Lint, Tests & Manifest)**. It runs on PRs and
pushes to `main`, and by manual dispatch. The `preflight` job classifies PR paths;
non-PR events run the full suite. `lightweight` and `dependency-constraints` run
for every classified change. Full runs add lint, Python 3.12 type checking,
Python 3.11/3.12 core tests, Python 3.11 ML tests, and manifest generation.
`frontdoor-contract` runs when the frontdoor classifier requests it.

`CI Gate` aggregates these results and permits intentional heavy-job skips in
lightweight mode. Main branch protection was read on **2026-09-18 UTC** and required
`CI Gate` and `Dependency Security`, both bound to GitHub Actions app `15368`,
with strict up-to-date checking. An active default-branch ruleset additionally
configures Copilot review, code quality, and CodeQL scanning; see the
[remote snapshot](../BRANCH_PROTECTION_VERIFIED.md). Re-read GitHub before
relying on that snapshot. See [change classification](../../docs/ci/CHANGE_AWARE_CI.md)
and [testing strategy](../../docs/testing/STRATEGY.md).

The former `python-app.yml` and `pylint.yml` are not active workflow files.

## Other maintained surfaces

| Workflow | Purpose and evidence boundary |
| --- | --- |
| [`ci.yml`](ci.yml) | Push validation on main/develop, including coverage and packaging. |
| [`ci-quality-firewall.yml`](ci-quality-firewall.yml) | Post-CI verification of the exact successful same-repository push/manual upstream commit; dispatch build.yml to initiate manual verification. |
| [`enforcement.yml`](enforcement.yml) | Action pins, dependency/artifact boundaries, layered tests and golden contracts. |
| [`codeql.yml`](codeql.yml), [`security-unified.yml`](security-unified.yml) | Separate static/security analysis; inspect each current job result. |
| [`dependency-update.yml`](dependency-update.yml) | Scheduled/manual generic dependency update transaction; target-owned ML/DA3 locks retain separate lanes. |
| [`dependency-pinning-check.yml`](dependency-pinning-check.yml) | Fails its job on exact-pin/constraint drift; distinct from branch protection. |
| [`secure-install-pilot.yml`](secure-install-pilot.yml) | Advisory hash-install pilot; does not replace checked-in ordinary locks. |
| [`docs.yml`](docs.yml) | Documentation build, Markdown, and navigation validation; not production-runtime evidence. |
| [`frontdoor-deployment-gate.yml`](frontdoor-deployment-gate.yml) | Manual shared-frontdoor deployment posture checks. |
| [`submit-pypi.yml`](submit-pypi.yml) | Tag/manual package publication through configured OIDC environments. |
| [`apex_performance.yml`](apex_performance.yml) | Synthetic PR/push matrix; scheduled/manual real mode requires backend execution evidence. Pages publication alone proves no inference result. |
| [`performance-monitor.yml`](performance-monitor.yml) | Scheduled/manual Lux smoke with four required writer tests/artifacts and parsed baselines. |
| [`nightly.yml`](nightly.yml), [`ml-slow-suite.yml`](ml-slow-suite.yml) | Separate deep/ML lanes with their own prerequisites; see the performance policy's legacy-nightly limitations. |
| [`ai-code-review.yml`](ai-code-review.yml), [`summary.yml`](summary.yml), [`smart-issue-management.yml`](smart-issue-management.yml) | Advisory AI automation; service-unavailable diagnostics are not approvals or successful reviews. |

## Package publishing prerequisites

[`submit-pypi.yml`](submit-pypi.yml) uses **Trusted Publishing** through OIDC;
no PyPI API-token secret is required. Configure a trusted publisher for
`RC219805/Transformation_Portal` and `submit-pypi.yml` on each package index,
matching the GitHub environment: `pypi` for PyPI and `testpypi` for Test PyPI.
Configure the corresponding GitHub environments and their required reviewers
before releasing. Both publishing jobs request `id-token: write` to obtain the
short-lived OIDC identity; the build job retains read-only repository access.

Pushing a version tag matching `v*` triggers production publication. For a
Test PyPI upload, use `workflow_dispatch` with `test_pypi=true` on the intended
ref. Manual dispatch never publishes to production PyPI, including when a
version-tag ref is selected. A dispatch with `test_pypi=false` builds packages
without publishing them.
Building packages or passing this repository's checks does not prove that the
external trusted-publisher configuration or release approvals are in place.

## Local validation

```bash
make validate-ci
make ci-quick
make test-fast
```

Use the exact job's command and interpreter when reproducing a hosted failure.
An `action_required` run with no jobs is an approval state, not a product-test
failure. Governance and permissions remain defined by the existing source and
[dependency review policy](../../docs/governance/DEPENDABOT_PR_GOVERNANCE.md).
