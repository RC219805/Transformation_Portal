# GitHub Configuration Forensic Audit

**Audit date:** 2026-09-18 UTC
**Source baseline:** `6e6c204456df404881c77f9db03585da3fc7f829`
**Scope:** tracked `.github` files, their execution consumers, focused tests,
current operator guidance, and a bounded read-only GitHub sample.

## Evidence and scope

The baseline contains **106 tracked files** under `.github`:

| Surface | Files | Disposition |
| --- | ---: | --- |
| Active workflow YAML | 31 | Review triggers, permissions, dependencies, cache inputs, runtime installs, evidence and release boundaries. |
| Workflow documents and legacy Python helpers | 10 | Align current guidance; label historical reports; retain compatibility helpers with explicit limitations. |
| Live agent guides/profiles | 8 | Correct current execution and model-selection guidance. |
| Live RAG runtime, templates and tests | 30 | Repair invocation and cache freshness; retain API/CLI shapes. |
| Agent archive trees | 16 | Preserve historical files; exclude archived profiles from live agent retrieval. |
| Disabled workflow archive | 1 | Preserve outside active workflow discovery; add a warning against reuse. |
| Root configuration, guidance and history | 10 | Verify ownership/remote-state boundaries; label documentation-only configurations. |

Counts use `git ls-tree -r --name-only` at the baseline. Active YAML contained
8,263 lines before changes; the maintained [workflow matrix](WORKFLOW_MATRIX.md)
records current source counts. An old report or an inactive file is not proof
of current enforcement.

Read-only GitHub checks confirmed main protection requires **CI Gate** and
**Dependency Security**, both associated with app `15368`, with strict
up-to-date checking, conversation resolution and admin enforcement enabled.
Active default-branch ruleset `9331244` also configures Copilot review on push,
code-quality severity `all`, and CodeQL scanning with both alert thresholds
`all`. Status contexts alone do not describe the entire merge policy. No remote
settings were changed.

```bash
gh api repos/RC219805/Transformation_Portal/branches/main/protection
gh api repos/RC219805/Transformation_Portal/rulesets/9331244
gh run list --repo RC219805/Transformation_Portal --branch main --limit 50 \
  --json databaseId,workflowName,status,conclusion,headSha,createdAt,updatedAt,event
```

The sampled main runs are baseline observations, not hosted acceptance of this
patch. At commit `ff44d93ab389db09b1e32f26f5ffd47f992a1c52`, both
[primary CI](https://github.com/RC219805/Transformation_Portal/actions/runs/35313917881)
and the [push firewall](https://github.com/RC219805/Transformation_Portal/actions/runs/35313917820)
completed successfully and independently ran lint, type checks, two core Python
legs and an ML leg. The post-CI firewall also retains unique resolution,
isolation and flake-analysis checks. Overlap is evidence for a future
coverage-equivalence audit, not sufficient evidence to delete a gate.

## Findings and implemented corrections

| Priority | Finding and evidence | Correction |
| --- | --- | --- |
| P1 | `build.yml` omitted root `app.py`, schema/migration inputs and several frontdoor dependencies from PR classification. Rename-away paths and incomplete API file lists could also under-classify changes. | Classify both rename paths, conservatively run full checks for incomplete/capped inventories, and cover current runtime/schema/agent/frontdoor dependencies. Ordinary documentation stays lightweight. Executable JavaScript regression tests exercise the actual workflow script. |
| P1 | `submit-pypi.yml` enabled production for any selected version-tag ref, including a manual Test PyPI request. | Production requires a tag push; manual dispatch can only publish to Test PyPI. OIDC environment boundaries remain intact. |
| P2 | The RAG cache had no source/config identity and could return old or deleted guidance indefinitely. Archived profiles were indexed as live agents; root `AGENTS.md` and Copilot instructions were omitted. | Bind cache reuse to deterministic source/config identity, include current authorities, prune archive/cache/generated directories, and skip empty chunks that the JSON decoder rejects. Legacy cache versions rebuild. Publication binds to the exact bytes chunked and rejects transient A-to-B-to-A edits or failed reads. |
| P2 | The RAG CLI imported modules as top-level even though components use package-relative imports. Quickstart template examples also used an unsupported flag or omitted required arguments. | Repair direct-script and module invocation, retain CLI flags, test real offline indexing through both entrypoints, and execute the corrected template examples. |
| P2 | Current Copilot/Specialist/APEX guidance still described old DA3 defaults or unqualified cache/precision advice. | Document commercial-safe `da3_metric`, research-only bare `da3`, prepared execution, identity-v3 cache authority, and backend/device-specific performance evidence. |
| P2 | CodeQL analyzed Actions and Python but omitted the maintained JavaScript/TypeScript frontdoor and Worker. | Add the `javascript-typescript` lane with `build-mode: none`; retain pinned actions and upload permissions. This intentionally adds scan cost to close a coverage gap; the language key follows [GitHub CodeQL guidance](https://docs.github.com/en/code-security/concepts/code-scanning/codeql/codeql-code-scanning). |
| P2 | APEX manual string inputs were interpolated directly into shell scripts. | Transport input values through step environments and quote shell uses; retain flags, mode selection, and synthetic PR/push behavior. A shell regression test proves metacharacters remain literal arguments. |
| P2 | Scheduled/manual dependency updates could concurrently publish the same output branch. | Serialize both triggers in one non-cancelling concurrency group and enforce this in the existing workflow validator. |
| P2 | Enforcement path classification needed PR API read permission; six independent jobs unnecessarily waited on that classifier. | Add job-scoped read permission and remove only unnecessary dependency edges. Conditional ML/golden jobs still depend on classification. |
| P2 | Test cache keys omitted transitive base-lock inputs; primary 3.11/3.12 core jobs shared an immutable cache key. | Include consumed dependency inputs and Python namespaces. Post-CI changes affect only cache dependency inputs; trusted checkout and trigger rules remain intact. |
| P2 | Bot comments allocated summarizer work and could cancel a human-triggered run; the summarizer checked out the repository without using its files. | Skip bot-comment jobs, isolate their concurrency groups, ignore deleted-comment events, remove checkout and set `GH_REPO` explicitly. Comment events summarize the comment body. |
| P2 | AI review's missing-key step returned success but subsequent setup continued. | Use an explicit availability output to gate checkout, changed-file retrieval, dependency setup and review. |
| P3 | Dependency submission installed unused pip-tools and restored a cache despite detection disabling pip cache; release cleanup allocated a fresh runner merely to report no cleanup was needed. | Remove unused setup/cache and the no-op cleanup job; retain submission triggers/retry and release build verification. |
| P3 | Documentation builds accumulated superseded runs; Python 3.11 dependency validation installed a needless `tomli` backport. | Add event/ref-scoped documentation concurrency and use the existing stdlib `tomllib` path. |
| P2 | Workflow README claimed only one required check. Historical score/approval reports and configuration-shaped YAML could be mistaken for current authority. | Refresh verified protection guidance and quality gates, label historical reports, add navigation, and explain which YAML files are documentation only. |

## Performance evidence and limits

One local RAG sample used 2,492 candidate sources and produced 29,570 chunks on the final cache implementation:
**0.962 seconds cold**, **0.366 seconds with a validated warm cache**, identical
results. The warm run replaced chunk parsing with a failure stub to prove cache
reuse. This is a single development-checkout observation, not a benchmark
budget or a comparison with the old unsafe cache. Content hashing reads source
bytes on each cache lookup; a valid hit avoids chunk parsing. Later edits can
change inventory counts and timings.

The workflow edits remove six unnecessary enforcement dependency edges, an
unused submission installation/cache, a summarizer checkout, missing-key AI
setup, and one no-op release runner allocation. They also cancel superseded
documentation work and correct cache invalidation. No hosted minute-saving
percentage is claimed. More accurate runtime classification and the new
CodeQL language lane can increase necessary validation work.

## Deliberately retained work and residual risks

- **Overlapping core workflows:** `build.yml`, `ci.yml`, and
  `ci-quality-firewall.yml` have different event/trust and evidence contracts.
  Any retirement needs an exact coverage map, preserved `develop` coverage,
  equivalent required checks and hosted proof. The matrix's old consolidation
  targets remain historical proposals.
- **Layer 1 versus Golden Regression:** enforcement repeats the same complete
  non-ML selector/install with different `--maxfail` values. Keep both named
  checks until their intended gate responsibility is reconciled.
- **Legacy nightly performance:** old runtime pins, filename-selected
  baselines, no-test acceptance and coarse failure classification remain the
  documented legacy limitations in [performance policy](../performance/GATE_POLICY.md).
  A green nightly summary does not become authoritative regression evidence.
  Retuning its runtime requires separate dependency/model validation.
- **Legacy helper compatibility:** `.github/workflows/quality_*.py` have no
  active workflow consumer; some exclude current source, recursively scan
  untracked environments, or rewrite files by default. They are explicitly
  deprecated in [quality guidance](../../.github/workflows/QUALITY_STANDARDS.md).
  The compatibility shell hook delegates to the maintained hook, but its old
  fallback still has auto-restaging/import-heuristic limitations. Retiring
  these callable entrypoints needs a dedicated compatibility migration.
- **Remote configuration:** `.github/copilot-firewall.yml` and
  `dependency-submission-config.yml` have no repository runtime consumer.
  GitHub's [Copilot firewall settings](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-the-firewall)
  and dependency-submission settings must be verified remotely. Their YAML
  keys are not enforcement evidence. Network access policy was not changed.
- **Hosted acceptance:** local contracts cannot prove CodeQL database/upload
  success, cross-version hosted cache behavior, OIDC approvals, model inference,
  or Actions timing. No release, workflow dispatch, PR comment or repository
  setting mutation was performed as part of this audit.

## Validation

Validation is recorded against the final local patch. Focused workflow tests
exercise the path classifier, permissions, concurrency, cache inputs, release
conditions, advisory behavior and preserved firewall trust. RAG tests exercise
source edits/additions/deletions/renames, chunk settings, cache migration,
archive exclusion, CLI invocation and concurrent-source changes, including A-to-B-to-A edits and failed reads.

**Proven green:** combined contracts **532 passed, 2 skipped** (the two skips
cover the intentionally disabled `python-app.yml` workflow); `ci-quick`
(27 tests), `test-fast` (77 tests), workflow validation (31 workflows),
pre-commit, documentation structure/navigation, CI dependency synchronization,
JSON/YAML governance and Python header checks.

The combined contract command was:

```bash
PYTHONPATH=src ./.venv/bin/pytest \
  tests/test_ci_change_classification.py tests/test_validate_ci_config.py \
  tests/test_custom_agent_config.py tests/test_rag_system.py \
  tests/test_rag_enhanced.py tests/test_rag_classifier.py \
  tests/test_rag_integration.py tests/test_rag_knowledge_engine.py \
  .github/agents/rag_system/tests/test_rag_pipeline.py \
  tests/test_codebase_structure.py tests/test_check_workflow_concurrency_contract.py \
  tests/test_check_dependency_update_workflow.py tests/test_summary_workflow.py \
  tests/test_ai_code_review_workflow.py tests/test_smart_issue_management_workflow.py \
  tests/test_apex_performance_workflow.py tests/test_apex_backend_deps.py \
  tests/test_pypi_workflows.py tests/structural/test_automation_workflow_boundaries.py \
  tests/validation/test_codeql_workflow_contract.py tests/test_lux_depth_v3_doc_sync.py -q
```

Additional commands:

```bash
PYTHONPATH=src make ci-quick
PYTHONPATH=src make test-fast
make validate-ci
make check-stale-docs check-doc-heading-links check-docs
python3 scripts/governance/check_docs_structure.py --all
make check-ci-sync check-json-serialization check-yaml-governance check-python-headers
PRE_COMMIT_HOME=/private/tmp/tp-github-precommit-cache \
  TP_LINT_VENV=/private/tmp/tp-github-lint PYTHONPATH=src make pre-commit
git diff --check
```

**Resolved during validation:** two release assertions reflected the previous
cleanup/manual-publication contract; they were updated to enforce the intended
new behavior. Independent review reproduced an A-to-B-to-A cache race; the
exact-read digest fix and dedicated regressions passed. The first pre-commit
bootstrap was blocked from writing the macOS Go build cache; an approved
rerun completed successfully without weakening hooks.
Publication review also corrected two template CLI examples and executed both
successfully without changing parser behavior.

**Not yet proven:** hosted execution of this patch, CodeQL upload, release
publication, real model inference, hosted duration savings, or equivalence for
retiring overlapping workflows. Existing route, selector, CLI flag, required
check and firewall checkout contracts were retained; the documented release
trigger restriction and cache-format migration are intentional changes.
