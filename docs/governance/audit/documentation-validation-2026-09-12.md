# Documentation Validation Ledger — September 12, 2026

Base: `faa3758fc3b5180bccf264c9a1b1d1b5ef927c52`.
Local worktree: `/private/tmp/tp-docs-refresh-20260912-local`.
Logs and command metadata: `/private/tmp/tp-docs-refresh-20260912-handoff/logs/`.
Final commit/tree identities and outcomes are recorded in the local handoff.

The existing repository Python 3.12.13 is selected by
`./scripts/setup/resolve_python_311.sh`; the worktree `.venv` points to that
existing environment, without installing into it. System Python is 3.9.6 and
is not the test runtime. Node is 22.22.2. A separate temporary docs environment
uses the Makefile's declared Sphinx requirements and read-only access to the
existing core packages. Its installed package versions are recorded locally.

## Reproduction

From the worktree, use the commands and environment captured in each adjacent
log JSON. The local `run_check.py` records command, interpreter, base HEAD/tree,
a hash of tested file contents, exit status, and output. A dirty-worktree content
hash is not a claim that HEAD already contains those changes.

```bash
make validate-ci check-docs check-stale-docs check-doc-heading-links
make check-todo-governance check-ci-sync check-piptools-cache
make check-requirements-lock-contract check-dependency-pinning
make check-json-serialization check-yaml-governance check-python-headers
./.venv/bin/python scripts/governance/check_docs_structure.py --all
./.venv/bin/python scripts/validation/generate_design_tokens_doc.py --check
./.venv/bin/pytest tests/test_custom_agent_config.py tests/test_codebase_structure.py -q
```

Explicit changed/canonical Markdown heading checks and local-target scans
supplement the default heading checker, whose default source set is narrow.
Reviewed shell blocks are checked with `bash -n`; placeholders and configuration
fragments are identified separately from executable snippets.

## Recorded Integration Results

These results were observed during local integration. The external handoff adds
the final committed HEAD/tree, repeat-check records, and exact commands so this
tracked ledger does not require a self-referential commit hash.

| Check | Observed result | Evidence |
| --- | --- | --- |
| Documentation structure and agent contracts | 98 passed | Included in `integration-final-candidate.log` |
| Combined docs, parser/producer/schema, graph, diagnostic, Lux plan/lifecycle/PBR and heading regressions | 357 passed; no skipped tests | `integration-final-candidate.log` and adjacent command metadata |
| Explicit changed/canonical Markdown heading sources | 124 sources passed | `heading-expanded-final-candidate.log` |
| Documentation structure | 933 files scanned; passed | `docs-structure-final-candidate.log` |
| Requested Make governance/docs targets | Passed | `governance-initial.log`; final committed repeat in handoff |
| Quick local CI / fast tests | Passed / 77 tests passed | `ci-quick.log`, `test-fast.log` |
| Pre-commit hooks, including gitleaks | Passed | `pre-commit-final.log`; scoped commit hooks also passed |
| Strict offline Sphinx HTML | Passed | `sphinx-core-imports.log`; final committed repeat in handoff |
| TODO generator / design-token generator check | 25 governed, 0 ungoverned, 1,811 scanned / passed | Existing generator output and local logs |
| Local Markdown file targets | Six unresolved occurrences across five original targets | Link backlog; historical target identity remains unavailable |
| Coordinator/operations shell review | 71 current/mixed blocks passed; historical placeholders excluded from runnable claims | `final-shell-review.json`; public and runtime reviewer reports contain their additional scopes |

Initial Sphinx attempts failed because the separate docs environment lacked
core imports; adding read-only access to the existing core packages resolved
that tooling failure. Initial hook setup encountered a sandbox cache restriction
and an incomplete concurrent lint bootstrap; approved cache setup and serial
initialization of the separate pinned lint environment resolved both. Product
contracts and dependency locks were not relaxed. These initial failures remain
distinguishable from subsequent passing checks in the local evidence.

## Evidence Limits

A strict `-W --keep-going` Sphinx HTML build uses the existing `SPHINX_OFFLINE=1`
configuration. External intersphinx inventories are excluded; configured optional
ML import mocks remain in place. This proves the documentation build under that
configuration, not live optional-runtime imports or inference.

No model download, optional runtime installation, live Postgres/Redis/S3 bring-up,
credential replacement, license acknowledgment, deployment, or publication was
performed. Skipped tests and unavailable runtime/service evidence are not passes.
The [finding ledger](documentation-findings-2026-09-12.md) supplies acceptance
criteria for those separate verification tasks.
