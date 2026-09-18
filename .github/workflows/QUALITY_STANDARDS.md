# Repository Quality Gates

Current authority is the workflow YAML, [AGENTS.md](../../AGENTS.md), and the
[workflow matrix](../../docs/ci/WORKFLOW_MATRIX.md). Historical Pylint scores,
test counts, disk savings, and percentage targets are not acceptance evidence.

## Local validation

From the repository root:

```bash
make install-hooks
make ci-quick
make test-fast
make pre-commit
make validate-ci
git diff --check
```

`make install-hooks` installs the repository-managed pre-commit and pre-push
hooks. Formatting belongs in local hooks; hosted CI verifies without rewriting
source files. Use the exact CI-pinned lint toolchain from
`requirements-lint.txt` through `scripts/setup/run_lint_tool.sh`.

`make quality-check` combines the shared lint runner, workflow validation, and
root placement policy. Root placement follows
`scripts/setup/pre-commit-check.sh`; a historical count of ten Markdown files
is not a substitute for its governed path allowlist.

## Hosted checks

- `build.yml` preserves the `CI Gate` aggregator. PR change classification
  selects lightweight or full execution; full execution includes Python
  3.11/3.12 CPU core tests and Python 3.11 CPU ML tests. There is no GPU matrix
  in this workflow.
- Lint uses Black/isort parity and the shared flake8/Pylint policy. Pylint
  fatal/error/usage findings block; warning-only scores are advisory.
- Blocking mypy paths are owned by
  [TYPE_CHECKING_POLICY.md](../../docs/ci/TYPE_CHECKING_POLICY.md).
- Core coverage has a 30% global floor plus per-package line/branch floors and
  touched-file evidence. The primary ML leg disables coverage; sampled ML
  coverage is advisory. Consult [testing strategy](../../docs/testing/STRATEGY.md)
  for the maintained evidence boundaries.
- `security-unified.yml` supplies the separate `Dependency Security` check.
  Both it and `CI Gate` were required by main protection on 2026-09-18 UTC;
  inspect live GitHub settings before relying on this snapshot.

## Legacy helpers

`quality_checker.py`, `quality_standards.py`, and `quality_enforcement.py` in
this directory are historical standalone helpers. No active workflow invokes
them. They exclude maintained source or scan untracked environments, and some
rewrite files by default. Use the commands above for current validation;
these helpers are not the canonical quality or formatting authority.

The older incident record remains in
[Code Quality Improvements](../../docs/guides/CODE_QUALITY_STANDARDS.md).
