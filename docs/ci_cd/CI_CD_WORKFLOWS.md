# CI/CD Workflows

This repository uses GitHub Actions to enforce correctness, security, and reproducibility for the Transformation Portal pipeline (including Lux Depth V3).

This document explains:
- how to interpret PR checks,
- which workflows exist and what they gate,
- the repository's workflow permission model,
- how to reproduce failures locally,
- common failure modes and what to do about them.

---

## Where workflows live

All GitHub Actions workflows live under:

- `.github/workflows/`

In PRs, check labels generally appear in this format:

- `<Workflow name> / <Job name>`

To map a check label to its YAML file:
1. List workflow files: `ls .github/workflows`
2. Search by workflow name: `rg -n 'name:\s*<Workflow name>' .github/workflows`
3. Open that workflow file and locate the job with the matching job name.

---

## Workflow groups

### 1) Core CI

Purpose: Validate that code is importable, tests pass, and manifests/config expectations hold.

Typical jobs include:
- Lint (repo-configured tooling)
- Unit tests (Python 3.11 baseline)
- "Manifest" / "contract" verification where applicable

Local equivalents:
- `python -m pytest -q`
- `pre-commit run -a` (if pre-commit is configured)

---

### 2) Quality Gate

Workflow file:
- `.github/workflows/quality-gate.yml`

Purpose: Enforce repo hygiene rules that prevent slow drift:
- formatting / critical lint checks
- root directory hygiene rules (example: maximum markdown files in repo root)
- guardrails around "what belongs where"

Important:
- Workflows should not "commit" changes locally unless they also push (and pushing from CI should be rare and explicitly approved).
- If the workflow currently creates a local commit without pushing, treat that as "signal-only" and prefer either:
  - failing with actionable instructions, or
  - uploading a patch artifact (recommended).

---

### 3) Security scanning and dependency hardening

Purpose: Detect vulnerable dependencies, unsafe patterns, or insecure workflow configuration.

Common checks you may see:
- CodeQL analysis (Python / Actions)
- Dependency submission / dependency scanning
- Repository "Security Unified" checks (org-specific aggregation)

Notes:
- Some security workflows require additional permissions (see "Permissions policy" below).
- If a security workflow flags the workflow file itself, fix workflow YAML first (the pipeline can't protect you if the pipeline is broken).

---

### 4) Enforcement gates

Purpose: Hard guarantees the organization wants to hold everywhere.

These checks commonly include:
- Verify Action Pins (ensure actions are pinned or meet policy)
- Verify Artifact Boundary (ensure output artifacts don't leak across boundaries)
- Verify No Banned Dependencies
- Layered test gates (fast vs ML tier vs golden regression)

Local equivalents vary by check, but typically include:
- `python -m pytest -q`
- `pre-commit run -a`
- any repo-specific golden/regression harnesses

---

### 5) Performance monitoring

Purpose: Detect regressions in:
- runtime
- memory usage
- throughput (if applicable)

These checks typically produce artifacts (e.g., benchmark JSON, memory reports). Prefer reviewing the artifact rather than guessing from the summary.

---

### 6) Automation / assistants

Purpose: Non-blocking automation such as:
- issue/PR summarizers
- AI triage classifiers
- AI code review bots

Important:
- These are frequently subject to rate limits or quota issues.
- Treat AI automation as advisory unless explicitly configured as a gate.

---

## Permissions policy (least privilege)

All workflows must declare explicit `permissions:` to avoid accidental privilege escalation.

### Baseline (recommended default)

Use read-only unless you are certain a job needs more:

- `contents: read`

### Additive permissions (only when needed)

Examples:
- Code scanning upload: `security-events: write`
- Creating or updating PR comments: `pull-requests: write`
- Writing to the repository (rare): `contents: write`

### Common pitfall: duplicate `permissions:` blocks

YAML workflows must not define `permissions:` more than once at the same level.
If you need different permissions for a specific job, override at the job level rather than repeating at the top level.

---

## Troubleshooting

### "Invalid workflow file: permissions is already defined"
Cause:
- duplicate `permissions:` blocks in the same scope (usually top-level)

Fix:
- keep exactly one top-level `permissions:` block, and move job-specific overrides into the specific job(s).

---

### AI triage / summarization errors (rate limit / quota)
Symptom:
- AI triage or summarization job fails with quota/rate limit errors

Action:
- Manually add labels/priority and proceed.
- Do not block merges on assistant-only checks unless intended.

---

### "Workflow awaiting approval"
Cause:
- new workflow or permission change requires maintainer approval (org policy)

Action:
- request approval from a maintainer
- avoid frequent workflow churn to reduce approval friction

---

## Adding or changing workflows: required checklist

Before merging workflow changes:
- [ ] Explicit `permissions:` is present and minimal.
- [ ] No duplicate `permissions:` blocks.
- [ ] Actions meet repository pinning policy.
- [ ] The workflow does not mutate the repository unless explicitly required and reviewed.
- [ ] The workflow name and purpose is documented (update this file if it changes behavior).
