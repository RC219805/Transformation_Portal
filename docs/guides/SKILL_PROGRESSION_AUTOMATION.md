# Skill Progression Automation

## Purpose

`skill-progression-map` is a recurring automation that reviews recent authored pull requests and review feedback, then ranks the next skills worth deepening.

The workflow is intentionally evidence-first:

1. Read automation memory from `$CODEX_HOME/automations/skill-progression-map/memory.md`.
2. Prefer GitHub connector data when the automation can fetch recent PRs and review threads directly.
3. If connector review-thread data is missing or unavailable, run the repo-local helper:

```bash
python scripts/automation/skill_progression_map.py --json
```

4. Fall back to local git history only when both connector and `gh`-backed collection fail.

The automation should not claim review threads were unreachable unless both GitHub tiers fail.

## Repo-Local Helper

The helper lives in:

- `src/transformation_portal/dev/skill_progression_map.py`
- `scripts/automation/skill_progression_map.py`

Stable CLI flags:

- `--repo`
- `--author`
- `--since`
- `--limit`
- `--json`

Default behavior:

- repo: resolve from `git remote.origin.url`
- author: resolve from active `gh auth status`
- since: latest timestamp in automation memory, otherwise trailing 7 days
- limit: `10`

## Evidence Sources

The helper emits one normalized report with a `source_status` block so the automation can explain evidence quality without guessing.

Primary evidence sources:

- `gh pr list` for recent authored PR discovery
- `gh pr view` for changed files and review summaries
- `gh api graphql` for inline review threads

Degraded evidence source:

- local `git log` plus changed-file inspection when GitHub evidence cannot be collected

The helper ignores non-actionable GitHub noise such as AI rate-limit comments from `github-actions` and `chatgpt-codex-connector`.

## Ranking Rubric

Evidence is normalized into:

- PR number, title, date, and URL
- file path and line when available
- review-thread status
- concise comment summary
- subsystem tag
- issue-class tag

Required issue classes:

- `contract parity`
- `fail-closed behavior`
- `timeout/runtime guard`
- `path normalization`
- `deterministic validation/preflight`
- `atomicity/concurrency`
- `optional-dependency/runtime isolation`

Theme scores are fixed-weight and deterministic:

- review threads score higher than review summaries
- review summaries score higher than changed-file-only signals
- local git fallback scores lowest
- recent evidence gets a higher multiplier
- repeated PR recurrence and review-thread density add explicit bonuses

This keeps genuine review pressure above raw file-touch counts.

## Output Contract

JSON output includes:

- repo, author, and analysis window
- `source_status`
- inspected PRs
- fallback commits when GitHub collection fails
- normalized evidence records
- ranked themes with numeric scores
- `top_skills` as the top five themes

Automation prose should turn that into:

1. the top five ranked skills
2. two concrete evidence anchors per skill
3. two training tasks per skill
4. one short confidence or evidence-quality line

The helper does not write training tasks itself. It provides the ranked evidence for the automation prompt to summarize.

## Memory and Failure Handling

Memory remains external to the repo:

- `$CODEX_HOME/automations/skill-progression-map/memory.md`

On a successful automation run, append a concise dated summary and current run time there.

If memory write fails:

- state the failure explicitly in the automation output
- do not redirect memory into the repo
- keep the ranked report visible so the run still provides value

## Manual Dry Run

For a recent validation pass against the April 2026 PR cluster:

```bash
python scripts/automation/skill_progression_map.py \
  --repo RC219805/Transformation_Portal \
  --author RC219805 \
  --since 2026-04-03T00:00:00Z \
  --limit 5 \
  --json
```
