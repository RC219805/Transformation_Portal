# APEX Performance Workflow - Production Deployment Guide

## Overview

The APEX Performance workflow (`apex_performance.yml`) runs both V1 and V2 performance lanes. Pull requests and pushes use deterministic synthetic observations; scheduled runs and manual `mode=real` dispatches execute the selected depth backend against the checked-in image fixtures. PR runs also publish a human-readable comment.

## Features Delivered

### ✅ Overall Gate Verdict
- Displays worst-case status across all buckets/zones at the top of PR comment
- Uses worst-of logic: fail > warn > pass
- Clear visual indicator: ❌ (fail), ⚠️ (warn), ✅ (pass)

### ✅ Per-Zone Heatmap
- Bucket × Zone status icon matrix in collapsed section
- Truncates to 8 zones max to keep PR compact
- Shows both per-zone and global results
- Example:
  ```markdown
  <details>
  <summary>📊 V1 Zone × Bucket Heatmap</summary>

  | Bucket | local | Global |
  |---|---|---|
  | depth_inference | ⚠️ | ⚠️ |
  | export_tiff | ❌ | ❌ |
  | load_image | ✅ | ✅ |
  ```

### ✅ Worst Offenders List
- Top-N (default: 10) worst performers by p95/limit ratio
- Sorted by severity (ratio > 1.0)
- Shows exact delta and ratio
- Collapsed by default to keep PR comment compact
- Example:
  ```markdown
  | Rank | Bucket | Zone | p95 | Limit | Over | Status |
  |------|--------|------|-----|-------|------|--------|
  | 1 | export_tiff | Global | 600ms | 500ms | 100ms (1.20×) | ❌ |
  ```

### ✅ Commit SHA Filtering
- Schema-aware: introspects table to check if `commit_sha` column exists
- Only filters by commit when column is present
- Backward-compatible with older ledger databases

### ✅ Size Guardrails
- Monitors GitHub's 65,536 character limit
- Truncates collapsed sections if needed
- Preserves header and overall verdict
- Adds truncation notice with link to full data

## Workflow Triggers

```yaml
on:
  pull_request:
    types: [opened, synchronize, reopened]
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      mode:
        options: [synthetic, real]
  schedule:
    - cron: "0 0 * * 0"
```

- **PR and push**: Force synthetic mode for fast, deterministic contract evidence.
- **Scheduled**: Run the real DA3 backend every Sunday at 00:00 UTC.
- **Manual**: Select synthetic or real mode plus backend, sample size, and device.

## Workflow Steps

### 1. Checkout & Setup
```yaml
- uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
- uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0
  with:
    python-version: "3.11"
    cache: "pip"
```

### 2. Prepare the Real Backend

Real DA3 lanes install the governed, commit-pinned subprocess runtime and expose its interpreter to the matrix runner:

```bash
./scripts/setup/install_da3_runtime.sh --profile baseline
export TRANSFORMATION_PORTAL_DA3_PYTHON="$PWD/.runtime/Depth-Anything-3/.venv-da3/bin/python"
export TP_STRICT_MODEL_LOCK=1
```

The host environment intentionally remains core-only for DA3 because the isolated runtime owns Torch and Transformers. Other real backends continue to install the repository's `ml` extra.

When `TRANSFORMATION_PORTAL_DA3_PYTHON` is configured, that exact executable is the runtime boundary: a missing or non-executable path fails closed instead of falling back to the host interpreter. The worker subprocess also inherits `TP_STRICT_MODEL_LOCK` and any explicit model-lock manifest override.

### 3. Run APEX Matrix
Executes performance benchmarks across V1/V2 workflows:
```bash
python scripts/apex_matrix_runner.py \
  --run-id "${{ github.run_id }}-${{ github.run_attempt }}" \
  --commit-sha "${{ github.sha }}" \
  --workflow-versions v1 v2 \
  --zones local \
  --output-dir ./apex_results \
  --ledger-db ./apex_performance.db
```

### 4. Generate PR Comment
Creates markdown report from ledger:
```bash
python scripts/apex_pr_comment.py \
  --run-id "${{ github.run_id }}-${{ github.run_attempt }}" \
  --commit-sha "${{ github.sha }}" \
  --ledger-db ./apex_performance.db \
  --output comment.md

# Always publish to job summary (visible even if PR comment fails)
cat comment.md >> "$GITHUB_STEP_SUMMARY"
```

### 5. Post PR Comment (Idempotent)
Updates existing APEX comment or creates new one:
```bash
# Find existing APEX comment
COMMENT_ID=$(gh pr view "$PR_NUMBER" \
  --json comments \
  --jq '.comments[] | select(.body | startswith("# 🎯 APEX Performance Report")) | .id' \
  | head -n 1)

if [ -n "$COMMENT_ID" ]; then
  # Update existing comment (idempotent)
  gh api --method PATCH "/repos/$REPO/issues/comments/$COMMENT_ID" \
    -f body="$(cat comment.md)"
else
  # Create new comment
  gh pr comment "$PR_NUMBER" --body-file comment.md
fi
```

### 6. Publish Current-Run Evidence

The gate reads the current run's matrix-step outcomes from the GitHub Jobs API and keeps runner execution, result upload, and result download as separate stages. It uploads a ledger only when all three stages succeeded, both expected V1/V2 observations arrived, and ledger rebuild/aggregation completed successfully. Dashboard deployment is limited to real results on `main`, and the scheduled backup downloads that same run's ledger rather than searching for an older successful artifact.

The ledger is retained for 90 days:
```yaml
- uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.0
  with:
    name: apex-ledger
    path: apex_performance.db
    retention-days: 90
```

## Permissions Required

```yaml
permissions:
  contents: read          # Checkout code
  pull-requests: write    # Post/update PR comments
  issues: write          # Update comment via API
```

## Testing Locally

### 1. Create Test Database
```bash
python3 << 'EOF'
import sqlite3
from datetime import datetime

conn = sqlite3.connect("test_apex_performance.db")
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS apex_runs (
        run_id TEXT NOT NULL,
        commit_sha TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        workflow_version TEXT NOT NULL,
        zone TEXT,
        bucket_name TEXT NOT NULL,
        p50 REAL NOT NULL,
        p95 REAL NOT NULL,
        p99 REAL,
        count INTEGER NOT NULL,
        threshold_p50 REAL NOT NULL,
        threshold_p95 REAL NOT NULL,
        pass_fail TEXT NOT NULL,
        raw_capsules_json TEXT,
        PRIMARY KEY (run_id, workflow_version, zone, bucket_name)
    )
""")

timestamp = datetime.utcnow().isoformat()
test_data = [
    ("test_123", "abc123def", timestamp, "v1", None, "load_image",
     0.050, 0.080, 0.100, 100, 0.120, 0.150, "pass", None),
    # ... add more test data
]

cursor.executemany("""
    INSERT INTO apex_runs
    (run_id, commit_sha, timestamp, workflow_version, zone, bucket_name,
     p50, p95, p99, count, threshold_p50, threshold_p95, pass_fail, raw_capsules_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", test_data)

conn.commit()
conn.close()
EOF
```

### 2. Generate Comment
```bash
python scripts/apex_pr_comment.py \
  --run-id test_123 \
  --commit-sha abc123 \
  --ledger-db test_apex_performance.db \
  --output test_comment.md
```

### 3. Verify Output
```bash
# Check character count
wc -c test_comment.md

# View comment
cat test_comment.md

# Verify it renders correctly
gh markdown-preview test_comment.md
```

## GitHub Actions Security

### Pinned Actions (SHA Hashes)
All actions are pinned to specific commit SHAs for supply-chain security:
```yaml
- uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
- uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0
- uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.0
```

### Token Permissions
Uses minimal required permissions:
- `contents: read` (not write)
- `pull-requests: write` (scoped to comments only)
- `issues: write` (scoped to comment updates only)

## Troubleshooting

### Issue: "Body is too long" GitHub API error
**Cause**: Comment exceeds 65,536 characters
**Solution**: Script automatically truncates and adds notice. Check job summary for full details.

### Issue: No comment posted to PR
**Cause**: Missing permissions or GitHub token issues
**Solution**:
1. Check workflow permissions
2. Verify `GH_TOKEN` is set correctly
3. Check job summary (comment is always written there)

### Issue: Database schema errors
**Cause**: Ledger DB missing expected columns
**Solution**: Script uses schema introspection and gracefully handles missing columns like `commit_sha`

### Issue: Empty or missing stats
**Cause**: No data in ledger for specified run_id/commit
**Solution**:
1. Check that `apex_matrix_runner.py` completed successfully
2. Verify ledger artifact was uploaded
3. Check run_id matches between runner and comment generator

The partial report identifies the earliest proven failed stage: matrix runner execution, result upload, result download, or incomplete V1/V2 files after successful transfer. If the Jobs API or step data is unavailable, the report says the outcome could not be resolved instead of attributing a failure. Start with that named stage; the aggregate matrix-job result is diagnostic context, not proof that runner execution failed.

## Maintenance

### Updating Character Limit
If GitHub changes comment size limits:
```python
# In apex_pr_comment.py
MAX_GITHUB_COMMENT_SIZE = 65_000  # Adjust as needed
```

### Adjusting Zone Display Limit
To show more/fewer zones in heatmap:
```python
heatmap = generate_zone_heatmap_from_stats(
    v1_stats,
    "📊 V1 Zone × Bucket Heatmap",
    max_zones=8  # Increase/decrease as needed
)
```

### Adjusting Worst Offenders Count
To show more/fewer offenders:
```python
offenders = generate_worst_offenders(
    v1_stats,
    "⚠️ V1 Worst Offenders (Top 10)",
    top_n=10  # Increase/decrease as needed
)
```

## Success Criteria

All requirements from the specification have been met:

- ✅ Overall gate verdict displayed prominently
- ✅ Per-zone heatmap generated in collapsed section
- ✅ Worst offenders list generated in collapsed section
- ✅ Commit SHA actually filters query results (schema-aware)
- ✅ Comment stays under 65K characters
- ✅ GitHub Actions workflow posts comment idempotently
- ✅ Job summary always populated (even if PR comment fails)

## References

- **Workflow**: `.github/workflows/apex_performance.yml`
- **Script**: `scripts/apex_pr_comment.py`
- **Matrix Runner**: `scripts/apex_matrix_runner.py`
- **Ledger Schema**: `src/transformation_portal/metrics/ledger.py`
