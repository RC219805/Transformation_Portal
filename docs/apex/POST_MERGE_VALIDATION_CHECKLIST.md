# Post-Merge Validation Checklist

**After merging Phase 2, execute these steps to verify runtime behavior.**

---

## Day 1: Immediate Validation (Within 1 Hour)

### 1. Trigger Manual Real Run

```bash
gh workflow run apex_performance.yml \
    -f mode=real \
    -f backend_id=da3 \
    -f sample_size=5 \
    -f device=cpu
```

### 2. Monitor Execution

```bash
# List recent runs
gh run list --workflow=apex_performance.yml --limit 5

# Watch the latest run
gh run watch

# Or watch specific run
gh run watch <run-id>
```

### 3. Verify Workflow Behavior

**Check workflow logs for:**
- ✅ ML dependencies installed (should see `torch`, `transformers` in pip install output)
- ✅ Real mode selected (logs should show "using real execution")
- ✅ Depth inference executed (not mocked/synthetic)
- ✅ No errors in matrix runner

**Get logs:**
```bash
gh run view <run-id> --log
```

### 4. Download and Inspect Artifacts

```bash
# Download all artifacts from the run
gh run download <run-id>

# Inspect ledger
cd apex-ledger-v*/
sqlite3 performance_ledger.db "SELECT run_id, commit_sha, COUNT(*) FROM performance_capsules GROUP BY run_id;"

# Inspect performance capsules
cd ../apex-capsules-v*/
ls -lh
cat performance_capsule_*.json | jq '.bucket_stats | keys'
```

### 5. Validate Metadata Completeness

**Check capsule JSON:**
```bash
cat performance_capsule_*.json | jq '{
  run_id,
  commit_sha,
  workflow_version,
  zone,
  mode,
  sample_size,
  timestamp
}'
```

**Verify:**
- ✅ run_id matches GitHub Actions run
- ✅ commit_sha is correct
- ✅ mode = "real" (not "synthetic")
- ✅ All expected fields present

---

## Day 2: PR Lane Validation

### 1. Open Trivial PR

Create a small documentation change and open PR.

### 2. Verify Synthetic Run Behavior

**Check workflow logs:**
- ✅ Mode forced to synthetic (should see "PR/push lane: forcing synthetic mode")
- ✅ ML dependencies NOT installed (pip install should skip `.[ml]`)
- ✅ Execution fast (<3 min total)
- ✅ No depth inference (synthetic data used)

### 3. Verify PR Comment

**Check PR comment includes:**
- ✅ Clear "[SYNTHETIC DATA]" indicator
- ✅ Performance summary table
- ✅ Link to workflow run
- ✅ No misleading performance claims

---

## Week 1: Scheduled Run Validation

### Wait for First Scheduled Run (Sunday 00:00 UTC)

### 1. Verify Automatic Mode Selection

**Check workflow logs:**
- ✅ Mode automatically set to "real" (should see "Scheduled run: using real execution")
- ✅ No manual input required

### 2. Download Artifacts

```bash
# Find the scheduled run
gh run list --workflow=apex_performance.yml --json conclusion,event,databaseId \
    | jq '.[] | select(.event == "schedule")'

# Download
gh run download <run-id>
```

### 3. Verify Ledger Accumulation

```bash
sqlite3 apex-ledger-v*/performance_ledger.db << SQL
-- Should show multiple runs now
SELECT run_id, timestamp, COUNT(*) as capsule_count
FROM performance_capsules
GROUP BY run_id
ORDER BY timestamp DESC;
SQL
```

---

## Regression Detection (Ongoing)

### Create Baseline

After 3-5 real runs, establish baseline:

```bash
# Query ledger for recent runs
sqlite3 performance_ledger.db << SQL
SELECT
    bucket,
    zone,
    AVG(p50_ms) as avg_p50,
    AVG(p95_ms) as avg_p95,
    COUNT(*) as sample_count
FROM apex_runs
WHERE workflow_version = 'v1'
    AND mode = 'real'
    AND timestamp > datetime('now', '-7 days')
GROUP BY bucket, zone;
SQL
```

### Monitor for Drift

Weekly check:
```bash
# Compare recent performance to baseline
# (Manual for now; automated changepoint detection in Step D)
```

---

## Failure Scenarios to Test

### 1. What Happens When ML Deps Missing in Real Mode?

**Expected:** Clear error message, not silent fallback

**Test:**
- Trigger real run on runner without ML deps
- Verify workflow fails fast with actionable error

### 2. What Happens When Backend Unavailable?

**Expected:** Graceful degradation or clear error

**Test:**
- Trigger with `backend_id=depth_pro` (not installed)
- Verify error handling

---

## Success Criteria

Phase 2 is **runtime verified** when:

- ✅ Manual real run completes successfully
- ✅ Artifacts downloaded and metadata validated
- ✅ PR synthetic run completes fast (<3 min)
- ✅ PR comment generated correctly
- ✅ Scheduled run triggers automatically
- ✅ Ledger accumulates data across runs
- ✅ No silent failures or mode confusion

---

## If Something Breaks

### Rollback Plan

```bash
git revert <merge-commit-sha>
git push origin main
```

### Investigation Steps

1. Capture workflow logs: `gh run view <run-id> --log > failure.log`
2. Capture artifacts if available: `gh run download <run-id>`
3. Create issue with:
   - Workflow run URL
   - Logs
   - Expected vs actual behavior
   - Commit SHA

### Temporary Bypass

If Phase 2 blocks work:
```bash
# Disable workflow temporarily
# (Edit .github/workflows/apex_performance.yml, comment out triggers)
git commit -m "temp: disable APEX while investigating"
git push
```

---

**Checklist Format:**

Copy this into an issue after merge:

```markdown
## Post-Merge Validation Tracking

### Day 1
- [ ] Manual real run triggered
- [ ] Workflow completed successfully
- [ ] Artifacts downloaded
- [ ] Metadata validated
- [ ] Ledger inspected

### Day 2
- [ ] Trivial PR opened
- [ ] Synthetic run verified
- [ ] PR comment inspected

### Week 1
- [ ] Scheduled run completed
- [ ] Automatic mode selection verified
- [ ] Ledger accumulation confirmed

### Baseline Established
- [ ] 3-5 real runs collected
- [ ] Baseline metrics calculated
- [ ] Documented in ledger
```
