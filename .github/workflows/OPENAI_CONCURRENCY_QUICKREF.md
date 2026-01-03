# OpenAI API Concurrency Control - Quick Reference

## 🎯 One-Liner
Repository-wide job-level concurrency queuing prevents OpenAI API RPM limit contention across 3 workflows.

## 🔑 Key Pattern (Copy-Paste for New Workflows)

```yaml
jobs:
  my-openai-job:
    timeout-minutes: 5  # Adjust based on expected duration
    concurrency:
      group: openai-api-${{ github.repository }}  # Must match exactly!
      cancel-in-progress: false                   # Queue, don't cancel!
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    steps:
      - name: Run AI feature
        run: |
          python - <<'PY'
          import os, sys
          from openai import OpenAI

          # CRITICAL: Strip whitespace from API key
          api_key = (os.getenv("OPENAI_API_KEY") or "").strip()

          try:
              client = OpenAI(api_key=api_key)
              response = client.chat.completions.create(...)
          except Exception as e:
              error_str = str(e).lower()

              # CRITICAL: Differentiate error types
              if "insufficient_quota" in error_str or "billing details" in error_str:
                  message = "Quota limit reached. Check billing."
              elif "429" in error_str or "rate_limit" in error_str:
                  message = "Rate limit exceeded. Try again later."
              elif "401" in error_str or "invalid_api_key" in error_str:
                  message = "Invalid API key."
              else:
                  message = "See CI logs for details."

              print(repr(e), file=sys.stderr)  # Full error to logs only
          PY
```

## 📋 Checklist for New Workflows

- [ ] Add `concurrency` block with `openai-api-${{ github.repository }}` group
- [ ] Set `cancel-in-progress: false` (queue, don't cancel)
- [ ] Add `timeout-minutes` (5-10 min recommended)
- [ ] Strip API key whitespace: `(os.getenv("...") or "").strip()`
- [ ] Implement error differentiation (quota/rate/auth/generic)
- [ ] Quote shell variables: `"$GITHUB_OUTPUT"`
- [ ] Add `continue-on-error: true` if AI is optional
- [ ] Test: Trigger multiple workflows simultaneously → verify queuing

## 🚨 Common Mistakes

| ❌ DON'T | ✅ DO |
|----------|-------|
| `group: ${{ github.workflow }}` | `group: openai-api-${{ github.repository }}` |
| `cancel-in-progress: true` | `cancel-in-progress: false` |
| Generic error: `"An error occurred"` | Specific: `"Quota limit reached. Check billing."` |
| `OPENAI_API_KEY = os.getenv("...")` | `OPENAI_API_KEY = (os.getenv("...") or "").strip()` |
| No timeout | `timeout-minutes: 5` |

## 📊 Affected Workflows

| Workflow | Job | Timeout | Purpose |
|----------|-----|---------|---------|
| `summary.yml` | `summarize` | 5 min | Issue/PR/comment summaries |
| `smart-issue-management.yml` | `smart-triage` | 5 min | Auto-labeling & classification |
| `ai-code-review.yml` | `ai-review` | 10 min | GPT-4o code review |

All use the same concurrency group: `openai-api-${{ github.repository }}`

## 🔍 Monitoring

**GitHub Actions UI:**
- Look for: `⏳ Waiting for a job to complete...`
- This indicates jobs are queuing (expected behavior)

**Logs:**
- Search for: `"Detailed error (redacted for posting)"`
- Indicates error occurred; check stderr for full trace

**Queue Health:**
- Normal: 0-2 jobs waiting
- Investigate if: >3 jobs waiting consistently (may need tier upgrade)

## 📚 Full Documentation

| Document | Purpose | Path |
|----------|---------|------|
| ADR-002 | Architectural decision record | `docs/architecture/ADR-002-OPENAI-API-CONCURRENCY-CONTROL.md` |
| Implementation Summary | Detailed overview | `.github/workflows/OPENAI_API_HARDENING_SUMMARY.md` |
| Integration Checklist | Step-by-step guide | `.github/workflows/OPENAI_WORKFLOW_INTEGRATION_CHECKLIST.md` |

## 🎯 Decision Summary

**Problem:** 3 workflows competing for 3 RPM → 429 errors
**Solution:** Repository-wide concurrency queuing
**Result:** Jobs queue in FIFO order → zero contention failures

**Date:** 2025-01-02
**Status:** Production-ready

---

**Quick Help:**
- New workflow? → Use `.github/workflows/OPENAI_WORKFLOW_INTEGRATION_CHECKLIST.md`
- Understanding why? → Read `docs/architecture/ADR-002-OPENAI-API-CONCURRENCY-CONTROL.md`
- Implementation details? → See `.github/workflows/OPENAI_API_HARDENING_SUMMARY.md`
