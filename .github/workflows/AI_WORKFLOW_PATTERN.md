# AI Advisory Workflow Pattern - Reference Card

This document defines the canonical pattern for AI-powered advisory workflows in the Transformation Portal repository.

---

## Pattern Overview

AI advisory workflows are **non-blocking** CI jobs that provide intelligent suggestions without gating PR merges. They must be timeout-bounded and failure-visible.

---

## Required Configuration

### Job Level
```yaml
jobs:
  ai-advisory-job:
    name: AI Advisory / <Purpose>
    runs-on: ubuntu-latest
    continue-on-error: true        # ← REQUIRED: Non-blocking
    timeout-minutes: 10            # ← REQUIRED: Hard ceiling
```

### AI Processing Step
```yaml
      - name: Run AI Processing
        timeout-minutes: 4         # ← REQUIRED: Step-level bound
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python << 'EOF'
          # Python script here
          EOF
```

### Python Exception Handler
```python
import sys

try:
    # AI API call with retries
    response = call_openai_with_retries(...)
except Exception as e:
    # Emit a fixed, non-sensitive warning annotation
    print("::warning::<Job Name> OpenAI request failed (non-blocking). See job logs for details.")
    # Optional: log redacted diagnostics to stderr (avoid raw exception strings in shared output)
    print("AI processing failed due to an internal advisory workflow error.", file=sys.stderr)
    # ↓ REQUIRED: Emit warning for UI visibility
    # ↓ REQUIRED: Preserve non-blocking behavior for expected AI/service failures
    sys.exit(0)
```

### Failure Detection Step
```yaml
      - name: Report non-blocking AI failure
        if: failure()              # ← REQUIRED: Captures timeouts/crashes
        run: |
          echo "::warning::<Job Name> failed (non-blocking). See job logs."
```

### Terminal Status Step
```yaml
      - name: Summary
        if: always()               # ← REQUIRED: Always runs
        run: |
          echo "<Job Name> completed (non-blocking)."
```

---

## Timeout Values

| Level | Value | Rationale |
|-------|-------|-----------|
| **Job** | 10 minutes | Hard ceiling; allows terminal steps after timeout |
| **AI Step** | 4 minutes | Primary defense; typical AI calls: 1-2 min |

**Critical**: Step timeout < Job timeout to allow terminal steps to run.

---

## Warning Emission Requirements

### Python Handler
- **When**: OpenAI API call fails after retries
- **Message**: `"<Job Name> OpenAI request failed (non-blocking). See job logs for details."`
- **Format**: `print("::warning::...")`

### Shell Handler
- **When**: Step fails (timeout, crash, Python exception)
- **Message**: `"<Job Name> failed (non-blocking). See job logs."`
- **Format**: `echo "::warning::..."`

### Visibility
- Warnings appear as annotations in GitHub Actions UI
- Detailed errors logged to job logs (not PR comments)
- Fallback AI-unavailable diagnostics stay in logs/output files, not PR comments
- No secrets or sensitive data in warnings

---

## Retry Logic Pattern

```python
def call_openai_with_retries(client, messages, model="gpt-4o-mini", max_retries=6, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
        except Exception as e:
            # Detect rate-limit (429)
            status_is_429 = False
            status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
            if status == 429 or "429" in str(e) or "Rate limit" in str(e):
                status_is_429 = True
            error_code = getattr(e, "code", None)
            error_type = getattr(e, "type", None)

            # Quota/billing exhaustion is terminal for the current workflow run.
            if status_is_429 and "insufficient_quota" in (error_code, error_type):
                raise

            # Retry with exponential backoff
            if status_is_429 and attempt < max_retries:
                base_wait = min(60, 2 ** attempt)
                jitter = random.uniform(0.5, 1.5)
                wait = base_wait * jitter
                print(f"⚠️ Rate limited (attempt {attempt}/{max_retries}). Waiting {wait:.1f}s...")
                time.sleep(wait)
                continue

            # Last attempt or non-retryable error
            raise
```

**Parameters:**
- `max_retries=6`: Sufficient for typical rate limits
- Exponential backoff: 2^attempt seconds (capped at 60s)
- Jitter: 0.5x to 1.5x to avoid thundering herd
- `insufficient_quota`: Terminal quota/billing condition; log once and do not retry

---

## Concurrency Control

```yaml
# Pull-request code-review style workflows
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true       # ← REQUIRED: Cancel outdated runs
```

```yaml
# Mixed schedule + push/pull_request workflows
concurrency:
  group: ${{ github.workflow }}-${{ github.event_name }}-${{ github.ref }}
  cancel-in-progress: true       # ← REQUIRED: Prevent cross-event cancellation
```

```yaml
# Issue/PR event workflows (issues, issue_comment, pull_request_target)
concurrency:
  group: ${{ github.workflow }}-${{ github.event_name }}-${{ github.event.issue.number || github.event.pull_request.number || github.run_id }}
  cancel-in-progress: true
```

**Rationale**: Reduces CI minutes and API costs by only processing latest relevant state while avoiding unrelated event cancellation.

```yaml
# Bot-event filtering for issue/PR advisory workflows
jobs:
  advisory-job:
    if: >-
      ${{ !(github.event_name == 'issue_comment' && github.event.comment.user.type == 'Bot') &&
      !(github.event_name == 'issues' && github.event.sender.type == 'Bot') }}
```

**Rationale**: Deployment bots (`cloudflare-workers-and-pages[bot]`, `vercel[bot]`) and
dependency bots comment on nearly every deploy/PR; running AI advisory jobs on those
events wastes runs and API calls and can leave spurious failed check runs. A missing
payload path evaluates to an empty string (`!= 'Bot'`), so the guard fails safe.

---

## Permissions (Minimal)

```yaml
permissions:
  contents: read                 # Read code
  issues: write                  # Post comments
  pull-requests: write           # Post PR comments
```

**Do not grant**: `write-all`, `admin`, `security-events`

---

## Exit Code Strategy

### Success Case
```python
# AI processing completed
sys.exit(0)                      # ← Success
```

### Failure Case
```python
# AI processing failed
print("::warning::...")          # ← Warning visible in UI
sys.exit(0)                      # ← Non-blocking (still success)
```

**Critical**: Expected AI/service failures should not block advisory workflows; use warning + non-blocking exit behavior for those paths.

---

## Validation Checklist

Before merging AI advisory workflows, verify:

- [ ] Job-level `timeout-minutes: 10`
- [ ] Step-level `timeout-minutes: 4` on AI processing
- [ ] Job-level `continue-on-error: true`
- [ ] Python exception handler emits `::warning::`
- [ ] Python script exits 0 on failure
- [ ] `if: failure()` step emits `::warning::`
- [ ] `if: always()` terminal step present
- [ ] Concurrency control configured
- [ ] Minimal permissions
- [ ] Retry logic with exponential backoff (required when workflow includes retry semantics)
- [ ] No secrets in output/warnings
- [ ] `make validate-ci` passes

---

## Example Implementation

See canonical implementations in:
- `.github/workflows/ai-code-review.yml`
- `.github/workflows/summary.yml`
- `.github/workflows/smart-issue-management.yml`

---

## Anti-Patterns (Do Not Use)

### ❌ Missing Step Timeout
```yaml
# BAD: No step-level timeout
- name: Run AI Processing
  run: python script.py           # ← Can run forever until job timeout
```

### ❌ Blocking Exit
```yaml
# BAD: Exit 1 blocks PR
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)                   # ← Blocks PR merge
```

### ❌ Silent Failure
```yaml
# BAD: No warning emission
except Exception as e:
    sys.exit(0)                   # ← Success, but no visibility
```

### ❌ No Terminal Step
```yaml
# BAD: No if: always() step
# When timeout occurs, no summary is posted
```

### ❌ Job Timeout Too Short
```yaml
# BAD: Job timeout < Step timeout
timeout-minutes: 3                # ← Job kills step before it can timeout
steps:
  - timeout-minutes: 4            # ← Never reached
```

---

## Failure Mode Reference

| Failure | Detection | Warning Source | Job Result |
|---------|-----------|----------------|------------|
| OpenAI transient rate limit | Exception after retries | Python `::warning::` | Success ✓ |
| OpenAI quota exhausted | Non-retryable 429 classification | Python `::warning::` | Success ✓ |
| OpenAI timeout | Step timeout (4 min) | Shell `::warning::` | Success ✓ |
| Python crash | Exception | Python + Shell `::warning::` | Success ✓ |
| Network error | Exception | Python `::warning::` | Success ✓ |
| Job timeout (10 min) | GitHub Actions | None (GitHub UI) | Failure ✗ |

**Note**: Job timeout failure is extremely rare (requires step timeout + 6 min overhead).

---

## Maintenance

### When to Update This Pattern
- OpenAI SDK breaking changes
- GitHub Actions timeout behavior changes
- New advisory workflow types introduced
- Security posture requirements change

### Pattern Approval Authority
Changes to this pattern require Architect approval (see `docs/architecture/agent_governance.md`).

---

## References

- **Governance**: `docs/architecture/agent_governance.md`
- **Status Document**: `.github/workflows/AI_WORKFLOWS_HARDENING_STATUS.md`
- **PR**: #1028 (AI Advisory Workflows Hardening)

---

*Canonical Pattern Version: 1.0*
*Last Updated: PR #1028*
*Authority: Transformation Portal Architect*
