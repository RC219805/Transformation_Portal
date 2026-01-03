# ADR-002: OpenAI API Concurrency Control for GitHub Actions Workflows

**Status:** Accepted
**Date:** 2026-01-02
**Decision Makers:** Transformation Portal Architect
**Affected Components:** `.github/workflows/summary.yml`, `.github/workflows/smart-issue-management.yml`, `.github/workflows/ai-code-review.yml`

---

## Context

The Transformation Portal repository employs multiple GitHub Actions workflows that leverage the OpenAI API for AI-powered features:

1. **Issue Summarizer** (`summary.yml`) - Generates concise summaries of issues, PRs, comments, and reviews
2. **Smart Issue Triage** (`smart-issue-management.yml`) - Automatically classifies and labels issues/PRs
3. **AI Code Review** (`ai-code-review.yml`) - Provides comprehensive GPT-4o code review analysis

### The Problem

These workflows run concurrently on various GitHub events (issue creation, PR updates, comments, etc.), causing **RPM (Requests Per Minute) limit contention** on the shared OpenAI API key:

- **Org RPM Limit:** 3 requests per minute (tier-1 free/low-tier account)
- **Concurrent Triggers:** A single PR can trigger all three workflows simultaneously
- **Failure Mode:** Rate limit (429) errors cause workflow failures and degraded user experience

### Observed Failures

```
⚠️ Rate limited by OpenAI (attempt 1/6). Waiting 2.3s before retrying...
⚠️ Rate limited by OpenAI (attempt 2/6). Waiting 5.7s before retrying...
```

While workflows implement exponential backoff + jitter retry logic, this is:
- **Inefficient:** Wastes CI minutes on retry delays
- **Unreliable:** Multiple workflows competing for the same 3 RPM creates race conditions
- **Non-deterministic:** Success depends on random timing

---

## Decision

We implement **repository-wide job-level concurrency queuing** for all OpenAI-consuming workflows using GitHub Actions' native `concurrency` feature.

### Implementation

Each OpenAI-dependent job now includes:

```yaml
concurrency:
  group: openai-api-${{ github.repository }}
  cancel-in-progress: false
```

**Key Properties:**

1. **Shared Concurrency Group:** All workflows use the same group identifier (`openai-api-<repo>`), ensuring serialization across workflow files
2. **Queuing Behavior:** `cancel-in-progress: false` makes jobs wait in a queue instead of canceling each other
3. **Job-Level Scope:** Applied to the actual job making API calls (not the `check-api-key` pre-flight job)
4. **Repository-Scoped:** Serialization is per-repository, not per-workflow

### Additional Hardening

#### 1. Enhanced Error Message Differentiation

Replaced generic error handling with specific error classification:

```python
error_str = str(e).lower()

# Quota/billing errors (permanent until account action)
if "insufficient_quota" in error_str or "billing details" in error_str:
    summary = "OpenAI quota/billing limit reached. Check plan/billing."

# Rate limit errors (temporary, retry later)
elif "429" in error_str or "rate_limit" in error_str:
    summary = "OpenAI rate limit exceeded. Please try again later."

# Authentication errors (invalid API key)
elif "401" in error_str or "invalid_api_key" in error_str:
    summary = "Invalid OPENAI_API_KEY. Verify the secret value."

# Generic fallback
else:
    summary = "An internal error occurred. See CI logs for details."
```

This allows users to distinguish between:
- **Actionable errors** (fix billing, update API key)
- **Transient errors** (wait and retry)
- **System errors** (investigate logs)

#### 2. Input Sanitization

```python
# Strip whitespace from API key to prevent configuration errors
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
```

Prevents silent failures from accidentally pasted keys with leading/trailing whitespace.

#### 3. Shell Variable Quoting

```bash
echo "has_key=true" >> "$GITHUB_OUTPUT"  # Was: >> $GITHUB_OUTPUT
```

Prevents path expansion issues in edge cases.

#### 4. Job Timeouts

```yaml
timeout-minutes: 5   # Issue Summarizer & Smart Triage
timeout-minutes: 10  # AI Code Review (larger payloads)
```

Prevents runaway jobs from blocking the concurrency queue indefinitely.

#### 5. Comment Noise Control (Issue Summarizer)

Implemented intelligent comment updates:

- **Hidden Marker:** `<!-- ai-summarizer-summary -->`
- **Update Strategy:** Check for existing summary comment and UPDATE instead of creating a new one
- **Fallback:** If update fails, post new comment
- **Diagnostic Deduplication:** Separate marker for "API key missing" messages

**Before:** Each trigger posted a new comment (noise)
**After:** Updates the previous summary in-place (clean)

---

## Consequences

### ✅ Positive

1. **Eliminates RPM Contention:** Only one OpenAI job runs at a time across all workflows
2. **Predictable Execution:** Jobs queue in FIFO order instead of racing
3. **Reduced CI Waste:** No more exponential backoff delays burning CI minutes
4. **Better User Experience:**
   - Clear error messages for different failure modes
   - Reduced comment spam from repeated summarizations
5. **Graceful Degradation:** Rate limit errors are handled gracefully with user-friendly messages
6. **Operational Clarity:** Maintainers can distinguish quota issues from transient rate limits

### ⚠️ Trade-offs

1. **Increased Latency:** Jobs wait in queue instead of running concurrently
   - **Mitigation:** Most triggers are infrequent (issue creation, PR updates), so queue depth is typically 1-2
   - **Acceptable:** A 30-second delay for a summary is preferable to a failure

2. **Single Point of Failure:** If one job hangs, it blocks all others
   - **Mitigation:** `timeout-minutes` ensures jobs cannot block indefinitely
   - **Monitoring:** CI logs clearly show queued jobs

3. **Per-Repository Serialization:** Multi-repo orgs still share the same API key limits
   - **Future Work:** Consider per-org concurrency groups if multiple repos adopt this pattern

### 📋 Operational Notes

- **No Breaking Changes:** Existing workflows continue to function; this is a reliability improvement
- **Backward Compatible:** Workflows gracefully handle missing API keys (skip AI features)
- **Monitoring:** GitHub Actions UI shows queued jobs with `⏳ Waiting for a job to complete...`

---

## Alternatives Considered

### 1. Increase OpenAI Tier/Quota

**Rejected:** This is a cost/budget decision, not an architectural one. The architecture should be resilient regardless of tier.

### 2. Workflow-Level Concurrency

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
```

**Rejected:** This only serializes jobs within the same workflow. Different workflows (summarizer, triage, code review) would still race.

### 3. Client-Side Rate Limiting

Implement a shared Redis/Memcached rate limiter.

**Rejected:** Over-engineered for CI environment. GitHub Actions' native concurrency feature is purpose-built for this use case.

### 4. Workflow Orchestrator

Create a single "AI dispatcher" workflow that calls others sequentially.

**Rejected:** Violates separation of concerns. Workflows should remain independent for maintainability.

---

## Validation

### Test Scenarios

1. **Concurrent Triggers:**
   - Open an issue → triggers Smart Triage
   - Immediately comment on it → triggers Summarizer
   - **Expected:** Jobs queue; both complete successfully without 429 errors

2. **Error Differentiation:**
   - Invalid API key → "Invalid OPENAI_API_KEY. Verify the secret value."
   - Quota exceeded → "OpenAI quota/billing limit reached. Check plan/billing."
   - Rate limit → "OpenAI rate limit exceeded. Please try again later."

3. **Comment Updates:**
   - Edit an issue description multiple times
   - **Expected:** Summary comment updates in-place (only 1 comment visible)

4. **Timeout Protection:**
   - Simulate hanging API call (mock network delay)
   - **Expected:** Job terminates after `timeout-minutes`, unblocking queue

---

## References

- [GitHub Actions Concurrency Documentation](https://docs.github.com/en/actions/using-jobs/using-concurrency)
- [OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
- Original Issue Review: Internal technical review (2025-01-02)

---

## Maintenance

**Review Schedule:** Quarterly or when OpenAI tier/quota changes
**Owner:** DevOps / CI/CD Maintainers
**Related ADRs:** None (first ADR for GitHub Actions architecture)

---

**Changelog:**
- 2025-01-02: Initial decision - Implemented concurrency control across 3 workflows
