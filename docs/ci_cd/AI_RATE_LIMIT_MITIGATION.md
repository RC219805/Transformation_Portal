# CI Workflow AI Rate-Limit Mitigation

**Issue:** GitHub Actions workflows using AI services (OpenAI) have experienced rate-limit errors (HTTP 429) during PR reviews and issue summarization, causing workflow failures or degraded experience.

**Current State:** AI code review is non-blocking, retries transient OpenAI 429 responses, treats `insufficient_quota` as terminal, and keeps fallback diagnostics in CI logs instead of posting fallback PR comments.

---

## Current Mitigation (Implemented)

### AI Code Review Workflow (`.github/workflows/ai-code-review.yml`)

✅ **Has retry logic with exponential backoff:**
- 6 retries with exponential backoff (2^attempt, capped at 60s)
- Jitter (0.5x-1.5x) to avoid thundering herd
- Transient 429 responses are retried
- `insufficient_quota` is treated as a terminal quota/billing condition and is not retried in the same workflow run
- Graceful degradation writes a diagnostic fallback response and emits a CI warning without posting a fallback PR comment

✅ **Has concurrency control:**
- `cancel-in-progress: true` reduces redundant API calls
- Only latest code in PR is reviewed

✅ **Skips gracefully:**
- Checks for `OPENAI_API_KEY` presence
- Exits cleanly if key missing

✅ **Suppresses noisy fallback comments:**
- Real AI-generated reviews are posted as PR comments
- AI-unavailable fallback messages stay in the workflow logs/output file
- Operators should inspect CI logs for terminal quota diagnostics

### Issue Summarizer Workflow (`.github/workflows/summary.yml`)

⚠️ **Partial mitigation:**
- Has concurrency control
- Skips gracefully if key missing
- **No retry logic** - single attempt only

---

## Recommended Improvements

### 1. Keep AI Workflows Non-Blocking (Implemented)

**Contract:** AI review/summary jobs are **informational only**, not required status checks.

**Implementation:**
```yaml
# In .github/workflows/ai-code-review.yml
jobs:
  ai-review:
    # ... existing config ...
    continue-on-error: true  # Don't block PR merge on AI failure
```

**Rationale:**
- AI services are external dependencies with variable availability
- Rate limits and quota exhaustion must not block PR merges when code is correct
- Human review is the authoritative gate; AI is advisory

### 2. Add Retry Logic to Issue Summarizer (Medium Priority)

**Change:** Add same retry/backoff pattern used in AI code review.

**Implementation:**
```python
def call_openai_with_retries(client, messages, model="gpt-4o-mini", max_retries=6, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=0.3,
                **kwargs
            )
        except Exception as e:
            err_str = str(e)
            status_is_429 = False
            status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
            if status == 429 or "429" in err_str or "Rate limit" in err_str:
                status_is_429 = True

            if status_is_429 and attempt < max_retries:
                base_wait = min(60, 2 ** attempt)
                jitter = random.uniform(0.5, 1.5)
                wait = base_wait * jitter
                print(f"⚠️ Rate limited (attempt {attempt}/{max_retries}). Waiting {wait:.1f}s...")
                time.sleep(wait)
                continue
            raise
```

**Rationale:**
- Consistent error handling across AI workflows
- Transient rate limits should not cause permanent failures
- Terminal quota errors such as `insufficient_quota` should be logged once and not retried

### 3. Add Rate-Limit Budget Monitoring (Medium Priority)

**Change:** Track API usage and warn when approaching quota.

**Implementation:**
```yaml
# New job in ai-code-review.yml
check-quota:
  runs-on: ubuntu-latest
  if: github.event.pull_request.draft == false
  steps:
    - name: Check OpenAI quota
      run: |
        # Query OpenAI API for current usage (if available via their API)
        # Post warning comment if approaching limit
        # Example:
        # - Green: <50% quota used
        # - Yellow: 50-80% quota used
        # - Red: >80% quota used
```

**Rationale:**
- Proactive warning before hitting hard limits
- Allows repo maintainers to upgrade plan or adjust usage

### 4. Add Fallback Degraded Mode (Low Priority)

**Change:** When rate-limited, provide lightweight summary instead of full AI review.

**Implementation:**
- Count changed lines, files, modified areas (git diff stats only)
- Write simple stats to the job summary/output; avoid PR comments for AI-unavailable fallback text
- No external API calls required

**Rationale:**
- Still provides some value when AI unavailable
- Keeps PR comment section consistent

### 5. Consider Alternative AI Providers (Low Priority)

**Change:** Add fallback to different AI provider if OpenAI rate-limited.

**Options:**
- Anthropic Claude API (separate rate limits)
- GitHub Copilot API (if available for Actions)
- Self-hosted LLM (higher latency, but no external rate limits)

**Rationale:**
- Diversifies risk across multiple providers
- May increase overall availability

---

## Implementation Priority

| Priority | Change | Impact | Effort | Recommended Timeline |
|----------|--------|--------|--------|---------------------|
| **Done** | Keep AI workflows non-blocking | Prevents PR merge blockage | Low (1 line change) | Implemented |
| **Medium** | Add retry to issue summarizer | Improves reliability | Medium (30 mins) | Next sprint |
| **Medium** | Add rate-limit budget monitoring | Proactive management | Medium (1-2 hours) | Next sprint |
| **Low** | Add degraded mode fallback | Better job-summary UX when limited | High (2-3 hours) | Future consideration |
| **Low** | Alternative AI providers | Risk diversification | High (4-6 hours) | Future consideration |

---

## Existing Contract: Non-Blocking AI Workflows

**Configured behavior:**

```diff
# .github/workflows/ai-code-review.yml
jobs:
  ai-review:
    name: AI-Powered Code Review
    runs-on: ubuntu-latest
+   continue-on-error: true  # Don't block PR merge on AI failure
    if: github.event.pull_request.draft == false
```

```diff
# .github/workflows/summary.yml
jobs:
  summarize:
+   continue-on-error: true  # Don't block on AI failures
    if: ${{ !(github.event_name == 'pull_request' && (github.actor == 'Copilot' || github.actor == 'copilot-swe-agent[bot]')) }}
    runs-on: ubuntu-latest
```

**Test plan:**
1. Create test PR with code changes
2. Trigger AI review workflow
3. Verify that even if AI fails, PR can still merge
4. Verify that AI comments still post when successful
5. Verify that `insufficient_quota` logs a warning without repeated retries or a fallback PR comment

---

## Monitoring and Alerting

**To detect ongoing rate-limit issues:**

1. **GitHub Actions logs:** Search for "Rate limited", "429", or `insufficient_quota` in workflow runs
2. **OpenAI dashboard:** Monitor API usage and quota consumption
3. **GitHub Issues:** Track rate-limit failures as incidents

**Alert thresholds:**
- **Warning:** >10 rate-limit errors per day
- **Critical:** >50 rate-limit errors per day or >5 consecutive failures

---

## Related Documentation

- **AI Code Review Workflow:** `.github/workflows/ai-code-review.yml`
- **Issue Summarizer Workflow:** `.github/workflows/summary.yml`
- **OpenAI Rate Limits:** https://platform.openai.com/docs/guides/rate-limits

---

## Status

**Current:** AI code review retries transient 429s, treats `insufficient_quota` as terminal, stays non-blocking, and avoids fallback PR comments.

**Proposed:** Add retry classification parity to issue summarizer and consider optional quota monitoring.

**Owner:** Repository maintainers (requires workflow permission to modify).

**Next Steps:**
1. Review and approve this recommendation
2. Implement high-priority changes (non-blocking)
3. Schedule medium-priority improvements
4. Monitor effectiveness and iterate
