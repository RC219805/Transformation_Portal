# CI Workflow AI Rate-Limit Mitigation

**Issue:** GitHub Actions workflows using AI services (OpenAI) have experienced rate-limit errors (HTTP 429) during PR reviews and issue summarization, causing workflow failures or degraded experience.

**Current State:** AI advisory workflows are non-blocking. AI code review retries transient OpenAI 429 responses, treats `insufficient_quota` as terminal, and keeps fallback diagnostics in CI logs instead of posting fallback PR comments. The issue summarizer retries bounded transient OpenAI failures and posts only successful AI summaries; fallback diagnostics stay in CI logs.

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

✅ **Has bounded retry logic and quiet fallback behavior:**
- Has concurrency control
- Skips gracefully if key missing
- Retries OpenAI HTTP 429 responses up to 3 total attempts
- Retries transient HTTP 5xx and network errors up to 3 total attempts
- Honors `Retry-After` when present, capped at 30 seconds
- Otherwise uses exponential backoff with jitter, capped at 20 seconds
- Writes fallback diagnostics to the workflow output/logs without posting PR comments
- Posts only successful AI-generated summaries as issue/PR comments

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

### 2. Keep Issue Summarizer Retry Bounded (Implemented)

**Contract:** `.github/workflows/summary.yml` keeps retries short enough for its 4-minute step timeout and 10-minute job timeout.

**Implementation:**
- Calls OpenAI with `max_retries=3`
- Retries HTTP 429, transient HTTP 5xx, and network errors
- Honors `Retry-After` for 429/5xx responses, capped at 30 seconds
- Falls back to exponential backoff plus jitter when `Retry-After` is absent, capped at 20 seconds
- Treats other HTTP 4xx responses as terminal for the current run
- Emits workflow warnings and exits non-blocking after exhausted retries
- Keeps diagnostic fallback bodies log-only by skipping comment posting when the `<!-- ai-summarizer-diagnostic -->` marker is present

**Rationale:**
- Transient rate limits should get a bounded retry window
- PRs should not accumulate comments for service-unavailable diagnostics
- Human review and normal CI remain authoritative when AI services are unavailable

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
| **Done** | Keep issue summarizer retries bounded and fallback comments suppressed | Improves reliability without PR noise | Low | Implemented |
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
5. Verify that rate-limit fallback diagnostics log a warning without posting a fallback PR comment

---

## Monitoring and Alerting

**To detect ongoing rate-limit issues:**

1. **GitHub Actions logs:** Search for "Rate limited", "429", or `insufficient_quota` in workflow runs
2. **OpenAI dashboard:** Monitor API usage and quota consumption
3. **GitHub Issues:** Track rate-limit failures as maintainer-created incidents when logs show sustained failures

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

**Current:** AI code review retries transient 429s, treats `insufficient_quota` as terminal, stays non-blocking, and avoids fallback PR comments. The issue summarizer retries 429/5xx/network failures up to 3 total attempts, keeps fallback diagnostics log-only, and posts successful AI summaries.

**Proposed:** Consider optional quota monitoring if logs show sustained OpenAI rate-limit pressure.

**Owner:** Repository maintainers (requires workflow permission to modify).

**Next Steps:**
1. Review and approve this recommendation
2. Implement high-priority changes (non-blocking)
3. Schedule medium-priority improvements
4. Monitor effectiveness and iterate
