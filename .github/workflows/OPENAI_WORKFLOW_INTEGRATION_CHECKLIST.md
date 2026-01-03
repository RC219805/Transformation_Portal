# OpenAI Workflow Integration Checklist

**Purpose:** Ensure new GitHub Actions workflows that consume the OpenAI API follow architectural best practices established in ADR-002.

**Reference:** `docs/architecture/ADR-002-OPENAI-API-CONCURRENCY-CONTROL.md`

---

## ✅ Pre-Integration Checklist

Before adding a new workflow that uses the OpenAI API, verify the following:

### 1. Concurrency Control (CRITICAL)

- [ ] Add job-level concurrency group to prevent RPM limit contention:
  ```yaml
  concurrency:
    group: openai-api-${{ github.repository }}
    cancel-in-progress: false
  ```
- [ ] Apply concurrency to the **job that makes API calls**, not the pre-flight check job
- [ ] Set `cancel-in-progress: false` to enable queuing (not cancellation)

**Rationale:** Multiple workflows share the same OpenAI API key with a 3 RPM limit. Concurrency queuing prevents race conditions and 429 errors.

---

### 2. Job Timeout Protection (CRITICAL)

- [ ] Add appropriate timeout to prevent runaway jobs from blocking the queue:
  ```yaml
  timeout-minutes: 5   # For summaries/triage
  timeout-minutes: 10  # For code reviews (larger payloads)
  ```

**Rationale:** A hanging job without timeout can block all other OpenAI workflows indefinitely.

---

### 3. API Key Pre-Flight Check (REQUIRED)

- [ ] Add a `check-api-key` job to verify key presence:
  ```yaml
  check-api-key:
    name: Check OpenAI API Key
    runs-on: ubuntu-latest
    outputs:
      has_key: ${{ steps.check.outputs.has_key }}
    steps:
      - id: check
        run: |
          if [ -n "${{ secrets.OPENAI_API_KEY }}" ]; then
            echo "has_key=true" >> "$GITHUB_OUTPUT"
          else
            echo "has_key=false" >> "$GITHUB_OUTPUT"
            echo "⚠️ OPENAI_API_KEY not configured - skipping AI feature"
          fi
  ```
- [ ] Make the main job conditional: `if: needs.check-api-key.outputs.has_key == 'true'`
- [ ] Quote `$GITHUB_OUTPUT` to prevent path expansion

**Rationale:** Graceful degradation—skip AI features instead of failing the workflow.

---

### 4. API Key Sanitization (REQUIRED)

- [ ] Strip whitespace from the API key in Python/Node scripts:
  ```python
  OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
  ```

**Rationale:** Prevents silent failures from accidentally pasted keys with leading/trailing whitespace.

---

### 5. Enhanced Error Differentiation (REQUIRED)

- [ ] Implement specific error handling for different failure modes:
  ```python
  except Exception as e:
      error_str = str(e).lower()

      # Quota/billing errors (permanent until account action)
      if "insufficient_quota" in error_str or "billing details" in error_str or "exceeded your current quota" in error_str:
          message = "OpenAI quota/billing limit reached. Check plan/billing for the API key's org."

      # Rate limit errors (temporary, retry later)
      elif "429" in error_str or "rate_limit" in error_str:
          message = "OpenAI rate limit exceeded. Please try again later."

      # Authentication errors (invalid API key)
      elif "401" in error_str or "invalid_api_key" in error_str or "incorrect api key" in error_str:
          message = "Invalid OPENAI_API_KEY. Verify the secret value."

      # Generic fallback
      else:
          message = "An internal error occurred. See CI logs for details."

      # Log full error to stderr (visible in CI logs, not user comments)
      print("Detailed error (redacted for posting):", file=sys.stderr)
      print(repr(e), file=sys.stderr)
  ```

**Rationale:** Users need actionable error messages to self-service issues.

---

### 6. Continue-On-Error (RECOMMENDED)

- [ ] Add `continue-on-error: true` to the main job if AI failures shouldn't block the workflow:
  ```yaml
  continue-on-error: true
  ```

**Rationale:** AI features are enhancements. Workflow should succeed even if AI fails.

---

### 7. Comment Posting Best Practices (RECOMMENDED)

If your workflow posts comments:

- [ ] Add a hidden marker to enable comment updates instead of spam:
  ```html
  <!-- my-workflow-name-marker -->
  ```
- [ ] Before posting, check for existing comments with the marker
- [ ] If found, **PATCH** the existing comment instead of posting new
- [ ] If PATCH fails, fallback to posting new comment

**Example:**
```bash
EXISTING_COMMENT_ID=$(gh api "repos/$REPO/issues/$ISSUE_NUMBER/comments" \
  --jq '.[] | select(.body | contains("<!-- my-marker -->")) | .id' | head -1)

if [ -n "$EXISTING_COMMENT_ID" ]; then
  gh api -X PATCH "repos/$REPO/issues/comments/$EXISTING_COMMENT_ID" \
    -f body="@$RESPONSE_FILE"
else
  gh issue comment "$ISSUE_NUMBER" --repo "$REPO" --body-file "$RESPONSE_FILE"
fi
```

**Rationale:** Reduces comment noise in issue/PR threads.

---

### 8. Retry Logic (OPTIONAL)

If implementing retry logic (less critical now with concurrency control):

- [ ] Use exponential backoff with jitter for transient errors
- [ ] Limit retries to 3-6 attempts (max 6 recommended)
- [ ] Only retry on **rate limits** (429), not quota/auth errors
- [ ] Cap wait times to avoid excessive delays (e.g., `min(60, 2**attempt)`)

**Example:**
```python
def call_openai_with_retries(client, messages, model="gpt-4o-mini", max_retries=6, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(model=model, messages=messages, **kwargs)
        except Exception as e:
            if is_rate_limit_error(e) and attempt < max_retries:
                wait = min(60, 2**attempt) * random.uniform(0.5, 1.5)
                time.sleep(wait)
                continue
            raise
```

**Rationale:** Graceful handling of transient failures (though concurrency control should eliminate most rate limit errors).

---

## 🧪 Testing Checklist

Before merging, verify:

### Syntax Validation

- [ ] YAML syntax is valid: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/my-workflow.yml'))"`
- [ ] Python syntax is valid (extract and run script independently)

### Behavioral Testing

- [ ] Workflow skips gracefully when `OPENAI_API_KEY` is not set
- [ ] Workflow queues properly when another OpenAI job is running
- [ ] Error messages are user-friendly and actionable
- [ ] Comments (if posted) update existing instead of creating new

### Concurrency Testing

- [ ] Trigger multiple workflows simultaneously (e.g., open issue + comment on it)
- [ ] Verify jobs queue in GitHub Actions UI: "⏳ Waiting for a job to complete..."
- [ ] Verify no 429 errors in logs

---

## 📋 Integration Template

### Minimal Workflow Template

```yaml
name: My AI Feature

on:
  issues:
    types: [opened]

jobs:
  check-api-key:
    name: Check OpenAI API Key
    runs-on: ubuntu-latest
    outputs:
      has_key: ${{ steps.check.outputs.has_key }}
    steps:
      - id: check
        run: |
          if [ -n "${{ secrets.OPENAI_API_KEY }}" ]; then
            echo "has_key=true" >> "$GITHUB_OUTPUT"
          else
            echo "has_key=false" >> "$GITHUB_OUTPUT"
            echo "⚠️ OPENAI_API_KEY not configured - skipping AI feature"
          fi

  my-ai-job:
    name: My AI-Powered Feature
    runs-on: ubuntu-latest
    needs: check-api-key
    if: needs.check-api-key.outputs.has_key == 'true'
    continue-on-error: true
    timeout-minutes: 5

    # CRITICAL: Repository-wide OpenAI API serialization
    concurrency:
      group: openai-api-${{ github.repository }}
      cancel-in-progress: false

    permissions:
      issues: write

    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      GH_TOKEN: ${{ github.token }}

    steps:
      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install openai requests

      - name: Run AI feature
        run: |
          python - <<'PY'
          import os, sys
          from openai import OpenAI

          # HARDENING: Strip whitespace
          api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
          if not api_key:
              print("No API key; skipping.")
              sys.exit(0)

          client = OpenAI(api_key=api_key)

          try:
              response = client.chat.completions.create(
                  model="gpt-4o-mini",
                  messages=[{"role": "user", "content": "Hello"}],
                  max_tokens=100
              )
              result = response.choices[0].message.content
          except Exception as e:
              error_str = str(e).lower()
              if "insufficient_quota" in error_str:
                  result = "Quota limit reached. Check billing."
              elif "429" in error_str:
                  result = "Rate limit exceeded. Try again later."
              elif "401" in error_str:
                  result = "Invalid API key."
              else:
                  result = "An error occurred. See logs."
              print(repr(e), file=sys.stderr)

          print(result)
          PY
```

---

## 📚 References

- **ADR-002:** `docs/architecture/ADR-002-OPENAI-API-CONCURRENCY-CONTROL.md`
- **Implementation Summary:** `.github/workflows/OPENAI_API_HARDENING_SUMMARY.md`
- **Existing Workflows:**
  - `summary.yml` (Issue Summarizer)
  - `smart-issue-management.yml` (AI Triage)
  - `ai-code-review.yml` (Code Review)

---

## 🚨 Common Pitfalls

### ❌ DON'T: Use workflow-level concurrency
```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}  # ❌ Wrong!
```
**Why:** Only serializes within the same workflow; different workflows still race.

### ❌ DON'T: Set `cancel-in-progress: true`
```yaml
concurrency:
  group: openai-api-${{ github.repository }}
  cancel-in-progress: true  # ❌ Wrong!
```
**Why:** Jobs cancel instead of queuing, defeating the purpose.

### ❌ DON'T: Retry on quota/billing errors
```python
if "insufficient_quota" in error_str:
    time.sleep(60)  # ❌ Won't help!
    retry()
```
**Why:** Quota errors require account action; retrying wastes time.

### ❌ DON'T: Post generic error messages
```python
except Exception as e:
    message = "An error occurred."  # ❌ Not actionable!
```
**Why:** Users can't self-service; increases support load.

---

## ✅ Approval Checklist

Before merging a new OpenAI workflow:

- [ ] All items in **Pre-Integration Checklist** completed
- [ ] All items in **Testing Checklist** verified
- [ ] Concurrency group matches existing workflows exactly
- [ ] Error messages are user-friendly and actionable
- [ ] Workflow degrades gracefully (no hard failures)
- [ ] YAML and Python syntax validated
- [ ] ADR-002 reviewed and understood

---

**Maintainer:** DevOps / CI/CD Team
**Last Updated:** 2026-01-02
**Version:** 1.0
