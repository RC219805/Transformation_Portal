from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
AI_CODE_REVIEW_PATH = REPO_ROOT / ".github" / "workflows" / "ai-code-review.yml"


def _workflow_text() -> str:
    return AI_CODE_REVIEW_PATH.read_text(encoding="utf-8")


def _load_workflow() -> dict:
    return yaml.load(_workflow_text(), Loader=yaml.BaseLoader)


def _run_ai_code_review_step() -> dict:
    workflow = _load_workflow()
    steps = workflow["jobs"]["ai-review"]["steps"]
    return next(step for step in steps if step.get("name") == "Run AI Code Review")


def test_ai_code_review_workflow_remains_advisory_and_timeout_bounded() -> None:
    workflow = _load_workflow()
    job = workflow["jobs"]["ai-review"]
    run_step = _run_ai_code_review_step()

    assert job["continue-on-error"] == "true"
    assert job["timeout-minutes"] == "10"
    assert run_step["timeout-minutes"] == "4"


def test_ai_code_review_classifies_insufficient_quota_as_non_retryable() -> None:
    workflow = _workflow_text()

    assert "def get_openai_error(response):" in workflow
    assert 'non_retryable_429_codes = {"insufficient_quota"}' in workflow
    assert "if response.status_code == 429:" in workflow
    assert "OpenAI HTTP 429 non-retryable" in workflow


def test_ai_code_review_preserves_transient_429_retry_behavior() -> None:
    workflow = _workflow_text()

    assert "OpenAI retryable 429" in workflow
    assert "base_wait = min(60, 2 ** attempt)" in workflow
    assert "jitter = random.uniform(0.5, 1.5)" in workflow
    assert "time.sleep(wait)" in workflow


def test_ai_code_review_preserves_network_exception_retries() -> None:
    workflow = _workflow_text()

    assert "except requests.RequestException as e:" in workflow
    assert "OpenAI network error" in workflow
    assert "OpenAI request failed after retries" in workflow


def test_ai_code_review_suppresses_fallback_pr_comments() -> None:
    workflow = _workflow_text()

    assert "should_post_review_comment = False" in workflow
    assert "should_post_review_comment = bool(review_text)" in workflow
    assert "if should_post_review_comment and review_text:" in workflow
    assert "Post comments only when OpenAI returned real review content." in workflow
    assert "attempt even if fallback review_text used" not in workflow
