from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
SMART_TRIAGE_PATH = REPO_ROOT / ".github" / "workflows" / "smart-issue-management.yml"


def _workflow_text() -> str:
    return SMART_TRIAGE_PATH.read_text(encoding="utf-8")


def _load_workflow() -> dict:
    return yaml.load(_workflow_text(), Loader=yaml.BaseLoader)


def _run_triage_step() -> dict:
    workflow = _load_workflow()
    steps = workflow["jobs"]["smart-triage"]["steps"]
    return next(step for step in steps if step.get("name") == "Smart Issue/PR Triage")


def test_smart_triage_workflow_remains_advisory_and_timeout_bounded() -> None:
    workflow = _load_workflow()
    job = workflow["jobs"]["smart-triage"]
    run_step = _run_triage_step()

    assert job["continue-on-error"] == "true"
    assert job["timeout-minutes"] == "10"
    assert run_step["timeout-minutes"] == "4"


def test_smart_triage_classifies_insufficient_quota_as_non_retryable() -> None:
    workflow = _workflow_text()

    assert "def get_openai_error(response):" in workflow
    assert 'non_retryable_429_codes = {"insufficient_quota"}' in workflow
    assert "if response.status_code == 429:" in workflow
    assert "OpenAI HTTP 429 non-retryable" in workflow


def test_smart_triage_preserves_transient_429_retry_behavior() -> None:
    workflow = _workflow_text()

    assert "OpenAI retryable 429" in workflow
    assert "base_wait = min(60, 2 ** attempt)" in workflow
    assert "jitter = random.uniform(0.5, 1.5)" in workflow
    assert "time.sleep(wait)" in workflow


def test_smart_triage_preserves_network_exception_retries() -> None:
    workflow = _workflow_text()

    assert "except requests.RequestException as e:" in workflow
    assert "OpenAI network error" in workflow
    assert "OpenAI request failed after retries" in workflow


def test_smart_triage_suppresses_fallback_pr_comments() -> None:
    workflow = _workflow_text()

    # Per AI_WORKFLOW_PATTERN.md, fallback AI-unavailable diagnostics stay in
    # logs and must not be posted as PR/issue comments. The previous
    # "## ⚠️ AI Triage Error" comment and its POST must both be gone.
    assert "AI Triage Error" not in workflow
    assert "error_comment" not in workflow
    assert "Fallback AI-unavailable diagnostics stay in logs only" in workflow


def test_smart_triage_trigger_excludes_label_events() -> None:
    workflow = _load_workflow()
    triggers = workflow["on"]

    issues_types = triggers["issues"]["types"]
    pr_types = triggers["pull_request_target"]["types"]

    assert "labeled" not in issues_types
    assert "unlabeled" not in issues_types
    assert "labeled" not in pr_types
    assert "unlabeled" not in pr_types
    assert set(issues_types) == {"opened", "reopened"}
    assert set(pr_types) == {"opened", "reopened"}
