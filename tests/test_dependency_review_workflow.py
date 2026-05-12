from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "dependency-review.yml"
DEPENDENCY_REVIEW_SHA = "a1d282b36b6f3519aa1f3fc636f609c47dddb294"


def _load_workflow() -> dict:
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_dependency_review_workflow_exists() -> None:
    assert WORKFLOW_PATH.exists()


def test_dependency_review_workflow_triggers_on_pull_requests_to_main() -> None:
    workflow = _load_workflow()

    assert "pull_request" in workflow["on"]
    assert workflow["on"]["pull_request"]["branches"] == ["main"]


def test_dependency_review_workflow_is_advisory_and_sha_pinned() -> None:
    workflow = _load_workflow()
    job = workflow["jobs"]["dependency-review"]
    steps = job["steps"]
    dependency_review_step = next(
        step for step in steps if step.get("uses", "").startswith("actions/dependency-review-action@")
    )

    assert dependency_review_step["uses"] == f"actions/dependency-review-action@{DEPENDENCY_REVIEW_SHA}"
    assert dependency_review_step["with"]["warn-only"] == "true"
    assert dependency_review_step["with"]["retry-on-snapshot-warnings"] == "true"
    assert dependency_review_step["with"]["retry-on-snapshot-warnings-timeout"] == "120"
    assert workflow["permissions"]["contents"] == "read"
