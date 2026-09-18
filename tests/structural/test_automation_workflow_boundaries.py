"""Protect release destinations and automation coverage after the workflow audit."""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW_ROOT = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def _workflow(name: str) -> dict:
    return yaml.load((WORKFLOW_ROOT / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_testpypi_dispatch_never_enables_production_publication() -> None:
    jobs = _workflow("submit-pypi.yml")["jobs"]

    assert jobs["pypi"]["if"] == "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
    assert jobs["test-pypi"]["if"] == "github.event_name == 'workflow_dispatch' && github.event.inputs.test_pypi == 'true'"
    for name, environment in (("pypi", "pypi"), ("test-pypi", "testpypi")):
        assert jobs[name]["environment"]["name"] == environment
        assert jobs[name]["permissions"] == {"id-token": "write"}
    assert "cleanup" not in jobs, "a fresh release runner has no prior job workspace to clean"


def test_codeql_covers_managed_frontdoor_and_worker_languages() -> None:
    job = _workflow("codeql.yml")["jobs"]["analyze"]
    lanes = {lane["language"]: lane["build-mode"] for lane in job["strategy"]["matrix"]["include"]}

    assert lanes == {"actions": "none", "python": "none", "javascript-typescript": "none"}
    assert job["permissions"]["security-events"] == "write"


def test_dependency_submission_retains_retry_and_trigger_contract_without_unused_cache() -> None:
    workflow = _workflow("dependency-submission.yml")
    steps = workflow["jobs"]["submit-pypi"]["steps"]
    submissions = [step for step in steps if step.get("id", "").startswith("dependency_submission_attempt_")]

    assert set(workflow["on"]) == {"push", "pull_request", "workflow_dispatch"}
    assert len(submissions) == 2
    assert submissions[1]["if"] == "steps.dependency_submission_attempt_1.outcome == 'failure'"
    assert all(step["env"]["PIP_NO_CACHE_DIR"] == "1" for step in submissions)
    assert not any(step.get("uses", "").startswith("actions/cache@") for step in steps)
    assert all("pip-tools" not in step.get("run", "") for step in steps)
