"""Contract for the independent, disposable orchestration service CI job."""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
JOB_ID = "orchestrator-services"


def _load_workflow(name: str) -> dict:
    return yaml.load((WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_service_job_is_advisory_and_preserves_real_failures() -> None:
    enforcement = _load_workflow("enforcement.yml")
    job = enforcement["jobs"][JOB_ID]

    assert {"pull_request", "push"} <= enforcement["on"].keys()
    assert enforcement["permissions"] == {"contents": "read"}
    assert job["if"] == "github.event_name == 'pull_request' || github.event_name == 'push'"
    assert job["timeout-minutes"] == "20"
    assert "continue-on-error" not in job
    assert all("continue-on-error" not in step for step in job["steps"])
    assert JOB_ID not in _load_workflow("build.yml")["jobs"]["ci_gate"]["needs"]


def test_service_job_uses_separate_databases_and_ephemeral_loopback_ports() -> None:
    job = _load_workflow("enforcement.yml")["jobs"][JOB_ID]
    steps = job["steps"]
    postgres, redis = (job["services"][service] for service in ("postgres", "redis"))
    setup_python = next(step for step in steps if step.get("uses", "").startswith("actions/setup-python@"))
    create = next(step for step in steps if step.get("name") == "Create isolated Postgres test databases")
    run = next(step for step in steps if step.get("name") == "Run live Postgres and Redis contracts")

    assert setup_python["with"]["python-version"] == "3.11"
    assert postgres["image"] == "postgres:16-alpine"
    assert redis["image"] == "redis:7-alpine"
    assert postgres["ports"] == ["5432/tcp"]
    assert redis["ports"] == ["6379/tcp"]
    assert create["env"] == {"POSTGRES_CONTAINER_ID": "${{ job.services.postgres.id }}"}
    database_names = {
        "TP_TEST_POSTGRES_URL": "tp_orchestrator_plain_test",
        "TP_DISPATCH_TEST_DATABASE_URL": "tp_orchestrator_migrated_test",
    }
    assert run["env"].keys() == {*database_names, "TP_TEST_REDIS_URL"}
    for variable, database in database_names.items():
        assert f"createdb --username postgres {database}" in create["run"]
        assert run["env"][variable].endswith(f"@127.0.0.1:${{{{ job.services.postgres.ports[5432] }}}}/{database}")
    assert run["env"]["TP_TEST_REDIS_URL"] == "redis://127.0.0.1:${{ job.services.redis.ports[6379] }}/0"
    assert steps.index(create) < steps.index(run)


def test_service_job_uploads_only_the_sanitized_contract_report() -> None:
    steps = _load_workflow("enforcement.yml")["jobs"][JOB_ID]["steps"]
    run = next(step for step in steps if step.get("name") == "Run live Postgres and Redis contracts")
    uploads = [step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@")]

    assert run["run"] == (
        "python scripts/validation/run_orchestrator_service_contracts.py ci "
        '--evidence-out "$RUNNER_TEMP/orchestrator-service-contracts.json"'
    )
    assert len(uploads) == 1
    assert uploads[0]["if"] == "${{ !cancelled() }}"
    assert uploads[0]["with"]["path"] == "${{ runner.temp }}/orchestrator-service-contracts.json"
