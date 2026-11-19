"""Evidence and target guards for the live orchestrator contract runner."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from scripts.validation import run_orchestrator_service_contracts as runner
from tests.orchestrator.test_dispatch_authority_services import _clear_dispatch_tables

pytestmark = pytest.mark.unit


@pytest.fixture
def targets(monkeypatch):
    monkeypatch.setenv("TP_TEST_POSTGRES_URL", "postgresql+asyncpg://ci:ephemeral@127.0.0.1/plain_test")
    monkeypatch.setenv("TP_DISPATCH_TEST_DATABASE_URL", "postgresql+asyncpg://ci:ephemeral@localhost/migrated_test")
    monkeypatch.setenv("TP_TEST_REDIS_URL", "redis://127.0.0.1/0")
    return (
        runner.endpoint("TP_TEST_POSTGRES_URL"),
        runner.endpoint("TP_DISPATCH_TEST_DATABASE_URL"),
        runner.endpoint("TP_TEST_REDIS_URL", redis=True),
    )


@pytest.mark.parametrize(
    "url",
    [
        "",
        "postgresql://db.example:5432/migrated_test",
        "postgresql+asyncpg://db.example:0/migrated_test",
        "postgresql+asyncpg://db.example/%6digrated_test",
        "postgresql+asyncpg://db.example/migrated_test?database=protected",
        "postgresql+asyncpg://db.example/migrated_test?HOST=protected.example",
    ],
)
def test_explicit_database_identity_is_required(monkeypatch, url):
    monkeypatch.setenv("TP_DISPATCH_TEST_DATABASE_URL", url)
    with pytest.raises(runner.ContractError, match="TP_DISPATCH_TEST_DATABASE_URL") as error:
        runner.endpoint("TP_DISPATCH_TEST_DATABASE_URL")
    assert "protected" not in str(error.value)


@pytest.mark.parametrize("name", ["plain_test", "migrated_test"])
def test_provider_same_database_name_fails_before_service_probe(monkeypatch, targets, name):
    monkeypatch.setenv("TP_DATABASE_URL", f"postgresql+asyncpg://operator:private@different-name.example/{name}")
    probe = AsyncMock()
    monkeypatch.setattr(runner, "service_info", probe)

    with pytest.raises(runner.ContractError, match="separate from staging"):
        asyncio.run(runner.preflight(*targets, ci=False))

    probe.assert_not_awaited()


def test_provider_rejects_staging_redis_alias_before_service_probe(monkeypatch, targets):
    monkeypatch.setenv("TP_DATABASE_URL", "postgresql+asyncpg://stage.example/staging")
    monkeypatch.setenv("TP_REDIS_URL", "rediss://localhost:6379/00")
    monkeypatch.setenv("TP_FRONTDOOR_REDIS_URL", "redis://session.example/1")
    probe = AsyncMock()
    monkeypatch.setattr(runner, "service_info", probe)

    with pytest.raises(runner.ContractError, match="disposable Redis"):
        asyncio.run(runner.preflight(*targets, ci=False))

    probe.assert_not_awaited()


def test_ci_rejects_non_loopback_before_service_probe(monkeypatch, targets):
    monkeypatch.setenv("TP_TEST_REDIS_URL", "redis://remote.example/0")
    redis = runner.endpoint("TP_TEST_REDIS_URL", redis=True)
    probe = AsyncMock()
    monkeypatch.setattr(runner, "service_info", probe)

    with pytest.raises(runner.ContractError, match="loopback services"):
        asyncio.run(runner.preflight(*targets[:2], redis, ci=True))

    probe.assert_not_awaited()


def test_live_probe_failure_never_echoes_provider_credentials(monkeypatch, targets):
    probe = AsyncMock(side_effect=RuntimeError("postgresql+asyncpg://operator:private@staging.example/database"))
    monkeypatch.setattr(runner, "database_info", probe)

    with pytest.raises(runner.ContractError, match="read-only PostgreSQL probe failed") as error:
        asyncio.run(runner.service_info(targets[1]))

    assert "private" not in str(error.value)


def test_new_test_processes_use_only_disposable_service_settings(monkeypatch, targets):
    monkeypatch.setenv("TP_DATABASE_URL", "postgresql+asyncpg://stage.example/staging")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "private")
    monkeypatch.setenv("PGHOST", "stage.example")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-n 8")
    monkeypatch.setenv("TP_DISPATCH_TEST_S3_URL", "https://provider.example")

    settings = runner.test_environment(*targets)

    assert settings["TP_TEST_POSTGRES_URL"] == targets[0].url
    assert settings["TP_PHOTOGRAPHY_TEST_DATABASE_URL"] == targets[1].url
    assert settings["TP_TENANT_TEST_DATABASE_URL"] == targets[1].url
    assert settings["TP_DISPATCH_TEST_REDIS_URL"] == targets[2].url
    assert settings["TP_ARTIFACT_STORE"] == "local"
    assert {"TP_DATABASE_URL", "AWS_ACCESS_KEY_ID", "PGHOST", "PYTEST_ADDOPTS", "TP_DISPATCH_TEST_S3_URL"}.isdisjoint(settings)


def test_dispatch_truncate_checks_the_connection_target():
    connection = AsyncMock()
    connection.scalar.return_value = "different_test"

    with pytest.raises(RuntimeError, match="configured dispatch test database"):
        asyncio.run(_clear_dispatch_tables(connection, "dispatch_test"))

    connection.execute.assert_not_awaited()


def _results_for_plain():
    return [
        {"nodeid": f"{runner.TESTS}/{test}", "outcome": "passed", "call": {"outcome": "passed"}}
        for test in runner.PLAIN.required_tests
    ]


def test_evidence_requires_real_backend_passes_and_keeps_expected_memory_skips(tmp_path):
    report = tmp_path / "report.json"
    tests = _results_for_plain()
    tests.append({"nodeid": f"{runner.TESTS}/{runner.PLAIN.allowed_skips[0]}", "outcome": "skipped"})
    report.write_text(json.dumps({"exitcode": 0, "tests": tests}))

    assert runner.check_report(report, runner.PLAIN) == {"passed": len(tests) - 1, "skipped": 1}


@pytest.mark.parametrize("change", ["skip", "deselect_backend", "deselect_canary", "missing_report", "no_call"])
def test_incomplete_service_evidence_fails_closed(tmp_path, change):
    report = tmp_path / "report.json"
    tests = _results_for_plain()
    if change == "skip":
        tests[0]["outcome"] = "skipped"
    elif change == "deselect_backend":
        tests.pop(0)
    elif change == "deselect_canary":
        tests.pop(2)
    elif change == "no_call":
        tests[0].pop("call")
    if change != "missing_report":
        report.write_text(json.dumps({"exitcode": 0, "tests": tests}))

    with pytest.raises(runner.ContractError, match="required|results"):
        runner.check_report(report, runner.PLAIN)


def test_unknown_service_skip_fails_even_when_all_known_canaries_pass(tmp_path):
    report = tmp_path / "report.json"
    tests = _results_for_plain()
    tests.append({"nodeid": f"{runner.TESTS}/test_new_provider.py::test_service_url_missing", "outcome": "skipped"})
    report.write_text(json.dumps({"exitcode": 0, "tests": tests}))

    with pytest.raises(runner.ContractError, match="unexpected non-passing test"):
        runner.check_report(report, runner.PLAIN)
