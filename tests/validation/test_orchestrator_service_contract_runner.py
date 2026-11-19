"""Evidence and target guards for the live orchestrator contract runner."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from subprocess import CompletedProcess
from types import ModuleType
from unittest.mock import AsyncMock

import pytest

from tests.orchestrator.test_dispatch_authority_services import _clear_dispatch_tables

pytestmark = pytest.mark.unit


def _load_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "validation" / "run_orchestrator_service_contracts.py"
    spec = importlib.util.spec_from_file_location("orchestrator_service_contract_runner_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load orchestrator service contract runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


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
    tests.append({"nodeid": f"{runner.TESTS}/private_provider_name.py::test_service_url_missing", "outcome": "skipped"})
    report.write_text(json.dumps({"exitcode": 0, "tests": tests}))

    with pytest.raises(runner.ContractError, match="unexpected non-passing test") as error:
        runner.check_report(report, runner.PLAIN)
    assert "private_provider_name" not in str(error.value)


def test_provider_pytest_failure_reports_only_verified_identifiers(monkeypatch, tmp_path, capsys):
    suite = runner.PROVIDER
    known_test = f"{runner.TESTS}/{suite.required_tests[0]}"
    selected_file = known_test.split("::", 1)[0]
    unexpected_test = f"{selected_file}::test_private_hostname"
    unselected_test = f"{runner.TESTS}/{runner.PLAIN.required_tests[0]}"
    deselected_test = f"{runner.TESTS}/{runner.DISPATCH_S3}"
    report = {
        "exitcode": 1,
        "collectors": [
            {"nodeid": selected_file, "outcome": "failed", "longrepr": "private_collection_message"},
            {"nodeid": "/private-provider-path/test_file.py", "outcome": "failed"},
        ],
        "tests": [
            {
                "nodeid": f"{known_test}[private_parameter_token]",
                "outcome": "error",
                "setup": {
                    "outcome": "failed",
                    "crash": {"message": "private_exception_message"},
                    "stderr": "private_captured_output",
                },
            },
            *[{"nodeid": node, "outcome": "failed"} for node in (unexpected_test, unselected_test, deselected_test)],
        ],
    }
    original_run = runner.subprocess.run

    def simulate_pytest(argv, **kwargs):
        if argv[0] == "git":
            return original_run(argv, **kwargs)
        (tmp_path / f"{suite.name}.json").write_text(json.dumps(report), encoding="utf-8")
        return CompletedProcess(argv, 1, "postgresql+asyncpg://operator:private_provider_password@private-host/database")

    monkeypatch.setattr(runner.subprocess, "run", simulate_pytest)

    with pytest.raises(runner.ContractError, match="provider-authority pytest run failed"):
        runner.run_suite(suite, {}, tmp_path, ci=False)

    captured = capsys.readouterr()
    assert captured.out == "Running provider-authority\n"
    assert captured.err.splitlines() == [
        "provider-authority exited with status 1",
        f"provider-authority: {selected_file} (collection failed)",
        f"provider-authority: {known_test} (setup error)",
    ]


def test_provider_pytest_failure_without_verified_report_has_safe_fallback(monkeypatch, tmp_path, capsys):
    def simulate_pytest(*_args, **_kwargs):
        (tmp_path / f"{runner.PLAIN.name}.json").write_text("private incomplete report", encoding="utf-8")
        return 2

    monkeypatch.setattr(runner, "run_command", simulate_pytest)

    with pytest.raises(runner.ContractError, match="plain-postgres pytest run failed"):
        runner.run_suite(runner.PLAIN, {}, tmp_path, ci=False)

    assert capsys.readouterr().err == "plain-postgres: no verified pytest failure identifiers available\n"


@pytest.mark.parametrize(
    ("operation", "error", "expected"),
    [
        ("upgrade", "asyncpg.exceptions.InsufficientPrivilegeError", "database privileges"),
        ("check", "FAILED: New upgrade operations detected", "database schema differs from the ORM metadata"),
        ("check", "UnknownFailure", None),
    ],
)
def test_provider_alembic_failure_prints_only_known_categories(monkeypatch, capsys, operation, error, expected):
    raw = f"{error}: private database at postgresql+asyncpg://operator:secret@private-host/db\n"
    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **kwargs: CompletedProcess(argv, 1, raw))

    status = runner.run_command(("alembic", operation), {}, label=f"alembic {operation}", ci=False)

    assert status == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    expected_lines = [f"alembic {operation} exited with status 1"]
    if expected is not None:
        expected_lines.append(f"alembic {operation}: {expected}")
    assert captured.err.splitlines() == expected_lines
