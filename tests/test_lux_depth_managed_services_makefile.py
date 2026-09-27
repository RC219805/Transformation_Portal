"""The unified service gate must execute or reject missing service configuration."""

from __future__ import annotations

import pytest

from tests.test_lux_depth_v6_managed_services_makefile import DATABASE_URL, REDIS_URL, _run_target

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "configured",
    [{}, {"TP_DISPATCH_TEST_DATABASE_URL": DATABASE_URL}, {"TP_DISPATCH_TEST_REDIS_URL": REDIS_URL}],
    ids=["both-missing", "redis-missing", "database-missing"],
)
def test_unified_service_gate_rejects_missing_services_before_pytest(tmp_path, configured):
    result, invocation = _run_target(tmp_path, configured, target="test-lux-depth-managed-services")
    assert result.returncode != 0
    assert "dedicated migrated *_test database" in result.stdout
    assert not invocation.exists()


@pytest.mark.parametrize("pytest_exit", [0, 7], ids=["pytest-success", "pytest-failure"])
def test_unified_service_gate_forwards_selected_services_and_pytest_status(tmp_path, pytest_exit):
    result, invocation = _run_target(
        tmp_path,
        {"TP_DISPATCH_TEST_DATABASE_URL": DATABASE_URL, "TP_DISPATCH_TEST_REDIS_URL": REDIS_URL},
        pytest_exit=pytest_exit,
        target="test-lux-depth-managed-services",
    )
    assert (result.returncode == 0) == (pytest_exit == 0)
    assert invocation.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "pytest",
        "-q",
        "tests/orchestrator/test_managed_lux_depth_services.py",
        "src",
        DATABASE_URL,
        DATABASE_URL,
        REDIS_URL,
    ]
