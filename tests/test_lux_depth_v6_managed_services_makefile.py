"""The explicit V6 service gate must execute tests or fail before pytest."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = "postgresql+asyncpg://test@127.0.0.1:5432/photography_test"
REDIS_URL = "redis://127.0.0.1:6379/15"


def _run_target(tmp_path: Path, configured: dict[str, str], *, pytest_exit: int = 0):
    invocation = tmp_path / "pytest-invocation.txt"
    python = tmp_path / "stub-python"
    python.write_text(
        "#!/bin/sh\n"
        "{\n"
        '  printf "%s\\n" "$@"\n'
        '  printf "%s\\n" "$PYTHONPATH" "$TP_DISPATCH_TEST_DATABASE_URL" '
        '"$TP_PHOTOGRAPHY_TEST_DATABASE_URL" "$TP_DISPATCH_TEST_REDIS_URL"\n'
        '} > "$TP_MAKE_TEST_LOG"\n'
        'exit "$TP_MAKE_TEST_EXIT"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    environment = os.environ.copy()
    for name in ("TP_DISPATCH_TEST_DATABASE_URL", "TP_DISPATCH_TEST_REDIS_URL", "MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES"):
        environment.pop(name, None)
    environment.update(
        TP_PHOTOGRAPHY_TEST_DATABASE_URL="postgresql+asyncpg://test@invalid/ambient_test",
        TP_MAKE_TEST_LOG=str(invocation),
        TP_MAKE_TEST_EXIT=str(pytest_exit),
    )
    result = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "test-lux-depth-v6-managed-services",
            f"PY={python}",
            *(f"{name}={value}" for name, value in configured.items()),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return result, invocation


@pytest.mark.parametrize(
    "configured",
    [{}, {"TP_DISPATCH_TEST_DATABASE_URL": DATABASE_URL}, {"TP_DISPATCH_TEST_REDIS_URL": REDIS_URL}],
    ids=["both-missing", "redis-missing", "database-missing"],
)
def test_managed_v6_service_gate_rejects_missing_services_before_pytest(tmp_path, configured):
    result, invocation = _run_target(tmp_path, configured)
    assert result.returncode != 0
    assert "dedicated migrated *_test database" in result.stdout
    assert "TP_DISPATCH_TEST_DATABASE_URL" in result.stdout
    assert "TP_DISPATCH_TEST_REDIS_URL" in result.stdout
    assert not invocation.exists()


@pytest.mark.parametrize("pytest_exit", [0, 7], ids=["pytest-success", "pytest-failure"])
def test_managed_v6_service_gate_forwards_selected_services_and_pytest_status(tmp_path, pytest_exit):
    result, invocation = _run_target(
        tmp_path,
        {"TP_DISPATCH_TEST_DATABASE_URL": DATABASE_URL, "TP_DISPATCH_TEST_REDIS_URL": REDIS_URL},
        pytest_exit=pytest_exit,
    )
    assert (result.returncode == 0) == (pytest_exit == 0)
    assert invocation.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "pytest",
        "-q",
        "tests/orchestrator/test_managed_v6_services.py",
        "src",
        DATABASE_URL,
        DATABASE_URL,
        REDIS_URL,
    ]
