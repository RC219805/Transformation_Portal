from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validation" / "run_managed_paid_pilot_gate.sh"


def _write_env_file(path: Path, *, extra: str = "") -> Path:
    path.write_text(
        "\n".join(
            [
                "TP_ORCHESTRATOR_STATE_BACKEND=postgres",
                "TP_DATABASE_URL=postgresql+asyncpg://staging_user:secret@staging-db.example.test:5432/tp_staging",
                "TP_TEST_POSTGRES_URL=postgresql+asyncpg://test_user:secret@test-db.example.test:5432/tp_validation",
                "TP_ORCHESTRATOR_QUEUE_BACKEND=redis",
                "TP_REDIS_URL=rediss://queue.example.test:6380/0",
                "TP_TEST_REDIS_URL=rediss://queue-validation.example.test:6380/0",
                "TP_FRONTDOOR_SESSION_STORE=redis",
                "TP_FRONTDOOR_REDIS_URL=rediss://sessions.example.test:6380/0",
                "TP_ARTIFACT_STORE=s3",
                "TP_ARTIFACT_ENDPOINT_URL=https://s3.example.test",
                "TP_TEST_S3_URL=https://s3-validation.example.test",
                "TP_ARTIFACT_BUCKET=tp-staging-artifacts",
                "TP_TEST_S3_BUCKET=tp-validation-artifacts",
                "TP_ARTIFACT_REGION=us-west-2",
                "AWS_ACCESS_KEY_ID=AKIATESTVALUE",
                "AWS_SECRET_ACCESS_KEY=test-secret-value",
                extra,
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _run_preflight(env_file: Path, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), "--env-file", str(env_file), "--preflight-only"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_managed_paid_pilot_preflight_runs_from_clean_env(tmp_path: Path) -> None:
    env_file = _write_env_file(tmp_path / "managed.env")
    contaminated_env = dict(os.environ)
    contaminated_env.update(
        {
            "TP_API_KEY": "local-dev-secret",
            "TP_ALLOW_LOCAL_ACCESS_BYPASS": "1",
            "TP_PORTAL_UPLOAD_ROOT": "/tmp/local-upload-root",
        }
    )

    result = _run_preflight(env_file, env=contaminated_env)

    assert result.returncode == 0, result.stderr + result.stdout
    assert "missing: []" in result.stdout
    assert "placeholder-like: []" in result.stdout
    assert "wrong selectors: {}" in result.stdout
    assert "leaked local-dev vars: []" in result.stdout
    assert "unsafe managed/test overlap: {}" in result.stdout
    assert "Managed paid-pilot clean-env preflight passed." in result.stdout


def test_managed_paid_pilot_preflight_rejects_local_dev_vars_in_env_file(tmp_path: Path) -> None:
    env_file = _write_env_file(
        tmp_path / "managed.env",
        extra="TP_ALLOW_LOCAL_ACCESS_BYPASS=1",
    )

    result = _run_preflight(env_file)

    assert result.returncode == 1
    assert "leaked local-dev vars: ['TP_ALLOW_LOCAL_ACCESS_BYPASS']" in result.stdout


def test_managed_paid_pilot_preflight_rejects_unsafe_secret_file_mode(tmp_path: Path) -> None:
    env_file = _write_env_file(tmp_path / "managed.env")
    env_file.chmod(0o644)

    result = _run_preflight(env_file)

    assert result.returncode == 1
    assert "managed provider env file must be chmod 600 or stricter" in result.stderr


def test_managed_paid_pilot_preflight_rejects_staging_test_overlap(tmp_path: Path) -> None:
    env_file = _write_env_file(
        tmp_path / "managed.env",
        extra="TP_TEST_S3_BUCKET=tp-staging-artifacts",
    )

    result = _run_preflight(env_file)

    assert result.returncode == 1
    assert "'TP_TEST_S3_BUCKET': 'must not equal TP_ARTIFACT_BUCKET'" in result.stdout
