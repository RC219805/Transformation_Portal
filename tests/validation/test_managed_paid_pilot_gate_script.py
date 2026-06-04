from __future__ import annotations

import os
import shlex
import subprocess
import sys
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


def _run_preflight(
    env_file: Path,
    *,
    env: dict[str, str] | None = None,
    evidence_out: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    fake_bin = env_file.parent / "fake-bin"
    fake_bin.mkdir(exist_ok=True)
    fake_node = fake_bin / "node"
    fake_node.write_text("#!/bin/sh\nprintf '%s\\n' 'v22.22.2'\n", encoding="utf-8")
    fake_node.chmod(0o755)

    run_env = dict(os.environ) if env is None else dict(env)
    run_env["PATH"] = f"{fake_bin}{os.pathsep}{run_env.get('PATH', '')}"

    args = ["bash", str(SCRIPT_PATH), "--env-file", str(env_file), "--preflight-only"]
    if evidence_out is not None:
        args.extend(["--evidence-out", str(evidence_out)])

    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=run_env,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_executable(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_fake_gate_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    fake_repo = tmp_path / "fake-repo"
    gate_script = fake_repo / "scripts" / "validation" / "run_managed_paid_pilot_gate.sh"
    _write_executable(gate_script, SCRIPT_PATH.read_text(encoding="utf-8"))

    real_python = shlex.quote(sys.executable)
    _write_executable(
        fake_repo / "scripts" / "setup" / "resolve_python_311.sh",
        f"#!/bin/sh\nprintf '%s\\n' {real_python}\n",
    )
    _write_executable(
        fake_repo / "scripts" / "setup" / "ensure_node_version.sh",
        "#!/bin/sh\nexit 0\n",
    )

    venv_bin = fake_repo / ".venv" / "bin"
    _write_executable(
        venv_bin / "activate",
        f"export PATH={shlex.quote(str(venv_bin))}:$PATH\n",
    )
    _write_executable(
        venv_bin / "python",
        "\n".join(
            [
                "#!/bin/sh",
                'if [ "$1" = "-m" ] && [ "$2" = "pytest" ] && [ "$4" = "tests/orchestrator/test_paid_pilot_services_contract.py" ]; then',
                '    printf "%s\\n" "integrated:$*" >> "$GATE_TEST_LOG"',
                '    exit "${GATE_INTEGRATED_EXIT:-0}"',
                "fi",
                f'exec {real_python} "$@"',
                "",
            ]
        ),
    )

    fake_bin = tmp_path / "fake-bin"
    _write_executable(fake_bin / "node", "#!/bin/sh\nprintf '%s\\n' 'v22.22.2'\n")
    _write_executable(
        fake_bin / "make",
        "\n".join(
            [
                "#!/bin/sh",
                'target="$1"',
                'printf "%s\\n" "make:$target" >> "$GATE_TEST_LOG"',
                'if [ "${GATE_FAIL_TARGET:-}" = "$target" ]; then',
                '    exit "${GATE_FAIL_EXIT:-42}"',
                "fi",
                "exit 0",
                "",
            ]
        ),
    )
    return gate_script, fake_bin, tmp_path / "gate-steps.log"


def _run_fake_full_gate(
    tmp_path: Path,
    *,
    make_fail_target: str = "",
    integrated_exit: int = 0,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    gate_script, fake_bin, step_log = _write_fake_gate_repo(tmp_path)
    env_file = _write_env_file(
        tmp_path / "managed.env",
        extra="\n".join(
            [
                f"GATE_TEST_LOG={shlex.quote(str(step_log))}",
                f"GATE_FAIL_TARGET={shlex.quote(make_fail_target)}",
                "GATE_FAIL_EXIT=42",
                f"GATE_INTEGRATED_EXIT={integrated_exit}",
            ]
        ),
    )
    evidence_out = tmp_path / "acceptance-note.md"

    run_env = dict(os.environ)
    run_env["PATH"] = f"{fake_bin}{os.pathsep}{run_env.get('PATH', '')}"
    result = subprocess.run(
        ["bash", str(gate_script), "--env-file", str(env_file), "--evidence-out", str(evidence_out)],
        cwd=gate_script.parents[2],
        env=run_env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, evidence_out, step_log


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


def test_managed_paid_pilot_preflight_writes_redacted_evidence_note(tmp_path: Path) -> None:
    env_file = _write_env_file(tmp_path / "managed.env")
    evidence_out = tmp_path / "acceptance-note.md"

    result = _run_preflight(env_file, evidence_out=evidence_out)

    assert result.returncode == 0, result.stderr + result.stdout
    note = evidence_out.read_text(encoding="utf-8")
    assert "# Managed Provider Paid-Pilot Acceptance Note" in note
    assert "gate_status: `preflight_passed`" in note
    assert "`TP_ORCHESTRATOR_STATE_BACKEND`: `postgres`" in note
    assert "`TP_DATABASE_URL`: `scheme=postgresql+asyncpg; tls=no; path=set`" in note
    assert "## Gate Step Results" in note
    assert "| Clean env preflight | `managed provider env validation` | `passed` | `0` |" in note
    assert "staging_user" not in note
    assert "staging-db.example.test" not in note
    assert "tp-staging-artifacts" not in note
    assert "test-secret-value" not in note


def test_managed_paid_pilot_preflight_rejects_repo_local_evidence_note(tmp_path: Path) -> None:
    env_file = _write_env_file(tmp_path / "managed.env")

    result = _run_preflight(env_file, evidence_out=REPO_ROOT / "managed-acceptance.md")

    assert result.returncode == 1
    assert "evidence output must live outside the repository" in result.stderr


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


def test_managed_paid_pilot_clean_reexec_uses_portable_bash_shell() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'SHELL="/bin/bash"' in script
    assert 'SHELL="/bin/zsh"' not in script


def test_managed_paid_pilot_preflight_rejects_staging_test_overlap(tmp_path: Path) -> None:
    env_file = _write_env_file(
        tmp_path / "managed.env",
        extra="TP_TEST_S3_BUCKET=tp-staging-artifacts",
    )

    result = _run_preflight(env_file)

    assert result.returncode == 1
    assert "'TP_TEST_S3_BUCKET': 'must not equal TP_ARTIFACT_BUCKET'" in result.stdout


def test_managed_paid_pilot_preflight_rejects_optional_placeholder_region(tmp_path: Path) -> None:
    env_file = _write_env_file(
        tmp_path / "managed.env",
        extra="TP_ARTIFACT_REGION='<region>'",
    )

    result = _run_preflight(env_file)

    assert result.returncode == 1
    assert "placeholder-like:" in result.stdout
    assert "TP_ARTIFACT_REGION" in result.stdout


def test_managed_paid_pilot_preflight_rejects_same_postgres_database_with_different_credentials(
    tmp_path: Path,
) -> None:
    env_file = _write_env_file(
        tmp_path / "managed.env",
        extra="\n".join(
            [
                "TP_DATABASE_URL=postgresql+asyncpg://staging_user:secret@db.example.test:5432/tp",
                "TP_TEST_POSTGRES_URL=postgresql+asyncpg://test_user:secret@db.example.test:5432/tp",
            ]
        ),
    )

    result = _run_preflight(env_file)

    assert result.returncode == 1
    assert "TP_TEST_POSTGRES_URL" in result.stdout
    assert "same Postgres host/port/database" in result.stdout


def test_managed_paid_pilot_full_gate_runs_components_before_integrated_smoke(
    tmp_path: Path,
) -> None:
    result, evidence_out, step_log = _run_fake_full_gate(tmp_path)

    assert result.returncode == 0, result.stderr + result.stdout
    assert step_log.read_text(encoding="utf-8").splitlines() == [
        "make:db-upgrade",
        "make:test-orchestrator-postgres-contract",
        "make:test-orchestrator-postgres-app-contract",
        "make:test-worker-redis-contract",
        "make:test-artifact-s3-contract",
        "make:test-frontdoor-redis-contract",
        "integrated:-m pytest -q tests/orchestrator/test_paid_pilot_services_contract.py -m unit",
    ]

    note = evidence_out.read_text(encoding="utf-8")
    assert "gate_status: `gate_passed`" in note
    assert "| Database migrations | `make db-upgrade` | `passed` | `0` |" in note
    assert (
        "| Orchestrator Postgres contract | "
        "`TP_ORCHESTRATOR_STATE_BACKEND=memory TP_ORCHESTRATOR_QUEUE_BACKEND=memory "
        "TP_ARTIFACT_STORE=local make test-orchestrator-postgres-contract` | "
        "`passed` | `0` |"
    ) in note
    assert "| Frontdoor Redis contract | `make test-frontdoor-redis-contract` | `passed` | `0` |" in note
    assert (
        "| Integrated paid-pilot smoke | "
        "`TP_RUN_PAID_PILOT_SERVICES_CONTRACT=1 python -m pytest -q "
        "tests/orchestrator/test_paid_pilot_services_contract.py -m unit` | "
        "`passed` | `0` |"
    ) in note
    assert note.index("Orchestrator Postgres contract") < note.index("Integrated paid-pilot smoke")
    assert "staging_user" not in note
    assert "staging-db.example.test" not in note
    assert "tp-staging-artifacts" not in note
    assert "test-secret-value" not in note


def test_managed_paid_pilot_full_gate_records_failed_component_and_stops(
    tmp_path: Path,
) -> None:
    result, evidence_out, step_log = _run_fake_full_gate(
        tmp_path,
        make_fail_target="test-worker-redis-contract",
    )

    assert result.returncode == 42, result.stderr + result.stdout
    assert step_log.read_text(encoding="utf-8").splitlines() == [
        "make:db-upgrade",
        "make:test-orchestrator-postgres-contract",
        "make:test-orchestrator-postgres-app-contract",
        "make:test-worker-redis-contract",
    ]

    note = evidence_out.read_text(encoding="utf-8")
    assert "gate_status: `gate_failed`" in note
    assert (
        "| Worker Redis contract | "
        "`TP_ORCHESTRATOR_STATE_BACKEND=memory TP_ORCHESTRATOR_QUEUE_BACKEND=memory "
        "TP_ARTIFACT_STORE=local make test-worker-redis-contract` | "
        "`failed` | `42` |"
    ) in note
    assert "| Artifact S3 contract | `make test-artifact-s3-contract` | `not_run` | `-` |" in note
    assert (
        "| Integrated paid-pilot smoke | "
        "`TP_RUN_PAID_PILOT_SERVICES_CONTRACT=1 python -m pytest -q "
        "tests/orchestrator/test_paid_pilot_services_contract.py -m unit` | "
        "`not_run` | `-` |"
    ) in note
