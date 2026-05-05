from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WRITE_LOCAL_ENV_SCRIPT = REPO_ROOT / "scripts" / "dev" / "write_local_env.sh"
STOP_LOCAL_STACK_SCRIPT = REPO_ROOT / "scripts" / "dev" / "stop_local_stack.sh"
RUN_CLOUDFLARED_SCRIPT = REPO_ROOT / "scripts" / "dev" / "run_cloudflared.sh"


def _source_local_env_key_pair(env_file: Path) -> list[str]:
    result = subprocess.run(
        [
            "bash",
            "-lc",
            f'source {shlex.quote(str(env_file))}; printf \'%s\\n%s\' "$TP_API_KEY" "$TP_BACKEND_API_KEY"',
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.splitlines()


@pytest.mark.unit
def test_write_local_env_shell_quotes_caller_supplied_values(tmp_path: Path) -> None:
    env_file = tmp_path / "local.env"
    users_file = tmp_path / "users$(touch users-pwned).json"
    session_db = tmp_path / "sessions`touch sessions-pwned`.db"
    marker = tmp_path / "key-pwned"
    malicious_key = f"secret$(touch {shlex.quote(str(marker))})"

    env = dict(os.environ)
    env.update(
        {
            "TP_LOCAL_ENV_FILE": str(env_file),
            "TP_FRONTDOOR_USERS_FILE": str(users_file),
            "TP_FRONTDOOR_SESSION_DB": str(session_db),
            "TP_LOCAL_API_KEY": malicious_key,
        }
    )

    subprocess.run(["bash", str(WRITE_LOCAL_ENV_SCRIPT)], check=True, env=env)
    content = env_file.read_text(encoding="utf-8")

    assert 'export TP_API_KEY="' not in content
    assert "$(" in content
    assert "`" in content

    result = subprocess.run(
        [
            "bash",
            "-lc",
            f"source {shlex.quote(str(env_file))}; printf '%s' \"$TP_API_KEY\"",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout == malicious_key
    assert not marker.exists()
    assert not (tmp_path / "users-pwned.json").exists()
    assert not (tmp_path / "sessions-pwned.db").exists()


@pytest.mark.unit
def test_write_local_env_reuses_generated_single_quoted_key(tmp_path: Path) -> None:
    env_file = tmp_path / "local.env"
    users_file = tmp_path / "users.json"
    session_db = tmp_path / "sessions.db"
    shell_single_quote_escape_fragment = "'\\''"
    key = f"idempotent{shell_single_quote_escape_fragment}key-with-trailing=="

    env = dict(os.environ)
    env.update(
        {
            "TP_LOCAL_ENV_FILE": str(env_file),
            "TP_FRONTDOOR_USERS_FILE": str(users_file),
            "TP_FRONTDOOR_SESSION_DB": str(session_db),
            "TP_LOCAL_API_KEY": key,
        }
    )

    subprocess.run(["bash", str(WRITE_LOCAL_ENV_SCRIPT)], check=True, env=env)

    env.pop("TP_LOCAL_API_KEY")
    result = subprocess.run(
        ["bash", str(WRITE_LOCAL_ENV_SCRIPT)],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "Reused existing key" in result.stdout
    assert _source_local_env_key_pair(env_file) == [key, key]


@pytest.mark.unit
def test_write_local_env_reuses_legacy_double_quoted_key(tmp_path: Path) -> None:
    env_file = tmp_path / "local.env"
    users_file = tmp_path / "users.json"
    session_db = tmp_path / "sessions.db"
    key = 'legacy"key\\with$dollar`tick=='
    env_file.write_text(
        'export TP_API_KEY="legacy\\"key\\\\with\\$dollar\\`tick=="\n',
        encoding="utf-8",
    )

    env = dict(os.environ)
    env.update(
        {
            "TP_LOCAL_ENV_FILE": str(env_file),
            "TP_FRONTDOOR_USERS_FILE": str(users_file),
            "TP_FRONTDOOR_SESSION_DB": str(session_db),
        }
    )

    result = subprocess.run(
        ["bash", str(WRITE_LOCAL_ENV_SCRIPT)],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "Reused existing key" in result.stdout
    assert _source_local_env_key_pair(env_file) == [key, key]


@pytest.mark.unit
def test_stop_local_stack_uses_portable_pid_joining() -> None:
    content = STOP_LOCAL_STACK_SCRIPT.read_text(encoding="utf-8")

    assert "xargs -r" not in content
    assert "join_pids()" in content


@pytest.mark.unit
def test_run_cloudflared_named_tunnel_preserves_exit_trap() -> None:
    content = RUN_CLOUDFLARED_SCRIPT.read_text(encoding="utf-8")

    assert "exec cloudflared tunnel" not in content
    assert 'cloudflared tunnel --protocol "${PROTOCOL}" run "${CLOUDFLARED_TUNNEL_NAME}"' in content
