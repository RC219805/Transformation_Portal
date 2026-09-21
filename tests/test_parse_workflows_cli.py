"""Regression tests for the governed workflow parser's actual public CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/parse_workflows.py"


def run_parser(
    tmp_path: Path,
    script: str,
    *,
    shell=None,
    job_defaults=None,
    defaults=None,
    runner="ubuntu-latest",
    container=None,
    **kwargs,
):
    """Create one workflow and invoke the implementation used by the guide."""
    step = {"run": script}
    if shell is not None:
        step["shell"] = shell
    job = {"runs-on": runner, "steps": [step]}
    if container is not None:
        job["container"] = container
    if job_defaults is not None:
        job["defaults"] = {"run": {"shell": job_defaults}}
    workflow = {"name": "Shell regression", "on": ["push"], "jobs": {"check": job}}
    if defaults is not None:
        workflow["defaults"] = {"run": {"shell": defaults}}
    (tmp_path / "workflow.yml").write_text(yaml.safe_dump(workflow), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--workflow-dir", str(tmp_path), "--format", "json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
        **kwargs,
    )


@pytest.mark.parametrize(
    "script",
    [
        "python - <<'PY'\nif True:\n    print('if in Python')\nPY\n",
        "python -c \"\nif True:\n    print('quoted Python')\n\"\n",
        "printf '%s\\n' 'if this is text' 'fi is also text'\n",
        "if true; then\n  echo 'if quoted';\nelif false; then\n  :\nelse\n  :\nfi\n",
        "cat <<'EOF'\nif arbitrary non-shell text\nEOF\n",
        "if true; then\n  python - <<'PY'\nif True:\n    print('inside shell conditional')\nPY\nfi\n",
    ],
)
def test_shell_parser_preserves_quotes_and_heredocs(tmp_path, script):
    result = run_parser(tmp_path, script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Shell syntax error" not in result.stdout


@pytest.mark.parametrize(
    "shell", [None, "bash", "sh", "/usr/bin/env -u BASH_ENV -u ENV /bin/bash --noprofile --norc -p -e -o pipefail {0}"]
)
def test_missing_fi_is_an_error_from_actual_cli(tmp_path, shell):
    result = run_parser(tmp_path, "if true; then\n  echo missing\n", shell=shell)
    assert result.returncode == 1, result.stdout + result.stderr
    # Preserve the public JSON diagnostic envelope, including shell context.
    json_start = result.stdout.index("[\n")
    diagnostics = json.loads(result.stdout[json_start:])
    assert len(diagnostics) == 1
    assert diagnostics[0]["severity"] == "error"
    assert "job 'check', step 1" in diagnostics[0]["message"]
    assert "syntax" in diagnostics[0]["context"].lower()


def test_quoted_fi_does_not_close_a_shell_conditional(tmp_path):
    result = run_parser(tmp_path, "if true; then\n  echo 'fi'\n")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Shell syntax error" in result.stdout


def test_container_default_is_sh_even_with_dynamic_runner(tmp_path):
    result = run_parser(tmp_path, "if true; then\n  :\n", runner="${{ matrix.os }}", container="python:3.12")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "(sh)" in result.stdout


@pytest.mark.parametrize(
    "settings,script,expected",
    [
        ({"shell": "python"}, "if True:\n    print('python')\n", 0),
        ({"defaults": "python"}, "if True:\n    print('python')\n", 0),
        ({"defaults": "bash", "job_defaults": "python"}, "if True:\n    print('python')\n", 0),
        ({"defaults": "python", "job_defaults": "bash"}, "if true; then\n  :\n", 1),
        ({"job_defaults": "python", "shell": "sh"}, "if true; then\n  :\n", 1),
        ({"runner": "windows-latest"}, "if ($true) { Write-Output 'pwsh' }\n", 0),
        ({"shell": "pwsh"}, "if ($true) { Write-Output 'pwsh' }\n", 0),
        ({"shell": "${{ matrix.shell }}"}, "if True:\n    print('dynamic')\n", 0),
        ({"runner": "${{ matrix.os }}"}, "if True:\n    print('unknown runner')\n", 0),
        ({"container": "${{ matrix.container }}"}, "if ($true) { Write-Output 'unknown' }\n", 0),
        ({"container": "${{ matrix.container }}", "defaults": "bash"}, "if true; then\n  :\n", 1),
    ],
)
def test_shell_precedence_and_non_shell_interpreters(tmp_path, settings, script, expected):
    result = run_parser(tmp_path, script, **settings)
    assert result.returncode == expected, result.stdout + result.stderr
    if expected == 0:
        assert "Shell syntax check skipped" in result.stdout


def test_check_never_executes_script_or_ambient_bash_startup(tmp_path):
    sentinel = tmp_path / "executed"
    startup = tmp_path / "startup.sh"
    startup.write_text(f"touch {sentinel}\n", encoding="utf-8")
    environment = dict(os.environ, BASH_ENV=str(startup), ENV=str(startup))
    script = f"touch {sentinel}\necho $(touch {sentinel})\n"
    result = run_parser(tmp_path, script, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not sentinel.exists()


def test_custom_shell_command_is_not_executed(tmp_path):
    sentinel = tmp_path / "executed"
    result = run_parser(tmp_path, "if True:\n    pass\n", shell=f"bash -c 'touch {sentinel}' {{0}}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Shell syntax check skipped" in result.stdout
    assert not sentinel.exists()


def test_static_model_hint_does_not_assert_api_validity(tmp_path):
    result = run_parser(tmp_path, "printf '%s' '{\"model\": \"gpt-4.1-mini\"}'\n")
    assert result.returncode == 0
    assert "legacy static hints" in result.stdout
    assert "does not establish API validity or account availability" in result.stdout
    assert "invalid OpenAI model" not in result.stdout


def test_documented_default_command_passes_current_workflows():
    guide = (ROOT / "docs/guides/parse_workflows_README.md").read_text(encoding="utf-8")
    assert ".venv/bin/python scripts/validation/parse_workflows.py" in guide
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, text=True, capture_output=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
