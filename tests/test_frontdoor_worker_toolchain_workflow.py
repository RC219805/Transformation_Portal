"""Execute blocking Frontdoor workflow shell steps without installing packages."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit, pytest.mark.regression]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OWNER_PATHS = ("", "cloudflare/transformationportal-worker", "web/secure-landing")
STEP_NAMES = (
    "Activate governed npm owner",
    "Install governed Worker dependencies",
    "Run Worker request security contracts",
    "Verify Worker toolchain and root shim",
    "Install frontdoor dependencies",
    "Build frontdoor production bundle",
    "Check built portal asset budgets",
)


def _job() -> dict:
    workflow = yaml.safe_load((PROJECT_ROOT / ".github/workflows/build.yml").read_text(encoding="utf-8"))
    return workflow["jobs"]["frontdoor-contract"]


def _run_owner_steps(tmp_path: Path, *, npm_version: str = "11.12.1", mismatched_owner: str | None = None, fail: str = ""):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the workflow manifest guard")
    for path in OWNER_PATHS:
        directory = tmp_path / path
        directory.mkdir(parents=True, exist_ok=True)
        owner = "npm@11.12.2" if path == mismatched_owner else "npm@11.12.1"
        (directory / "package.json").write_text(json.dumps({"packageManager": owner}), encoding="utf-8")
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    log = tmp_path / "commands.jsonl"
    for name in ("npm", "python"):
        executable = binary_dir / name
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "command = pathlib.Path(sys.argv[0]).name + ' ' + ' '.join(sys.argv[1:])\n"
            "with open(os.environ['COMMAND_LOG'], 'a', encoding='utf-8') as stream:\n"
            "    stream.write(json.dumps({'cwd':os.getcwd(), 'command':command}) + '\\n')\n"
            "if command == 'npm --version': print(os.environ['FAKE_NPM_VERSION'])\n"
            "sys.exit(17 if command == os.environ['FAIL_COMMAND'] else 0)\n",
            encoding="utf-8",
        )
        executable.chmod(0o700)
    environment = {
        **os.environ,
        "PATH": str(binary_dir) + os.pathsep + str(Path(node).parent) + os.pathsep + os.defpath,
        "COMMAND_LOG": str(log),
        "FAKE_NPM_VERSION": npm_version,
        "FAIL_COMMAND": fail,
    }
    selected = [step for step in _job()["steps"] if step.get("name") in STEP_NAMES]
    assert [step["name"] for step in selected] == list(STEP_NAMES)
    for step in selected:
        result = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-c", step["run"]],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if result.returncode:
            break
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    return result, events


def test_frontdoor_job_caches_all_owner_locks_and_retains_bounded_blocking_checks() -> None:
    job = _job()
    assert job["needs"] == ["preflight"]
    assert job["if"] == "needs.preflight.outputs.run_frontdoor == 'true'"
    assert job["timeout-minutes"] == 15
    assert not job.get("continue-on-error", False)
    setup = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/setup-node@"))
    assert setup["with"]["node-version"] == "22.23.5"
    assert setup["with"]["cache"] == "npm"
    assert setup["with"]["cache-dependency-path"].splitlines() == [
        str(Path(path) / "package-lock.json") for path in OWNER_PATHS
    ]
    owner_index = next(index for index, step in enumerate(job["steps"]) if step.get("name") == STEP_NAMES[0])
    assert job["steps"].index(setup) < owner_index
    parity_index = next(index for index, step in enumerate(job["steps"]) if step.get("name") == STEP_NAMES[3])
    assert any(
        step.get("uses", "").startswith("actions/setup-python@") and step.get("with", {}).get("python-version") == "3.11"
        for step in job["steps"][:parity_index]
    )
    for step in job["steps"]:
        if step.get("name") in STEP_NAMES:
            assert "if" not in step
            assert not step.get("continue-on-error", False)


def test_job_installs_owned_toolchains_and_checks_budgets_after_the_frontdoor_build(tmp_path: Path) -> None:
    result, events = _run_owner_steps(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [event["command"] for event in events] == [
        "npm install --global npm@11.12.1",
        "npm --version",
        "npm ci",
        "npm --prefix cloudflare/transformationportal-worker ci",
        "npm --prefix cloudflare/transformationportal-worker test",
        "python -I scripts/validation/check_worker_dependency_parity.py",
        "npm --prefix cloudflare/transformationportal-worker run typecheck",
        "npm --prefix cloudflare/transformationportal-worker run dry-run",
        "npm run worker:dry-run",
        "npm ci",
        "npm run build",
        "python -I scripts/validation/check_portal_asset_budgets.py",
    ]
    workdirs = [Path(event["cwd"]) for event in events]
    assert workdirs == [tmp_path] * 9 + [tmp_path / "web/secure-landing"] * 2 + [tmp_path]


@pytest.mark.parametrize(
    "npm_version,mismatched_owner",
    [("10.9.9", None), ("11.12.1", ""), ("11.12.1", OWNER_PATHS[1]), ("11.12.1", OWNER_PATHS[2])],
)
def test_owner_version_or_manifest_mismatch_prevents_installs(
    tmp_path: Path, npm_version: str, mismatched_owner: str | None
) -> None:
    result, events = _run_owner_steps(tmp_path, npm_version=npm_version, mismatched_owner=mismatched_owner)
    assert result.returncode != 0
    assert [event["command"] for event in events] == ["npm install --global npm@11.12.1", "npm --version"]


@pytest.mark.parametrize(
    "failure",
    [
        "npm ci",
        "npm --prefix cloudflare/transformationportal-worker ci",
        "python -I scripts/validation/check_worker_dependency_parity.py",
        "npm --prefix cloudflare/transformationportal-worker run typecheck",
        "npm --prefix cloudflare/transformationportal-worker run dry-run",
        "npm run worker:dry-run",
        "npm run build",
        "python -I scripts/validation/check_portal_asset_budgets.py",
    ],
)
def test_failed_owner_or_bundle_check_stops_the_job(tmp_path: Path, failure: str) -> None:
    result, events = _run_owner_steps(tmp_path, fail=failure)
    assert result.returncode == 17
    assert events[-1]["command"] == failure
