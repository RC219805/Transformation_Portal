"""Execute Nightly's actual shell with controlled tools, never models or installs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github/workflows/nightly.yml"
ML_INSTALLER = REPO_ROOT / "scripts/ci/install_ml_test_dependencies.sh"
NIGHTLY_INSTALLER = REPO_ROOT / "scripts/ci/install_nightly_dependencies.sh"
WRITERS = (
    "test_single_image_cold_start_p95",
    "test_single_image_steady_state_p95",
    "test_batch_throughput_baseline",
    "test_memory_peak_rss_baseline",
)
ARTIFACTS = (
    "baseline_cold_start.json",
    "baseline_steady_state.json",
    "baseline_batch.json",
    "baseline_memory.json",
)
FULL_JOBS = {
    "stress-tests",
    "performance-benchmarks",
    "memory-leak-detection",
    "dependency-audit-deep",
    "integration-tests-full",
    "nightly-summary",
}


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _step(name: str) -> dict:
    return next(step for step in _workflow()["jobs"]["performance-benchmarks"]["steps"] if step.get("name") == name)


def _shell(script: str, env: dict[str, str], cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    assert bash is not None, "Nightly requires bash"
    return subprocess.run(
        [bash, "--noprofile", "--norc", "-c", script],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _benchmark_environment(tmp_path: Path, scenario: str) -> dict[str, str]:
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "python").symlink_to(sys.executable)
    stub = tools / "pytest"
    stub.write_text(
        f"#!{sys.executable}\n" + textwrap.dedent(f"""\
            import json
            import os
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            Path(os.environ["STUB_PYTEST_ARGS"]).write_text(json.dumps(args))
            scenario = os.environ["STUB_SCENARIO"]
            if scenario in {{"collection", "import"}}:
                print("collection error" if scenario == "collection" else "ModuleNotFoundError: controlled import failure")
                raise SystemExit(2)
            tests = [{{"nodeid": "controlled.py::" + name, "outcome": "passed"}} for name in {WRITERS!r}]
            if scenario == "empty":
                tests = []
            elif scenario == "all_skipped":
                for test in tests:
                    test["outcome"] = "skipped"
            elif scenario == "writer_regression":
                tests[0]["outcome"] = "failed"
            elif scenario == "other_suite_failure":
                tests.append({{"nodeid": "controlled.py::test_unrelated_benchmark", "outcome": "failed"}})
            report = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--json-report-file="))
            Path(report).write_text(json.dumps({{"tests": tests}}))
            if scenario not in {{"empty", "all_skipped"}}:
                for name in {ARTIFACTS!r}:
                    if scenario == "missing_artifact" and name == "baseline_memory.json":
                        continue
                    (Path(os.environ["BENCHMARK_ARTIFACTS_DIR"]) / name).write_text(json.dumps({{"controlled_test_fixture": True}}))
            raise SystemExit({{"empty": 5, "writer_regression": 1, "other_suite_failure": 1, "raw_failure": 4}}.get(scenario, 0))
            """),
        encoding="utf-8",
    )
    stub.chmod(0o755)
    output = tmp_path / "github-output"
    output.touch()
    env = dict(os.environ)
    env.pop("BASH_ENV", None)
    env.update(
        PATH=str(tools) + os.pathsep + os.environ.get("PATH", ""),
        BENCHMARK_EVIDENCE_DIR=str(tmp_path / "fresh-evidence"),
        GITHUB_OUTPUT=str(output),
        STUB_PYTEST_ARGS=str(tmp_path / "pytest-args.json"),
        STUB_SCENARIO=scenario,
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def _outputs(env: dict[str, str]) -> dict[str, str]:
    return dict(line.split("=", 1) for line in Path(env["GITHUB_OUTPUT"]).read_text(encoding="utf-8").splitlines())


def _enforce(env: dict[str, str], outputs: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return _shell(
        _step("Enforce nightly benchmark evidence")["run"],
        {
            **env,
            "BENCHMARK_STATUS": outputs.get("status", ""),
            "PYTEST_EXIT": outputs.get("pytest_exit", ""),
            "EVIDENCE_EXIT": outputs.get("evidence_exit", ""),
        },
    )


@pytest.mark.parametrize(
    ("scenario", "status", "pytest_exit", "evidence_exit", "final_exit"),
    [
        ("passed", "passed", "0", "0", 0),
        ("empty", "failed", "5", "1", 1),
        ("all_skipped", "failed", "0", "1", 1),
        ("collection", "failed", "2", "1", 1),
        ("import", "failed", "2", "1", 1),
        ("writer_regression", "regression", "1", "2", 1),
        ("other_suite_failure", "failed", "1", "3", 1),
        ("raw_failure", "failed", "4", "0", 1),
        ("missing_artifact", "failed", "0", "1", 1),
    ],
)
def test_actual_benchmark_shell_classifies_and_enforces(
    tmp_path: Path, scenario: str, status: str, pytest_exit: str, evidence_exit: str, final_exit: int
) -> None:
    env = _benchmark_environment(tmp_path, scenario)
    result = _shell(_step("Run performance benchmarks")["run"], env)
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = _outputs(env)
    assert outputs == {"status": status, "pytest_exit": pytest_exit, "evidence_exit": evidence_exit}
    assert _enforce(env, outputs).returncode == final_exit
    args = json.loads(Path(env["STUB_PYTEST_ARGS"]).read_text(encoding="utf-8"))
    assert args[:4] == ["tests/", "-m", "benchmark", "--json-report"]
    assert "--benchmark-only" not in args and "--collect-only" not in args
    assert not any(arg.startswith("--benchmark-compare") for arg in args)
    evidence = Path(env["BENCHMARK_EVIDENCE_DIR"])
    assert (evidence / "pytest.log").is_file()
    assert (evidence / "evidence-gate.log").is_file()


def test_actual_shell_refuses_existing_evidence_namespace(tmp_path: Path) -> None:
    env = _benchmark_environment(tmp_path, "passed")
    evidence = Path(env["BENCHMARK_EVIDENCE_DIR"])
    evidence.mkdir()
    sentinel = evidence / "retained.json"
    sentinel.write_text('{"retained": true}', encoding="utf-8")
    result = _shell(_step("Run performance benchmarks")["run"], env)
    assert result.returncode != 0
    assert not Path(env["STUB_PYTEST_ARGS"]).exists()
    assert not _outputs(env)
    assert _enforce(env, {}).returncode == 1
    assert sentinel.read_text(encoding="utf-8") == '{"retained": true}'


@pytest.mark.parametrize(
    "outputs",
    [{}, {"status": "passed"}, {"status": "passed", "pytest_exit": "1", "evidence_exit": "0"}],
)
def test_final_shell_fails_closed_on_missing_or_inconsistent_outputs(tmp_path: Path, outputs: dict[str, str]) -> None:
    assert _enforce(_benchmark_environment(tmp_path, "passed"), outputs).returncode == 1


def test_workflow_keeps_default_full_mode_and_isolates_manual_install_only() -> None:
    workflow = _workflow()
    trigger = workflow.get("on", workflow.get(True))  # YAML 1.1 treats unquoted "on" as a boolean.
    assert trigger["schedule"] == [{"cron": "0 2 * * *"}]
    assert set(trigger) == {"schedule", "workflow_dispatch"}
    option = trigger["workflow_dispatch"]["inputs"]["install_only"]
    assert option["type"] == "boolean" and option["default"] is False and option["required"] is False
    jobs = workflow["jobs"]
    assert set(jobs) == FULL_JOBS | {"install-only"}
    for name in FULL_JOBS - {"nightly-summary"}:
        assert jobs[name]["if"] == "${{ github.event_name != 'workflow_dispatch' || !inputs.install_only }}"
    assert (
        jobs["nightly-summary"]["if"]
        == "${{ always() && (github.event_name != 'workflow_dispatch' || !inputs.install_only) }}"
    )
    assert set(jobs["nightly-summary"]["needs"]) == FULL_JOBS - {"nightly-summary"}
    qualification = jobs["install-only"]
    assert qualification["if"] == "${{ github.event_name == 'workflow_dispatch' && inputs.install_only }}"
    assert qualification["timeout-minutes"] == 30 and qualification["permissions"] == {"contents": "read"}
    assert qualification["env"] == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    install = "bash scripts/ci/install_nightly_dependencies.sh\n"
    for name in {"stress-tests", "performance-benchmarks", "memory-leak-detection", "integration-tests-full", "install-only"}:
        assert next(step for step in jobs[name]["steps"] if step.get("name") == "Install dependencies")["run"] == install
    assert [step["run"] for step in qualification["steps"] if "run" in step] == [install]


def test_benchmark_upload_and_final_gate_always_run_in_order() -> None:
    job = _workflow()["jobs"]["performance-benchmarks"]
    evidence_dir = _step("Run performance benchmarks")["env"]["BENCHMARK_EVIDENCE_DIR"]
    assert "github.run_id" in evidence_dir
    assert "github.run_attempt" in evidence_dir
    upload = _step("Upload benchmark results")
    final = _step("Enforce nightly benchmark evidence")
    assert upload["if"] == final["if"] == "always()"
    assert upload["with"]["path"] == evidence_dir + "/"
    assert job["steps"].index(upload) < job["steps"].index(final) == len(job["steps"]) - 1
    assert all(not step.get("continue-on-error", False) for step in job["steps"])


def _audit_step() -> dict:
    return next(step for step in _workflow()["jobs"]["dependency-audit-deep"]["steps"] if step.get("id") == "audit")


def _audit_environment(
    tmp_path: Path, report: str | None, raw_exit: int, python_optimize: str, *, oversized: bool = False
) -> dict[str, str]:
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "python").symlink_to(sys.executable)
    stub = tools / "pip-audit"
    stub.write_text(
        f"#!{sys.executable}\n" + textwrap.dedent("""\
            import json
            import os
            import sys
            from pathlib import Path

            Path(os.environ["AUDIT_CALLS"]).write_text(json.dumps(sys.argv[1:]))
            Path(os.environ["AUDIT_OPTIMIZATION"]).write_text(str(sys.flags.optimize))
            if os.environ["AUDIT_WRITE_REPORT"] == "1":
                content = b"x" * (4 * 1024 * 1024 + 1) if os.environ["AUDIT_OVERSIZED"] == "1" else os.environ["AUDIT_REPORT"].encode()
                Path("audit-report.json").write_bytes(content)
            raise SystemExit(int(os.environ["AUDIT_RAW_EXIT"]))
            """),
        encoding="utf-8",
    )
    stub.chmod(0o755)
    output = tmp_path / "github-output"
    output.touch()
    env = dict(os.environ)
    env.pop("BASH_ENV", None)
    env.update(
        PATH=str(tools) + os.pathsep + os.environ.get("PATH", ""),
        GITHUB_OUTPUT=str(output),
        GITHUB_STEP_SUMMARY=str(tmp_path / "github-summary"),
        AUDIT_CALLS=str(tmp_path / "audit-calls.json"),
        AUDIT_OPTIMIZATION=str(tmp_path / "audit-optimization"),
        AUDIT_WRITE_REPORT="0" if report is None else "1",
        AUDIT_REPORT=report or "",
        AUDIT_RAW_EXIT=str(raw_exit),
        AUDIT_OVERSIZED="1" if oversized else "0",
        PYTHONOPTIMIZE=python_optimize,
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def _audit_report(vulnerabilities: list[dict] | None = None) -> str:
    return json.dumps({"dependencies": [{"name": "controlled", "version": "1.0", "vulns": vulnerabilities or []}]})


@pytest.mark.parametrize("python_optimize", ["0", "1"])
@pytest.mark.parametrize("findings", [False, True])
def test_actual_audit_shell_keeps_complete_findings_advisory(tmp_path: Path, python_optimize: str, findings: bool) -> None:
    report = _audit_report([{"id": "CONTROLLED-ADVISORY"}] if findings else [])
    env = _audit_environment(tmp_path, report, int(findings), python_optimize)
    result = _shell(_audit_step()["run"], env, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _outputs(env) == {
        "status": "advisory_findings" if findings else "clean",
        "audit_exit": str(int(findings)),
        "advisory_count": str(int(findings)),
    }
    assert json.loads(Path(env["AUDIT_CALLS"]).read_text()) == [
        "--desc",
        "-r",
        "requirements.txt",
        "--format",
        "json",
        "--output",
        "audit-report.json",
    ]
    assert Path(env["AUDIT_OPTIMIZATION"]).read_text() == python_optimize
    assert (tmp_path / "audit-report.json").read_text() == report
    summary = Path(env["GITHUB_STEP_SUMMARY"]).read_text()
    assert "Security Unified remains the blocking installed-environment authority." in summary
    assert f"Raw scanner exit: {int(findings)}" in summary
    assert ("::warning::" in result.stdout) is findings


@pytest.mark.parametrize("python_optimize", ["0", "1"])
@pytest.mark.parametrize(
    ("report", "raw_exit"),
    [
        pytest.param(None, 0, id="missing-success-report"),
        pytest.param(None, 1, id="fatal-exit-one"),
        pytest.param(None, 2, id="usage-error"),
        pytest.param("", 0, id="empty-report"),
        pytest.param("{", 0, id="malformed-json"),
        pytest.param("[]", 0, id="wrong-top-level"),
        pytest.param("{}", 0, id="missing-inventory"),
        pytest.param('{"dependencies":[]}', 0, id="empty-inventory"),
        pytest.param('{"dependencies":[null]}', 0, id="malformed-dependency"),
        pytest.param('{"dependencies":[{"name":"controlled","skip_reason":"unresolved"}]}', 0, id="skipped-dependency"),
        pytest.param('{"dependencies":[{"name":"","version":"1","vulns":[]}]}', 0, id="empty-name"),
        pytest.param('{"dependencies":[{"name":"controlled","version":null,"vulns":[]}]}', 0, id="missing-version"),
        pytest.param('{"dependencies":[{"name":"controlled","version":"1"}]}', 0, id="missing-vulnerabilities"),
        pytest.param(
            '{"dependencies":[{"name":"controlled","version":"1","vulns":{}}]}', 0, id="malformed-vulnerability-list"
        ),
        pytest.param(_audit_report([None]), 1, id="malformed-vulnerability"),
        pytest.param(_audit_report([{"id": " "}]), 1, id="empty-vulnerability-id"),
        pytest.param(_audit_report(), 1, id="clean-report-fatal-exit"),
        pytest.param(_audit_report([{"id": "CONTROLLED-ADVISORY"}]), 0, id="findings-success-exit"),
        pytest.param(_audit_report([{"id": "CONTROLLED-ADVISORY"}]), 2, id="findings-tool-error-exit"),
        pytest.param(_audit_report(), 2, id="clean-report-tool-error-exit"),
        pytest.param('{"dependencies":[],"dependencies":[]}', 0, id="duplicate-field"),
        pytest.param(_audit_report()[:-1] + ',"unused":NaN}', 0, id="non-finite-value"),
        pytest.param(
            '{"dependencies":[{"name":"controlled","version":"1","vulns":[]},{"skip_reason":"unresolved"}]}',
            0,
            id="mixed-incomplete-report",
        ),
    ],
)
def test_actual_audit_shell_rejects_invalid_or_inconsistent_evidence(
    tmp_path: Path, report: str | None, raw_exit: int, python_optimize: str
) -> None:
    env = _audit_environment(tmp_path, report, raw_exit, python_optimize)
    result = _shell(_audit_step()["run"], env, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert _outputs(env) == {"status": "failed", "audit_exit": str(raw_exit), "advisory_count": "not_checked"}
    assert Path(env["AUDIT_OPTIMIZATION"]).read_text() == python_optimize
    assert "::error::Invalid advisory audit evidence:" in result.stdout
    if report is not None:
        assert (tmp_path / "audit-report.json").read_text() == report


def test_actual_audit_shell_bounds_report_size_without_removing_evidence(tmp_path: Path) -> None:
    env = _audit_environment(tmp_path, "oversized", 0, "1", oversized=True)
    result = _shell(_audit_step()["run"], env, tmp_path)
    assert result.returncode == 1
    assert "audit report exceeds 4 MiB" in result.stdout
    assert (tmp_path / "audit-report.json").stat().st_size == 4 * 1024 * 1024 + 1


@pytest.mark.parametrize("existing", ["file", "symlink", "dangling-symlink", "directory"])
def test_actual_audit_shell_refuses_stale_report_without_mutation(tmp_path: Path, existing: str) -> None:
    env = _audit_environment(tmp_path, _audit_report(), 0, "1")
    report = tmp_path / "audit-report.json"
    retained = tmp_path / "retained.json"
    if existing == "file":
        report.write_text("retained report", encoding="utf-8")
    elif existing == "directory":
        report.mkdir()
        (report / "retained").write_text("retained report", encoding="utf-8")
    else:
        if existing == "symlink":
            retained.write_text("retained report", encoding="utf-8")
        report.symlink_to(retained)
    result = _shell(_audit_step()["run"], env, tmp_path)
    assert result.returncode == 1
    assert not Path(env["AUDIT_CALLS"]).exists()
    assert not _outputs(env)
    assert "scanner not run" in Path(env["GITHUB_STEP_SUMMARY"]).read_text()
    if existing == "file":
        assert report.read_text() == "retained report"
    elif existing == "directory":
        assert (report / "retained").read_text() == "retained report"
    else:
        assert report.is_symlink() and report.readlink() == retained
        assert retained.exists() is (existing == "symlink")
        if retained.exists():
            assert retained.read_text() == "retained report"


def test_audit_evidence_is_always_retained_and_all_failed_suites_notify() -> None:
    jobs = _workflow()["jobs"]
    upload = next(
        step for step in jobs["dependency-audit-deep"]["steps"] if step.get("name") == "Upload SBOM and audit reports"
    )
    assert upload["if"] == "always()"
    assert upload["with"]["path"].splitlines() == ["sbom.json", "audit-report.json"]
    summary = jobs["nightly-summary"]
    notify = next(step for step in summary["steps"] if step.get("name") == "Create issue on failure (deduplicated)")
    expected = {f"needs.{name}.result == 'failure'" for name in FULL_JOBS - {"nightly-summary"}}
    assert {clause.strip() for clause in notify["if"].strip().split("||")} == expected


def _installer_environment(tmp_path: Path, **updates: str) -> dict[str, str]:
    interpreter = tmp_path / "controlled-python"
    interpreter.write_text(
        f"#!{sys.executable}\n" + textwrap.dedent("""\
            import json
            import os
            import platform
            import sys
            import types

            with open(os.environ["INSTALLER_CALLS"], "a") as stream:
                stream.write(json.dumps(sys.argv[1:]) + "\\n")
            with open(os.environ["INSTALLER_OPTIMIZATIONS"], "a") as stream:
                stream.write(json.dumps(sys.flags.optimize) + "\\n")
            if sys.argv[1:4] == ["-m", "pip", "check"] and os.environ.get("FAIL_PIP_CHECK"):
                raise SystemExit(1)
            if sys.argv[1:2] == ["-c"]:
                platform.machine = lambda: "x86_64"
                sys.platform = os.environ.get("CONTROLLED_PLATFORM", "linux")
                sys.modules["torch"] = types.SimpleNamespace(__version__=os.environ.get("CONTROLLED_TORCH", "2.13.0+cpu"), version=types.SimpleNamespace(cuda=os.environ.get("CONTROLLED_CUDA") or None))
                sys.modules["torchvision"] = types.SimpleNamespace(__version__=os.environ.get("CONTROLLED_TORCHVISION", "0.28.0+cpu"))
                sys.modules["transformers"] = types.SimpleNamespace(__version__="5.10.1")
                for name in ("memory_profiler", "psutil", "pytest_benchmark", "pytest_jsonreport", "pytest_memray", "pytest_timeout"):
                    sys.modules[name] = types.ModuleType(name)
                exec(sys.argv[2])
            """),
        encoding="utf-8",
    )
    interpreter.chmod(0o755)
    env = dict(os.environ)
    for name in ("BASH_ENV", "TP_CI_ML_LOCKFILE", "PYTHON_BIN", "CI_REQUIREMENTS_FILE", "TP_CI_ML_RAW_REQUIREMENTS_FILE"):
        env.pop(name, None)
    env.update(
        PYTHON_BIN=str(interpreter),
        INSTALLER_CALLS=str(tmp_path / "calls.jsonl"),
        INSTALLER_OPTIMIZATIONS=str(tmp_path / "optimizations.jsonl"),
        **updates,
    )
    return env


def _install(script: Path, env: dict[str, str], *args: str) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    result = subprocess.run(
        ["bash", str(script), *args], cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False, timeout=30
    )
    log = Path(env["INSTALLER_CALLS"])
    return result, [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []


@pytest.mark.parametrize("python_optimize", ["0", "1"])
def test_cpu_installer_uses_exact_official_pair_then_checks_after_metadata_extras(
    tmp_path: Path, python_optimize: str
) -> None:
    env = _installer_environment(tmp_path, PYTHONOPTIMIZE=python_optimize)
    result, calls = _install(ML_INSTALLER, env, "--cpu", "--skip-ci-requirements", "--include-rawpy")
    assert result.returncode == 0, result.stdout + result.stderr
    pair = [
        "-m",
        "pip",
        "install",
        "torch==2.13.0+cpu",
        "torchvision==0.28.0+cpu",
        "--index-url",
        "https://download.pytorch.org/whl/cpu",
    ]
    assert pair in calls
    assert ["-m", "pip", "install", "-r", "requirements-ci.txt"] not in calls
    assert ["-m", "pip", "install", "-r", "requirements/ml-raw.in"] in calls
    assert calls.index(pair) < calls.index(["-m", "pip", "install", "-e", ".[ml-core]"])
    assert calls[-3] == ["-m", "pip", "check"]
    assert "Verified governed CPU torch/torchvision pair" in result.stdout
    assert {json.loads(line) for line in Path(env["INSTALLER_OPTIMIZATIONS"]).read_text().splitlines()} == {
        int(python_optimize)
    }


def test_default_installer_keeps_existing_non_cpu_baseline(tmp_path: Path) -> None:
    result, calls = _install(ML_INSTALLER, _installer_environment(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert ["-m", "pip", "install", "torch==2.13.0", "torchvision==0.28.0"] in calls
    assert not any("--index-url" in call or any("+cpu" in value for value in call) for call in calls)
    assert ["-m", "pip", "install", "-r", "requirements-ci.txt"] in calls


def test_non_cpu_native_lock_path_preserves_existing_owned_lock_semantics(tmp_path: Path) -> None:
    lock = tmp_path / "controlled.lock"
    lock.write_text("# controlled installer fixture, never installed\n", encoding="utf-8")
    result, calls = _install(ML_INSTALLER, _installer_environment(tmp_path, TP_CI_ML_LOCKFILE=str(lock)))
    assert result.returncode == 0, result.stdout + result.stderr
    assert ["-m", "pip", "install", "-r", str(lock)] in calls
    assert ["-m", "pip", "install", "-e", ".", "--no-deps"] in calls
    assert ["-m", "pip", "check"] in calls
    assert not any(
        "--index-url" in call or ".[ml-core]" in call or any("torch==" in value for value in call) for call in calls
    )


@pytest.mark.parametrize("args", [("--cpu",), ("--cpu", "--skip-ci-requirements")])
def test_cpu_and_native_lock_reject_before_any_interpreter_or_pip_call(tmp_path: Path, args: tuple[str, ...]) -> None:
    result, calls = _install(
        ML_INSTALLER, _installer_environment(tmp_path, TP_CI_ML_LOCKFILE="never-install-this.lock"), *args
    )
    assert result.returncode == 2 and calls == []
    assert "--cpu cannot be combined with TP_CI_ML_LOCKFILE" in result.stderr


@pytest.mark.parametrize(
    "updates",
    [
        {"CONTROLLED_PLATFORM": "darwin"},
        {"CONTROLLED_TORCH": "2.13.0"},
        {"CONTROLLED_TORCHVISION": "0.28.0"},
        {"CONTROLLED_CUDA": "12.6"},
        {"FAIL_PIP_CHECK": "1"},
    ],
)
@pytest.mark.parametrize("python_optimize", ["0", "1"])
def test_cpu_platform_resolved_pair_and_dependency_failures_propagate(
    tmp_path: Path, updates: dict[str, str], python_optimize: str
) -> None:
    env = _installer_environment(tmp_path, PYTHONOPTIMIZE=python_optimize, **updates)
    result, calls = _install(ML_INSTALLER, env, "--cpu")
    assert result.returncode != 0
    assert {json.loads(line) for line in Path(env["INSTALLER_OPTIMIZATIONS"]).read_text().splitlines()} == {
        int(python_optimize)
    }
    if "CONTROLLED_PLATFORM" in updates:
        assert not any(call[:2] == ["-m", "pip"] for call in calls)


def test_shared_nightly_installation_uses_owned_cpu_path_and_existing_tooling(tmp_path: Path) -> None:
    result, calls = _install(NIGHTLY_INSTALLER, _installer_environment(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert [
        "-m",
        "pip",
        "install",
        "torch==2.13.0+cpu",
        "torchvision==0.28.0+cpu",
        "--index-url",
        "https://download.pytorch.org/whl/cpu",
    ] in calls
    assert [
        "-m",
        "pip",
        "install",
        "memory-profiler",
        "psutil",
        "pytest-timeout",
        "pytest-benchmark",
        "pytest-memray",
    ] in calls
    assert calls[-2] == ["-m", "pip", "check"]
    assert calls[-1][0] == "-c" and "Nightly test tooling imports passed" in calls[-1][1]


def test_shared_nightly_installer_rejects_arguments_without_installing(tmp_path: Path) -> None:
    result, calls = _install(NIGHTLY_INSTALLER, _installer_environment(tmp_path), "--unexpected")
    assert result.returncode == 2 and calls == []
