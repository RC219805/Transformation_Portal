"""Keep named Enforcement gates while avoiding duplicate full-suite execution."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from itertools import product
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "enforcement.yml"
PROJECT_ROOT = WORKFLOW.parents[2]
LAYER1_MARKERS = "not ml and not slow and not benchmark"
GOLDEN_MARKERS = "golden and not ml and not slow and not benchmark"
PIP_CACHE_INPUTS = (
    "requirements-dev.txt",
    "requirements-ci.txt",
    "requirements.txt",
    "requirements/base.txt",
    "requirements/constraints.txt",
    "pyproject.toml",
    ".github/workflows/enforcement.yml",
)


def _jobs() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def _test_command(job_id: str) -> list[str]:
    steps = [step for step in _jobs()[job_id]["steps"] if step.get("run", "").startswith("pytest ")]
    assert len(steps) == 1
    assert "if" not in steps[0] and not steps[0].get("continue-on-error", False)
    assert "\n" not in steps[0]["run"].strip()
    return shlex.split(steps[0]["run"])


def test_enforcement_named_gates_and_admission_remain_intact() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML also accepts the YAML 1.1 boolean spelling of the Actions `on` key.
    triggers = workflow.get("on", workflow.get(True))
    assert triggers == {
        "push": {"branches": ["main", "develop"]},
        "pull_request": {"branches": ["main", "develop"]},
        "schedule": [{"cron": "15 4 * * *"}],
    }
    assert workflow["permissions"] == {"contents": "read"}
    jobs = _jobs()
    layer1, golden = jobs["test-layer1"], jobs["golden-regression"]
    assert layer1["name"] == "Layer 1 Tests (Fast)"
    assert golden["name"] == "Golden Regression Tests"
    assert "needs" not in layer1 and "if" not in layer1
    assert golden["needs"] == ["changes"]
    assert golden["if"] == "needs.changes.outputs.core_changed == 'true' || github.event_name == 'schedule'"
    for job in (layer1, golden):
        assert job["runs-on"] == "ubuntu-latest"
        assert not job.get("continue-on-error", False)
        assert "env" not in job and "permissions" not in job
    assert _test_command("test-layer1") == ["pytest", "tests/", "-m", LAYER1_MARKERS, "--maxfail=3"]
    assert _test_command("golden-regression") == ["pytest", "tests/", "-m", GOLDEN_MARKERS, "--maxfail=1"]


def test_enforcement_golden_uses_the_same_dependency_and_checkout_policy() -> None:
    jobs = _jobs()
    steps = {key: jobs[key]["steps"] for key in ("test-layer1", "golden-regression")}
    for action in ("actions/checkout@", "actions/setup-python@"):
        layer1 = [step for step in steps["test-layer1"] if step.get("uses", "").startswith(action)]
        golden = [step for step in steps["golden-regression"] if step.get("uses", "").startswith(action)]
        assert len(layer1) == len(golden) == 1
        assert layer1 == golden
        assert "if" not in layer1[0] and not layer1[0].get("continue-on-error", False)
        if action == "actions/checkout@":
            assert layer1[0]["with"] == {"fetch-depth": 1}
        else:
            assert set(layer1[0]["with"]) == {"python-version", "cache", "cache-dependency-path"}
            assert layer1[0]["with"]["python-version"] == "3.11"
            assert layer1[0]["with"]["cache"] == "pip"
            assert layer1[0]["with"]["cache-dependency-path"].splitlines() == list(PIP_CACHE_INPUTS)
    installs = {key: [step for step in values if step.get("name", "").startswith("Install")] for key, values in steps.items()}
    assert all(len(values) == 1 for values in installs.values())
    assert installs["test-layer1"][0]["run"] == installs["golden-regression"][0]["run"]
    for values in installs.values():
        assert "if" not in values[0] and not values[0].get("continue-on-error", False)
        assert "python -m pip check" in values[0]["run"]
        assert "--build-constraint requirements/constraints.txt" in values[0]["run"]


def _nested_requirement_inputs(path: Path, seen: set[Path] | None = None) -> set[str]:
    seen = set() if seen is None else seen
    path = path.resolve()
    assert path.is_relative_to(PROJECT_ROOT)
    if path in seen:
        return set()
    seen.add(path)
    inputs = {path.relative_to(PROJECT_ROOT).as_posix()}
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = shlex.split(line, comments=True)
        if not tokens:
            continue
        reference = None
        for option in ("-r", "--requirement", "-c", "--constraint", "--build-constraint"):
            if tokens[0] == option:
                assert len(tokens) == 2
                reference = tokens[1]
            elif tokens[0].startswith(option + "="):
                reference = tokens[0].split("=", 1)[1]
            elif option in {"-r", "-c"} and tokens[0].startswith(option):
                reference = tokens[0][2:]
            if reference is not None:
                inputs.update(_nested_requirement_inputs(path.parent / reference, seen))
                break
    return inputs


@pytest.mark.parametrize("job_id", ["test-layer1", "golden-regression", "test-layer2-ml"])
def test_pip_cache_tracks_actual_nested_requirements_and_inline_policy(job_id: str) -> None:
    setup = [step for step in _jobs()[job_id]["steps"] if step.get("with", {}).get("cache") == "pip"]
    assert len(setup) == 1
    cached = setup[0]["with"]["cache-dependency-path"].splitlines()
    assert cached == list(PIP_CACHE_INPUTS)
    consumed = _nested_requirement_inputs(PROJECT_ROOT / "requirements-dev.txt")
    consumed.update({"requirements/constraints.txt", "pyproject.toml", ".github/workflows/enforcement.yml"})
    assert set(cached) == consumed


def _write_selection_config(project: Path) -> None:
    (project / "pytest.ini").write_text(
        "[pytest]\naddopts = --strict-markers\nmarkers =\n"
        "    unit\n    golden\n    ml\n    slow\n    benchmark\n    stress\n",
        encoding="utf-8",
    )


@pytest.fixture
def selection_project(tmp_path: Path) -> Path:
    _write_selection_config(tmp_path)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_selection.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.unit\ndef test_unit(): pass\n\n"
        "def test_unmarked(): pass\n\n"
        "@pytest.mark.golden\ndef test_golden(): pass\n\n"
        "@pytest.mark.ml\n@pytest.mark.golden\ndef test_ml_golden(): pass\n\n"
        "@pytest.mark.slow\n@pytest.mark.golden\ndef test_slow_golden(): pass\n\n"
        "@pytest.mark.benchmark\n@pytest.mark.golden\ndef test_benchmark_golden(): pass\n\n"
        "@pytest.mark.stress\ndef test_stress(): pass\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def marker_matrix_project(tmp_path: Path) -> tuple[Path, dict[str, frozenset[str]]]:
    project = tmp_path / "marker_matrix"
    project.mkdir()
    _write_selection_config(project)
    tests = project / "tests"
    tests.mkdir()
    names = ("unit", "golden", "ml", "slow", "benchmark", "stress")
    cases = {}
    source = ["import pytest\n"]
    for assignment in product((False, True), repeat=len(names)):
        name = "test_markers_" + "".join("1" if present else "0" for present in assignment)
        markers = frozenset(marker for marker, present in zip(names, assignment) if present)
        cases[f"tests/test_selection.py::{name}"] = markers
        source.extend(f"@pytest.mark.{marker}\n" for marker in names if marker in markers)
        source.append(f"def {name}(): pass\n\n")
    (tests / "test_selection.py").write_text("".join(source), encoding="utf-8")
    return project, cases


def _run_gate(project: Path, job_id: str, *, collect: bool = False) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    args = [sys.executable, "-I", "-B", "-m", "pytest", *_test_command(job_id)[1:], "-q", "-p", "no:cacheprovider"]
    args.extend(["--confcutdir", str(project)])
    if collect:
        args.append("--collect-only")
    return subprocess.run(args, cwd=project, env=env, capture_output=True, text=True, timeout=30, check=False)


def _nodeids(result: subprocess.CompletedProcess[str]) -> set[str]:
    assert result.returncode == 0, result.stdout + result.stderr
    return {line for line in result.stdout.splitlines() if line.startswith("tests/test_selection.py::")}


def test_real_pytest_selectors_preserve_the_full_non_ml_union(selection_project: Path) -> None:
    layer1 = _nodeids(_run_gate(selection_project, "test-layer1", collect=True))
    golden = _nodeids(_run_gate(selection_project, "golden-regression", collect=True))
    expected = {f"tests/test_selection.py::test_{name}" for name in ("unit", "unmarked", "golden", "stress")}
    assert layer1 == expected
    assert golden == {"tests/test_selection.py::test_golden"}
    assert layer1 | golden == expected


def test_real_pytest_union_covers_every_marker_combination(
    marker_matrix_project: tuple[Path, dict[str, frozenset[str]]],
) -> None:
    project, cases = marker_matrix_project
    assert len(cases) == 64
    layer1 = _nodeids(_run_gate(project, "test-layer1", collect=True))
    golden = _nodeids(_run_gate(project, "golden-regression", collect=True))
    excluded = {"ml", "slow", "benchmark"}
    full_non_ml = {nodeid for nodeid, markers in cases.items() if markers.isdisjoint(excluded)}
    golden_non_ml = {nodeid for nodeid in full_non_ml if "golden" in cases[nodeid]}
    assert len(full_non_ml) == 8 and len(golden_non_ml) == 4
    assert layer1 == full_non_ml
    assert golden == golden_non_ml
    assert layer1 | golden == full_non_ml


def test_empty_golden_selection_fails_closed(selection_project: Path) -> None:
    (selection_project / "tests" / "test_selection.py").write_text("def test_unmarked(): pass\n", encoding="utf-8")
    result = _run_gate(selection_project, "golden-regression")
    assert result.returncode == pytest.ExitCode.NO_TESTS_COLLECTED
    assert "1 deselected" in result.stdout


@pytest.mark.parametrize("job_id", ["test-layer1", "golden-regression"])
def test_both_named_gates_propagate_golden_failures(selection_project: Path, job_id: str) -> None:
    (selection_project / "tests" / "test_selection.py").write_text(
        "import pytest\n@pytest.mark.golden\ndef test_golden_failure():\n    raise AssertionError('golden sentinel')\n",
        encoding="utf-8",
    )
    result = _run_gate(selection_project, job_id)
    assert result.returncode == pytest.ExitCode.TESTS_FAILED
    assert "golden sentinel" in result.stdout


def test_layer1_retains_non_golden_failure_detection(selection_project: Path) -> None:
    path = selection_project / "tests" / "test_selection.py"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "def test_unmarked(): pass", "def test_unmarked(): raise ValueError('unit sentinel')"
        ),
        encoding="utf-8",
    )
    layer1 = _run_gate(selection_project, "test-layer1")
    golden = _run_gate(selection_project, "golden-regression")
    assert layer1.returncode == pytest.ExitCode.TESTS_FAILED
    assert "unit sentinel" in layer1.stdout
    assert golden.returncode == pytest.ExitCode.OK
    assert "1 passed" in golden.stdout
