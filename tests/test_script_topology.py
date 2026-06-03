from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = REPO_ROOT / "scripts/governance/check_script_topology.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_script_topology_under_test", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()
COMPATIBILITY_WRAPPERS = CHECKER.COMPATIBILITY_WRAPPERS
CLI_COMPATIBILITY_WRAPPERS = CHECKER.CLI_COMPATIBILITY_WRAPPERS
SCRIPT_PACKAGE_COMPATIBILITY_WRAPPERS = CHECKER.SCRIPT_PACKAGE_COMPATIBILITY_WRAPPERS
SOURCE_PACKAGE_COMPATIBILITY_WRAPPERS = CHECKER.SOURCE_PACKAGE_COMPATIBILITY_WRAPPERS
src_root_bootstrap_expression = CHECKER._src_root_bootstrap_expression
repo_root_parent_expression = CHECKER._repo_root_parent_expression
validate_script_topology = CHECKER.validate_script_topology


def _reader(contents: dict[str, str]):
    def read_text(path: str) -> str:
        return contents[path]

    return read_text


def _valid_wrapper_text(wrapper: str, marker: str) -> str:
    lines = ["#!/usr/bin/env python3"]
    if wrapper in SCRIPT_PACKAGE_COMPATIBILITY_WRAPPERS:
        lines.extend(
            [
                "import sys",
                "from pathlib import Path",
                f"REPO_ROOT = {repo_root_parent_expression(wrapper)}",
                "sys.path.insert(0, str(REPO_ROOT))",
            ]
        )
    if wrapper in SOURCE_PACKAGE_COMPATIBILITY_WRAPPERS:
        lines.extend(
            [
                "import sys",
                "from pathlib import Path",
                f"SRC_ROOT = {src_root_bootstrap_expression(wrapper)}",
                "sys.path.insert(0, str(SRC_ROOT))",
            ]
        )
    lines.append(f"{marker} main")
    if wrapper in CLI_COMPATIBILITY_WRAPPERS:
        lines.append("if __name__ == '__main__':")
        lines.append("    raise SystemExit(_main())")
    return "\n".join(lines) + "\n"


def test_script_topology_accepts_compatibility_wrappers() -> None:
    tracked_paths = set()
    contents = {}

    for wrapper, (canonical, marker) in COMPATIBILITY_WRAPPERS.items():
        tracked_paths.add(wrapper)
        tracked_paths.add(canonical)
        contents[wrapper] = _valid_wrapper_text(wrapper, marker)

    assert validate_script_topology(tracked_paths, read_text=_reader(contents)) == []


def test_script_topology_rejects_retired_organizer_paths() -> None:
    violations = validate_script_topology(
        {"scripts/organize_outputs.sh"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/organize_outputs.sh"
    assert "retired broad-mutating" in violations[0].reason


def test_script_topology_rejects_script_root_historical_reports() -> None:
    violations = validate_script_topology(
        {"scripts/PIPELINE_OPTIMIZATION_REPORT.md"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].suggestion == "move historical evidence to docs/historical/script-audits/"


def test_script_topology_requires_wrapper_to_delegate_to_canonical_path() -> None:
    violations = validate_script_topology(
        {
            "scripts/install_models.py",
            "scripts/setup/install_models.py",
        },
        read_text=_reader(
            {
                "scripts/install_models.py": (
                    "from pathlib import Path\n"
                    "REPO_ROOT = Path(__file__).resolve().parents[1]\n"
                    "if __name__ == '__main__':\n"
                    "    raise SystemExit(_main())\n"
                )
            }
        ),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/install_models.py"
    assert "does not delegate" in violations[0].reason


def test_script_topology_requires_cli_wrappers_to_propagate_exit_status() -> None:
    violations = validate_script_topology(
        {
            "scripts/visualize_material_assignments.py",
            "scripts/utilities/visualize_material_assignments.py",
        },
        read_text=_reader(
            {
                "scripts/visualize_material_assignments.py": (
                    "from scripts.utilities.visualize_material_assignments import main as _main\n"
                    "from pathlib import Path\n"
                    "REPO_ROOT = Path(__file__).resolve().parents[1]\n"
                    "if __name__ == '__main__':\n"
                    "    _main()\n"
                )
            }
        ),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/visualize_material_assignments.py"
    assert "does not propagate canonical exit status" in violations[0].reason


def test_script_topology_requires_script_package_wrappers_to_bootstrap_repo_root() -> None:
    violations = validate_script_topology(
        {
            "scripts/visualize_material_assignments.py",
            "scripts/utilities/visualize_material_assignments.py",
        },
        read_text=_reader(
            {
                "scripts/visualize_material_assignments.py": (
                    "from scripts.utilities.visualize_material_assignments import main as _main\n"
                    "if __name__ == '__main__':\n"
                    "    raise SystemExit(_main())\n"
                )
            }
        ),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/visualize_material_assignments.py"
    assert "does not bootstrap repository root" in violations[0].reason


def test_script_topology_requires_source_package_wrappers_to_bootstrap_src_root() -> None:
    violations = validate_script_topology(
        {
            "scripts/synthetic_viewer.py",
            "src/transformation_portal/perceptual/synthetic_viewer.py",
        },
        read_text=_reader(
            {"scripts/synthetic_viewer.py": ("from transformation_portal.perceptual.synthetic_viewer import *\n")}
        ),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/synthetic_viewer.py"
    assert "does not bootstrap src package root" in violations[0].reason


def test_script_topology_requires_nested_source_package_wrappers_to_bootstrap_repo_src_root() -> None:
    violations = validate_script_topology(
        {
            "scripts/pipelines/lux_render_pipeline.py",
            "src/transformation_portal/pipelines/lux_render_pipeline.py",
        },
        read_text=_reader(
            {
                "scripts/pipelines/lux_render_pipeline.py": (
                    "from pathlib import Path\n"
                    'SRC_ROOT = Path(__file__).resolve().parents[1] / "src"\n'
                    "from transformation_portal.pipelines.lux_render_pipeline import main\n"
                    "if __name__ == '__main__':\n"
                    "    raise SystemExit(main())\n"
                )
            }
        ),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/pipelines/lux_render_pipeline.py"
    assert "does not bootstrap src package root" in violations[0].reason
    assert 'parents[2] / "src"' in violations[0].suggestion


def test_repository_compatibility_wrappers_reference_canonical_modules() -> None:
    for wrapper, (_canonical, marker) in COMPATIBILITY_WRAPPERS.items():
        wrapper_path = REPO_ROOT / wrapper
        assert wrapper_path.exists(), f"Missing compatibility wrapper: {wrapper}"
        assert marker in wrapper_path.read_text(encoding="utf-8")


def test_repository_cli_wrappers_propagate_exit_status() -> None:
    for wrapper in CLI_COMPATIBILITY_WRAPPERS:
        wrapper_text = (REPO_ROOT / wrapper).read_text(encoding="utf-8")
        assert "raise SystemExit(" in wrapper_text, f"Wrapper must propagate CLI status: {wrapper}"


def test_repository_script_package_wrappers_bootstrap_repo_root() -> None:
    for wrapper in SCRIPT_PACKAGE_COMPATIBILITY_WRAPPERS:
        wrapper_text = (REPO_ROOT / wrapper).read_text(encoding="utf-8")
        expected = repo_root_parent_expression(wrapper)
        assert expected in wrapper_text, f"Wrapper must bootstrap repo root via {expected}: {wrapper}"


def test_repository_source_package_wrappers_bootstrap_src_root() -> None:
    for wrapper in SOURCE_PACKAGE_COMPATIBILITY_WRAPPERS:
        wrapper_text = (REPO_ROOT / wrapper).read_text(encoding="utf-8")
        expected = src_root_bootstrap_expression(wrapper)
        assert expected in wrapper_text, f"Wrapper must bootstrap src root via {expected}: {wrapper}"


def test_synthetic_viewer_wrapper_imports_from_raw_checkout() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "from scripts.synthetic_viewer import SyntheticViewer; print(SyntheticViewer.__name__)",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "SyntheticViewer"


def test_visualize_material_assignments_wrapper_matches_canonical_missing_file_exit() -> None:
    missing_image = REPO_ROOT / "missing-material-input.jpg"
    wrapper = REPO_ROOT / "scripts/visualize_material_assignments.py"
    canonical = REPO_ROOT / "scripts/utilities/visualize_material_assignments.py"

    wrapper_result = subprocess.run(
        [sys.executable, str(wrapper), str(missing_image)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    canonical_result = subprocess.run(
        [sys.executable, str(canonical), str(missing_image)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert wrapper_result.returncode == canonical_result.returncode == 1
    assert "Input image not found" in wrapper_result.stderr
    assert wrapper_result.stderr == canonical_result.stderr
