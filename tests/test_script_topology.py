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
SHELL_COMPATIBILITY_WRAPPERS = CHECKER.SHELL_COMPATIBILITY_WRAPPERS
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


def _valid_shell_wrapper_text(canonical: str) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"\n'
        f'exec "${{REPO_ROOT}}/{canonical}" "$@"\n'
    )


def test_script_topology_accepts_compatibility_wrappers() -> None:
    tracked_paths = set()
    contents = {}

    for wrapper, (canonical, marker) in COMPATIBILITY_WRAPPERS.items():
        tracked_paths.add(wrapper)
        tracked_paths.add(canonical)
        contents[wrapper] = _valid_wrapper_text(wrapper, marker)

    assert validate_script_topology(tracked_paths, read_text=_reader(contents)) == []


def test_script_topology_accepts_shell_compatibility_wrappers() -> None:
    tracked_paths = set()
    contents = {}

    for wrapper, canonical in SHELL_COMPATIBILITY_WRAPPERS.items():
        tracked_paths.add(wrapper)
        tracked_paths.add(canonical)
        contents[wrapper] = _valid_shell_wrapper_text(canonical)

    assert validate_script_topology(tracked_paths, read_text=_reader(contents)) == []


def test_script_topology_rejects_retired_organizer_paths() -> None:
    violations = validate_script_topology(
        {"scripts/organize_outputs.sh"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/organize_outputs.sh"
    assert "retired broad-mutating" in violations[0].reason


def test_script_topology_rejects_retired_branch_cleanup_paths() -> None:
    violations = validate_script_topology(
        {"scripts/FIX_CRITICAL_ISSUES.sh"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/FIX_CRITICAL_ISSUES.sh"
    assert "retired broad-mutating" in violations[0].reason
    assert violations[0].suggestion == ("move to archive/scripts/legacy-organization/fix_critical_issues_legacy.sh")


def test_script_topology_rejects_retired_pr_specific_verifier() -> None:
    violations = validate_script_topology(
        {"scripts/verify_pr98.sh"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/verify_pr98.sh"
    assert "retired broad-mutating" in violations[0].reason
    assert violations[0].suggestion == "move to archive/scripts/legacy-organization/verify_pr98_legacy.sh"


def test_script_topology_rejects_retired_context_quickstart() -> None:
    violations = validate_script_topology(
        {"scripts/context_aware_quickstart.sh"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/context_aware_quickstart.sh"
    assert "retired broad-mutating" in violations[0].reason
    assert violations[0].suggestion == "move to archive/scripts/legacy-organization/context_aware_quickstart_legacy.sh"


def test_script_topology_rejects_retired_tool_backups() -> None:
    violations = validate_script_topology(
        {"tools/performance_ledger_v1.0_backup.py"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].path == "tools/performance_ledger_v1.0_backup.py"
    assert "retired or relocated tool path" in violations[0].reason
    assert violations[0].suggestion == "move to archive/scripts/performance_ledger_v1_0_backup.py"


def test_script_topology_rejects_promoted_validation_tools() -> None:
    violations = validate_script_topology(
        {"tools/test_16bit_implementation.py"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].path == "tools/test_16bit_implementation.py"
    assert "retired or relocated tool path" in violations[0].reason
    assert violations[0].suggestion == ("move to scripts/validation/validate_lux_depth_v3_16bit_output.py")


def test_script_topology_rejects_relocated_unicode_checker() -> None:
    violations = validate_script_topology(
        {"tools/check_unicode.py"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].path == "tools/check_unicode.py"
    assert "retired or relocated tool path" in violations[0].reason
    assert violations[0].suggestion == "move to scripts/validation/check_unicode_controls.py"


def test_script_topology_rejects_relocated_rag_workflow_runner() -> None:
    violations = validate_script_topology(
        {"scripts/pipelines/run_rag_workflow.py"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/pipelines/run_rag_workflow.py"
    assert "architecture-inconsistent location" in violations[0].reason
    assert violations[0].suggestion == "move to examples/rag/run_rag_workflow.py"


def test_script_topology_rejects_retired_picacho_batch_wrapper() -> None:
    violations = validate_script_topology(
        {"scripts/pipelines/process_750_picacho_batch.py"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/pipelines/process_750_picacho_batch.py"
    assert "architecture-inconsistent location" in violations[0].reason
    assert violations[0].suggestion == "move to archive/scripts/legacy-organization/process_750_picacho_batch_legacy.py"


def test_script_topology_rejects_script_root_historical_reports() -> None:
    violations = validate_script_topology(
        {"scripts/PIPELINE_OPTIMIZATION_REPORT.md"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].suggestion == "move historical evidence to docs/historical/script-audits/"


def test_script_topology_rejects_new_root_shell_implementations() -> None:
    violations = validate_script_topology(
        {"scripts/new_quality_gate.sh"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/new_quality_gate.sh"
    assert "root shell script is not a governed compatibility wrapper" in violations[0].reason


def test_script_topology_rejects_new_root_python_implementations() -> None:
    violations = validate_script_topology(
        {"scripts/new_quality_gate.py"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/new_quality_gate.py"
    assert "root Python script is not a governed compatibility wrapper" in violations[0].reason


def test_script_topology_rejects_pipeline_local_validation_test_names() -> None:
    violations = validate_script_topology(
        {"scripts/pipelines/test_luxury_estate_pipeline.py"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/pipelines/test_luxury_estate_pipeline.py"
    assert "test/example helper naming pattern" in violations[0].reason
    assert violations[0].suggestion == "move validation or diagnostic helpers to scripts/validation or scripts/analysis"


def test_script_topology_rejects_pipeline_local_examples_prefix() -> None:
    violations = validate_script_topology(
        {"scripts/pipelines/examples_luxury_estate_pipeline.py"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/pipelines/examples_luxury_estate_pipeline.py"
    assert "test/example helper naming pattern" in violations[0].reason
    assert violations[0].suggestion == "move usage examples to examples/pipelines/ and name by subject"


def test_script_topology_rejects_pipeline_local_examples_suffix() -> None:
    violations = validate_script_topology(
        {"scripts/pipelines/elite_pipeline_examples.py"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/pipelines/elite_pipeline_examples.py"
    assert "test/example helper naming pattern" in violations[0].reason
    assert violations[0].suggestion == "move usage examples to examples/pipelines/ and name by subject"


def test_script_topology_allows_contract_bound_enhance_image_entrypoint() -> None:
    assert validate_script_topology({"scripts/enhance_image.py"}, read_text=_reader({})) == []


def test_script_topology_requires_wrapper_to_delegate_to_canonical_path() -> None:
    violations = validate_script_topology(
        {
            "scripts/test_metadata_extraction.py",
            "scripts/validation/validate_metadata_extraction.py",
        },
        read_text=_reader(
            {
                "scripts/test_metadata_extraction.py": (
                    "from pathlib import Path\n"
                    "REPO_ROOT = Path(__file__).resolve().parents[1]\n"
                    "if __name__ == '__main__':\n"
                    "    raise SystemExit(_main())\n"
                )
            }
        ),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/test_metadata_extraction.py"
    assert "does not delegate" in violations[0].reason


def test_script_topology_requires_shell_wrappers_to_delegate_to_canonical_path() -> None:
    violations = validate_script_topology(
        {
            "scripts/lint_runner.sh",
            "scripts/ci/lint_runner.sh",
        },
        read_text=_reader({"scripts/lint_runner.sh": '#!/usr/bin/env bash\nexec ./elsewhere.sh "$@"\n'}),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/lint_runner.sh"
    assert "does not delegate" in violations[0].reason


def test_script_topology_requires_shell_wrappers_to_exec_canonical_command() -> None:
    violations = validate_script_topology(
        {
            "scripts/lint_runner.sh",
            "scripts/ci/lint_runner.sh",
        },
        read_text=_reader(
            {"scripts/lint_runner.sh": ("#!/usr/bin/env bash\n" 'bash "${REPO_ROOT}/scripts/ci/lint_runner.sh" "$@"\n')}
        ),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/lint_runner.sh"
    assert "does not replace itself" in violations[0].reason


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


def test_repository_shell_wrappers_reference_canonical_scripts() -> None:
    for wrapper, canonical in SHELL_COMPATIBILITY_WRAPPERS.items():
        wrapper_path = REPO_ROOT / wrapper
        wrapper_text = wrapper_path.read_text(encoding="utf-8")
        assert canonical in wrapper_text, f"Shell wrapper must reference {canonical}: {wrapper}"
        assert "exec " in wrapper_text, f"Shell wrapper must preserve signals and exit status: {wrapper}"


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
