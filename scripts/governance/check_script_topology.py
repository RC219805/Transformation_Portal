#!/usr/bin/env python3
"""Validate governed script placement and compatibility-wrapper contracts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

COMPATIBILITY_WRAPPERS = {
    "scripts/apex_aggregate_ledger.py": (
        "scripts/ci/apex/aggregate_ledger.py",
        "from scripts.ci.apex.aggregate_ledger import",
    ),
    "scripts/apex_dashboard_generator.py": (
        "scripts/ci/apex/dashboard_generator.py",
        "from scripts.ci.apex.dashboard_generator import",
    ),
    "scripts/apex_enforce_gate.py": (
        "scripts/ci/apex/enforce_gate.py",
        "from scripts.ci.apex.enforce_gate import",
    ),
    "scripts/apex_matrix_runner.py": (
        "scripts/ci/apex/matrix_runner.py",
        "from scripts.ci.apex.matrix_runner import",
    ),
    "scripts/apex_pr_comment.py": (
        "scripts/ci/apex/pr_comment.py",
        "from scripts.ci.apex.pr_comment import",
    ),
    "scripts/apex_rebuild_ledger.py": (
        "scripts/ci/apex/rebuild_ledger.py",
        "from scripts.ci.apex.rebuild_ledger import",
    ),
    "scripts/apex_validate_policy.py": (
        "scripts/ci/apex/validate_policy.py",
        "from scripts.ci.apex.validate_policy import",
    ),
    "scripts/apex_verify_contract.py": (
        "scripts/ci/apex/verify_contract.py",
        "from scripts.ci.apex.verify_contract import",
    ),
    "scripts/analyze_flakes.py": (
        "scripts/ci/analyze_flakes.py",
        "from scripts.ci.analyze_flakes import",
    ),
    "scripts/architectural_context_extractor.py": (
        "scripts/analysis/architectural_context_extractor.py",
        "from scripts.analysis.architectural_context_extractor import",
    ),
    "scripts/auto_fix_quality.py": (
        "scripts/maintenance/auto_fix_quality.py",
        "from scripts.maintenance.auto_fix_quality import",
    ),
    "scripts/benchmark_phase2.py": (
        "scripts/analysis/benchmark_phase2.py",
        "from scripts.analysis.benchmark_phase2 import",
    ),
    "scripts/benchmark_phase3.py": (
        "scripts/analysis/benchmark_phase3.py",
        "from scripts.analysis.benchmark_phase3 import",
    ),
    "scripts/create_board_textures.py": (
        "scripts/utilities/create_board_textures.py",
        "from scripts.utilities.create_board_textures import main",
    ),
    "scripts/deprecate_docs.py": (
        "scripts/maintenance/deprecate_docs.py",
        "from scripts.maintenance.deprecate_docs import",
    ),
    "scripts/check_image_processing_readiness.py": (
        "scripts/validation/check_image_processing_readiness.py",
        "from scripts.validation.check_image_processing_readiness import",
    ),
    "scripts/codebase_philosophy_auditor.py": (
        "src/transformation_portal/analyzers/codebase_philosophy_auditor.py",
        "from transformation_portal.analyzers.codebase_philosophy_auditor import",
    ),
    "scripts/decision_decay_dashboard.py": (
        "src/transformation_portal/analyzers/decision_decay_dashboard.py",
        "from transformation_portal.analyzers.decision_decay_dashboard import",
    ),
    "scripts/depth_pro_export.py": (
        "scripts/pipelines/depth_pro_export.py",
        "from scripts.pipelines.depth_pro_export import",
    ),
    "scripts/download_depth_models.py": (
        "scripts/setup/download_depth_models.py",
        "from scripts.setup.download_depth_models import",
    ),
    "scripts/download_samples.py": (
        "scripts/setup/download_samples.py",
        "from scripts.setup.download_samples import",
    ),
    "scripts/download_sam2_checkpoint.py": (
        "scripts/setup/download_sam2_checkpoint.py",
        "from scripts.setup.download_sam2_checkpoint import",
    ),
    "scripts/evolutionary_checkpoint.py": (
        "src/transformation_portal/streaming/checkpoint.py",
        "from transformation_portal.streaming.checkpoint import",
    ),
    "scripts/extract_architectural_context.py": (
        "scripts/analysis/extract_architectural_context.py",
        "from scripts.analysis.extract_architectural_context import",
    ),
    "scripts/install_models.py": (
        "scripts/setup/install_models.py",
        "from scripts.setup.install_models import",
    ),
    "scripts/install_models_auto.py": (
        "scripts/setup/install_models_auto.py",
        "from scripts.setup.install_models_auto import",
    ),
    "scripts/migrate_imports.py": (
        "scripts/maintenance/migrate_imports.py",
        "from scripts.maintenance.migrate_imports import",
    ),
    "scripts/parse_workflows.py": (
        "scripts/validation/parse_workflows.py",
        "from scripts.validation.parse_workflows import",
    ),
    "scripts/run_aerial_enhancement.py": (
        "scripts/pipelines/run_aerial_enhancement.py",
        "from scripts.pipelines.run_aerial_enhancement import",
    ),
    "scripts/run_depth_estimation.py": (
        "scripts/pipelines/run_depth_estimation.py",
        "from scripts.pipelines.run_depth_estimation import",
    ),
    "scripts/validate_ci_config.py": (
        "scripts/validation/validate_ci_config.py",
        "from scripts.validation.validate_ci_config import",
    ),
    "scripts/validate_depth_pro_checkpoint.py": (
        "scripts/validation/validate_depth_pro_checkpoint.py",
        "from scripts.validation.validate_depth_pro_checkpoint import",
    ),
    "scripts/validate_ingest_contract.py": (
        "scripts/validation/validate_ingest_contract.py",
        "from scripts.validation.validate_ingest_contract import",
    ),
    "scripts/validate_pbr_phase5d.py": (
        "scripts/validation/validate_pbr_phase5d.py",
        "from scripts.validation.validate_pbr_phase5d import",
    ),
    "scripts/validate_phase1_optimizations.py": (
        "scripts/validation/validate_phase1_optimizations.py",
        "from scripts.validation.validate_phase1_optimizations import",
    ),
    "scripts/validate_phase2.py": (
        "scripts/validation/validate_phase2.py",
        "from scripts.validation.validate_phase2 import",
    ),
    "scripts/validate_path_filters.py": (
        "scripts/validation/validate_path_filters.py",
        "from scripts.validation.validate_path_filters import",
    ),
    "scripts/verify_setup.py": (
        "scripts/verification/verify_setup.py",
        "from scripts.verification.verify_setup import",
    ),
    "scripts/verify_lux_depth_v3_surface.py": (
        "scripts/verification/verify_lux_depth_v3_surface.py",
        "from scripts.verification.verify_lux_depth_v3_surface import",
    ),
    "scripts/verify_depth_pro.py": (
        "scripts/verification/verify_depth_pro.py",
        "from scripts.verification.verify_depth_pro import",
    ),
    "scripts/verify_performance_ledger_fixes.py": (
        "scripts/verification/verify_performance_ledger_fixes.py",
        "from scripts.verification.verify_performance_ledger_fixes import",
    ),
    "scripts/verify_run_card_integrity.py": (
        "scripts/verification/verify_run_card_integrity.py",
        "from scripts.verification.verify_run_card_integrity import",
    ),
    "scripts/test_metadata_extraction.py": (
        "scripts/validation/validate_metadata_extraction.py",
        "from scripts.validation.validate_metadata_extraction import",
    ),
    "scripts/track_test_flakes.py": (
        "scripts/ci/track_test_flakes.py",
        "from scripts.ci.track_test_flakes import",
    ),
    "scripts/synthetic_viewer.py": (
        "src/transformation_portal/perceptual/synthetic_viewer.py",
        "from transformation_portal.perceptual.synthetic_viewer import",
    ),
    "scripts/temporal_evolution.py": (
        "src/transformation_portal/analyzers/temporal_evolution.py",
        "from transformation_portal.analyzers.temporal_evolution import",
    ),
    "scripts/simple_image_processor.py": (
        "scripts/utilities/simple_image_processor.py",
        "from scripts.utilities.simple_image_processor import",
    ),
    "scripts/pipelines/lux_render_pipeline.py": (
        "src/transformation_portal/pipelines/lux_render_pipeline.py",
        "from transformation_portal.pipelines.lux_render_pipeline import",
    ),
    "scripts/visualize_material_assignments.py": (
        "scripts/utilities/visualize_material_assignments.py",
        "from scripts.utilities.visualize_material_assignments import",
    ),
}

NON_CLI_COMPATIBILITY_WRAPPERS = {
    "scripts/codebase_philosophy_auditor.py",
    "scripts/evolutionary_checkpoint.py",
    "scripts/synthetic_viewer.py",
    "scripts/temporal_evolution.py",
}
CLI_COMPATIBILITY_WRAPPERS = set(COMPATIBILITY_WRAPPERS) - NON_CLI_COMPATIBILITY_WRAPPERS
SCRIPT_PACKAGE_COMPATIBILITY_WRAPPERS = {
    wrapper for wrapper, (_canonical, marker) in COMPATIBILITY_WRAPPERS.items() if marker.startswith("from scripts.")
}
SOURCE_PACKAGE_COMPATIBILITY_WRAPPERS = {
    wrapper
    for wrapper, (_canonical, marker) in COMPATIBILITY_WRAPPERS.items()
    if marker.startswith("from transformation_portal.")
}

SHELL_COMPATIBILITY_WRAPPERS = {
    "scripts/check_ml_test_isolation.sh": "scripts/validation/check_ml_test_isolation.sh",
    "scripts/lint_runner.sh": "scripts/ci/lint_runner.sh",
    "scripts/local_ci_check.sh": "scripts/ci/local_ci_check.sh",
    "scripts/organize_docs.sh": "scripts/maintenance/organize_docs.sh",
    "scripts/pre_commit_hook.sh": "scripts/maintenance/pre_commit_hook.sh",
    "scripts/process_750_picacho_elite.sh": "scripts/pipelines/process_750_picacho_elite.sh",
    "scripts/process_750_picacho_elite_batch.sh": "scripts/pipelines/process_750_picacho_elite_batch.sh",
    "scripts/security_scan.sh": "scripts/validation/security_scan.sh",
    "scripts/test_v2_integration.sh": "scripts/validation/test_v2_integration.sh",
    "scripts/validate_dependency_constraints.sh": "scripts/validation/validate_dependency_constraints.sh",
    "scripts/validate_phase6a.sh": "scripts/validation/validate_phase6a.sh",
}

ALLOWED_ROOT_PYTHON_IMPLEMENTATIONS = {
    # Contract-bound Lux Depth V3 subprocess entrypoint. V2Runner and operator
    # docs intentionally look up this exact root path.
    "scripts/enhance_image.py",
}

ALLOWED_ROOT_SHELL_IMPLEMENTATIONS: set[str] = set()

RETIRED_ORGANIZER_PATHS = {
    "archive/.organize_docs.sh": "archive/scripts/legacy-organization/organize_docs_root_legacy.sh",
    "scripts/ADD_REMAINING_FILES.sh": "archive/scripts/legacy-organization/add_remaining_files_legacy.sh",
    "scripts/context_aware_quickstart.sh": "archive/scripts/legacy-organization/context_aware_quickstart_legacy.sh",
    "scripts/create_optimized_structure.sh": "archive/scripts/legacy-organization/create_optimized_structure_legacy.sh",
    "scripts/EXECUTE_CLEANUP.sh": "archive/scripts/legacy-organization/execute_cleanup_legacy.sh",
    "scripts/execute_phase_2_extraction.sh": "archive/scripts/legacy-organization/execute_phase_2_extraction_legacy.sh",
    "scripts/FIX_CRITICAL_ISSUES.sh": "archive/scripts/legacy-organization/fix_critical_issues_legacy.sh",
    "scripts/FIX_TIFF_PUSH.sh": "archive/scripts/legacy-organization/fix_tiff_push_legacy.sh",
    "scripts/install_models_old_backup.py": "archive/scripts/legacy-organization/install_models_old_backup.py",
    "scripts/organize_outputs.sh": "archive/scripts/legacy-organization/organize_outputs.sh",
    "scripts/organize_remaining.sh": "archive/scripts/legacy-organization/organize_remaining.sh",
    "scripts/organize_root_files.sh": "archive/scripts/legacy-organization/organize_root_files.sh",
    "scripts/organize_scripts.sh": "archive/scripts/legacy-organization/organize_scripts.sh",
    "scripts/verify_pr98.sh": "archive/scripts/legacy-organization/verify_pr98_legacy.sh",
}

RETIRED_TOOL_PATHS = {
    "tools/check_unicode.py": "scripts/validation/check_unicode_controls.py",
    "tools/performance_ledger_v1.0_backup.py": "archive/scripts/performance_ledger_v1_0_backup.py",
    "tools/test_16bit_implementation.py": "scripts/validation/validate_lux_depth_v3_16bit_output.py",
    "tools/verify_16bit_handoff.py": "scripts/validation/verify_lux_depth_v3_16bit_handoff.py",
}

RELOCATED_SCRIPT_PATHS = {
    "scripts/pipelines/process_750_picacho_batch.py": "archive/scripts/legacy-organization/process_750_picacho_batch_legacy.py",
    "scripts/pipelines/run_rag_workflow.py": "examples/rag/run_rag_workflow.py",
}

ALLOWED_SCRIPT_DOCS = {
    "scripts/README.md",
    "scripts/README_QUALITY_CONTROL.md",
    "scripts/QUICKSTART_QUALITY.md",
    "scripts/TEST_V2_INTEGRATION_README.md",
}

MISPLACED_PIPELINE_HELPER_PATTERNS = (
    "test_",
    "examples_",
)


@dataclass(frozen=True)
class TopologyViolation:
    """Single script-topology violation."""

    path: str
    reason: str
    suggestion: str


def _git_ls_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def _read_repo_text(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _repo_root_parent_expression(wrapper_path: str) -> str:
    wrapper_parent_depth = len(Path(wrapper_path).parent.parts)
    return f"Path(__file__).resolve().parents[{wrapper_parent_depth}]"


def _src_root_bootstrap_expression(wrapper_path: str) -> str:
    return f'{_repo_root_parent_expression(wrapper_path)} / "src"'


def validate_script_topology(
    tracked_paths: Iterable[str],
    *,
    read_text: Callable[[str], str] = _read_repo_text,
) -> list[TopologyViolation]:
    """Return deterministic script-topology violations for tracked paths."""
    tracked = set(tracked_paths)
    violations: list[TopologyViolation] = []

    for path, destination in sorted(RETIRED_ORGANIZER_PATHS.items()):
        if path in tracked:
            violations.append(
                TopologyViolation(
                    path=path,
                    reason="retired broad-mutating organization helper remains active",
                    suggestion=f"move to {destination}",
                )
            )

    for path, destination in sorted(RETIRED_TOOL_PATHS.items()):
        if path in tracked:
            violations.append(
                TopologyViolation(
                    path=path,
                    reason="retired or relocated tool path remains in active tools root",
                    suggestion=f"move to {destination}",
                )
            )

    for path, destination in sorted(RELOCATED_SCRIPT_PATHS.items()):
        if path in tracked:
            violations.append(
                TopologyViolation(
                    path=path,
                    reason="script remains in an architecture-inconsistent location",
                    suggestion=f"move to {destination}",
                )
            )

    for path in sorted(p for p in tracked if p.startswith("scripts/") and Path(p).parent == Path("scripts")):
        if path in ALLOWED_SCRIPT_DOCS:
            continue
        suffix = Path(path).suffix.lower()
        if suffix in {".md", ".txt"}:
            violations.append(
                TopologyViolation(
                    path=path,
                    reason="historical script report is stored in active scripts root",
                    suggestion="move historical evidence to docs/historical/script-audits/",
                )
            )
        if (
            suffix == ".py"
            and path != "scripts/__init__.py"
            and path not in COMPATIBILITY_WRAPPERS
            and path not in ALLOWED_ROOT_PYTHON_IMPLEMENTATIONS
        ):
            violations.append(
                TopologyViolation(
                    path=path,
                    reason="root Python script is not a governed compatibility wrapper or allowed contract entrypoint",
                    suggestion=(
                        "move implementation to scripts/analysis, scripts/ci, scripts/validation, scripts/setup, "
                        "scripts/verification, scripts/pipelines, scripts/maintenance, or src/, and keep a thin "
                        "wrapper only if the public path is contract-bound"
                    ),
                )
            )
        if (
            suffix == ".sh"
            and path not in SHELL_COMPATIBILITY_WRAPPERS
            and path not in ALLOWED_ROOT_SHELL_IMPLEMENTATIONS
            and path not in RETIRED_ORGANIZER_PATHS
        ):
            violations.append(
                TopologyViolation(
                    path=path,
                    reason="root shell script is not a governed compatibility wrapper or allowed legacy entrypoint",
                    suggestion=(
                        "move implementation to scripts/ci, scripts/validation, scripts/setup, or "
                        "scripts/pipelines and keep a thin wrapper only if the public path is contract-bound"
                    ),
                )
            )

    for path in sorted(p for p in tracked if p.startswith("scripts/pipelines/") and Path(p).suffix == ".py"):
        script_name = Path(path).name
        if (
            script_name.startswith(MISPLACED_PIPELINE_HELPER_PATTERNS)
            or script_name.endswith("_test.py")
            or script_name.endswith("_examples.py")
        ):
            if script_name.startswith("examples_") or script_name.endswith("_examples.py"):
                suggestion = "move usage examples to examples/pipelines/ and name by subject"
            else:
                suggestion = "move validation or diagnostic helpers to scripts/validation or scripts/analysis"
            violations.append(
                TopologyViolation(
                    path=path,
                    reason="pipeline implementation directory contains a test/example helper naming pattern",
                    suggestion=suggestion,
                )
            )

    for wrapper, (canonical, import_marker) in sorted(COMPATIBILITY_WRAPPERS.items()):
        wrapper_present = wrapper in tracked
        canonical_present = canonical in tracked
        if wrapper_present and not canonical_present:
            violations.append(
                TopologyViolation(
                    path=wrapper,
                    reason="compatibility wrapper has no tracked canonical implementation",
                    suggestion=f"restore canonical implementation at {canonical}",
                )
            )
            continue
        if canonical_present and not wrapper_present:
            violations.append(
                TopologyViolation(
                    path=canonical,
                    reason="canonical implementation is missing its public compatibility wrapper",
                    suggestion=f"restore wrapper at {wrapper}",
                )
            )
            continue
        if not wrapper_present:
            continue
        try:
            wrapper_text = read_text(wrapper)
        except OSError as exc:
            violations.append(
                TopologyViolation(
                    path=wrapper,
                    reason=f"compatibility wrapper could not be read: {exc}",
                    suggestion="restore a readable wrapper file",
                )
            )
            continue
        if import_marker not in wrapper_text:
            violations.append(
                TopologyViolation(
                    path=wrapper,
                    reason="compatibility wrapper does not delegate to canonical implementation",
                    suggestion=f"import and delegate to {canonical}",
                )
            )
        if wrapper in CLI_COMPATIBILITY_WRAPPERS and "raise SystemExit(" not in wrapper_text:
            violations.append(
                TopologyViolation(
                    path=wrapper,
                    reason="CLI compatibility wrapper does not propagate canonical exit status",
                    suggestion="call the canonical main through raise SystemExit(...)",
                )
            )
        if (
            wrapper in SCRIPT_PACKAGE_COMPATIBILITY_WRAPPERS
            and _repo_root_parent_expression(wrapper) not in wrapper_text
        ):
            violations.append(
                TopologyViolation(
                    path=wrapper,
                    reason="script-package compatibility wrapper does not bootstrap repository root",
                    suggestion=(
                        f"insert the repository root ({_repo_root_parent_expression(wrapper)}) into sys.path "
                        "before importing scripts.*"
                    ),
                )
            )
        if (
            wrapper in SOURCE_PACKAGE_COMPATIBILITY_WRAPPERS
            and _src_root_bootstrap_expression(wrapper) not in wrapper_text
        ):
            violations.append(
                TopologyViolation(
                    path=wrapper,
                    reason="source-package compatibility wrapper does not bootstrap src package root",
                    suggestion=(
                        f"insert the src root ({_src_root_bootstrap_expression(wrapper)}) into sys.path "
                        "before importing transformation_portal.*"
                    ),
                )
            )

    for wrapper, canonical in sorted(SHELL_COMPATIBILITY_WRAPPERS.items()):
        wrapper_present = wrapper in tracked
        canonical_present = canonical in tracked
        if wrapper_present and not canonical_present:
            violations.append(
                TopologyViolation(
                    path=wrapper,
                    reason="shell compatibility wrapper has no tracked canonical implementation",
                    suggestion=f"restore canonical implementation at {canonical}",
                )
            )
            continue
        if canonical_present and not wrapper_present:
            violations.append(
                TopologyViolation(
                    path=canonical,
                    reason="canonical shell implementation is missing its public compatibility wrapper",
                    suggestion=f"restore wrapper at {wrapper}",
                )
            )
            continue
        if not wrapper_present:
            continue
        try:
            wrapper_text = read_text(wrapper)
        except OSError as exc:
            violations.append(
                TopologyViolation(
                    path=wrapper,
                    reason=f"shell compatibility wrapper could not be read: {exc}",
                    suggestion="restore a readable wrapper file",
                )
            )
            continue
        if canonical not in wrapper_text:
            violations.append(
                TopologyViolation(
                    path=wrapper,
                    reason="shell compatibility wrapper does not delegate to canonical implementation",
                    suggestion=f"exec {canonical}",
                )
            )
        if "exec " not in wrapper_text:
            violations.append(
                TopologyViolation(
                    path=wrapper,
                    reason="shell compatibility wrapper does not replace itself with canonical command",
                    suggestion="delegate with exec so signals and exit status are preserved",
                )
            )

    return sorted(violations, key=lambda item: item.path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate governed script topology.")
    parser.add_argument("--verbose", action="store_true", help="Print extra pass/fail detail.")
    args = parser.parse_args()

    try:
        violations = validate_script_topology(_git_ls_files())
    except RuntimeError as exc:
        print(f"Unable to collect tracked paths: {exc}", file=sys.stderr)
        return 2

    if violations:
        print("Script topology violations detected:")
        for violation in violations:
            print(f"  - {violation.path}")
            print(f"    reason: {violation.reason}")
            print(f"    suggested: {violation.suggestion}")
        return 1

    if args.verbose:
        print("Script topology check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
