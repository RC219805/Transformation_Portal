#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests to validate codebase structure and organization.

These tests ensure that the repository maintains good structure and
prevents accumulation of clutter or structural issues.
"""

import hashlib
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

# Repository root for structure validation
_repo_root = Path(__file__).parent.parent


class TestDirectoryStructure:
    """Tests for proper directory organization."""

    def test_src_package_exists(self):
        """Test that main package exists in src/."""
        package_path = _repo_root / "src" / "transformation_portal"
        assert package_path.exists(), "Main package should exist in src/"
        assert package_path.is_dir(), "transformation_portal should be a directory"

    def test_docs_directory_exists(self):
        """Test that docs directory exists."""
        docs_path = _repo_root / "docs"
        assert docs_path.exists(), "docs/ directory should exist"
        assert docs_path.is_dir(), "docs should be a directory"

    def test_tests_directory_exists(self):
        """Test that tests directory exists."""
        tests_path = _repo_root / "tests"
        assert tests_path.exists(), "tests/ directory should exist"
        assert tests_path.is_dir(), "tests should be a directory"

    def test_assets_directory_exists(self):
        """Test that assets directory exists."""
        assets_path = _repo_root / "assets"
        assert assets_path.exists(), "assets/ directory should exist"
        assert assets_path.is_dir(), "assets should be a directory"

    def test_config_directory_exists(self):
        """Test that config directory exists."""
        config_path = _repo_root / "config"
        assert config_path.exists(), "config/ directory should exist"


class TestPackageStructure:
    """Tests for proper package organization in src/."""

    def test_main_init_exists(self):
        """Test that main package has __init__.py."""
        init_file = _repo_root / "src" / "transformation_portal" / "__init__.py"
        assert init_file.exists(), "Main __init__.py should exist"

    def test_subpackages_have_init(self):
        """Test that all subpackages have __init__.py."""
        base_path = _repo_root / "src" / "transformation_portal"

        # Check key subpackages
        subpackages = [
            "depth",
            "processors",
            "pipelines",
            "utils",
            "analyzers",
        ]

        for subpkg in subpackages:
            subpkg_path = base_path / subpkg
            if subpkg_path.exists() and subpkg_path.is_dir():
                init_file = subpkg_path / "__init__.py"
                assert init_file.exists(), f"{subpkg} should have __init__.py"

    def test_utils_performance_module_exists(self):
        """Test that performance utilities module exists."""
        perf_module = _repo_root / "src" / "transformation_portal" / "utils" / "performance.py"
        assert perf_module.exists(), "performance.py should exist in utils/"

    def test_utils_error_handling_module_exists(self):
        """Test that error handling utilities module exists."""
        error_module = _repo_root / "src" / "transformation_portal" / "utils" / "error_handling.py"
        assert error_module.exists(), "error_handling.py should exist in utils/"


class TestDocumentationOrganization:
    """Tests for documentation organization."""

    def test_main_readme_exists(self):
        """Test that main README exists in root."""
        readme = _repo_root / "README.md"
        assert readme.exists(), "README.md should exist in root"

    def test_no_excessive_root_markdown_files(self):
        """Test that tracked root paths conform to the canonical placement policy."""
        result = subprocess.run(
            ["./scripts/setup/pre-commit-check.sh", "--all"],
            cwd=_repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_docs_subdirectories_exist(self):
        """Test that docs has proper subdirectories."""
        docs_path = _repo_root / "docs"

        # Check for key documentation directories
        expected_subdirs = [
            "depth_pipeline",
            "workflow",
            "brand",
        ]

        for subdir in expected_subdirs:
            subdir_path = docs_path / subdir
            if not subdir_path.exists():
                # Not all may exist in every branch, but log it
                print(f"Note: {subdir} directory not found in docs/")


class TestRootGovernanceMetadata:
    """Tests for root governance metadata that is allowed to stay in root."""

    @staticmethod
    def _make_targets(makefile: Path) -> set[str]:
        return {match.group(1) for match in re.finditer(r"(?m)^([A-Za-z0-9_.%/+@-]+):", makefile.read_text())}

    @staticmethod
    def _relative_markdown_links(markdown_path: Path) -> list[tuple[int, str, Path]]:
        links: list[tuple[int, str, Path]] = []
        text = markdown_path.read_text(encoding="utf-8")
        for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text):
            raw_target = match.group(1).strip()
            link_target = raw_target.split("#", 1)[0].split("?", 1)[0].strip("<>")
            if not link_target or link_target.startswith("#") or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", link_target):
                continue
            line_number = text.count("\n", 0, match.start()) + 1
            links.append((line_number, raw_target, markdown_path.parent / link_target))
        return links

    def test_root_operator_markdown_links_resolve_to_tracked_targets(self):
        """Root-facing operator docs should not carry broken relative links."""
        relative_docs = [
            "README.md",
            "AGENTS.md",
            "CLAUDE.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "CHANGELOG.md",
            "docs/ci/BRANCH_PROTECTION_SETUP.md",
            "docs/ci/BRANCH_PROTECTION_COMMANDS.md",
            "docs/deployment/PRODUCTION_READINESS.md",
        ]
        broken_links = []

        for relative_path in relative_docs:
            markdown_path = _repo_root / relative_path
            for line_number, raw_target, target_path in self._relative_markdown_links(markdown_path):
                if not target_path.exists():
                    broken_links.append(f"{relative_path}:{line_number}: {raw_target}")

        assert not broken_links, "Broken root-facing Markdown links:\n" + "\n".join(sorted(broken_links))

    def test_root_readme_status_points_to_current_documentation_navigation(self):
        """Root status prose should not depend on stale PR-specific snapshots."""
        readme = (_repo_root / "README.md").read_text()

        assert "building on `main` through PR #" not in readme
        required_navigation = [
            "docs/README.md",
            "docs/governance/DOCUMENTATION_MAP.md",
            "May 11, 2026 repo-wide refresh audit",
            "Last Updated: 2026-06-11",
        ]
        for expected in required_navigation:
            assert expected in readme
        assert "Last Updated: 2026-05-11" not in readme

    def test_operator_guides_use_governed_local_input_surface(self):
        """Current CLI examples should not create untracked input directories."""
        readme = (_repo_root / "README.md").read_text()
        setup_readme = (_repo_root / "scripts" / "setup" / "README.md").read_text()
        setup_guide = (_repo_root / "docs" / "guides" / "SETUP_GUIDE.md").read_text()
        current_workflow_guides = [
            (_repo_root / "docs" / "decisions" / "ADR_019_INTEGRATION_COMPLETE.md").read_text(),
            (_repo_root / "docs" / "investigations" / "PHASE2_16BIT_QUICKREF.md").read_text(),
            (_repo_root / "docs" / "workflows" / "PBR_ONLY_WORKFLOW.md").read_text(),
        ]
        runtime_installers = [
            (_repo_root / "scripts" / "setup" / "install_da3_runtime.sh").read_text(),
            (_repo_root / "scripts" / "setup" / "install_depth_pro_runtime.sh").read_text(),
            (_repo_root / "scripts" / "setup" / "install_raw_runtime.sh").read_text(),
        ]
        compose = (_repo_root / "docker-compose.yml").read_text()
        active_input_guidance = {
            "docs/decisions/ADR_019_INTEGRATION_COMPLETE.md": (
                _repo_root / "docs" / "decisions" / "ADR_019_INTEGRATION_COMPLETE.md"
            ).read_text(),
            "docs/guides/AI_ENHANCEMENT_GUIDE.md": (_repo_root / "docs" / "guides" / "AI_ENHANCEMENT_GUIDE.md").read_text(),
            "docs/guides/FLUX_INTEGRATION.md": (_repo_root / "docs" / "guides" / "FLUX_INTEGRATION.md").read_text(),
            "docs/guides/LUX_DEPTH_V3_TROUBLESHOOTING.md": (
                _repo_root / "docs" / "guides" / "LUX_DEPTH_V3_TROUBLESHOOTING.md"
            ).read_text(),
            "docs/guides/PBR_ENHANCE_CONFIG_GUIDE.md": (
                _repo_root / "docs" / "guides" / "PBR_ENHANCE_CONFIG_GUIDE.md"
            ).read_text(),
            "docs/reference/PBR_PRESETS_QUICK_REFERENCE.md": (
                _repo_root / "docs" / "reference" / "PBR_PRESETS_QUICK_REFERENCE.md"
            ).read_text(),
        }

        for guide in (readme, setup_readme, setup_guide, *current_workflow_guides, *runtime_installers):
            assert "--input-dir ./input_images" in guide
            assert "--input-dir ./input " not in guide
            assert "--input-dir ./input\n" not in guide
        stale_input_snippets = [
            '--input-dir "./input"',
            'Path("./input/',
            "Path('./input/",
            'Path("./input")',
            "Path('./input')",
            'input_root=Path("./input")',
            "input_root=Path('./input')",
            'glob.glob("input/*.jpg")',
            "glob.glob('input/*.jpg')",
        ]
        for guide_name, guide in active_input_guidance.items():
            assert "input_images" in guide, guide_name
            for stale_snippet in stale_input_snippets:
                assert stale_snippet not in guide, f"{guide_name} contains {stale_snippet}"
        assert "./input_images:/app/input:ro" in compose
        assert "./input:/app/input" not in compose

    def test_root_ml_setup_guidance_prefers_make_targets(self):
        """Root/current setup guidance should not present bootstrap profiles as the primary install path."""
        guides = {
            "README.md": (_repo_root / "README.md").read_text(),
            "CLAUDE.md": (_repo_root / "CLAUDE.md").read_text(),
            "docs/guides/SETUP_GUIDE.md": (_repo_root / "docs" / "guides" / "SETUP_GUIDE.md").read_text(),
        }

        for guide_name, guide in guides.items():
            assert "make install-ml-core" in guide, guide_name
            assert "Advanced Apple Silicon" in guide or "bootstrap-profile work" in guide, guide_name

        stale_primary_guidance = [
            "For example: `make install-ml-core` or, on Apple Silicon,",
            "or use the target-specific bootstrap flow on macOS Apple Silicon",
            "Or use the Apple Silicon bootstrap profiles directly",
            "use `make install-ml-core` or `./scripts/bootstrap/install_ml_stack.sh --profile core-mps`",
        ]
        combined = "\n".join(guides.values())
        for stale_fragment in stale_primary_guidance:
            assert stale_fragment not in combined

    def test_active_setup_guidance_uses_repo_managed_python(self):
        """Active setup docs should not depend on ambient Python or raw requirements installs."""
        guidance = {
            "README.md": (_repo_root / "README.md").read_text(),
            "CONTRIBUTING.md": (_repo_root / "CONTRIBUTING.md").read_text(),
            "SECURITY.md": (_repo_root / "SECURITY.md").read_text(),
            "docs/architecture/ADR-033-test-flake-management.md": (
                _repo_root / "docs" / "architecture" / "ADR-033-test-flake-management.md"
            ).read_text(),
            "docs/deployment/ROLLBACK_PROCEDURES.md": (
                _repo_root / "docs" / "deployment" / "ROLLBACK_PROCEDURES.md"
            ).read_text(),
            "docs/deployment/STAGING_VALIDATION.md": (
                _repo_root / "docs" / "deployment" / "STAGING_VALIDATION.md"
            ).read_text(),
            "docs/guides/SETUP_GUIDE.md": (_repo_root / "docs" / "guides" / "SETUP_GUIDE.md").read_text(),
            "docs/guides/TROUBLESHOOTING.md": (_repo_root / "docs" / "guides" / "TROUBLESHOOTING.md").read_text(),
            "docs/status/READY_TO_PROCESS.md": (_repo_root / "docs" / "status" / "READY_TO_PROCESS.md").read_text(),
            "pyproject.toml": (_repo_root / "pyproject.toml").read_text(),
            "requirements.txt": (_repo_root / "requirements.txt").read_text(),
            "Makefile": (_repo_root / "Makefile").read_text(),
            "scripts/analyze_flakes.py": (_repo_root / "scripts" / "analyze_flakes.py").read_text(),
            "scripts/README.md": (_repo_root / "scripts" / "README.md").read_text(),
            "scripts/setup/download_depth_models.py": (
                _repo_root / "scripts" / "setup" / "download_depth_models.py"
            ).read_text(),
            "scripts/setup/install_models.py": (_repo_root / "scripts" / "setup" / "install_models.py").read_text(),
            "scripts/setup/install_models_auto.py": (_repo_root / "scripts" / "setup" / "install_models_auto.py").read_text(),
            "scripts/setup/README.md": (_repo_root / "scripts" / "setup" / "README.md").read_text(),
        }
        combined = "\n".join(guidance.values())

        forbidden_fragments = [
            "pip install -r requirements.txt",
            "pip install transformers torch",
            "Real-ESRGAN (upscaling)",
            "git clone https://github.com/NVlabs/nvdiffrast && pip install -e .",
        ]
        for forbidden_fragment in forbidden_fragments:
            assert forbidden_fragment not in combined

        forbidden_ambient_python = [
            r"(?<![/\w.-])python scripts/analyze_flakes\.py",
            r"(?<![/\w.-])python scripts/setup/download_depth_models\.py",
            r"(?<![/\w.-])python scripts/setup/install_models\.py",
            r"(?<![/\w.-])python scripts/setup/install_models_auto\.py",
            r"(?<![/\w.-])python scripts/test_metadata_extraction\.py",
            r"(?<![/\w.-])python scripts/verification/verify_core\.py",
            r"(?<![/\w.-])python scripts/verification/verify_ml_deps\.py",
            r"(?<![/\w.-])python scripts/check_image_processing_readiness\.py",
            r"(?<![/\w.-])python scripts/simple_image_processor\.py",
            r"(?<![/\w.-])python scripts/validation/verify_3dgs_artifacts\.py",
            r"(?<![/\w.-])python -m transformation_portal\.",
            r"(?m)^python -m build$",
            r"(?m)^twine check dist/\*$",
            r"(?m)^pip install dist/\*\.whl --force-reinstall$",
            r"(?<!python -m )(?<![/\w.-])pip install -r requirements/security\.txt",
            r"(?<!python -m )(?<![/\w.-])pip install diff-cover",
            r"pylint --enable=security",
            r"(?m)^black --(?:check --)?line-length=127 src/ tests/$",
            r"(?m)^isort --(?:check-only --)?profile=black --line-length=127 src/ tests/$",
            r"(?m)^flake8 src/ tests/ --max-line-length=127$",
        ]
        for forbidden_pattern in forbidden_ambient_python:
            assert re.search(forbidden_pattern, combined) is None

        required_fragments = [
            "make install-core",
            "make repair-core-venv",
            ".venv/bin/python scripts/analyze_flakes.py",
            ".venv/bin/python scripts/setup/download_depth_models.py",
            ".venv/bin/python scripts/setup/install_models.py",
            ".venv/bin/python scripts/setup/install_models_auto.py",
            ".venv/bin/python scripts/test_metadata_extraction.py",
            ".venv/bin/python scripts/verification/verify_core.py",
            ".venv/bin/python scripts/verification/verify_ml_deps.py",
            ".venv/bin/python scripts/check_image_processing_readiness.py",
            ".venv/bin/python scripts/simple_image_processor.py",
            ".venv/bin/python scripts/validation/verify_3dgs_artifacts.py",
            ".venv/bin/python -m transformation_portal.lux_depth_v3 --help",
            ".venv/bin/python -m transformation_portal.lux_depth_v3.pbr_cli --help",
            ".venv/bin/python -m transformation_portal.cli depth-estimation",
            '.venv/bin/python -m pip install "git+https://github.com/RC219805/Transformation_Portal.git@<release-tag>"',
            ".venv/bin/python -m build",
            ".venv/bin/twine check dist/*",
            ".venv/bin/python -m pip install dist/*.whl --force-reinstall",
            ".venv/bin/python -m pip install -r requirements/security.txt",
            ".venv/bin/bandit -r src/ -ll",
            ".venv/bin/pip-audit",
            ".venv/bin/bandit -r src/",
            "make lint-parity",
            "make test-fast",
            "./scripts/setup/run_lint_tool.sh black src/ tests/",
            "./scripts/setup/run_lint_tool.sh isort src/ tests/",
            "Run 'make install-core' to install the repo-managed tooling",
            "git clone https://github.com/NVlabs/nvdiffrast external/nvdiffrast",
            ".venv/bin/python -m pip install -e external/nvdiffrast",
            "input_images/image.CR2",
            "external `realesrgan`",
        ]
        for required_fragment in required_fragments:
            assert required_fragment in combined

        assert "/input/image.CR2" not in combined

    def test_machine_mode_guidance_uses_repo_managed_python_and_local_paths(self):
        """Machine-mode examples should mirror the repo-managed local execution contract."""
        guidance = {
            "README.md": (_repo_root / "README.md").read_text(),
            "docs/api/MACHINE_MODE_CONTRACT.md": (_repo_root / "docs" / "api" / "MACHINE_MODE_CONTRACT.md").read_text(),
            "docs/quick_references/MACHINE_MODE_JSON.md": (
                _repo_root / "docs" / "quick_references" / "MACHINE_MODE_JSON.md"
            ).read_text(),
            "tools/parse_machine_json.py": (_repo_root / "tools" / "parse_machine_json.py").read_text(),
            "tools/parse_machine_json_examples.sh": (_repo_root / "tools" / "parse_machine_json_examples.sh").read_text(),
        }
        combined = "\n".join(guidance.values())

        forbidden_patterns = [
            r"(?<![/\w.-])python scripts/test_metadata_extraction\.py",
            r"(?<![/\w.-])python tools/parse_machine_json\.py",
            r"(?<![/\w.-])python parse_machine_json\.py",
        ]
        for pattern in forbidden_patterns:
            assert re.search(pattern, combined) is None

        forbidden_fragments = [
            "/input/image.CR2",
            "/input/IMG_1234.CR2",
            "/input/missing.CR2",
            '"/input"',
            '"/output"',
            "extract-batch /input",
            "validate /output/sidecar.json",
        ]
        for fragment in forbidden_fragments:
            assert fragment not in combined

        required_fragments = [
            ".venv/bin/python scripts/test_metadata_extraction.py",
            ".venv/bin/python tools/parse_machine_json.py",
            "input_images/image.CR2",
            "input_images/IMG_1234.CR2",
            "extract-batch input_images --output output",
            "validate output/sidecar.json",
        ]
        for fragment in required_fragments:
            assert fragment in combined

    def test_root_changelog_keeps_pr1562_snapshot_historical(self):
        """Root changelog should not present the PR #1562 snapshot as current state."""
        changelog = (_repo_root / "CHANGELOG.md").read_text()

        stale_current_phrases = [
            "Repository State Through PR #1562",
            "Current documentation and operator surfaces have been refreshed to match the April 27, 2026",
            "Live custom-agent, Copilot, and RAG-template instructions now align with the PR #1562",
        ]
        for phrase in stale_current_phrases:
            assert phrase not in changelog

        required_current_context = [
            "Documentation checkpoint through PR #1562",
            "Current documentation navigation is governed by `docs/README.md` and `docs/governance/DOCUMENTATION_MAP.md`",
            "May 11, 2026 repo-wide refresh through PR #1721",
            "May 12 architecture and CLI overlays",
            "## [3.4.0] - 2026-02-24",
            "Canonical evidence-bundle root anchoring",
            "Optional notarization validation support",
            "Governed ML Baseline Alignment",
            "torch: supported lock baseline `2.12.0`",
            "torchvision: paired supported lock baseline `0.27.0`",
            "diffusers: metadata floor `0.38.0`; supported lock baseline `0.38.0`",
            "transformers: metadata floor `5.0.0`; supported lock baseline `5.0.0`",
            "## [2.0.0] - 2026-01-02",
            "[Unreleased]: https://github.com/RC219805/Transformation_Portal/compare/v3.4.0...HEAD",
            "[3.4.0]: https://github.com/RC219805/Transformation_Portal/releases/tag/v3.4.0",
        ]
        for phrase in required_current_context:
            assert phrase in changelog
        assert "## [2.0.0] - 2025-11-14" not in changelog
        assert "compare/v2.0.0...HEAD" not in changelog
        assert "torch: 2.4.1" not in changelog
        assert "torchvision: 0.19.1" not in changelog
        assert "diffusers: 0.31.0" not in changelog
        assert "transformers: 4.53.0" not in changelog
        assert "docs/guides/coverage-improvement-plan.md" not in changelog
        assert "33% by Q2 2026" not in changelog
        assert "docs/testing/test_coverage_improvement_plan.md" in changelog
        assert "docs/testing/COLD_ZONE_COVERAGE_PROGRAM.md" in changelog

    def test_root_readme_performance_links_use_current_authorities(self):
        """README performance navigation should point to maintained policy docs."""
        readme = (_repo_root / "README.md").read_text()

        assert "docs/decisions/ADR-024-performance-regression-authority-canonicalization.md" not in readme
        assert "docs/guides/APEX_REAL_PIPELINE_INTEGRATION.md" not in readme
        required_links = [
            "docs/performance/README.md",
            "docs/apex/APEX_CONTRACT.md",
            "docs/performance/GATE_POLICY.md",
        ]
        for relative_path in required_links:
            assert relative_path in readme
            assert (_repo_root / relative_path).exists()

    def test_root_readme_machine_mode_links_use_current_contracts(self):
        """README machine-mode navigation should avoid mixed quick-reference docs."""
        readme = (_repo_root / "README.md").read_text()

        assert "docs/quick_references/MACHINE_MODE_JSON.md" not in readme
        assert "docs/api/MACHINE_MODE_CONTRACT.md" in readme
        assert (_repo_root / "docs/api/MACHINE_MODE_CONTRACT.md").exists()

    def test_security_policy_footer_tracks_current_root_policy_update(self):
        """Root security policy metadata should track current policy edits."""
        security_policy = (_repo_root / "SECURITY.md").read_text()

        assert "*Last Updated: March 2026*" not in security_policy
        assert "*Next Review: June 2026*" not in security_policy
        assert "*Last Updated: 2026-06-03*" in security_policy
        assert "*Next Review: 2026-09-03*" in security_policy
        assert "*Security Policy Version: 1.2*" in security_policy

    def test_security_policy_supported_versions_track_release_channels(self):
        """Root security policy should not name retired 0.x lines as current stable."""
        security_policy = (_repo_root / "SECURITY.md").read_text()

        stale_version_claims = [
            "| 0.1.x",
            "| < 0.1",
            "Current stable release",
        ]
        for stale_claim in stale_version_claims:
            assert stale_claim not in security_policy

        required_policy_fragments = [
            "release channels are currently supported with security updates",
            "| main    | :white_check_mark: | Active development branch; security fixes prioritized |",
            "| Latest semantic product release tag | :white_check_mark: | Supported for security updates until superseded by a newer semantic product release tag |",
            "| Older release tags | :x: | Unsupported unless an explicit security advisory or maintenance branch says otherwise |",
        ]
        for policy_fragment in required_policy_fragments:
            assert policy_fragment in security_policy

    def test_security_policy_resources_point_to_current_authorities(self):
        """Root security resources should avoid historical or mixed guidance."""
        security_policy = (_repo_root / "SECURITY.md").read_text()

        stale_links = [
            "docs/guides/BEST_PRACTICES.md",
            "docs/version_history/changelog.md",
        ]
        for stale_link in stale_links:
            assert stale_link not in security_policy

        required_links = [
            "CONTRIBUTING.md",
            "CHANGELOG.md",
            "docs/architecture/ARCHITECTURE.md",
        ]
        for relative_path in required_links:
            assert relative_path in security_policy
            assert (_repo_root / relative_path).exists()

    def test_security_policy_deployment_guidance_uses_repo_runtime_contract(self):
        """Root security deployment guidance should not advertise stale runtime commands."""
        security_policy = (_repo_root / "SECURITY.md").read_text()

        stale_fragments = [
            "sudo -u nobody python -m transformation_portal.cli",
            "# User=nobody",
            "# Group=nogroup",
        ]
        for stale_fragment in stale_fragments:
            assert stale_fragment not in security_policy

        required_fragments = [
            "sudo -u tp .venv/bin/python -m transformation_portal.cli serve --host 127.0.0.1 --port 8000",
            "# User=tp",
            "# Group=tp",
        ]
        for required_fragment in required_fragments:
            assert required_fragment in security_policy

    def test_retired_pygments_exception_stays_removed_from_security_scans(self):
        """The root security policy and CI scanners should agree on retired CVE exceptions."""
        security_policy = (_repo_root / "SECURITY.md").read_text()
        lockfiles = [
            "requirements/base.txt",
            "requirements/ci.txt",
            "requirements/security.txt",
        ]
        workflows = [
            ".github/workflows/ci.yml",
            ".github/workflows/ci-quality-firewall.yml",
            ".github/workflows/security-unified.yml",
            ".github/workflows/dependency-update.yml",
            ".github/workflows/nightly.yml",
        ]

        for relative_path in lockfiles:
            assert "pygments==2.20.0" in (_repo_root / relative_path).read_text()

        forbidden_fragments = [
            "--ignore-vuln CVE-2026-4539",
            "CVE-2026-4539 (pygments): No fix available yet",
            "No upstream fix available as of March 2026",
        ]
        scanned_text = security_policy + "\n" + "\n".join((_repo_root / path).read_text() for path in workflows)
        for fragment in forbidden_fragments:
            assert fragment not in scanned_text

        assert "Pygments==2.20.0" in security_policy
        assert "None active. New exceptions require an explicit expiry condition" in security_policy

    def test_security_baseline_versions_match_current_locks(self):
        """Root security guidance should describe the current governed lock baselines."""
        security_policy = (_repo_root / "SECURITY.md").read_text()
        requirements_ci = (_repo_root / "requirements-ci.txt").read_text()
        requirements_dev = (_repo_root / "requirements-dev.txt").read_text()
        requirements_lint = (_repo_root / "requirements-lint.txt").read_text()
        contributing = (_repo_root / "CONTRIBUTING.md").read_text()

        assert "cryptography==47.0.0" in (_repo_root / "requirements/ci.txt").read_text()
        assert "cryptography==47.0.0" in (_repo_root / "requirements/dev.txt").read_text()
        assert "cryptography==47.0.0" in (_repo_root / "requirements/all.txt").read_text()
        assert "pillow==12.2.0" in (_repo_root / "requirements/base.txt").read_text()

        assert "**cryptography==47.0.0**" in security_policy
        assert "**cryptography==46.0.5**" not in security_policy
        assert "cryptography>=47.0.0,<48" in requirements_ci
        assert "cryptography>=47.0.0,<48" in requirements_dev
        assert "cryptography>=46.0,<48" not in requirements_ci
        assert "cryptography>=46.0,<48" not in requirements_dev
        assert "`cryptography`          | >=47.0.0" in contributing
        assert "`cryptography`          | >=46.0.5" not in contributing
        assert "`starlette`             | >=1.0.1" in contributing
        assert "Starlette==1.0.1" in security_policy
        assert "current governed lock baseline is pillow==12.2.0" in requirements_lint
        assert "allows pillow==12.1.1 in lockfiles" not in requirements_lint

    def test_pyproject_security_extras_track_current_cryptography_floor(self):
        """Public optional metadata should not expose retired cryptography floors."""
        pyproject = tomllib.loads((_repo_root / "pyproject.toml").read_text())
        optional_dependencies = pyproject["project"]["optional-dependencies"]
        tools_archive_in = (_repo_root / "requirements" / "tools-archive.in").read_text()
        tools_archive_lock = (_repo_root / "requirements" / "tools-archive.txt").read_text()

        for extra_name in ("archive-signing", "dev"):
            assert "cryptography>=47.0.0,<48" in optional_dependencies[extra_name]
            assert "cryptography>=46.0,<48" not in optional_dependencies[extra_name]

        assert "cryptography>=47.0.0,<48" in tools_archive_in
        assert "cryptography>=46.0,<48" not in tools_archive_in
        assert "cryptography==47.0.0" in tools_archive_lock

    def test_pyproject_core_dependencies_track_governed_base_requirements(self):
        """Installable package metadata should not trail the governed core dependency surface."""
        pyproject = tomllib.loads((_repo_root / "pyproject.toml").read_text())
        dependencies = set(pyproject["project"]["dependencies"])
        contributing = (_repo_root / "CONTRIBUTING.md").read_text()

        required_dependency_ranges = {
            "Pillow>=10.3.0,<13",
            "scikit-learn>=1.8.0,<2",
            "fastapi>=0.136.1,<0.137",
            "starlette>=1.0.1,<1.1",
            "uvicorn>=0.48.0,<0.49",
            "aiofiles>=25.1.0,<26",
            "SQLAlchemy[asyncio]>=2.0.50,<2.2",
            "asyncpg>=0.29,<1",
            "alembic>=1.13,<2",
            "redis>=5.0,<7",
        }
        for dependency in required_dependency_ranges:
            assert dependency in dependencies

        stale_dependency_ranges = {
            "Pillow>=10.0.0,<13",
            "scikit-learn>=1.0,<2",
            "fastapi>=0.121.0,<0.137",
            "starlette>=0.49.1,<1.1",
            "uvicorn>=0.29.0,<0.49",
            "aiofiles>=23.2.1,<26",
        }
        for dependency in stale_dependency_ranges:
            assert dependency not in dependencies

        assert "`Pillow`                | >=10.3.0" in contributing
        assert "`Pillow`                | >=10.0.0" not in contributing

    def test_pyproject_dev_extra_tracks_governed_dev_requirements(self):
        """The public dev extra should not silently trail requirements/dev.in."""
        pyproject = tomllib.loads((_repo_root / "pyproject.toml").read_text())
        dev_extra = pyproject["project"]["optional-dependencies"]["dev"]
        requirements_dev_in = (_repo_root / "requirements" / "dev.in").read_text().splitlines()
        requirements_dev = (_repo_root / "requirements-dev.txt").read_text()
        contributing = (_repo_root / "CONTRIBUTING.md").read_text()

        def package_names(lines: list[str]) -> set[str]:
            names: set[str] = set()
            for line in lines:
                normalized_line = line.strip()
                if not normalized_line or normalized_line.startswith(("#", "-r")):
                    continue
                match = re.match(r"^([a-zA-Z0-9._-]+)", normalized_line)
                if match:
                    names.add(match.group(1).lower().replace("_", "-").replace(".", "-"))
            return names

        governed_dev_packages = package_names(requirements_dev_in)
        metadata_dev_packages = package_names(dev_extra)

        assert governed_dev_packages <= metadata_dev_packages
        assert "black>=26.3.1" in dev_extra
        assert "black>=24.8" not in dev_extra
        assert "black>=26.3.1" in requirements_dev
        assert "black>=24.8" not in contributing
        assert "mypy>=1.10  # Type checker" in contributing

        adr_032 = (_repo_root / "docs" / "architecture" / "ADR-032-dependency-pinning-strategy.md").read_text()
        assert "black>=24.8" not in adr_032
        assert "| `black`       | dev.in  | `>=24.8`" not in adr_032
        assert "| `black`       | dev.in  | `>=26.3.1`" in adr_032
        assert "pylint>=3.0" in adr_032

    def test_contributing_dependency_audit_schedule_is_current(self):
        """Canonical contribution guidance should not point to a past audit date."""
        contributing = (_repo_root / "CONTRIBUTING.md").read_text()

        assert "Next audit: **2026-05-16 (Q2 2026)**" not in contributing
        assert "Next audit: **2026-08-16 (Q3 2026)**" in contributing

    def test_contributing_coverage_guidance_matches_active_gates(self):
        """Root contribution guidance should match current coverage gate policy."""
        contributing = (_repo_root / "CONTRIBUTING.md").read_text()
        testing_plan = (_repo_root / "docs" / "testing" / "test_coverage_improvement_plan.md").read_text()
        guide_plan = (_repo_root / "docs" / "guides" / "coverage-improvement-plan.md").read_text()

        assert "--cov-fail-under=30" in contributing
        assert "**New/changed lines should target 85%+ covered**" in contributing
        assert "required PR CI currently enforces the" in contributing
        assert "`build.yml`, not `diff-cover`" in contributing
        assert "Coverage Gates (ENFORCED)" not in contributing
        assert "Enforced via `diff-cover` tool" not in contributing
        assert "| **Diff coverage** | diff-cover | ≥85% |" not in testing_plan
        assert "### Local / Review Targets (non-blocking today)" in testing_plan
        assert "Wire diff coverage into the required `build.yml` PR gate" in testing_plan
        assert "Required PR CI may still pass today" in guide_plan
        assert "local target / future PR gate" in guide_plan
        assert "docs/testing/test_coverage_improvement_plan.md" in contributing
        assert "docs/testing/COLD_ZONE_COVERAGE_PROGRAM.md" in contributing
        assert "**New/changed lines must be 80%+ covered**" not in contributing
        assert "80% diff coverage" not in contributing
        assert "Q2 2026 target: 28%" not in contributing

    def test_contributing_dependency_workflow_matches_requirements_contract(self):
        """Root contribution guidance should not route dependency edits through retired ML flows."""
        contributing = (_repo_root / "CONTRIBUTING.md").read_text()
        requirements_readme = (_repo_root / "requirements" / "README.md").read_text()
        requirements_ml = (_repo_root / "requirements" / "ml.in").read_text()
        requirements_ml_coreml = (_repo_root / "requirements" / "ml-coreml.in").read_text()
        requirements_ml_cpu = (_repo_root / "requirements" / "ml-cpu.in").read_text()
        requirements_ml_cuda = (_repo_root / "requirements" / "ml-cuda.in").read_text()
        requirements_ml_raw = (_repo_root / "requirements" / "ml-raw.in").read_text()
        requirements_ml_research = (_repo_root / "requirements" / "ml-research.in").read_text()

        stale_fragments = [
            "vim requirements/ml.in         # Optional ML/AI deps",
            "# 3. Recompile all .txt files",
            "git add requirements/*.in requirements/*.txt",
        ]
        for stale_fragment in stale_fragments:
            assert stale_fragment not in contributing

        required_fragments = [
            "vim requirements/ml-core.in",
            "vim requirements/ml-core-darwin-arm64.in",
            "make -C requirements compile",
            "make -C requirements compile-ml-darwin-arm64",
            "./scripts/validate_dependency_constraints.sh",
        ]
        for required_fragment in required_fragments:
            assert required_fragment in contributing

        assert "compile           Compile generic checked-in lockfiles only" in requirements_readme
        assert "compile-ml-darwin-arm64    Compile the Darwin arm64 ML lock" in requirements_readme
        assert "ml-cuda.in              # Retired unsupported CUDA lane stub" in requirements_readme
        assert "linux-x86_64-cuda` (retired unsupported CUDA lane; `core-cuda` fails closed)" in requirements_readme
        assert "Do not install CUDA PyTorch packages ad hoc into the repo .venv" in requirements_readme
        assert "PYTORCH_INDEX=https://download.pytorch.org/whl/cu121" not in requirements_readme
        assert "No checked-in umbrella ML lock is generated from this file" in requirements_ml
        assert "make install-ml-raw     # disabled: no trusted RAW lock contract" in requirements_ml
        assert "make install-ml         # disabled: no trusted umbrella lock contract" in requirements_ml
        assert "Run `make compile` in this directory to generate pinned ml.txt" not in requirements_ml
        assert "make install-ml         # ml-core + ml-raw (this umbrella)" not in requirements_ml
        assert "torch==2.12.0, torchvision==0.27.0, open-clip-torch==3.3.0" in requirements_ml
        assert "target-owned locks are install support promises" in requirements_ml
        assert "torch==2.8.0, torchvision==0.23.0" not in requirements_ml
        assert "CUDA is a retired unsupported lane" in requirements_ml_cpu
        assert "GPU-specific packages are in ml-cuda.in" not in requirements_ml_cpu
        assert "Retired unsupported CUDA ML lane" in requirements_ml_cuda
        assert "core-cuda` in scripts/bootstrap/install_ml_stack.sh fails closed" in requirements_ml_cuda
        assert "nvidia-cublas-cu12" not in requirements_ml_cuda
        assert "triton ;" not in requirements_ml_cuda
        assert "`make install-ml-raw` target fails closed" in requirements_ml_raw
        assert "Install via: make install-ml-raw" not in requirements_ml_raw
        assert "`make install-ml-coreml` fails closed unless a trusted CoreML lock exists" in requirements_ml_coreml
        assert "Install via: make install-ml-coreml" not in requirements_ml_coreml
        assert "Reserved metadata only" in requirements_ml_research
        assert "make install-ml-research" not in requirements_ml_research

    def test_root_requirements_guidance_preserves_target_owned_ml_install_contract(self):
        """Root requirements entrypoint should not advertise metadata extras as ML install support."""
        root_requirements = (_repo_root / "requirements.txt").read_text()
        pyproject = (_repo_root / "pyproject.toml").read_text()

        assert "make install-ml-core" in root_requirements
        assert "requirements/README.md" in root_requirements
        assert "pyproject.toml extras are metadata pointers" in root_requirements
        assert "not the governed operator install surface" in root_requirements
        assert "or use pyproject.toml extras" not in root_requirements
        assert "pyproject.toml extras are metadata/convenience pointers, not install support" in pyproject

    def test_contributing_setup_uses_repo_managed_environment(self):
        """Root contribution setup should use the current Makefile-managed .venv contract."""
        contributing = (_repo_root / "CONTRIBUTING.md").read_text()

        stale_fragments = [
            "python3.11 -m venv venv",
            "source venv/bin/activate",
            "pip install -r requirements-dev.txt\npip install -e .",
            "pytest --version\npython -c",
        ]
        for stale_fragment in stale_fragments:
            assert stale_fragment not in contributing

        required_fragments = [
            "make venv",
            "make install-core",
            "make install-hooks",
            "make ci-quick",
            "repo-managed .venv with Python 3.11+",
        ]
        for required_fragment in required_fragments:
            assert required_fragment in contributing

    def test_setup_guide_tracks_current_ml_runtime_contract(self):
        """Root-linked setup guidance should not advertise retired ML lanes."""
        setup_guide = (_repo_root / "docs" / "guides" / "SETUP_GUIDE.md").read_text()

        stale_fragments = [
            "Linux + NVIDIA only",
            "Linux with NVIDIA GPU:** use",
            "CPU only (macOS/Linux)",
            "currently supported on macOS and Linux only",
            "pip install transformers huggingface-hub",
            "pip install coremltools",
            "Install CUDA-enabled PyTorch",
            "Use mixed precision",
        ]
        for stale_fragment in stale_fragments:
            assert stale_fragment not in setup_guide

        required_fragments = [
            "checked-in ML core lock is target-owned for macOS Apple Silicon",
            "Linux and macOS Intel ML lanes are retired unsupported lanes",
            "core-cuda` fails closed",
            "Do not install CUDA PyTorch packages ad hoc into the repo `.venv`",
            "./scripts/setup/install_da3_runtime.sh",
            "./scripts/setup/install_depth_pro_runtime.sh",
        ]
        for required_fragment in required_fragments:
            assert required_fragment in setup_guide

    def test_contributing_branch_protection_links_use_current_setup_doc(self):
        """Contribution guidance should not route to historical verification reports."""
        contributing = (_repo_root / "CONTRIBUTING.md").read_text()

        assert "docs/governance/BRANCH_PROTECTION_VERIFICATION.md" not in contributing
        assert "docs/ci/BRANCH_PROTECTION_SETUP.md" in contributing
        assert (_repo_root / "docs/ci/BRANCH_PROTECTION_SETUP.md").exists()

    def test_architect_directive_status_points_to_current_authorities(self):
        """Root architect metadata should not masquerade as live PR/CI status."""
        status_path = _repo_root / ".architect_directive_status.yml"
        content = status_path.read_text()

        assert 'status: "SUPERSEDED"' in content
        assert "binding: false" in content

        stale_live_claims = [
            "PR #822",
            "PR #823",
            "ARCHITECT_RESPONSE_SUMMARY.md",
            "ci_status:",
            "open_prs:",
            "projected_completion:",
        ]
        for stale_claim in stale_live_claims:
            assert stale_claim not in content

        required_authorities = [
            "AGENTS.md",
            "docs/governance/DOCUMENTATION_MAP.md",
            "docs/architecture/ARCHITECTURE_CLEANUP_BOARD.md",
            "docs/architecture/agent_governance.md",
            "docs/ci/TYPE_CHECKING_POLICY.md",
        ]
        for relative_path in required_authorities:
            assert relative_path in content
            assert (_repo_root / relative_path).exists()

    def test_root_branch_protection_guidance_matches_current_contract(self):
        """Root branch-protection prose should not preserve stale setup snapshots."""
        contributing = (_repo_root / "CONTRIBUTING.md").read_text()
        setup_doc = (_repo_root / "docs" / "ci" / "BRANCH_PROTECTION_SETUP.md").read_text()
        historical_commands_doc = (_repo_root / "docs" / "ci" / "BRANCH_PROTECTION_COMMANDS.md").read_text()

        required_current_claims = [
            "Current Branch Protection Rules (Verified 2026-06-03)",
            '"enforce_admins": true',
            '"required_linear_history": false',
            '"required_conversation_resolution": true',
            "Linear history**: Not required by branch protection",
            "Require conversation resolution**: ✅ Enabled",
            "**Resolve all review conversations** (required)",
        ]
        for claim in required_current_claims:
            assert claim in contributing

        stale_claims = [
            "Verified 2026-02-10",
            '"enforce_admins": false',
            '"required_linear_history": true',
            "Require conversation resolution**: ❌ Not enabled",
            "Linear history**: ✅ Required",
            "**Resolve all review conversations** (recommended)",
        ]
        for stale_claim in stale_claims:
            assert stale_claim not in contributing

        assert "Last verified: 2026-06-03." in setup_doc
        assert "CI Gate" in setup_doc
        assert "lint (3.12)" not in setup_doc
        assert "test-core (3.10)" not in setup_doc
        assert "quality-summary" not in setup_doc
        assert "Admin enforcement is currently enabled" in setup_doc
        assert "[CONTRIBUTING.md](../../CONTRIBUTING.md)" in setup_doc
        assert "[Production Readiness](../deployment/PRODUCTION_READINESS.md)" in setup_doc
        assert "[CONTRIBUTING.md](../CONTRIBUTING.md)" not in setup_doc
        assert "[Production Readiness](./PRODUCTION_READINESS.md)" not in setup_doc

        assert "Historical record. Do not run these commands as current configuration." in historical_commands_doc
        assert "[BRANCH_PROTECTION_SETUP.md](./BRANCH_PROTECTION_SETUP.md)" in historical_commands_doc
        assert "[CI Workflow](../../.github/workflows/build.yml)" in historical_commands_doc
        assert "[CONTRIBUTING.md](../../CONTRIBUTING.md)" in historical_commands_doc
        assert "[CODE_QUALITY_SYSTEM.md](../guides/CODE_QUALITY_SYSTEM.md)" in historical_commands_doc
        assert "[CI Workflow](../.github/workflows/ci.yml)" not in historical_commands_doc
        assert "[CONTRIBUTING.md](../CONTRIBUTING.md)" not in historical_commands_doc
        assert "[CODE_QUALITY_SYSTEM.md](./CODE_QUALITY_SYSTEM.md)" not in historical_commands_doc

    def test_production_readiness_guidance_tracks_current_runtime_and_gate_contract(self):
        """Deployment readiness guidance should not advertise retired runtime or branch-protection state."""
        readiness_doc = (_repo_root / "docs" / "deployment" / "PRODUCTION_READINESS.md").read_text()

        assert "Python 3.11 and 3.12; Python 3.10 is retired" in readiness_doc
        assert "Python 3.11+; the supported CI matrix covers Python 3.11 and 3.12" in readiness_doc
        assert "CI Gate" in readiness_doc
        assert "Branch protection: ✅" in readiness_doc
        assert "Branch protection: 📝 Needs configuration" not in readiness_doc
        assert "Python 3.10, 3.11, 3.12" not in readiness_doc
        assert "Python 3.10+" not in readiness_doc
        assert "make test-fast" in readiness_doc
        assert "make ci-quick" in readiness_doc
        assert "[CONTRIBUTING.md](../../CONTRIBUTING.md)" in readiness_doc
        assert "[SECURITY.md](../../SECURITY.md)" in readiness_doc

    def test_ci_hygiene_checks_cover_split_coverage_artifacts(self):
        """CI hygiene checks should track every ignored root coverage artifact."""
        gitignore = (_repo_root / ".gitignore").read_text()
        workflows = [
            ".github/workflows/ci.yml",
            ".github/workflows/ci-quality-firewall.yml",
        ]
        required_patterns = [
            ".coverage",
            ".coverage.*",
            "coverage.json",
            "htmlcov/",
        ]

        for pattern in required_patterns:
            assert pattern in gitignore

        for relative_path in workflows:
            workflow_text = (_repo_root / relative_path).read_text()
            match = re.search(r'patterns=\("(?P<patterns>[^"]+(?:" "[^"]+)*)"\)', workflow_text)
            assert match is not None, f"{relative_path} must define root hygiene .gitignore coverage patterns"
            workflow_patterns = set(re.findall(r'"([^"]+)"', match.group(0)))
            for pattern in required_patterns:
                assert pattern in workflow_patterns, f"{relative_path} must check .gitignore coverage for {pattern}"

    def test_root_guidance_make_commands_are_defined(self):
        """Root guidance should not route operators to nonexistent Make targets."""
        root_targets = self._make_targets(_repo_root / "Makefile")
        requirements_targets = self._make_targets(_repo_root / "requirements" / "Makefile")

        root_guides = [
            "AGENTS.md",
            "CLAUDE.md",
            "README.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
        ]
        missing_targets = []

        for relative_path in root_guides:
            guide_path = _repo_root / relative_path
            in_fenced_block = False

            for line_number, line in enumerate(guide_path.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_fenced_block = not in_fenced_block
                    continue

                command_context = (
                    in_fenced_block
                    or "`make " in line
                    or stripped.startswith("make ")
                    or "&& make " in line
                    or "/ make " in line
                )
                if not command_context:
                    continue

                for match in re.finditer(
                    r"(?<![A-Za-z0-9_.-])make(?:\s+-C\s+([A-Za-z0-9_./-]+))?\s+([A-Za-z0-9_.%/+@-]+)",
                    line,
                ):
                    make_directory = match.group(1)
                    target = match.group(2)
                    uses_requirements_makefile = make_directory == "requirements" or "cd requirements" in line
                    target_source = requirements_targets if uses_requirements_makefile else root_targets
                    if target not in target_source:
                        missing_targets.append(f"{relative_path}:{line_number}: make {target}")

        assert not missing_targets, "Root guidance references undefined Make targets:\n" + "\n".join(sorted(missing_targets))

    def test_makefile_help_matches_quality_targets(self):
        """Makefile help should describe the actual quality target boundaries."""
        makefile = (_repo_root / "Makefile").read_text()

        assert "quality-check: lint validate-ci" in makefile
        assert "Run all quality checks (lint + structure + tests)" not in makefile
        assert "quality-check      Run lint, workflow validation, and root placement checks" in makefile
        assert "check-quality" in self._make_targets(_repo_root / "Makefile")
        assert "fix-quality check-quality validate-ci" in makefile
        assert "check-quality      Dry-run common quality issue fixes" in makefile

    def test_docs_build_tooling_uses_bounded_repo_contract(self):
        """Local and hosted docs builds should share the same bounded Sphinx lane."""
        makefile = (_repo_root / "Makefile").read_text()
        requirements_dev = (_repo_root / "requirements-dev.txt").read_text()
        docs_workflow = (_repo_root / ".github" / "workflows" / "docs.yml").read_text()

        docs_specs = [
            "sphinx>=7.2,<9",
            "sphinx-rtd-theme>=2.0,<3",
            "sphinx-autodoc-typehints>=2.0,<4",
        ]
        expected_make_var = (
            'DOCS_TOOL_REQUIREMENTS := "sphinx>=7.2,<9" ' '"sphinx-rtd-theme>=2.0,<3" "sphinx-autodoc-typehints>=2.0,<4"'
        )

        assert expected_make_var in makefile
        assert '"$(PY)" -m pip install -q $(DOCS_TOOL_REQUIREMENTS)' in makefile
        assert "pip install -q sphinx sphinx-rtd-theme sphinx-autodoc-typehints" not in makefile

        for spec in docs_specs:
            assert spec in requirements_dev
            assert f'"{spec}"' in docs_workflow

        stale_specs = [
            "sphinx>=7.2,<10",
            "sphinx-rtd-theme>=3.1.0,<4",
            "sphinx-autodoc-typehints>=3.6.1,<4",
        ]
        for spec in stale_specs:
            assert spec not in requirements_dev
            assert spec not in docs_workflow

        assert re.search(r"(?m)^\s+pip install ", docs_workflow) is None


class TestReferenceQuickstartGuidance:
    """Tests for active quick-start reference guidance."""

    STALE_OPERATOR_GUIDANCE_FRAGMENTS = [
        "pip install -r requirements.txt",
        'pip install -e ".[all]"',
        'pip install -e ".[ml]"',
        'pip install -e ".[tiff]"',
        "pip install realesrgan",
        "from depth_pipeline import ArchitecturalDepthPipeline",
        "from lux_render_pipeline import",
        "python examples/simple_process.py",
        "python luxury_tiff_batch_processor_cli.py",
        "python lux_render_pipeline.py",
        "python luxury_video_master_grader.py",
        "python pro_pipeline.py",
        "python scripts/pipelines/lux_render_pipeline.py",
        "python scripts/context_aware_rendering.py",
        "python scripts/utilities/luxury_tiff_batch_processor.py",
    ]

    def test_quickstart_cheatsheet_uses_current_repo_managed_entrypoints(self):
        """Reference quick-start guidance should not route operators to retired root CLIs."""
        cheat_sheet = (_repo_root / "docs" / "reference" / "QUICKSTART_CHEATSHEET.md").read_text()

        stale_fragments = [
            *self.STALE_OPERATOR_GUIDANCE_FRAGMENTS,
            "[README.md](../README.md)",
            "[DEPTH_PIPELINE_README.md](../DEPTH_PIPELINE_README.md)",
        ]
        for stale_fragment in stale_fragments:
            assert stale_fragment not in cheat_sheet

        required_fragments = [
            "make venv",
            "make install-core",
            "make check-environment",
            ".venv/bin/lux-depth-v3",
            ".venv/bin/luxury-tiff-batch",
            ".venv/bin/lux_render",
            "--input-glob",
            ".venv/bin/luxury_video_grader",
            "scripts/check_image_processing_readiness.py",
            "scripts/simple_image_processor.py",
            "../guides/SETUP_GUIDE.md",
            "../cli/LUX_DEPTH_V3_CLI_GUIDE.md",
            "Linux and macOS Intel ML lanes are retired unsupported lanes",
        ]
        for required_fragment in required_fragments:
            assert required_fragment in cheat_sheet

    def test_pipeline_operations_guide_uses_current_repo_managed_entrypoints(self):
        """Active operations guidance should use maintained CLI and setup contracts."""
        operations_guide = (_repo_root / "docs" / "pipeline_docs" / "PIPELINE_OPERATIONS_GUIDE.md").read_text()

        for stale_fragment in self.STALE_OPERATOR_GUIDANCE_FRAGMENTS:
            assert stale_fragment not in operations_guide

        required_fragments = [
            "make venv",
            "make install-core",
            "make check-environment",
            ".venv/bin/python scripts/check_image_processing_readiness.py",
            ".venv/bin/python scripts/simple_image_processor.py",
            ".venv/bin/lux-depth-v3",
            "--model-key da3-metric",
            ".venv/bin/luxury-tiff-batch",
            ".venv/bin/lux_render",
            "--input-glob",
            ".venv/bin/luxury_video_grader",
            "./scripts/setup/install_da3_runtime.sh --profile baseline",
            "./scripts/setup/install_depth_pro_runtime.sh",
            "./scripts/setup/install_raw_runtime.sh",
            "./scripts/setup/install_fastvlm_runtime.sh",
            "Linux and macOS Intel ML lanes are retired unsupported lanes",
            "../cli/LUX_DEPTH_V3_CLI_GUIDE.md",
            "../governance/DOCUMENTATION_MAP.md",
        ]
        for required_fragment in required_fragments:
            assert required_fragment in operations_guide

    def test_scripts_reference_describes_governed_script_topology(self):
        """Script reference guidance should describe current topology instead of stale root scripts."""
        scripts_reference = (_repo_root / "docs" / "reference" / "SCRIPTS_REFERENCE.md").read_text()

        for stale_fragment in self.STALE_OPERATOR_GUIDANCE_FRAGMENTS:
            assert stale_fragment not in scripts_reference

        stale_inventory_claims = [
            "Complete Script Inventory",
            "Total Scripts",
            "Real-ESRGAN 4x upscaling",
            "python script.py",
            "scripts/install_modules.py",
        ]
        for stale_fragment in stale_inventory_claims:
            assert stale_fragment not in scripts_reference

        required_fragments = [
            "scripts/setup/",
            "scripts/pipelines/",
            "scripts/utilities/",
            "scripts/validation/",
            "scripts/governance/",
            "src/transformation_portal/",
            "archive/scripts/legacy-organization/",
            ".venv/bin/lux-depth-v3",
            ".venv/bin/luxury-tiff-batch",
            ".venv/bin/lux_render",
            ".venv/bin/luxury_video_grader",
            "python3 scripts/governance/check_script_topology.py --verbose",
            "./.auto-organize.sh --check --verbose",
            "raise SystemExit(main())",
            "../governance/DOCUMENTATION_MAP.md",
        ]
        for required_fragment in required_fragments:
            assert required_fragment in scripts_reference

    def test_supported_file_formats_guide_uses_current_entrypoints(self):
        """Maintained format guidance should route through current processing surfaces."""
        supported_formats = (_repo_root / "docs" / "guides" / "SUPPORTED_FILE_FORMATS.md").read_text()

        stale_fragments = [
            *self.STALE_OPERATOR_GUIDANCE_FRAGMENTS,
            "python depth_pipeline/pipeline.py",
            "python material_response.py",
            "board_material_aerial_enhancer.py",
            "pip install Pillow",
            "pip install tifffile",
            "Included in requirements.txt",
            "[README.md](README.md)",
            "[DEPTH_PIPELINE_README.md](DEPTH_PIPELINE_README.md)",
        ]
        for stale_fragment in stale_fragments:
            assert stale_fragment not in supported_formats

        required_fragments = [
            ".venv/bin/python scripts/simple_image_processor.py",
            ".venv/bin/lux-depth-v3",
            "--model-key da3-metric",
            ".venv/bin/luxury-tiff-batch",
            ".venv/bin/lux_render",
            "--input-glob",
            ".venv/bin/luxury_video_grader",
            "./scripts/setup/install_raw_runtime.sh",
            "transformation_portal.utils.format_utils",
            "make install-core",
            "python3 scripts/governance/check_docs_structure.py --all",
            "../pipeline_docs/PIPELINE_OPERATIONS_GUIDE.md",
            "../governance/DOCUMENTATION_MAP.md",
        ]
        for required_fragment in required_fragments:
            assert required_fragment in supported_formats

    def test_format_sidecar_guides_use_current_entrypoints(self):
        """Format overview and quick reference should align with the maintained format authority."""
        guide_paths = [
            _repo_root / "docs" / "guides" / "FORMAT_SUPPORT_OVERVIEW.md",
            _repo_root / "docs" / "guides" / "FILE_FORMAT_QUICK_REFERENCE.md",
        ]

        for guide_path in guide_paths:
            content = guide_path.read_text()
            stale_fragments = [
                *self.STALE_OPERATOR_GUIDANCE_FRAGMENTS,
                "python depth_pipeline/pipeline.py",
                "python material_response.py",
                "pip install Pillow",
                "pip install tifffile",
                "pip install -e .",
                "requirements.txt",
                "from format_utils import",
                "[README.md](../README.md)",
                "[DEPTH_PIPELINE_README.md]",
            ]
            for stale_fragment in stale_fragments:
                assert stale_fragment not in content, f"{guide_path.name}: {stale_fragment}"

            required_fragments = [
                "make install-core",
                ".venv/bin/python examples/validate_file_formats.py",
                ".venv/bin/lux-depth-v3",
                ".venv/bin/luxury-tiff-batch",
                ".venv/bin/lux_render",
                "--input-glob",
                ".venv/bin/luxury_video_grader",
                "transformation_portal.utils.format_utils",
                "../governance/DOCUMENTATION_MAP.md",
            ]
            for required_fragment in required_fragments:
                assert required_fragment in content, f"{guide_path.name}: {required_fragment}"

    def test_image_processing_readiness_guide_uses_current_entrypoints(self):
        """Image-readiness guidance should mirror maintained processing and setup surfaces."""
        readiness_guide = (_repo_root / "docs" / "guides" / "IMAGE_PROCESSING_READINESS.md").read_text()

        stale_fragments = [
            *self.STALE_OPERATOR_GUIDANCE_FRAGMENTS,
            "pip install numpy Pillow",
            "pip install scipy",
            "pip install tifffile",
            "pip install torch",
        ]
        for stale_fragment in stale_fragments:
            assert stale_fragment not in readiness_guide

        raw_python_script_commands = re.findall(r"(?m)^python scripts/.*$", readiness_guide)
        assert raw_python_script_commands == []

        required_fragments = [
            "make venv",
            "make install-core",
            "make check-environment",
            ".venv/bin/python scripts/check_image_processing_readiness.py",
            ".venv/bin/python scripts/simple_image_processor.py",
            ".venv/bin/python scripts/setup/download_depth_models.py",
            ".venv/bin/python scripts/download_samples.py",
            ".venv/bin/lux-depth-v3",
            "--model-key da3-metric",
            ".venv/bin/luxury-tiff-batch",
            ".venv/bin/lux_render",
            "--input-glob",
            ".venv/bin/luxury_video_grader",
            "./scripts/setup/install_da3_runtime.sh --profile baseline",
            "./scripts/setup/install_depth_pro_runtime.sh",
            "./scripts/setup/install_raw_runtime.sh",
            "./scripts/setup/install_fastvlm_runtime.sh",
            "../governance/DOCUMENTATION_MAP.md",
        ]
        for required_fragment in required_fragments:
            assert required_fragment in readiness_guide


class TestRootEnvironmentTemplate:
    """Tests for the root Docker/FastAPI environment template."""

    def test_env_example_covers_current_backend_and_compose_contracts(self):
        """The root env template should describe the current local stack seams."""
        env_example = _repo_root / ".env.example"
        content = env_example.read_text()

        expected_variables = [
            "TP_API_KEY",
            "TP_ENFORCE_JOB_API_KEY",
            "TP_API_KEY_HEADER",
            "TP_ALLOWED_ORIGINS",
            "TP_TRUSTED_HOSTS",
            "TP_ENABLE_TRUSTED_HOSTS",
            "TP_TRUST_X_FORWARDED_FOR",
            "TP_TRUSTED_PROXY_IPS",
            "TP_MAX_REQUEST_BYTES",
            "TP_PORTAL_MAX_UPLOAD_REQUEST_BYTES",
            "TP_PORTAL_UPLOAD_MAX_FILES",
            "TP_PORTAL_UPLOAD_MAX_FIELDS",
            "TP_PORTAL_UPLOAD_MAX_PART_BYTES",
            "TP_RATE_LIMIT_PER_MINUTE",
            "TP_MAX_CONCURRENT_JOBS",
            "TP_ALLOWED_INPUT_ROOTS",
            "TP_ALLOWED_OUTPUT_ROOTS",
            "TP_JOB_RETENTION_SECONDS",
            "TP_CLEANUP_INTERVAL_SECONDS",
            "TP_JOB_CLEANUP_SCAN_LIMIT",
            "TP_CANCEL_GRACE_SECONDS",
            "TP_JOB_LIST_LIMIT",
            "TP_MAX_INDEXED_ARTIFACTS",
            "TP_WORKER_ID",
            "TP_WORKER_LEASE_SECONDS",
            "TP_WORKER_HEARTBEAT_SECONDS",
            "TP_WORKER_POLL_SECONDS",
            "TP_WORKER_MAX_BACKOFF_SECONDS",
            "TP_WORKER_LOG_LEVEL",
            "TP_ORCHESTRATOR_WORKER_SHUTDOWN_GRACE_SECONDS",
            "TP_ORCHESTRATOR_RECLAIM_SWEEP_INTERVAL_SECONDS",
            "TP_ENABLE_API_DOCS",
            "TP_READY_VERBOSE",
            "TP_UID",
            "TP_GID",
            "DEVICE",
            "TP_ORCHESTRATOR_STATE_BACKEND",
            "TP_DATABASE_URL",
            "TP_ORCHESTRATOR_QUEUE_BACKEND",
            "TP_REDIS_URL",
            "TP_REDIS_KEY_PREFIX",
            "TP_ARTIFACT_STORE",
            "TP_ARTIFACT_LOCAL_ROOT",
            "TP_ARTIFACT_PREFIX",
            "TP_ARTIFACT_PRESIGN_EXPIRES_SECONDS",
            "TP_ARTIFACT_RETENTION_SECONDS",
            "TP_ARTIFACT_BUCKET",
            "TP_ARTIFACT_ENDPOINT_URL",
            "TP_ARTIFACT_REGION",
            "POSTGRES_DB",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "TP_POSTGRES_PUBLIC_PORT",
            "TP_REDIS_PUBLIC_PORT",
            "MINIO_ROOT_USER",
            "MINIO_ROOT_PASSWORD",
            "TP_MINIO_API_PUBLIC_PORT",
            "TP_MINIO_CONSOLE_PUBLIC_PORT",
            "TP_MINIO_IMAGE",
            "TP_MINIO_MC_IMAGE",
            "TP_TEST_POSTGRES_URL",
            "TP_TEST_REDIS_URL",
            "TP_TEST_S3_URL",
            "TP_TEST_S3_BUCKET",
            "TRANSFORMATION_PORTAL_DA3_PYTHON",
            "TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON",
            "TRANSFORMATION_PORTAL_RAW_PYTHON",
        ]
        for variable in expected_variables:
            assert variable in content

        expected_runtime_claims = [
            "Lux Depth V3 auto-discover the repo-local runtime contract paths",
            '.venv/bin/python -c "import secrets; print(secrets.token_urlsafe(32))"',
            "./.runtime/Depth-Anything-3/.venv-da3/bin/python",
            "./.venv-depth-pro/bin/python",
            "./.venv-raw/bin/python",
            "keep TP_ENABLE_TRUSTED_HOSTS=true outside exceptional local",
            "Keep TP_PORTAL_UPLOAD_ROOT inside",
            "Canonical worker names are TP_WORKER_*",
            "standalone `python -m transformation_portal.orchestrator.worker`",
        ]
        for claim in expected_runtime_claims:
            assert claim in content

        expected_authorities = [
            "web/secure-landing/.env.example",
            "docs/deployment/managed_paid_pilot_staging_runbook.md",
        ]
        for relative_path in expected_authorities:
            assert relative_path in content
            assert (_repo_root / relative_path).exists()

        stale_claims = [
            "Inventory below covers Python-orchestrator env vars sourced from",
            "grep -oE",
            "app.py. Run this",
            "orchestrator picks `python3` from PATH",
            '#   python -c "import secrets; print(secrets.token_urlsafe(32))"',
            "TP_ORCHESTRATOR_USE_QUEUE_BROKER",
        ]
        for stale_claim in stale_claims:
            assert stale_claim not in content


class TestRootSecurityPolicy:
    """Tests for the root security policy's live runtime claims."""

    def test_security_policy_matches_current_backend_runtime_defaults(self):
        """Root security guidance should describe current backend controls."""
        security_policy = _repo_root / "SECURITY.md"
        content = security_policy.read_text()

        expected_claims = [
            "TP_MAX_REQUEST_BYTES=1048576",
            "TP_PORTAL_MAX_UPLOAD_REQUEST_BYTES=1048576",
            "TP_PORTAL_UPLOAD_MAX_FILES=256",
            "TP_PORTAL_UPLOAD_MAX_FIELDS=32",
            "TP_PORTAL_UPLOAD_MAX_PART_BYTES=1048576",
            "Protected `/v1` endpoints enforce API-key auth by",
            "TP_ENFORCE_JOB_API_KEY=true",
            "Default: 60 requests/minute per client via `TP_RATE_LIMIT_PER_MINUTE`",
            "Admission cap: 4 concurrent jobs via `TP_MAX_CONCURRENT_JOBS`",
            "[`.env.example`](.env.example)",
            "TP_ALLOWED_INPUT_ROOTS=.",
            "TP_ALLOWED_OUTPUT_ROOTS=.",
        ]
        for claim in expected_claims:
            assert claim in content

        stale_claims = [
            "Maximum file size limits (default: 500MB for images, 5GB for videos)",
            "Implement API key or OAuth 2.0",
            "Default: 100 requests/minute per IP",
            "Heavy operations: 10 requests/hour",
            "Recommended security configuration (not currently implemented)",
            "These settings should be added to application configuration for production use",
            "max_file_size: 524288000",
        ]
        for stale_claim in stale_claims:
            assert stale_claim not in content


class TestAssetOrganization:
    """Tests for asset organization."""

    def test_luts_directory_exists(self):
        """Test that LUTs directory exists."""
        luts_path = _repo_root / "assets" / "luts"
        assert luts_path.exists(), "assets/luts/ should exist"

    def test_luts_subdirectories(self):
        """Test that LUTs are organized in subdirectories."""
        luts_path = _repo_root / "assets" / "luts"

        if luts_path.exists():
            # Check for expected LUT categories
            expected_categories = [
                "film_emulation",
                "location_aesthetic",
                "material_response",
            ]

            for category in expected_categories:
                category_path = luts_path / category
                if not category_path.exists():
                    print(f"Note: {category} not found in assets/luts/")

    def test_duplicate_data_luts_removed(self):
        """LUTs have one canonical repo location."""
        assert not (_repo_root / "data" / "luts").exists(), "data/luts duplicates assets/luts and must stay removed"

    def test_board_material_textures_are_under_assets(self):
        """Board material textures live with other repo assets."""
        texture_path = _repo_root / "assets" / "textures" / "board_materials"
        assert texture_path.is_dir(), "assets/textures/board_materials/ should exist"
        assert not (_repo_root / "textures").exists(), "root textures/ should not be recreated"

        expected_textures = {
            "plaster_marmorino_westwood_beige.png",
            "stone_bokara_coastal.png",
            "cladding_sculptform_warm.png",
            "screens_grey_gum.png",
            "equitone_lt85.png",
            "bison_weathered_ipe.png",
            "dark_bronze_anodized.png",
            "louvretec_powder_white.png",
        }
        actual_textures = {path.name for path in texture_path.glob("*.png")}
        assert expected_textures <= actual_textures


class TestWrapperFiles:
    """Tests for console script entrypoints instead of wrapper files."""

    def test_console_scripts_defined_in_pyproject(self):
        """Test that console_scripts are properly defined in pyproject.toml."""
        pyproject_path = _repo_root / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist"

        content = pyproject_path.read_text()
        assert "[project.scripts]" in content, "project.scripts section should exist"
        assert "luxury-tiff-batch" in content, "luxury-tiff-batch entrypoint should be defined"


class TestGitignore:
    """Tests for .gitignore coverage."""

    def test_gitignore_exists(self):
        """Test that .gitignore exists."""
        gitignore = _repo_root / ".gitignore"
        assert gitignore.exists(), ".gitignore should exist"

    def test_gitignore_covers_python_artifacts(self):
        """Test that .gitignore covers Python artifacts."""
        gitignore = _repo_root / ".gitignore"
        content = gitignore.read_text()

        # Check for important patterns
        patterns = [
            "__pycache__",
            "*.pyc",
            "*.egg-info",
            ".venv",
            "dist/",
            "build/",
        ]

        for pattern in patterns:
            assert pattern in content, f".gitignore should include pattern: {pattern}"

    def test_gitignore_covers_outputs(self):
        """Test that .gitignore covers processing outputs."""
        gitignore = _repo_root / ".gitignore"
        content = gitignore.read_text()

        # Check for output patterns
        output_patterns = [
            "*.log",
            "_depth.npy",
            "_enhanced.png",
            "processed_output/",
        ]

        for pattern in output_patterns:
            assert pattern in content, f".gitignore should include output pattern: {pattern}"

    def test_gitignore_covers_frontdoor_temp_dirs(self):
        """Test that .gitignore covers frontdoor transient build directories."""
        gitignore = _repo_root / ".gitignore"
        content = gitignore.read_text()

        temp_dir_patterns = [
            "web/secure-landing/.next/",
            "web/secure-landing/.next-build-verify/",
            "web/secure-landing/.next-smoke-*/",
            "web/secure-landing/.next-codex-*/",
            "web/secure-landing/.metafiles/",
            "web/secure-landing/tmp/",
        ]

        for pattern in temp_dir_patterns:
            assert pattern in content, f".gitignore should include frontdoor temp dir pattern: {pattern}"

    def test_gitignore_covers_local_env_variants(self):
        """Test that .gitignore excludes local env files without hiding templates."""
        gitignore = _repo_root / ".gitignore"
        content = gitignore.read_text()

        assert ".env" in content
        assert ".env.*" in content
        assert "!.env.example" in content

    def test_gitignore_covers_local_credential_files(self):
        """Test that local package-manager credentials and certificate bundles stay ignored."""
        gitignore = _repo_root / ".gitignore"
        content = gitignore.read_text()

        sensitive_patterns = [
            ".npmrc",
            ".pypirc",
            "*.pem",
            "*.key",
            "*.p12",
            "*.pfx",
        ]

        for pattern in sensitive_patterns:
            assert pattern in content, f".gitignore should include local credential pattern: {pattern}"

    def test_gitignore_does_not_hide_tracked_files(self):
        """Tracked governance docs and fixtures should not be masked by ignore rules."""
        result = subprocess.run(
            ["git", "ls-files", "-c", "-i", "--exclude-standard"],
            cwd=_repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == ""


class TestNoOrphanedFiles:
    """Tests to prevent orphaned or redundant files."""

    def test_retired_generated_root_directories_absent(self):
        """Generated/demo roots should not be committed as top-level directories."""
        retired_roots = [
            "dashboard",
            "linear_ingest_demo",
            "projects",
            "test_sky_fix",
        ]
        present_roots = [path for path in retired_roots if (_repo_root / path).exists()]
        assert not present_roots, f"Retired root directories should stay removed: {present_roots}"

    def test_board_texture_generator_has_canonical_utility_module(self):
        """Root board-texture script delegates to the canonical utility module."""
        wrapper = _repo_root / "scripts" / "create_board_textures.py"
        canonical = _repo_root / "scripts" / "utilities" / "create_board_textures.py"

        wrapper_text = wrapper.read_text(encoding="utf-8")
        canonical_text = canonical.read_text(encoding="utf-8")

        assert "from scripts.utilities.create_board_textures import main" in wrapper_text
        assert 'DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets" / "textures" / "board_materials"' in canonical_text

    def test_sky_comparison_requires_explicit_inputs_and_ignored_default_output(self):
        """The sky comparison tool no longer depends on removed root investigation files."""
        script = _repo_root / "tools" / "investigations" / "materials_v3" / "create_sky_comparison.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=_repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "--input INPUT" in result.stdout
        assert "--before BEFORE" in result.stdout
        assert "--after AFTER" in result.stdout
        assert "--amplify AMPLIFY" in result.stdout
        assert "output/materials_v3/sky_fix_comparison.jpg" in result.stdout
        assert "test_sky_fix" not in result.stdout

    def test_sky_comparison_preserves_deprecated_input_mode(self):
        """Historical --input/--amplify flags still parse without restoring root defaults."""
        script = _repo_root / "tools" / "investigations" / "materials_v3" / "create_sky_comparison.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--input",
                "input_images/test_sky.jpg",
                "--output",
                "output/materials_v3/sky_fix_comparison.jpg",
                "--amplify",
                "10",
            ],
            cwd=_repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "--input is deprecated" in result.stdout
        assert "Use --before and --after" in result.stdout

    def test_no_duplicate_material_response(self):
        """Test that material_response isn't duplicated unnecessarily."""
        root_file = _repo_root / "material_response.py"
        src_file = _repo_root / "src" / "transformation_portal" / "processors" / "material_response" / "core.py"

        # Both may exist for backward compatibility, but root should be thin
        if root_file.exists() and src_file.exists():
            root_size = root_file.stat().st_size
            src_size = src_file.stat().st_size

            # Root file should not be a full duplicate
            # (Allow some size for a substantial wrapper, but not full copy)
            assert root_size < src_size * 0.5, (
                "Root material_response.py seems to be a full duplicate. " "Should be a thin wrapper."
            )

            # Check that the root file actually delegates to the src implementation
            root_text = root_file.read_text(encoding="utf-8")
            # Look for an import from transformation_portal (allow whitespace, case-insensitive)
            import_found = "from transformation_portal" in root_text or "import transformation_portal" in root_text
            assert import_found, (
                "Root material_response.py should import from transformation_portal "
                "to delegate implementation, not duplicate it."
            )

    def test_no_build_artifacts_in_root(self):
        """Test that build artifacts aren't in root."""
        # Check for common build artifacts
        artifacts = [
            "dist",
            "build",
            "*.egg-info",
        ]

        for pattern in artifacts:
            matching = list(_repo_root.glob(pattern))
            # Some may be in .gitignore, but shouldn't be tracked
            if matching:
                print(f"Note: Found build artifacts matching {pattern}")

    def test_no_nested_git_worktrees_in_root(self):
        """Fail if a copied repo/worktree lands directly under repo root."""
        nested_git_roots = []

        for child in _repo_root.iterdir():
            if not child.is_dir() or child.name == ".git":
                continue
            if (child / ".git").exists():
                nested_git_roots.append(child.relative_to(_repo_root).as_posix())

        assert not nested_git_roots, (
            "Unexpected nested git repositories/worktrees found in repo root. "
            "Remove copied worktrees so this checkout remains the single source of truth:\n"
            + "\n".join(sorted(nested_git_roots))
        )

    def test_no_exact_duplicate_python_modules_between_src_and_scripts(self):
        """Keep `scripts/` wrappers thin and prevent copy/paste module drift."""
        src_root = _repo_root / "src" / "transformation_portal"
        scripts_root = _repo_root / "scripts"

        if not src_root.exists() or not scripts_root.exists():
            return

        src_hashes = {}
        for src_file in src_root.rglob("*.py"):
            if src_file.is_file():
                digest = hashlib.sha1(src_file.read_bytes()).hexdigest()
                src_hashes.setdefault(digest, []).append(src_file.relative_to(_repo_root))

        duplicates = []
        for script_file in scripts_root.rglob("*.py"):
            if not script_file.is_file():
                continue
            digest = hashlib.sha1(script_file.read_bytes()).hexdigest()
            if digest in src_hashes:
                for src_match in src_hashes[digest]:
                    duplicates.append(f"{script_file.relative_to(_repo_root)} == {src_match}")

        assert not duplicates, (
            "Exact duplicate Python implementations found between scripts/ and src/. "
            "Convert script copies to thin wrappers that import canonical src modules.\n" + "\n".join(sorted(duplicates))
        )


class TestConfigurationFiles:
    """Tests for configuration files."""

    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists."""
        pyproject = _repo_root / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml should exist"

    def test_makefile_exists(self):
        """Test that Makefile exists."""
        makefile = _repo_root / "Makefile"
        assert makefile.exists(), "Makefile should exist for common tasks"

    def test_requirements_files_exist(self):
        """Test that requirements files exist."""
        requirements = _repo_root / "requirements.txt"
        assert requirements.exists(), "requirements.txt should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
