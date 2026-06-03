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

    def test_root_readme_status_points_to_current_documentation_navigation(self):
        """Root status prose should not depend on stale PR-specific snapshots."""
        readme = (_repo_root / "README.md").read_text()

        assert "building on `main` through PR #" not in readme
        required_navigation = [
            "docs/README.md",
            "docs/governance/DOCUMENTATION_MAP.md",
            "May 11, 2026 repo-wide refresh audit",
            "Last Updated: 2026-06-03",
        ]
        for expected in required_navigation:
            assert expected in readme
        assert "Last Updated: 2026-05-11" not in readme

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
        ]
        for phrase in required_current_context:
            assert phrase in changelog

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
        assert "`cryptography`          | >=46.0.5" in contributing
        assert "`starlette`             | >=1.0.1" in contributing
        assert "Starlette==1.0.1" in security_policy
        assert "current governed lock baseline is pillow==12.2.0" in requirements_lint
        assert "allows pillow==12.1.1 in lockfiles" not in requirements_lint

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

    def test_contributing_dependency_audit_schedule_is_current(self):
        """Canonical contribution guidance should not point to a past audit date."""
        contributing = (_repo_root / "CONTRIBUTING.md").read_text()

        assert "Next audit: **2026-05-16 (Q2 2026)**" not in contributing
        assert "Next audit: **2026-08-16 (Q3 2026)**" in contributing

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

                target_source = requirements_targets if "cd requirements" in line else root_targets
                for match in re.finditer(r"(?<![A-Za-z0-9_.-])make\s+([A-Za-z0-9_.%/+@-]+)", line):
                    target = match.group(1)
                    if target not in target_source:
                        missing_targets.append(f"{relative_path}:{line_number}: make {target}")

        assert not missing_targets, "Root guidance references undefined Make targets:\n" + "\n".join(sorted(missing_targets))


class TestRootEnvironmentTemplate:
    """Tests for the root Docker/FastAPI environment template."""

    def test_env_example_covers_current_backend_and_compose_contracts(self):
        """The root env template should describe the current local stack seams."""
        env_example = _repo_root / ".env.example"
        content = env_example.read_text()

        expected_variables = [
            "TP_API_KEY",
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
            "TP_ARTIFACT_BUCKET",
            "TP_ARTIFACT_ENDPOINT_URL",
            "POSTGRES_DB",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "TP_POSTGRES_PUBLIC_PORT",
            "TP_REDIS_PUBLIC_PORT",
            "MINIO_ROOT_USER",
            "MINIO_ROOT_PASSWORD",
            "TP_MINIO_API_PUBLIC_PORT",
            "TP_MINIO_CONSOLE_PUBLIC_PORT",
            "TP_TEST_POSTGRES_URL",
            "TP_TEST_REDIS_URL",
            "TP_TEST_S3_URL",
            "TP_TEST_S3_BUCKET",
        ]
        for variable in expected_variables:
            assert variable in content

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
