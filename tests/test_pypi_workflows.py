"""
Tests for PyPI workflow configurations.
"""

# pylint: disable=redefined-outer-name  # pytest fixtures

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit


@pytest.fixture
def workflows_dir():
    """Get the workflows directory path."""
    repo_root = Path(__file__).parent.parent
    return repo_root / ".github" / "workflows"


class TestPyPIWorkflows:
    """Test PyPI-related workflow configurations."""

    def test_submit_pypi_workflow_exists(self, workflows_dir):
        """Test that submit-pypi.yml workflow exists."""
        workflow_file = workflows_dir / "submit-pypi.yml"
        assert workflow_file.exists(), "submit-pypi.yml workflow file should exist"

    def test_submit_pypi_workflow_valid_yaml(self, workflows_dir):
        """Test that submit-pypi.yml is valid YAML."""
        workflow_file = workflows_dir / "submit-pypi.yml"

        with open(workflow_file, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        assert workflow is not None, "Workflow should parse as valid YAML"
        assert "name" in workflow, "Workflow should have a name"
        assert "jobs" in workflow, "Workflow should have jobs"

    def test_submit_pypi_workflow_structure(self, workflows_dir):
        """Test that submit-pypi.yml has correct structure."""
        workflow_file = workflows_dir / "submit-pypi.yml"

        with open(workflow_file, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        # Check workflow name
        assert workflow["name"] == "Submit to PyPI"

        # Check triggers (YAML may parse 'on' as True boolean)
        triggers = workflow.get("on", workflow.get(True, {}))
        assert triggers, "Workflow should have triggers"
        assert "push" in triggers and "tags" in triggers["push"]
        assert "workflow_dispatch" in triggers

        # Check required jobs
        jobs = workflow["jobs"]
        assert "build" in jobs, "Should have build job"
        assert "test-pypi" in jobs, "Should have test-pypi job"
        assert "pypi" in jobs, "Should have pypi job"
        assert "cleanup" in jobs, "Should have cleanup job"

        # Check build job structure
        build_job = jobs["build"]
        assert "steps" in build_job
        assert "runs-on" in build_job

        # Verify build steps include essential tasks
        step_names = [step.get("name", "") for step in build_job["steps"]]
        assert any("build" in name.lower() for name in step_names), "Build job should include build step"
        assert any(
            "twine" in name.lower() or "check" in name.lower() for name in step_names
        ), "Build job should include distribution check"

    def test_submit_pypi_workflow_triggers(self, workflows_dir):
        """Test that submit-pypi.yml has correct triggers."""
        workflow_file = workflows_dir / "submit-pypi.yml"

        with open(workflow_file, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        triggers = workflow.get("on", workflow.get(True, {}))

        # Version tag trigger
        assert "push" in triggers
        assert "tags" in triggers["push"]
        assert "v*" in triggers["push"]["tags"]

        # Manual dispatch
        assert "workflow_dispatch" in triggers
        assert "inputs" in triggers["workflow_dispatch"]
        assert "test_pypi" in triggers["workflow_dispatch"]["inputs"]

    def test_python_app_workflow_has_cleanup(self, workflows_dir):
        """Test that python-app.yml has cleanup job."""
        workflow_file = workflows_dir / "python-app.yml"

        # Skip if workflow has been disabled/removed (CI-001 consolidation)
        if not workflow_file.exists():
            pytest.skip("python-app.yml has been disabled (CI consolidation)")

        with open(workflow_file, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        jobs = workflow["jobs"]
        assert "cleanup" in jobs, "python-app.yml should have cleanup job"

        cleanup_job = jobs["cleanup"]
        assert cleanup_job.get("if") == "always()", "Cleanup should always run"
        assert "needs" in cleanup_job, "Cleanup should depend on other jobs"

    def test_python_app_workflow_has_test_pypi(self, workflows_dir):
        """Test that python-app.yml has Test PyPI deployment."""
        workflow_file = workflows_dir / "python-app.yml"

        # Skip if workflow has been disabled/removed (CI-001 consolidation)
        if not workflow_file.exists():
            pytest.skip("python-app.yml has been disabled (CI consolidation)")

        with open(workflow_file, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        jobs = workflow["jobs"]
        assert "deploy" in jobs, "python-app.yml should have deploy job"

        deploy_job = jobs["deploy"]
        step_names = [step.get("name", "") for step in deploy_job["steps"]]

        assert any("test pypi" in name.lower() for name in step_names), "Deploy job should include Test PyPI upload"

    def test_workflows_use_modern_actions(self, workflows_dir):
        """Test that workflows use modern action versions.

        Actions may be pinned using either version tags (e.g., @v6) or
        commit SHAs with version comments (e.g., @<sha> # v6).
        """
        pypi_workflow = workflows_dir / "submit-pypi.yml"

        with open(pypi_workflow, "r", encoding="utf-8") as f:
            content = f.read()

        # checkout: require v4+
        # Match either @v6 or @<sha> # v6 patterns
        m = re.search(r"actions/checkout@(?:v(\d+)|[a-f0-9]+\s*#\s*v(\d+))", content)
        if m:
            version = int(m.group(1) or m.group(2))
            assert version >= 4, "Should use recent checkout action (v4+)"
        else:
            raise AssertionError("Should use recent checkout action (v4+)")

        # setup-python: require v5+
        # Match either @v5/@v6 or @<sha> # v5/@<sha> # v6 patterns
        setup_python_match = re.search(
            r"actions/setup-python@(?:v([56])|[a-f0-9]+\s*#\s*v([56]))", content
        )
        assert setup_python_match, "Should use recent setup-python action (v5+)"

        # upload-artifact: require v4+
        # Match either @v7 or @<sha> # v7 patterns
        m = re.search(r"actions/upload-artifact@(?:v(\d+)|[a-f0-9]+\s*#\s*v(\d+))", content)
        if m:
            version = int(m.group(1) or m.group(2))
            assert version >= 4, "Should use recent upload-artifact action (v4+)"
        else:
            raise AssertionError("Should use recent upload-artifact action (v4+)")

    def test_submit_pypi_has_package_verification(self, workflows_dir):
        """Test that submit-pypi.yml verifies package contents."""
        workflow_file = workflows_dir / "submit-pypi.yml"

        with open(workflow_file, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        build_steps = workflow["jobs"]["build"]["steps"]
        step_names = [step.get("name", "") for step in build_steps]

        assert any(
            "verify" in name.lower() or "check" in name.lower() for name in step_names
        ), "Build job should verify package contents"

    def test_cleanup_job_prevents_failures(self, workflows_dir):
        """Test that cleanup jobs use error suppression."""
        workflows_to_check = ["submit-pypi.yml", "python-app.yml"]

        for workflow_name in workflows_to_check:
            workflow_file = workflows_dir / workflow_name

            # Skip if workflow has been disabled/removed (CI-001 consolidation)
            if not workflow_file.exists():
                continue

            with open(workflow_file, "r", encoding="utf-8") as f:
                content = f.read()

            if "cleanup" in content.lower():
                assert (
                    "|| true" in content or "continue-on-error" in content
                ), f"{workflow_name} cleanup should suppress errors"


class TestWorkflowDocumentation:
    """Test that workflow documentation is up to date."""

    def test_readme_documents_submit_pypi(self, workflows_dir):
        """Test that README.md documents the submit-pypi workflow."""
        readme_file = workflows_dir / "README.md"

        with open(readme_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "submit-pypi" in content.lower(), "README should document submit-pypi workflow"
        assert "pypi" in content.lower(), "README should mention PyPI"

    def test_readme_documents_usage(self, workflows_dir):
        """Test that README.md provides usage examples."""
        readme_file = workflows_dir / "README.md"

        with open(readme_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "tag" in content.lower() or "version" in content.lower(), "README should explain how to trigger PyPI uploads"
        assert "secret" in content.lower() or "token" in content.lower(), "README should mention required secrets"
