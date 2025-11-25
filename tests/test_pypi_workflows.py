"""
Tests for PyPI workflow configurations.
"""
# pylint: disable=redefined-outer-name  # pytest fixtures

from pathlib import Path

import pytest
import yaml


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

        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)

        assert workflow is not None, "Workflow should parse as valid YAML"
        assert 'name' in workflow, "Workflow should have a name"
        assert 'jobs' in workflow, "Workflow should have jobs"

    def test_submit_pypi_workflow_structure(self, workflows_dir):
        """Test that submit-pypi.yml has correct structure."""
        workflow_file = workflows_dir / "submit-pypi.yml"

        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)

        # Check workflow name
        assert workflow['name'] == "Submit to PyPI"

        # Check triggers (YAML may parse 'on' as True boolean)
        triggers = workflow.get('on', workflow.get(True, {}))
        assert triggers, "Workflow should have triggers"
        assert 'push' in triggers and 'tags' in triggers['push']
        assert 'workflow_dispatch' in triggers

        # Check required jobs
        jobs = workflow['jobs']
        assert 'build' in jobs, "Should have build job"
        assert 'test-pypi' in jobs, "Should have test-pypi job"
        assert 'pypi' in jobs, "Should have pypi job"
        assert 'cleanup' in jobs, "Should have cleanup job"

        # Check build job structure
        build_job = jobs['build']
        assert 'steps' in build_job
        assert 'runs-on' in build_job

        # Verify build steps include essential tasks
        step_names = [step.get('name', '') for step in build_job['steps']]
        assert any('build' in name.lower() for name in step_names), \
            "Build job should include build step"
        assert any('twine' in name.lower() or 'check' in name.lower() for name in step_names), \
            "Build job should include distribution check"

    def test_submit_pypi_workflow_triggers(self, workflows_dir):
        """Test that submit-pypi.yml has correct triggers."""
        workflow_file = workflows_dir / "submit-pypi.yml"

        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)

        # Check triggers (YAML may parse 'on' as True boolean)
        triggers = workflow.get('on', workflow.get(True, {}))

        # Check version tag trigger
        assert 'push' in triggers
        assert 'tags' in triggers['push']
        assert 'v*' in triggers['push']['tags']

        # Check manual dispatch
        assert 'workflow_dispatch' in triggers
        assert 'inputs' in triggers['workflow_dispatch']
        assert 'test_pypi' in triggers['workflow_dispatch']['inputs']

    def test_python_app_workflow_has_cleanup(self, workflows_dir):
        """Test that python-app.yml has cleanup job."""
        workflow_file = workflows_dir / "python-app.yml"

        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)

        jobs = workflow['jobs']
        assert 'cleanup' in jobs, "python-app.yml should have cleanup job"

        # Check cleanup job structure
        cleanup_job = jobs['cleanup']
        assert cleanup_job.get('if') == 'always()', \
            "Cleanup should always run"
        assert 'needs' in cleanup_job, \
            "Cleanup should depend on other jobs"

    def test_python_app_workflow_has_test_pypi(self, workflows_dir):
        """Test that python-app.yml has Test PyPI deployment."""
        workflow_file = workflows_dir / "python-app.yml"

        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)

        # Check deploy job exists
        jobs = workflow['jobs']
        assert 'deploy' in jobs, "python-app.yml should have deploy job"

        # Check for Test PyPI upload step
        deploy_job = jobs['deploy']
        step_names = [step.get('name', '') for step in deploy_job['steps']]

        # Should have Test PyPI upload step (not commented out)
        assert any('test pypi' in name.lower() for name in step_names), \
            "Deploy job should include Test PyPI upload"

    def test_workflows_use_modern_actions(self, workflows_dir):
        """Test that workflows use modern action versions."""
        pypi_workflow = workflows_dir / "submit-pypi.yml"

        with open(pypi_workflow, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for modern action versions (v4, v5, or v6 are acceptable)
        assert any(f'actions/checkout@v{v}' in content for v in ['4', '5', '6']), \
            "Should use recent checkout action (v4, v5, or v6)"
        assert any(f'actions/setup-python@v{v}' in content for v in ['5', '6']), \
            "Should use recent setup-python action (v5 or v6)"
        assert any(f'actions/upload-artifact@v{v}' in content for v in ['4', '5']), \
            "Should use recent upload-artifact action (v4 or v5)"

    def test_submit_pypi_has_package_verification(self, workflows_dir):
        """Test that submit-pypi.yml verifies package contents."""
        workflow_file = workflows_dir / "submit-pypi.yml"

        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)

        build_steps = workflow['jobs']['build']['steps']
        step_names = [step.get('name', '') for step in build_steps]

        # Should verify package contents
        assert any('verify' in name.lower() or 'check' in name.lower()
                   for name in step_names), \
            "Build job should verify package contents"

    def test_cleanup_job_prevents_failures(self, workflows_dir):
        """Test that cleanup jobs use error suppression."""
        workflows_to_check = ['submit-pypi.yml', 'python-app.yml']

        for workflow_name in workflows_to_check:
            workflow_file = workflows_dir / workflow_name

            with open(workflow_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Cleanup commands should have error suppression
            if 'cleanup' in content.lower():
                assert '|| true' in content or 'continue-on-error' in content, \
                    f"{workflow_name} cleanup should suppress errors to prevent timeouts"


class TestWorkflowDocumentation:
    """Test that workflow documentation is up to date."""

    def test_readme_documents_submit_pypi(self, workflows_dir):
        """Test that README.md documents the submit-pypi workflow."""
        readme_file = workflows_dir / "README.md"

        with open(readme_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should document submit-pypi workflow
        assert 'submit-pypi' in content.lower(), \
            "README should document submit-pypi workflow"
        assert 'pypi' in content.lower(), \
            "README should mention PyPI"

    def test_readme_documents_usage(self, workflows_dir):
        """Test that README.md provides usage examples."""
        readme_file = workflows_dir / "README.md"

        with open(readme_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should provide usage instructions
        assert 'tag' in content.lower() or 'version' in content.lower(), \
            "README should explain how to trigger PyPI uploads"
        assert 'secret' in content.lower() or 'token' in content.lower(), \
            "README should mention required secrets"
