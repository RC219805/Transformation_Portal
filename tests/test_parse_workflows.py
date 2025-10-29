"""
Tests for the workflow parser that detects bugs in GitHub Actions workflows.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from parse_workflows import WorkflowParser, WorkflowBug


@pytest.fixture
def temp_workflow_dir():
    """Create a temporary directory for workflow files."""
    temp_dir = tempfile.mkdtemp()
    workflow_dir = Path(temp_dir) / ".github" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    yield workflow_dir
    shutil.rmtree(temp_dir)


class TestWorkflowParser:
    """Test the WorkflowParser class."""
    
    def test_valid_workflow_no_bugs(self, temp_workflow_dir):
        """Test that a valid workflow has no bugs."""
        workflow_content = """
name: Test Workflow
on:
  push:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: echo "Testing"
"""
        workflow_file = temp_workflow_dir / "test.yml"
        workflow_file.write_text(workflow_content)
        
        parser = WorkflowParser(temp_workflow_dir)
        bugs = parser.parse_all_workflows()
        
        assert len(bugs) == 0
    
    def test_unclosed_conditional_detected(self, temp_workflow_dir):
        """Test that unclosed if/fi statements are detected."""
        workflow_content = """
name: Test Workflow
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Test with unclosed if
        run: |
          if [ -z "$VAR" ]; then
            echo "empty"
"""
        workflow_file = temp_workflow_dir / "test.yml"
        workflow_file.write_text(workflow_content)
        
        parser = WorkflowParser(temp_workflow_dir)
        bugs = parser.parse_all_workflows()
        
        # Should find unclosed conditional
        assert len(bugs) > 0
        assert any('Unclosed conditional' in bug.message for bug in bugs)
        assert any(bug.severity == 'error' for bug in bugs)
    
    def test_missing_step_id_detected(self, temp_workflow_dir):
        """Test that missing step IDs are detected when outputs are referenced."""
        workflow_content = """
name: Test Workflow
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Generate output
        run: echo "value=test" >> "$GITHUB_OUTPUT"
      - name: Use output
        run: echo ${{ steps.generate.outputs.value }}
"""
        workflow_file = temp_workflow_dir / "test.yml"
        workflow_file.write_text(workflow_content)
        
        parser = WorkflowParser(temp_workflow_dir)
        bugs = parser.parse_all_workflows()
        
        # Should find missing step ID
        assert len(bugs) > 0
        assert any('step id' in bug.message.lower() for bug in bugs)
    
    def test_valid_step_id_reference(self, temp_workflow_dir):
        """Test that valid step ID references don't trigger bugs."""
        workflow_content = """
name: Test Workflow
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Generate output
        id: generate
        run: echo "value=test" >> "$GITHUB_OUTPUT"
      - name: Use output
        run: echo ${{ steps.generate.outputs.value }}
"""
        workflow_file = temp_workflow_dir / "test.yml"
        workflow_file.write_text(workflow_content)
        
        parser = WorkflowParser(temp_workflow_dir)
        bugs = parser.parse_all_workflows()
        
        # Should not find any bugs
        assert len(bugs) == 0
    
    def test_invalid_job_dependency(self, temp_workflow_dir):
        """Test that invalid job dependencies are detected."""
        workflow_content = """
name: Test Workflow
on: [push]
jobs:
  job1:
    runs-on: ubuntu-latest
    steps:
      - run: echo "job1"
  job2:
    runs-on: ubuntu-latest
    needs: nonexistent_job
    steps:
      - run: echo "job2"
"""
        workflow_file = temp_workflow_dir / "test.yml"
        workflow_file.write_text(workflow_content)
        
        parser = WorkflowParser(temp_workflow_dir)
        bugs = parser.parse_all_workflows()
        
        # Should find invalid dependency
        assert len(bugs) > 0
        assert any('nonexistent_job' in bug.message for bug in bugs)
    
    def test_inefficient_matrix_usage(self, temp_workflow_dir):
        """Test that inefficient matrix usage is detected."""
        workflow_content = """
name: Test Workflow
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        task: [lint, test]
        device: [cpu, gpu]
    steps:
      - name: Lint
        if: matrix.task == 'lint'
        run: echo "linting"
      - name: Test
        if: matrix.task == 'test'
        run: echo "testing"
"""
        workflow_file = temp_workflow_dir / "test.yml"
        workflow_file.write_text(workflow_content)
        
        parser = WorkflowParser(temp_workflow_dir)
        bugs = parser.parse_all_workflows()
        
        # Should find inefficient matrix
        assert len(bugs) > 0
        assert any('device matrix' in bug.message for bug in bugs)
    
    def test_matrix_with_exclusions_ok(self, temp_workflow_dir):
        """Test that matrix with proper exclusions doesn't trigger warnings."""
        workflow_content = """
name: Test Workflow
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        task: [lint, test]
        device: [cpu, gpu]
        exclude:
          - task: lint
            device: gpu
    steps:
      - name: Lint
        if: matrix.task == 'lint'
        run: echo "linting"
      - name: Test
        if: matrix.task == 'test'
        run: echo "testing"
"""
        workflow_file = temp_workflow_dir / "test.yml"
        workflow_file.write_text(workflow_content)
        
        parser = WorkflowParser(temp_workflow_dir)
        bugs = parser.parse_all_workflows()
        
        # Should not find matrix warning
        assert not any('device matrix' in bug.message for bug in bugs)
    
    def test_invalid_openai_model(self, temp_workflow_dir):
        """Test that invalid OpenAI model names are detected."""
        workflow_content = """
name: Test Workflow
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Call OpenAI
        run: |
          curl -X POST https://api.openai.com/v1/chat/completions \\
            -d '{"model": "gpt-4.1-mini", "messages": []}'
"""
        workflow_file = temp_workflow_dir / "test.yml"
        workflow_file.write_text(workflow_content)
        
        parser = WorkflowParser(temp_workflow_dir)
        bugs = parser.parse_all_workflows()
        
        # Should find invalid model
        assert len(bugs) > 0
        assert any('gpt-4.1-mini' in bug.message for bug in bugs)
    
    def test_valid_openai_model(self, temp_workflow_dir):
        """Test that valid OpenAI model names don't trigger warnings."""
        workflow_content = """
name: Test Workflow
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Call OpenAI
        run: |
          curl -X POST https://api.openai.com/v1/chat/completions \\
            -d '{"model": "gpt-4o-mini", "messages": []}'
"""
        workflow_file = temp_workflow_dir / "test.yml"
        workflow_file.write_text(workflow_content)
        
        parser = WorkflowParser(temp_workflow_dir)
        bugs = parser.parse_all_workflows()
        
        # Should not find model bugs
        assert not any('model' in bug.message.lower() for bug in bugs)
    
    def test_yaml_syntax_error(self, temp_workflow_dir):
        """Test that YAML syntax errors are caught."""
        workflow_content = """
name: Test Workflow
on: [push
jobs:
  test:
    runs-on: ubuntu-latest
"""
        workflow_file = temp_workflow_dir / "test.yml"
        workflow_file.write_text(workflow_content)
        
        parser = WorkflowParser(temp_workflow_dir)
        bugs = parser.parse_all_workflows()
        
        # Should find YAML error
        assert len(bugs) > 0
        assert any('YAML syntax error' in bug.message for bug in bugs)
    
    def test_multiple_workflows(self, temp_workflow_dir):
        """Test parsing multiple workflow files."""
        # Create first workflow with a bug
        workflow1 = """
name: Workflow 1
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: |
          if [ true ]; then
            echo "missing fi"
"""
        (temp_workflow_dir / "workflow1.yml").write_text(workflow1)
        
        # Create second workflow without bugs
        workflow2 = """
name: Workflow 2
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "ok"
"""
        (temp_workflow_dir / "workflow2.yml").write_text(workflow2)
        
        parser = WorkflowParser(temp_workflow_dir)
        bugs = parser.parse_all_workflows()
        
        # Should find bug only in workflow1
        assert len(bugs) > 0
        assert any('workflow1.yml' in bug.file_path for bug in bugs)
        assert not any('workflow2.yml' in bug.file_path for bug in bugs)


class TestWorkflowBug:
    """Test the WorkflowBug class."""
    
    def test_bug_string_representation(self):
        """Test string representation of bugs."""
        bug = WorkflowBug(
            file_path="/path/to/workflow.yml",
            line_number=42,
            severity="error",
            message="Test error message"
        )
        
        bug_str = str(bug)
        assert "/path/to/workflow.yml:42" in bug_str
        assert "ERROR" in bug_str
        assert "Test error message" in bug_str
    
    def test_bug_without_line_number(self):
        """Test bug representation without line number."""
        bug = WorkflowBug(
            file_path="/path/to/workflow.yml",
            line_number=None,
            severity="warning",
            message="Test warning"
        )
        
        bug_str = str(bug)
        assert "/path/to/workflow.yml" in bug_str
        assert ":" not in bug_str.split("/path/to/workflow.yml")[1]  # Should not have any line number
        assert "WARNING" in bug_str
