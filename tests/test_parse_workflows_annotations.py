from pathlib import Path

import pytest

from transformation_portal.analyzers.parse_workflows import WorkflowBug, WorkflowParser, render_github_annotations

# pylint: disable=redefined-outer-name  # pytest fixtures


@pytest.fixture()
def workflow_dir(tmp_path: Path) -> Path:
    path = tmp_path / ".github" / "workflows"
    path.mkdir(parents=True)
    return path


def test_duplicate_job_names_are_reported_as_yaml_errors(
    workflow_dir: Path,
) -> None:
    workflow = """
name: Duplicate Jobs
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "first"
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "second"
"""
    (workflow_dir / "duplicate.yml").write_text(workflow, encoding="utf-8")

    bugs = WorkflowParser(workflow_dir).parse_all_workflows()

    assert any("duplicate key" in bug.message.lower() for bug in bugs)
    assert any(bug.severity == "error" for bug in bugs)


def test_render_github_annotations_escapes_multiline_messages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bug = WorkflowBug(
        file_path="/tmp/workflow.yml",
        line_number=3,
        severity="error",
        message="YAML syntax error: bad%\nline",
    )

    render_github_annotations([bug])

    rendered = capsys.readouterr().out.strip()
    assert rendered == ("::error file=/tmp/workflow.yml,line=3," "title=Workflow Issue::YAML syntax error: bad%25%0Aline")
