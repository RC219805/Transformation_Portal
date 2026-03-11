from pathlib import Path

import pytest

from transformation_portal.analyzers.parse_workflows import (
    WorkflowBug,
    WorkflowParser,
    render_github_annotations,
)

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


def test_yaml_merge_keys_are_not_rejected(
    workflow_dir: Path,
) -> None:
    """Workflows using YAML merge keys (<<: *anchor) must parse without errors."""
    workflow = """
name: Merge Key Workflow
on: [push]

.defaults: &defaults
  runs-on: ubuntu-latest
  timeout-minutes: 10

jobs:
  test:
    <<: *defaults
    steps:
      - run: echo "hello"
"""
    (workflow_dir / "merge.yml").write_text(workflow, encoding="utf-8")

    bugs = WorkflowParser(workflow_dir).parse_all_workflows()

    yaml_errors = [b for b in bugs if "yaml" in b.message.lower()]
    assert not yaml_errors, f"Unexpected YAML errors for merge-key workflow: {yaml_errors}"


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


def test_non_mapping_yaml_root_reports_structured_error(
    workflow_dir: Path,
) -> None:
    """A YAML root that is a list should produce a structured error indicating
    the root must be a mapping (not the generic 'Failed to parse file' handler)."""
    (workflow_dir / "list_root.yml").write_text("- foo\n- bar\n", encoding="utf-8")

    bugs = WorkflowParser(workflow_dir).parse_all_workflows()

    assert len(bugs) == 1
    assert "mapping" in bugs[0].message.lower()
    assert bugs[0].severity == "error"


def test_yaml_merge_keys_are_preserved(
    workflow_dir: Path,
) -> None:
    """Workflows using YAML merge keys (<<: *anchor) should parse without error,
    and the merged keys must be visible to the validator."""
    import yaml  # noqa: PLC0415  (local import to keep top-level imports minimal)

    workflow_text = """
name: Merge Keys
on: [push]
_defaults: &defaults
  runs-on: ubuntu-latest
jobs:
  test:
    <<: *defaults
    steps:
      - run: echo "ok"
"""
    (workflow_dir / "merge_keys.yml").write_text(workflow_text, encoding="utf-8")

    bugs = WorkflowParser(workflow_dir).parse_all_workflows()

    # Merge keys should not produce any parse errors
    assert not any(bug.severity == "error" for bug in bugs)

    # Verify the merge key was actually applied (runs-on inherited from anchor)
    from transformation_portal.analyzers.parse_workflows import _DuplicateKeySafeLoader  # noqa: PLC0415

    parsed = yaml.load(workflow_text, Loader=_DuplicateKeySafeLoader)
    assert parsed["jobs"]["test"]["runs-on"] == "ubuntu-latest"
