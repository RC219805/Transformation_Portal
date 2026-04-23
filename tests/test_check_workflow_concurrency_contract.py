import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "scripts" / "validation" / "check_workflow_concurrency_contract.py"
SPEC = importlib.util.spec_from_file_location("check_workflow_concurrency_contract", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
workflow_contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow_contract)


def test_branch_only_group_fails_for_mixed_schedule_push_workflow() -> None:
    text = """
name: Broken Workflow
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 1 * * *'
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo ok
"""
    errors = workflow_contract.validate_workflow_concurrency_contract_text("broken.yml", text)
    assert errors == [
        "broken.yml: mixed schedule/push workflow must include an interpolated '${{ github.event_name }}' token "
        "in concurrency.group when cancel-in-progress is enabled (current: '${{ github.workflow }}-${{ github.ref }}')"
    ]


def test_event_namespaced_group_passes_for_mixed_schedule_push_workflow() -> None:
    text = """
name: Safe Workflow
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 1 * * *'
concurrency:
  group: ${{ github.workflow }}-${{ github.event_name }}-${{ github.ref }}
  cancel-in-progress: true
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo ok
"""
    assert workflow_contract.validate_workflow_concurrency_contract_text("safe.yml", text) == []


def test_literal_event_name_text_without_expression_fails() -> None:
    text = """
name: Fake Event Namespace
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 1 * * *'
concurrency:
  group: mywf-github.event_name-${{ github.ref }}
  cancel-in-progress: true
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo ok
"""
    errors = workflow_contract.validate_workflow_concurrency_contract_text("fake.yml", text)
    assert errors == [
        "fake.yml: mixed schedule/push workflow must include an interpolated '${{ github.event_name }}' token "
        "in concurrency.group when cancel-in-progress is enabled (current: 'mywf-github.event_name-${{ github.ref }}')"
    ]


def test_schedule_only_workflow_is_exempt() -> None:
    text = """
name: Schedule Only
on:
  schedule:
    - cron: '0 1 * * *'
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo ok
"""
    assert workflow_contract.validate_workflow_concurrency_contract_text("schedule-only.yml", text) == []


def test_mixed_schedule_push_workflow_with_cancel_disabled_is_exempt() -> None:
    text = """
name: Non Cancelling Workflow
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 1 * * *'
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: false
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo ok
"""
    assert workflow_contract.validate_workflow_concurrency_contract_text("no-cancel.yml", text) == []


def test_expression_based_cancel_in_progress_requires_event_namespace() -> None:
    text = """
name: Conditional Cancellation
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 1 * * *'
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.ref == 'refs/heads/main' }}
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo ok
"""
    errors = workflow_contract.validate_workflow_concurrency_contract_text("conditional.yml", text)
    assert errors == [
        "conditional.yml: mixed schedule/push workflow must include an interpolated '${{ github.event_name }}' token "
        "in concurrency.group when cancel-in-progress is enabled (current: '${{ github.workflow }}-${{ github.ref }}')"
    ]


def test_invalid_yaml_is_reported_as_file_scoped_contract_violation() -> None:
    errors = workflow_contract.validate_workflow_concurrency_contract_text("broken.yml", "on: [")
    assert len(errors) == 1
    assert errors[0].startswith("broken.yml: invalid YAML:")


def test_missing_pyyaml_is_reported_as_file_scoped_contract_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workflow_contract, "yaml", None)
    errors = workflow_contract.validate_workflow_concurrency_contract_text("broken.yml", "on: push")
    assert errors == ["broken.yml: PyYAML not installed (pip install PyYAML)"]


def test_repo_validation_fails_closed_when_workflows_dir_is_missing(tmp_path: Path) -> None:
    missing_dir = tmp_path / "workflows"
    assert workflow_contract.validate_repo_workflow_concurrency_contract(missing_dir) == [
        f"{missing_dir}: workflows directory is missing or not a directory"
    ]


def test_repo_validation_fails_closed_when_workflows_dir_has_no_yaml_files(tmp_path: Path) -> None:
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    (workflows_dir / "README.md").write_text("docs only", encoding="utf-8")

    assert workflow_contract.validate_repo_workflow_concurrency_contract(workflows_dir) == [
        f"{workflows_dir}: no workflow files found"
    ]
