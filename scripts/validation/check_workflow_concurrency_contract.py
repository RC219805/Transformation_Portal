#!/usr/bin/env python3
"""Validate workflow concurrency isolation for mixed schedule/event workflows.

This guard prevents scheduled workflows from cancelling push or pull_request
workflows on the same branch when cancel-in-progress is enabled.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

EVENT_NAME_SNIPPET = "github.event_name"


def _normalize_workflow_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return a workflow mapping with YAML 1.1 boolean keys normalized."""
    if not isinstance(config, dict):
        return {}
    if True in config and "on" not in config:
        config = dict(config)
        config["on"] = config.pop(True)
    return config


def _has_trigger(config: dict[str, Any], trigger_name: str) -> bool:
    """Return True when the workflow declares the given trigger."""
    triggers = config.get("on")
    if isinstance(triggers, dict):
        return trigger_name in triggers
    if isinstance(triggers, list):
        return trigger_name in triggers
    if isinstance(triggers, str):
        return triggers == trigger_name
    return False


def validate_workflow_concurrency_contract_text(workflow_name: str, text: str) -> list[str]:
    """Return concurrency contract violations for a single workflow file."""
    config = _normalize_workflow_config(yaml.safe_load(text))
    errors: list[str] = []

    has_schedule = _has_trigger(config, "schedule")
    has_push = _has_trigger(config, "push")
    has_pull_request = _has_trigger(config, "pull_request")

    if not has_schedule or not (has_push or has_pull_request):
        return errors

    concurrency = config.get("concurrency")
    if not isinstance(concurrency, dict):
        return errors

    if concurrency.get("cancel-in-progress") is not True:
        return errors

    group = concurrency.get("group")
    if not isinstance(group, str):
        errors.append(
            f"{workflow_name}: mixed schedule/push workflow must define a string concurrency.group when "
            "cancel-in-progress is true"
        )
        return errors

    if EVENT_NAME_SNIPPET not in group:
        errors.append(
            f"{workflow_name}: mixed schedule/push workflow must include {EVENT_NAME_SNIPPET!r} in "
            f"concurrency.group when cancel-in-progress is true (current: {group!r})"
        )

    return errors


def validate_repo_workflow_concurrency_contract(workflows_dir: Path = WORKFLOWS_DIR) -> list[str]:
    """Validate all workflow files in the repository."""
    errors: list[str] = []

    for workflow_path in sorted(workflows_dir.glob("*.yml")) + sorted(workflows_dir.glob("*.yaml")):
        errors.extend(validate_workflow_concurrency_contract_text(workflow_path.name, workflow_path.read_text()))

    return errors


def main() -> int:
    errors = validate_repo_workflow_concurrency_contract()

    if errors:
        print("ERROR: workflow concurrency contract validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("workflow concurrency contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
