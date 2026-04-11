#!/usr/bin/env python3
"""Validate dependency-update workflow contract.

This guard prevents the scheduled dependency-update workflow from drifting away
from the repository's actual checked-in dependency contract.

Contracts enforced:
1. The workflow must audit the governed lockfile targets listed in
   REQUIRED_AUDIT_TARGETS.
2. The workflow must use explicit generic/Linux target-owned update commands
   instead of broad multi-target regeneration.
3. The workflow must run the ownership validator for the Ubuntu generic/Linux
   authoritative contexts.
4. The Create Pull Request body must mention the required lockfile / ML
   contract references and must not mention superseded references.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "dependency-update.yml"

AUDIT_TARGETS_BLOCK_RE = re.compile(r"(?ms)^[ \t]*audit_targets\s*=\s*\(\s*$\n(?P<body>.*?)^\s*\)\s*$")

REQUIRED_AUDIT_TARGETS = (
    "requirements/all.txt",
    "requirements/base.txt",
    "requirements/dev.txt",
    "requirements/ci.txt",
    "requirements/security.txt",
    "requirements/tools-archive.txt",
    "requirements/ml-core-linux.txt",
)

REQUIRED_PR_BODY_REFERENCES = (
    "requirements/base.txt",
    "requirements/dev.txt",
    "requirements/ci.txt",
    "requirements/security.txt",
    "requirements/tools-archive.txt",
    "requirements/all.txt",
    "requirements/ml-core-linux.txt",
)

REQUIRED_PR_BODY_SNIPPETS = (
    "scheduled automation updates generic locks + `ml-core-linux.txt`",
    "`ml-core-darwin-arm64.txt` is not updated by scheduled automation",
    "`ml-core-darwin-x86_64.txt` is frozen pending an authoritative lane decision",
)

REQUIRED_WORKFLOW_SNIPPETS = (
    "make update-generic LOCK_PYTHON_VERSION=3.11",
    "make check-generic LOCK_PYTHON_VERSION=3.11",
    "make update-ml-linux-x86_64 LOCK_PYTHON_VERSION=3.11",
    "make check-ml-linux-x86_64 LOCK_PYTHON_VERSION=3.11",
    "scripts/validation/check_lock_ownership.py",
    "--context ubuntu-x64-generic",
    "--context ubuntu-x64-linux",
    "requirements/ml-core-darwin-arm64.txt",
    "requirements/ml-core-darwin-x86_64.txt",
)

REQUIRED_INSTALL_TOOLCHAIN_SNIPPETS = (
    'python -m pip install --upgrade "pip<26"',
    'python -m pip install "pip-tools==7.5.2"',
    "python -m pip install -r requirements/security.txt",
)

FORBIDDEN_WORKFLOW_SNIPPETS = (
    "make update LOCK_PYTHON_VERSION=3.11",
    "make check LOCK_PYTHON_VERSION=3.11",
    "make update-ml-darwin-arm64",
    "make check-ml-darwin-arm64",
    "make update-ml-darwin-x86_64",
    "make check-ml-darwin-x86_64",
    "make compile-ml-layers",
)

FORBIDDEN_PR_BODY_REFERENCES = (
    "requirements/ml.txt",
    "requirements/ml-core.txt",
    "requirements/ml-raw.txt",
    "requirements/ml-coreml.txt",
    "requirements/ml-research.txt",
)

FORBIDDEN_REVIEW_TEXT = (
    "ml-core.txt, ml.txt",
    "requirements/ml.txt",
    "requirements/ml-core.txt",
)


def _extract_create_pr_body(text: str) -> str | None:
    """Return the create-pull-request body block content, if present."""
    lines = text.splitlines()
    create_pr_start: int | None = None

    for index, line in enumerate(lines):
        if re.match(r"^\s*-\s+name:\s+Create Pull Request\s*$", line):
            create_pr_start = index
            break

    if create_pr_start is None:
        return None

    body_indent: int | None = None
    body_lines: list[str] = []

    for line in lines[create_pr_start + 1 :]:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))

        if body_indent is None:
            if re.match(r"^\s*body:\s*\|\s*$", line):
                body_indent = indent
            continue

        if stripped and indent <= body_indent:
            break

        body_lines.append(line[body_indent + 2 :] if line.startswith(" " * (body_indent + 2)) else "")

    if body_indent is None:
        return None

    return "\n".join(body_lines)


def _extract_audit_targets_block(text: str) -> str | None:
    """Return the audit_targets block content, if present."""
    match = AUDIT_TARGETS_BLOCK_RE.search(text)
    if match is None:
        return None
    return match.group("body")


def validate_dependency_update_workflow(text: str) -> list[str]:
    """Return workflow contract violations."""
    errors: list[str] = []
    audit_targets_block = _extract_audit_targets_block(text)
    pr_body = _extract_create_pr_body(text)

    if audit_targets_block is None:
        errors.append("dependency-update workflow must define an audit_targets block")
    else:
        for target in REQUIRED_AUDIT_TARGETS:
            if target not in audit_targets_block:
                errors.append(f"dependency-update workflow must audit governed lockfile target {target!r}")

    for snippet in REQUIRED_WORKFLOW_SNIPPETS:
        if snippet not in text:
            errors.append(f"dependency-update workflow must include snippet {snippet!r}")

    for snippet in REQUIRED_INSTALL_TOOLCHAIN_SNIPPETS:
        if snippet not in text:
            errors.append(f"dependency-update workflow must include install-tool snippet {snippet!r}")

    for snippet in FORBIDDEN_WORKFLOW_SNIPPETS:
        if snippet in text:
            errors.append(f"dependency-update workflow must not include snippet {snippet!r}")

    if pr_body is None:
        return errors + ["dependency-update workflow must define a Create Pull Request body block"]

    for ref in REQUIRED_PR_BODY_REFERENCES:
        if ref not in pr_body:
            errors.append(f"dependency-update PR body must reference checked-in contract file {ref!r}")

    for snippet in REQUIRED_PR_BODY_SNIPPETS:
        if snippet not in pr_body:
            errors.append(f"dependency-update PR body must include snippet {snippet!r}")

    for ref in FORBIDDEN_PR_BODY_REFERENCES:
        if ref in pr_body:
            errors.append(f"dependency-update workflow still references non-contract ML lockfile {ref!r}")

    for snippet in FORBIDDEN_REVIEW_TEXT:
        if snippet in pr_body:
            errors.append(
                "dependency-update workflow review checklist still points at deprecated " f"ML lockfile text {snippet!r}"
            )

    return errors


def main() -> int:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    errors = validate_dependency_update_workflow(text)

    if errors:
        print("ERROR: dependency-update workflow contract validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("dependency-update workflow contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
