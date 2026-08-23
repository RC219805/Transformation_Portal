#!/usr/bin/env python3
"""Validate dependency-update workflow contract.

This guard prevents the scheduled dependency-update workflow from drifting away
from the repository's actual checked-in dependency contract.

Contracts enforced:
1. The workflow must audit the governed generic lockfile targets listed in
   REQUIRED_AUDIT_TARGETS.
2. The workflow must use explicit generic update commands instead of broad or
   target-owned regeneration.
3. The workflow must run the ownership validator for the Ubuntu generic
   authoritative context only.
4. The Create Pull Request body must mention the required lockfile / ML
   contract references and must not mention superseded, retired, or frozen-lane
   updates as installable manifests.
5. Dependency-update audit reports must be written outside the git checkout
   and uploaded from that temp location only.
6. The update transaction must bypass pip's Simple API cache so pip 26.2 does
   not hide newly published releases from scheduled resolution.
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
)

REQUIRED_PR_BODY_REFERENCES = (
    "requirements/base.txt",
    "requirements/dev.txt",
    "requirements/ci.txt",
    "requirements/security.txt",
    "requirements/tools-archive.txt",
    "requirements/all.txt",
)

REQUIRED_PR_BODY_SNIPPETS = (
    "scheduled automation updates generic locks only",
    "`ml-core-darwin-arm64.txt` remains a manual Apple Silicon authoritative lane",
    "retired Linux/macOS Intel ML lanes are not checked-in installable requirements",
)

REQUIRED_WORKFLOW_SNIPPETS = (
    "make update-generic LOCK_PYTHON_VERSION=3.11",
    "make check-generic LOCK_PYTHON_VERSION=3.11",
    "scripts/validation/check_lock_ownership.py",
    "--context ubuntu-x64-generic",
    "requirements/ml-core-darwin-arm64.txt",
)

REQUIRED_INSTALL_TOOLCHAIN_SNIPPETS = (
    'python -m pip install --upgrade "pip==26.2.1"',
    'python -m pip install "pip-tools==7.6.1"',
    "python -m pip install -r requirements/security.txt",
)

REQUIRED_AUDIT_REPORT_SNIPPETS = (
    'audit_reports_dir="${{ runner.temp }}/dependency-update-audit-reports"',
    'rm -rf "${audit_reports_dir}"',
    'mkdir -p "${audit_reports_dir}"',
    'report_path="${audit_reports_dir}/$(basename "${requirement_file%.txt}").json"',
    "path: ${{ runner.temp }}/dependency-update-audit-reports/",
)

FORBIDDEN_WORKFLOW_SNIPPETS = (
    "make update LOCK_PYTHON_VERSION=3.11",
    "make check LOCK_PYTHON_VERSION=3.11",
    "make update-ml-linux-x86_64",
    "make check-ml-linux-x86_64",
    "make update-ml-darwin-arm64",
    "make check-ml-darwin-arm64",
    "make update-ml-darwin-x86_64",
    "make check-ml-darwin-x86_64",
    "make compile-ml-layers",
    "mkdir -p audit-reports",
    'report_path="audit-reports/$(basename "${requirement_file%.txt}").json"',
    "path: audit-reports/",
    "--ignore-vuln CVE-2026-4539",
    "CVE-2026-4539 (pygments): No fix available yet",
)

FORBIDDEN_PR_BODY_REFERENCES = (
    "requirements/ml.txt",
    "requirements/ml-core.txt",
    "requirements/ml-raw.txt",
    "requirements/ml-coreml.txt",
    "requirements/ml-research.txt",
)

FORBIDDEN_PR_BODY_SNIPPETS = (
    "- `requirements/ml-core-linux.txt` - Linux x86_64 ML core contract",
    "scheduled automation updates generic locks + `ml-core-linux.txt`",
    "- `ml-core-linux.txt` is the Linux x86_64 target-owned ML baseline",
    "`ml-core-darwin-x86_64.txt` is frozen pending an authoritative lane decision",
    "`ml-core-linux.txt` is frozen as an unsupported historical lane",
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


def _extract_named_step(text: str, name: str) -> str | None:
    """Return one workflow step, bounded by its YAML indentation."""
    lines = text.splitlines()
    step_pattern = re.compile(rf"^\s*-\s+name:\s+{re.escape(name)}\s*$")

    for index, line in enumerate(lines):
        if not step_pattern.match(line):
            continue

        step_indent = len(line) - len(line.lstrip(" "))
        step_lines = [line]
        for candidate in lines[index + 1 :]:
            stripped = candidate.strip()
            indent = len(candidate) - len(candidate.lstrip(" "))
            if stripped and indent <= step_indent:
                break
            step_lines.append(candidate)
        return "\n".join(step_lines)

    return None


def _extract_step_env_value(step: str, key: str) -> str | None:
    """Return a scalar from a step-local env map, without accepting run text."""
    lines = step.splitlines()
    for index, line in enumerate(lines):
        if not re.match(r"^\s*env:\s*$", line):
            continue

        env_indent = len(line) - len(line.lstrip(" "))
        for candidate in lines[index + 1 :]:
            stripped = candidate.strip()
            indent = len(candidate) - len(candidate.lstrip(" "))
            if stripped and indent <= env_indent:
                break
            match = re.match(rf"^\s*{re.escape(key)}:\s*(?P<value>.+?)\s*$", candidate)
            if match is None:
                continue
            value = match.group("value").split("#", 1)[0].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            return value
        return None

    return None


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
    update_step = _extract_named_step(text, "Update dependencies")

    if audit_targets_block is None:
        errors.append("dependency-update workflow must define an audit_targets block")
    else:
        for target in REQUIRED_AUDIT_TARGETS:
            if target not in audit_targets_block:
                errors.append(f"dependency-update workflow must audit governed lockfile target {target!r}")

    for snippet in REQUIRED_WORKFLOW_SNIPPETS:
        if snippet not in text:
            errors.append(f"dependency-update workflow must include snippet {snippet!r}")

    if update_step is None or _extract_step_env_value(update_step, "PIP_NO_CACHE_DIR") != "1":
        errors.append("dependency-update Update dependencies step must set env.PIP_NO_CACHE_DIR to '1'")

    for snippet in REQUIRED_INSTALL_TOOLCHAIN_SNIPPETS:
        if snippet not in text:
            errors.append(f"dependency-update workflow must include install-tool snippet {snippet!r}")

    for snippet in REQUIRED_AUDIT_REPORT_SNIPPETS:
        if snippet not in text:
            errors.append(f"dependency-update workflow must include audit-report snippet {snippet!r}")

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

    for snippet in FORBIDDEN_PR_BODY_SNIPPETS:
        if snippet in pr_body:
            errors.append(f"dependency-update workflow still references frozen-lane PR body text {snippet!r}")

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
