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
6. Repository-owned dependency targets must be validated before lock generation.
7. The update transaction must bypass pip's Simple API cache so pip 26.2 does
   not hide newly published releases from scheduled resolution.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "dependency-update.yml"

AUDIT_TARGETS_BLOCK_RE = re.compile(r"(?ms)^[ \t]*audit_targets\s*=\s*\(\s*$\n(?P<body>.*?)^\s*\)\s*$")
PREFLIGHT_STEP_NAME = "Preflight dependency update targets"
INSTALL_STEP_NAME = "Install lock generation tools"
UPDATE_STEP_NAME = "Update dependencies"
AUDIT_STEP_NAME = "Check for vulnerabilities"
UPLOAD_STEP_NAME = "Upload pip-audit report"
CREATE_PR_STEP_NAME = "Create Pull Request"
FREE_DISK_STEP_NAME = "Free disk space"
LOCK_AUTHORITY_STEP_NAME = "Check lock ownership authority"
VERIFY_LOCK_STEP_NAME = "Verify lockfile contract"
TRUSTED_CHECKOUT_ACTION = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
TRUSTED_SETUP_PYTHON_ACTION = "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"
TRUSTED_UPLOAD_ACTION = "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
TRUSTED_CREATE_PR_ACTION = "peter-evans/create-pull-request@5f6978faf089d4d20b00c7766989d076bb2fc7f1"
TRUSTED_SETUP_PYTHON_ID = "setup-python"
TRUSTED_PYTHON = '"${{ steps.setup-python.outputs.python-path }}" -I'
TRUSTED_PREFLIGHT_SHELL = "/usr/bin/env -u BASH_ENV -u ENV /bin/bash --noprofile --norc -p -e -o pipefail {0}"
EXPECTED_TRIGGER_CONFIG = {
    "schedule": [{"cron": "0 9 * * 1"}],
    "workflow_dispatch": None,
}

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

REQUIRED_PR_ADD_PATHS = "".join(f"{path}\n" for path in REQUIRED_AUDIT_TARGETS)

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

REQUIRED_PREFLIGHT_COMMANDS = (
    "set -euo pipefail",
    f"{TRUSTED_PYTHON} scripts/validation/check_dependabot_config.py",
    f"{TRUSTED_PYTHON} scripts/validation/check_worker_dependency_parity.py",
    f"{TRUSTED_PYTHON} scripts/validation/check_lock_ownership.py --context ubuntu-x64-generic",
)

REQUIRED_UPDATE_COMMANDS = (
    "set -euo pipefail",
    'if [ ! -d "requirements" ]; then',
    'echo "requirements/ directory not found"',
    "exit 1",
    "fi",
    'echo "Updating generic governed requirements with pip-compile..."',
    "cd requirements",
    "make update-generic LOCK_PYTHON_VERSION=3.11",
    "make check-generic LOCK_PYTHON_VERSION=3.11",
    "cd ..",
)

REQUIRED_FREE_DISK_COMMANDS = (
    'echo "Disk space before cleanup:"',
    "df -h",
    "sudo rm -rf /usr/share/dotnet || true",
    "sudo rm -rf /opt/ghc || true",
    "sudo rm -rf /usr/local/lib/android || true",
    "sudo rm -rf /opt/hostedtoolcache/CodeQL || true",
    "sudo rm -rf /usr/local/share/boost || true",
    "sudo docker image prune --all --force || true",
    "sudo docker container prune --force || true",
    "sudo apt-get clean || true",
    'echo "Disk space after cleanup:"',
    "df -h",
)

REQUIRED_LOCK_AUTHORITY_COMMANDS = (
    "set -euo pipefail",
    'baseline_sha="${{ github.sha }}"',
    "current_sha=\"$(/usr/bin/git rev-parse --verify 'HEAD^{commit}')\"",
    'if [ "${current_sha}" != "${baseline_sha}" ]; then',
    'echo "ERROR: dependency automation must not create or amend commits before publication."',
    "exit 1",
    "fi",
    'changed_files_path="$(mktemp)"',
    "trap 'rm -f \"${changed_files_path}\"' EXIT",
    '/usr/bin/git diff --name-only "${baseline_sha}" > "${changed_files_path}"',
    '/usr/bin/git ls-files --others --exclude-standard >> "${changed_files_path}"',
    'LC_ALL=C sort -u -o "${changed_files_path}" "${changed_files_path}"',
    f"{TRUSTED_PYTHON} scripts/validation/check_lock_ownership.py --context ubuntu-x64-generic "
    '--changed-files-file "${changed_files_path}"',
    "current_sha=\"$(/usr/bin/git rev-parse --verify 'HEAD^{commit}')\"",
    'if [ "${current_sha}" != "${baseline_sha}" ]; then',
    'echo "ERROR: dependency automation must not create or amend commits during publication validation."',
    "exit 1",
    "fi",
    '/usr/bin/git diff --name-only "${baseline_sha}" > "${changed_files_path}"',
    '/usr/bin/git ls-files --others --exclude-standard >> "${changed_files_path}"',
    'LC_ALL=C sort -u -o "${changed_files_path}" "${changed_files_path}"',
    "if grep -Eq '^requirements/ml-core-darwin-arm64.txt$' \"${changed_files_path}\"; then",
    'echo "ERROR: scheduled dependency automation must not modify target-owned ML locks."',
    "exit 1",
    "fi",
    "if grep -Eqv '^requirements/(all|base|ci|dev|security|tools-archive)\\.txt$' " '"${changed_files_path}"; then',
    'echo "ERROR: scheduled dependency automation produced unexpected requirements changes:"',
    "grep -Ev '^requirements/(all|base|ci|dev|security|tools-archive)\\.txt$' " '"${changed_files_path}"',
    "exit 1",
    "fi",
)

REQUIRED_VERIFY_LOCK_COMMANDS = (f"{TRUSTED_PYTHON} scripts/validation/check_requirements_lock_contract.py",)

REQUIRED_AUDIT_COMMANDS = (
    "set -euo pipefail",
    'audit_reports_dir="${{ runner.temp }}/dependency-update-audit-reports"',
    'rm -rf "${audit_reports_dir}"',
    'mkdir -p "${audit_reports_dir}"',
    "audit_targets=(",
    *REQUIRED_AUDIT_TARGETS,
    ")",
    'for requirement_file in "${audit_targets[@]}"; do',
    'report_path="${audit_reports_dir}/$(basename "${requirement_file%.txt}").json"',
    'echo "Auditing ${requirement_file} -> ${report_path}"',
    'pip-audit -r "${requirement_file}" --format json --output "${report_path}" || true',
    "done",
)

EXPECTED_STEP_IDENTITIES = (
    TRUSTED_CHECKOUT_ACTION,
    TRUSTED_SETUP_PYTHON_ACTION,
    INSTALL_STEP_NAME,
    PREFLIGHT_STEP_NAME,
    FREE_DISK_STEP_NAME,
    UPDATE_STEP_NAME,
    VERIFY_LOCK_STEP_NAME,
    AUDIT_STEP_NAME,
    UPLOAD_STEP_NAME,
    LOCK_AUTHORITY_STEP_NAME,
    CREATE_PR_STEP_NAME,
)

REQUIRED_INSTALL_TOOLCHAIN_SNIPPETS = (
    f'{TRUSTED_PYTHON} -m pip --isolated install --upgrade "pip==26.2.1"',
    f'{TRUSTED_PYTHON} -m pip --isolated install "pip-tools==7.6.1"',
    f"{TRUSTED_PYTHON} -m pip --isolated install -r requirements/security.txt",
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


def _extract_audit_targets_block(text: str) -> str | None:
    """Return the audit_targets block content, if present."""
    match = AUDIT_TARGETS_BLOCK_RE.search(text)
    if match is None:
        return None
    return match.group("body")


def _github_actions_yaml_loader() -> type[Any]:
    """Return a safe loader that preserves GitHub Actions' literal ``on`` key."""
    if yaml is None:
        raise ValueError("PyYAML not installed (pip install PyYAML)")

    loader = type("_GitHubActionsLoader", (yaml.SafeLoader,), {})
    bool_tag = "tag:yaml.org,2002:bool"
    loader.yaml_implicit_resolvers = {
        initial: [(tag, pattern) for tag, pattern in resolvers if tag != bool_tag]
        for initial, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }
    loader.add_implicit_resolver(
        bool_tag,
        re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
        list("tTfF"),
    )
    return loader


def _load_dependency_job(text: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Parse and return the workflow, updater job, and its real step mappings."""
    if yaml is None:
        raise ValueError("PyYAML not installed (pip install PyYAML)")
    try:
        config = yaml.load(text, Loader=_github_actions_yaml_loader())
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("dependency-update workflow must be a YAML mapping")
    jobs = config.get("jobs")
    if not isinstance(jobs, dict):
        raise ValueError("dependency-update workflow must define a jobs mapping")
    job = jobs.get("update-dependencies")
    if not isinstance(job, dict):
        raise ValueError("dependency-update workflow must define jobs.update-dependencies")
    raw_steps = job.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("dependency-update workflow must define jobs.update-dependencies.steps as a list")
    if any(not isinstance(step, dict) for step in raw_steps):
        raise ValueError("dependency-update workflow steps must all be mappings")
    return config, job, raw_steps


def _named_step_matches(steps: list[dict[str, Any]], name: str) -> list[tuple[int, dict[str, Any]]]:
    """Return structurally parsed steps with the exact display name."""
    return [(index, step) for index, step in enumerate(steps) if step.get("name") == name]


def _single_named_step(
    steps: list[dict[str, Any]],
    name: str,
    *,
    errors: list[str],
) -> tuple[int, dict[str, Any]] | None:
    """Return one named step and report missing or duplicate mappings."""
    matches = _named_step_matches(steps, name)
    if len(matches) != 1:
        errors.append(f"dependency-update workflow must define exactly one {name!r} step mapping")
        return None
    return matches[0]


def _logical_shell_commands(run_block: str) -> tuple[str, ...]:
    """Return normalized executable lines, joining shell continuations."""
    commands: list[str] = []
    continuation: list[str] = []
    for raw_line in run_block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        continued = line.endswith("\\")
        if continued:
            line = line[:-1].rstrip()
        continuation.append(line)
        if continued:
            continue
        commands.append(re.sub(r"\s+", " ", " ".join(continuation)).strip())
        continuation = []
    if continuation:
        commands.append(re.sub(r"\s+", " ", " ".join(continuation)).strip())
    return tuple(commands)


def validate_dependency_update_workflow(text: str) -> list[str]:
    """Return workflow contract violations."""
    errors: list[str] = []
    try:
        config, update_job, steps = _load_dependency_job(text)
    except ValueError as exc:
        return [str(exc)]

    if set(config) != {"name", "on", "permissions", "jobs"}:
        errors.append("dependency-update workflow must retain only its governed top-level fields")
    if config.get("name") != "Dependency Updates":
        errors.append("dependency-update workflow must retain its governed display name")
    if config.get("on") != EXPECTED_TRIGGER_CONFIG:
        errors.append("dependency-update workflow must retain the exact weekly and manual trigger envelope")
    if "env" in config or "defaults" in config:
        errors.append("dependency-update workflow must not define inherited env or run defaults")
    jobs = config.get("jobs")
    if not isinstance(jobs, dict) or set(jobs) != {"update-dependencies"}:
        errors.append("dependency-update workflow must define only the governed updater job")
    if config.get("permissions") != {"contents": "write", "pull-requests": "write"}:
        errors.append("dependency-update workflow must retain only its governed write permissions")
    unexpected_job_fields = set(update_job) - {"name", "runs-on", "steps"}
    if unexpected_job_fields:
        errors.append("dependency-update job must not define execution overrides: " + ", ".join(sorted(unexpected_job_fields)))
    if update_job.get("name") != "Update Python Dependencies":
        errors.append("dependency-update job must retain its governed display name")
    if update_job.get("runs-on") != "ubuntu-24.04":
        errors.append("dependency-update job must run on the trusted GitHub-hosted ubuntu-24.04 runner")

    actual_step_identities = tuple(step.get("name", step.get("uses")) for step in steps)
    if actual_step_identities != EXPECTED_STEP_IDENTITIES:
        errors.append("dependency-update steps must match the exact governed sequence")

    checkout_matches = [(index, step) for index, step in enumerate(steps) if step.get("uses") == TRUSTED_CHECKOUT_ACTION]
    if len(checkout_matches) != 1 or checkout_matches[0][0] != 0:
        errors.append("dependency-update must start with exactly one pinned checkout step")
    elif checkout_matches[0][1] != {
        "uses": TRUSTED_CHECKOUT_ACTION,
        "with": {"token": "${{ secrets.GITHUB_TOKEN }}"},
    }:
        errors.append("dependency-update checkout must use only the current repository and trusted token")

    setup_matches = [
        (index, step)
        for index, step in enumerate(steps)
        if step.get("id") == TRUSTED_SETUP_PYTHON_ID or step.get("uses") == TRUSTED_SETUP_PYTHON_ACTION
    ]
    if len(setup_matches) != 1:
        errors.append("dependency-update must define exactly one trusted setup-python step")
    else:
        setup_index, setup_step = setup_matches[0]
        if setup_index != 1:
            errors.append("dependency-update setup-python must immediately follow checkout")
        if setup_step != {
            "uses": TRUSTED_SETUP_PYTHON_ACTION,
            "id": TRUSTED_SETUP_PYTHON_ID,
            "with": {"python-version": "3.11"},
        }:
            errors.append("dependency-update setup-python must expose the pinned Python 3.11 interpreter output")

    install_match = _single_named_step(steps, INSTALL_STEP_NAME, errors=errors)
    install_run = ""
    if install_match is not None:
        install_index, install_step = install_match
        if install_index != 2:
            errors.append("dependency-update tool installation must immediately follow setup-python")
        if set(install_step) != {"name", "shell", "run"} or install_step.get("shell") != TRUSTED_PREFLIGHT_SHELL:
            errors.append("dependency-update tool installation must use only the trusted sanitized Bash shell")
        install_run = install_step.get("run", "")
        if not isinstance(install_run, str):
            errors.append("dependency-update tool installation must define a run script")
            install_run = ""
        if _logical_shell_commands(install_run) != REQUIRED_INSTALL_TOOLCHAIN_SNIPPETS:
            errors.append("dependency-update tool installation must use only the exact isolated pinned commands")

    preflight_match = _single_named_step(steps, PREFLIGHT_STEP_NAME, errors=errors)
    preflight_run = ""
    preflight_index: int | None = None
    if preflight_match is not None:
        preflight_index, preflight_step = preflight_match
        if preflight_index != 3:
            errors.append("dependency-update preflight must immediately follow trusted tool installation")
        preflight_run = preflight_step.get("run", "")
        if not isinstance(preflight_run, str):
            errors.append("dependency-update preflight must define a literal run block")
            preflight_run = ""
        preflight_commands = _logical_shell_commands(preflight_run)
        for command in REQUIRED_PREFLIGHT_COMMANDS:
            if command not in preflight_commands:
                errors.append("dependency-update preflight is missing a governed command")
        for command in preflight_commands:
            if command not in REQUIRED_PREFLIGHT_COMMANDS:
                errors.append("dependency-update preflight contains an unexpected command")
        if set(preflight_commands) == set(REQUIRED_PREFLIGHT_COMMANDS) and preflight_commands != REQUIRED_PREFLIGHT_COMMANDS:
            errors.append("dependency-update preflight commands must run in the required order")
        if "if" in preflight_step:
            errors.append("dependency-update preflight must not be conditionally skipped")
        if "continue-on-error" in preflight_step:
            errors.append("dependency-update preflight must not continue on error")
        if preflight_step.get("shell") != TRUSTED_PREFLIGHT_SHELL:
            errors.append("dependency-update preflight must explicitly use the trusted sanitized Bash shell")
        unexpected_step_fields = set(preflight_step) - {"name", "run", "shell"}
        if unexpected_step_fields:
            errors.append(
                "dependency-update preflight must not define execution overrides: " + ", ".join(sorted(unexpected_step_fields))
            )
        if re.search(r"(?:\|\|\s*(?:true|:)|;\s*true\s*$|^\s*set\s+\+e\s*$)", preflight_run, re.MULTILINE):
            errors.append("dependency-update preflight must not suppress command failures")

    free_disk_match = _single_named_step(steps, FREE_DISK_STEP_NAME, errors=errors)
    if free_disk_match is not None:
        free_disk_index, free_disk_step = free_disk_match
        free_disk_run = free_disk_step.get("run", "")
        if (
            free_disk_index != 4
            or set(free_disk_step) != {"name", "shell", "run"}
            or free_disk_step.get("shell") != TRUSTED_PREFLIGHT_SHELL
            or not isinstance(free_disk_run, str)
            or _logical_shell_commands(free_disk_run) != REQUIRED_FREE_DISK_COMMANDS
        ):
            errors.append("dependency-update disk cleanup must use only the exact sanitized commands")

    update_match = _single_named_step(steps, UPDATE_STEP_NAME, errors=errors)
    if update_match is not None and preflight_index is not None:
        update_index, update_step = update_match
        if preflight_index >= update_index:
            errors.append("dependency-update target preflight must run before generic lock generation")
        if (
            update_index != 5
            or set(update_step) != {"name", "shell", "env", "run"}
            or update_step.get("shell") != TRUSTED_PREFLIGHT_SHELL
            or update_step.get("env") != {"PIP_NO_CACHE_DIR": "1"}
        ):
            errors.append("dependency-update update step must use only the governed environment")
        update_run = update_step.get("run", "")
        if not isinstance(update_run, str) or _logical_shell_commands(update_run) != REQUIRED_UPDATE_COMMANDS:
            errors.append("dependency-update update step must use only the exact ordered generic lock commands")

    lock_authority_match = _single_named_step(steps, LOCK_AUTHORITY_STEP_NAME, errors=errors)
    if lock_authority_match is not None:
        lock_authority_index, lock_authority_step = lock_authority_match
        lock_authority_run = lock_authority_step.get("run", "")
        if (
            lock_authority_index != 9
            or set(lock_authority_step) != {"name", "shell", "run"}
            or lock_authority_step.get("shell") != TRUSTED_PREFLIGHT_SHELL
            or not isinstance(lock_authority_run, str)
            or _logical_shell_commands(lock_authority_run) != REQUIRED_LOCK_AUTHORITY_COMMANDS
        ):
            errors.append("dependency-update lock authority check must use only the exact sanitized commands")

    verify_lock_match = _single_named_step(steps, VERIFY_LOCK_STEP_NAME, errors=errors)
    if verify_lock_match is not None:
        verify_lock_index, verify_lock_step = verify_lock_match
        verify_lock_run = verify_lock_step.get("run", "")
        if (
            verify_lock_index != 6
            or set(verify_lock_step) != {"name", "shell", "run"}
            or verify_lock_step.get("shell") != TRUSTED_PREFLIGHT_SHELL
            or not isinstance(verify_lock_run, str)
            or _logical_shell_commands(verify_lock_run) != REQUIRED_VERIFY_LOCK_COMMANDS
        ):
            errors.append("dependency-update lock verification must use only the exact sanitized command")

    audit_match = _single_named_step(steps, AUDIT_STEP_NAME, errors=errors)
    audit_run = ""
    if audit_match is not None:
        audit_index, audit_step = audit_match
        audit_run = audit_step.get("run", "")
        if (
            audit_index != 7
            or set(audit_step) != {"name", "shell", "run"}
            or audit_step.get("shell") != TRUSTED_PREFLIGHT_SHELL
            or not isinstance(audit_run, str)
            or _logical_shell_commands(audit_run) != REQUIRED_AUDIT_COMMANDS
        ):
            errors.append("dependency-update vulnerability audit must use only the exact sanitized commands")
        if not isinstance(audit_run, str):
            audit_run = ""
    audit_targets_block = _extract_audit_targets_block(audit_run)

    upload_match = _single_named_step(steps, UPLOAD_STEP_NAME, errors=errors)
    upload_path = ""
    if upload_match is not None:
        upload_index, upload_step = upload_match
        upload_with = upload_step.get("with")
        if isinstance(upload_with, dict) and isinstance(upload_with.get("path"), str):
            upload_path = upload_with["path"]
        if upload_index != 8 or upload_step != {
            "name": UPLOAD_STEP_NAME,
            "uses": TRUSTED_UPLOAD_ACTION,
            "with": {
                "name": "pip-audit-report",
                "path": "${{ runner.temp }}/dependency-update-audit-reports/",
                "if-no-files-found": "warn",
                "retention-days": 30,
            },
        }:
            errors.append("dependency-update audit upload must use only the pinned action and governed inputs")

    create_pr_match = _single_named_step(steps, CREATE_PR_STEP_NAME, errors=errors)
    pr_body: str | None = None
    if create_pr_match is not None:
        create_pr_index, create_pr_step = create_pr_match
        create_pr_with = create_pr_step.get("with")
        if isinstance(create_pr_with, dict) and isinstance(create_pr_with.get("body"), str):
            pr_body = create_pr_with["body"]
        expected_pr_inputs = {
            "token": "${{ secrets.GITHUB_TOKEN }}",
            "commit-message": "chore: update dependencies (automated)",
            "title": "🔄 Automated Dependency Updates",
            "branch": "automated/dependency-updates",
            "add-paths": REQUIRED_PR_ADD_PATHS,
            "delete-branch": True,
            "labels": "dependencies\nautomated\n",
        }
        actual_pr_inputs = (
            {key: value for key, value in create_pr_with.items() if key != "body"}
            if isinstance(create_pr_with, dict)
            else None
        )
        if (
            create_pr_index != 10
            or set(create_pr_step) != {"name", "uses", "with"}
            or create_pr_step.get("uses") != TRUSTED_CREATE_PR_ACTION
            or not isinstance(create_pr_with, dict)
            or set(create_pr_with) != {*expected_pr_inputs, "body"}
            or actual_pr_inputs != expected_pr_inputs
        ):
            errors.append("dependency-update PR creation must use only the pinned action and governed inputs")

    workflow_commands = tuple(
        command for step in steps if isinstance(step.get("run"), str) for command in _logical_shell_commands(step["run"])
    )
    workflow_command_text = "\n".join(workflow_commands)
    workflow_run_text = "\n".join(step["run"] for step in steps if isinstance(step.get("run"), str))

    if audit_targets_block is None:
        errors.append("dependency-update workflow must define an audit_targets block")
    else:
        for target in REQUIRED_AUDIT_TARGETS:
            if target not in audit_targets_block:
                errors.append(f"dependency-update workflow must audit governed lockfile target {target!r}")

    for snippet in REQUIRED_WORKFLOW_SNIPPETS:
        if snippet not in workflow_command_text:
            errors.append(f"dependency-update workflow must include snippet {snippet!r}")

    for snippet in REQUIRED_AUDIT_REPORT_SNIPPETS[:-1]:
        if snippet not in audit_run:
            errors.append(f"dependency-update workflow must include audit-report snippet {snippet!r}")
    if upload_path != "${{ runner.temp }}/dependency-update-audit-reports/":
        errors.append(f"dependency-update workflow must include audit-report snippet {REQUIRED_AUDIT_REPORT_SNIPPETS[-1]!r}")

    for snippet in FORBIDDEN_WORKFLOW_SNIPPETS:
        if snippet in workflow_run_text or snippet in f"path: {upload_path}":
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
