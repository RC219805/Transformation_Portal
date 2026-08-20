import importlib.util
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "scripts" / "validation" / "check_dependency_update_workflow.py"
SPEC = importlib.util.spec_from_file_location("check_dependency_update_workflow", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
workflow_contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow_contract)


def valid_workflow_text() -> str:
    required_pr_refs = "\n".join(f"          {ref}" for ref in workflow_contract.REQUIRED_PR_BODY_REFERENCES)
    required_pr_snippets = "\n".join(f"          {snippet}" for snippet in workflow_contract.REQUIRED_PR_BODY_SNIPPETS)
    required_update_commands = "\n".join(f"        {command}" for command in workflow_contract.REQUIRED_UPDATE_COMMANDS)
    required_free_disk_commands = "\n".join(f"        {command}" for command in workflow_contract.REQUIRED_FREE_DISK_COMMANDS)
    required_lock_authority_commands = "\n".join(
        f"        {command}" for command in workflow_contract.REQUIRED_LOCK_AUTHORITY_COMMANDS
    )
    required_verify_lock_commands = "\n".join(
        f"        {command}" for command in workflow_contract.REQUIRED_VERIFY_LOCK_COMMANDS
    )
    required_preflight_snippets = "\n".join(f"        {command}" for command in workflow_contract.REQUIRED_PREFLIGHT_COMMANDS)
    required_install_snippets = "\n".join(
        f"        {snippet}" for snippet in workflow_contract.REQUIRED_INSTALL_TOOLCHAIN_SNIPPETS
    )
    required_audit_commands = "\n".join(f"        {command}" for command in workflow_contract.REQUIRED_AUDIT_COMMANDS)
    return f"""
name: Dependency Updates

on:
  schedule:
    - cron: '0 9 * * 1'
  workflow_dispatch:

permissions:
  contents: write
  pull-requests: write

jobs:
  update-dependencies:
    name: Update Python Dependencies
    runs-on: ubuntu-24.04
    steps:
    - uses: {workflow_contract.TRUSTED_CHECKOUT_ACTION}
      with:
        token: ${{{{ secrets.GITHUB_TOKEN }}}}
    - uses: {workflow_contract.TRUSTED_SETUP_PYTHON_ACTION}
      id: {workflow_contract.TRUSTED_SETUP_PYTHON_ID}
      with:
        python-version: "3.11"
    - name: Install lock generation tools
      shell: {workflow_contract.TRUSTED_PREFLIGHT_SHELL}
      run: |
{required_install_snippets}
    - name: Preflight dependency update targets
      shell: {workflow_contract.TRUSTED_PREFLIGHT_SHELL}
      run: |
{required_preflight_snippets}
    - name: Free disk space
      shell: {workflow_contract.TRUSTED_PREFLIGHT_SHELL}
      run: |
{required_free_disk_commands}
    - name: Update dependencies
      shell: {workflow_contract.TRUSTED_PREFLIGHT_SHELL}
      env:
        PIP_NO_CACHE_DIR: "1"
      run: |
{required_update_commands}
    - name: Verify lockfile contract
      shell: {workflow_contract.TRUSTED_PREFLIGHT_SHELL}
      run: |
{required_verify_lock_commands}
    - name: Check for vulnerabilities
      shell: {workflow_contract.TRUSTED_PREFLIGHT_SHELL}
      run: |
{required_audit_commands}
    - name: Upload pip-audit report
      uses: {workflow_contract.TRUSTED_UPLOAD_ACTION}
      with:
        name: pip-audit-report
        path: ${{{{ runner.temp }}}}/dependency-update-audit-reports/
        if-no-files-found: warn
        retention-days: 30
    - name: Check lock ownership authority
      shell: {workflow_contract.TRUSTED_PREFLIGHT_SHELL}
      run: |
{required_lock_authority_commands}
    - name: Create Pull Request
      uses: {workflow_contract.TRUSTED_CREATE_PR_ACTION}
      with:
        token: ${{{{ secrets.GITHUB_TOKEN }}}}
        commit-message: "chore: update dependencies (automated)"
        title: "🔄 Automated Dependency Updates"
        body: |
{required_pr_refs}
{required_pr_snippets}
          Confirm target-owned ML lock contracts
        branch: automated/dependency-updates
        add-paths: |
          requirements/all.txt
          requirements/base.txt
          requirements/dev.txt
          requirements/ci.txt
          requirements/security.txt
          requirements/tools-archive.txt
        delete-branch: true
        labels: |
          dependencies
          automated
    """


def remove_from_pr_body(text: str, needle: str) -> str:
    before_body, body = text.split("        body: |\n", maxsplit=1)
    body_content, after_body = body.split("        branch: automated/dependency-updates\n", maxsplit=1)
    body_without_ref = body_content.replace(f"          {needle}\n", "", 1)
    return f"{before_body}        body: |\n{body_without_ref}        branch: automated/dependency-updates\n{after_body}"


def remove_from_audit_targets(text: str, needle: str) -> str:
    before_block, rest = text.split("        audit_targets=(\n", maxsplit=1)
    audit_targets_block, after_block = rest.split("        )\n", maxsplit=1)
    audit_targets_block = audit_targets_block.replace(f"{needle}\n", "", 1)
    return f"{before_block}        audit_targets=(\n{audit_targets_block}        )\n{after_block}"


def remove_audit_targets_block(text: str) -> str:
    before_block, rest = text.split("        audit_targets=(\n", maxsplit=1)
    _, after_block = rest.split("        )\n", maxsplit=1)
    return f"{before_block}{after_block}"


def add_to_pr_body(text: str, line: str) -> str:
    before_body, body = text.split("        body: |\n", maxsplit=1)
    body_content, after_body = body.split("        branch: automated/dependency-updates\n", maxsplit=1)
    body_with_ref = body_content + f"          {line}\n"
    return f"{before_body}        body: |\n{body_with_ref}        branch: automated/dependency-updates\n{after_body}"


def remove_workflow_snippet(text: str, snippet: str) -> str:
    return text.replace(f"        {snippet}\n", "", 1)


def add_to_step_run(text: str, step_name: str, *lines: str) -> str:
    marker = f"    - name: {step_name}\n" f"      shell: {workflow_contract.TRUSTED_PREFLIGHT_SHELL}\n" "      run: |\n"
    addition = "".join(f"        {line}\n" for line in lines)
    return text.replace(marker, marker + addition, 1)


def test_valid_dependency_update_workflow_contract_passes() -> None:
    assert workflow_contract.validate_dependency_update_workflow(valid_workflow_text()) == []


def test_missing_required_audit_target_is_reported() -> None:
    broken = remove_from_audit_targets(valid_workflow_text(), "requirements/security.txt")
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert ("dependency-update workflow must audit governed lockfile target " "'requirements/security.txt'") in errors


def test_stale_noncontract_ml_reference_is_reported() -> None:
    broken = add_to_pr_body(valid_workflow_text(), "requirements/ml.txt")
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert ("dependency-update workflow still references non-contract ML lockfile " "'requirements/ml.txt'") in errors


def test_missing_required_pr_body_reference_is_reported() -> None:
    broken = remove_from_pr_body(valid_workflow_text(), "requirements/security.txt")
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert ("dependency-update PR body must reference checked-in contract file " "'requirements/security.txt'") in errors


def test_missing_required_workflow_snippet_is_reported() -> None:
    broken = remove_workflow_snippet(valid_workflow_text(), "make update-generic LOCK_PYTHON_VERSION=3.11")
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert ("dependency-update workflow must include snippet " "'make update-generic LOCK_PYTHON_VERSION=3.11'") in errors


def test_missing_preflight_reference_check_is_reported() -> None:
    command = workflow_contract.REQUIRED_PREFLIGHT_COMMANDS[2]
    broken = remove_workflow_snippet(valid_workflow_text(), command)
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert f"dependency-update preflight must include exact command {command!r}" in errors


def test_commented_preflight_reference_check_is_not_executable_evidence() -> None:
    command = workflow_contract.REQUIRED_PREFLIGHT_COMMANDS[2]
    broken = valid_workflow_text().replace(f"        {command}\n", f"        # {command}\n", 1)

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert f"dependency-update preflight must include exact command {command!r}" in errors


@pytest.mark.parametrize(
    ("field", "expected_error"),
    (
        ("      if: ${{ false }}\n", "dependency-update preflight must not be conditionally skipped"),
        ("      continue-on-error: true\n", "dependency-update preflight must not continue on error"),
        (
            "      env:\n        BASH_ENV: /tmp/untrusted-bash-env\n",
            "dependency-update preflight must not define execution overrides: env",
        ),
    ),
)
def test_dependency_target_preflight_rejects_fail_open_step_fields(field: str, expected_error: str) -> None:
    broken = valid_workflow_text().replace(
        "    - name: Preflight dependency update targets\n",
        "    - name: Preflight dependency update targets\n" + field,
        1,
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert expected_error in errors


@pytest.mark.parametrize(
    "replacement",
    (
        pytest.param("", id="missing-shell"),
        pytest.param("      shell: bash\n", id="inherited-bash-env"),
        pytest.param("      shell: /bin/true {0}\n", id="custom-shell"),
        pytest.param(
            f"      shell: {workflow_contract.TRUSTED_PREFLIGHT_SHELL}\n" "      shell: /bin/true {0}\n",
            id="duplicate-shell",
        ),
    ),
)
def test_dependency_target_preflight_requires_exact_sanitized_shell(replacement: str) -> None:
    broken = valid_workflow_text().replace(
        "    - name: Preflight dependency update targets\n" f"      shell: {workflow_contract.TRUSTED_PREFLIGHT_SHELL}\n",
        "    - name: Preflight dependency update targets\n" + replacement,
        1,
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update preflight must explicitly use the trusted sanitized Bash shell" in errors


def test_dependency_target_preflight_rejects_scalar_decoy() -> None:
    text = valid_workflow_text()
    before, rest = text.split("    - name: Preflight dependency update targets\n", maxsplit=1)
    _, after = rest.split("    - name: Update dependencies\n", maxsplit=1)
    decoy = (
        "    - name: Preflight decoy documentation\n"
        "      run: |\n"
        "        - name: Preflight dependency update targets\n"
        f"          shell: {workflow_contract.TRUSTED_PREFLIGHT_SHELL}\n"
        "          run: |\n" + "".join(f"            {command}\n" for command in workflow_contract.REQUIRED_PREFLIGHT_COMMANDS)
    )
    broken = before + decoy + "    - name: Update dependencies\n" + after

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update workflow must define exactly one 'Preflight dependency update targets' step mapping" in errors


def test_dependency_target_preflight_rejects_inherited_job_environment() -> None:
    broken = valid_workflow_text().replace(
        "  update-dependencies:\n",
        "  update-dependencies:\n    env:\n      PYTHONPATH: /tmp/untrusted\n",
        1,
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update job must not define execution overrides: env" in errors


@pytest.mark.parametrize(
    ("field", "expected_error"),
    (
        ("    runs-on: self-hosted\n", "must run on the trusted GitHub-hosted ubuntu-24.04 runner"),
        ("    container: attacker/image\n", "must not define execution overrides: container"),
        ("    if: ${{ false }}\n", "must not define execution overrides: if"),
        ("    strategy:\n      fail-fast: false\n", "must not define execution overrides: strategy"),
    ),
)
def test_dependency_update_job_rejects_execution_context_mutations(field: str, expected_error: str) -> None:
    if field.startswith("    runs-on:"):
        broken = valid_workflow_text().replace("    runs-on: ubuntu-24.04\n", field, 1)
    else:
        broken = valid_workflow_text().replace("    runs-on: ubuntu-24.04\n", "    runs-on: ubuntu-24.04\n" + field, 1)

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert any(expected_error in error for error in errors)


def test_dependency_update_rejects_additional_write_capable_job() -> None:
    broken = valid_workflow_text().replace(
        "  update-dependencies:\n",
        "  ungoverned:\n"
        "    runs-on: ubuntu-24.04\n"
        "    steps:\n"
        "      - run: echo ungoverned\n"
        "  update-dependencies:\n",
        1,
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update workflow must define only the governed updater job" in errors


def test_dependency_update_rejects_expanded_workflow_permissions() -> None:
    broken = valid_workflow_text().replace(
        "  pull-requests: write\n",
        "  pull-requests: write\n  id-token: write\n",
        1,
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update workflow must retain only its governed write permissions" in errors


@pytest.mark.parametrize(
    "replacement",
    (
        "on:\n  push:\n",
        "on:\n  pull_request_target:\n",
        "on:\n  workflow_dispatch:\n",
    ),
)
def test_dependency_update_requires_exact_trigger_envelope(replacement: str) -> None:
    text = valid_workflow_text()
    before, rest = text.split("on:\n", maxsplit=1)
    _, after = rest.split("permissions:\n", maxsplit=1)
    broken = before + replacement + "\npermissions:\n" + after

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update workflow must retain the exact weekly and manual trigger envelope" in errors


def test_dependency_target_preflight_rejects_checkout_repository_override() -> None:
    broken = valid_workflow_text().replace(
        "      with:\n        token: ${{ secrets.GITHUB_TOKEN }}\n",
        "      with:\n        token: ${{ secrets.GITHUB_TOKEN }}\n        repository: attacker/untrusted\n",
        1,
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update checkout must use only the current repository and trusted token" in errors


def test_dependency_target_preflight_requires_pinned_python_output() -> None:
    trusted_command = workflow_contract.REQUIRED_PREFLIGHT_COMMANDS[1]
    broken = valid_workflow_text().replace(
        f"        {trusted_command}\n",
        "        python3 scripts/validation/check_dependabot_config.py\n",
        1,
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert f"dependency-update preflight must include exact command {trusted_command!r}" in errors


def test_dependency_update_rejects_heredoc_command_decoy() -> None:
    text = valid_workflow_text()
    update_start = "    - name: Update dependencies\n"
    before, rest = text.split(update_start, maxsplit=1)
    _, after = rest.split("    - name: Check lock ownership authority\n", maxsplit=1)
    decoy = (
        update_start
        + f"      shell: {workflow_contract.TRUSTED_PREFLIGHT_SHELL}\n"
        + "      env:\n"
        + '        PIP_NO_CACHE_DIR: "1"\n'
        + "      run: |\n"
        + "        cat <<'EXPECTED' >/dev/null\n"
        + "".join(f"        {command}\n" for command in workflow_contract.REQUIRED_UPDATE_COMMANDS)
        + "        EXPECTED\n"
    )
    broken = before + decoy + "    - name: Check lock ownership authority\n" + after

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update update step must use only the exact ordered generic lock commands" in errors


def test_dependency_target_preflight_rejects_shell_failure_suppression() -> None:
    command = workflow_contract.REQUIRED_PREFLIGHT_COMMANDS[2]
    broken = valid_workflow_text().replace(
        f"        {command}\n",
        f"        {command} || true\n",
        1,
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update preflight must not suppress command failures" in errors


def test_dependency_target_preflight_does_not_absorb_following_nameless_step() -> None:
    command = workflow_contract.REQUIRED_PREFLIGHT_COMMANDS[2]
    broken = remove_workflow_snippet(valid_workflow_text(), command).replace(
        "    - name: Update dependencies\n",
        f"    - run: |\n        {command}\n" "    - name: Update dependencies\n",
        1,
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert f"dependency-update preflight must include exact command {command!r}" in errors


def test_dependency_target_preflight_must_run_before_lock_generation() -> None:
    text = valid_workflow_text()
    before_preflight, after_preflight = text.split("    - name: Preflight dependency update targets\n", maxsplit=1)
    preflight_body, after_update = after_preflight.split("    - name: Update dependencies\n", maxsplit=1)
    broken = (
        before_preflight
        + "    - name: Update dependencies\n"
        + after_update.rstrip()
        + "\n"
        + "    - name: Preflight dependency update targets\n"
        + preflight_body
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update target preflight must run before generic lock generation" in errors


def test_missing_required_install_toolchain_snippet_is_reported() -> None:
    command = workflow_contract.REQUIRED_INSTALL_TOOLCHAIN_SNIPPETS[0]
    broken = remove_workflow_snippet(valid_workflow_text(), command)
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert "dependency-update tool installation must use only the exact isolated pinned commands" in errors


def test_missing_required_audit_report_temp_dir_snippet_is_reported() -> None:
    broken = remove_workflow_snippet(
        valid_workflow_text(), 'audit_reports_dir="${{ runner.temp }}/dependency-update-audit-reports"'
    )
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert (
        "dependency-update workflow must include audit-report snippet "
        "'audit_reports_dir=\"${{ runner.temp }}/dependency-update-audit-reports\"'"
    ) in errors


def test_missing_required_audit_report_upload_path_is_reported() -> None:
    broken = remove_workflow_snippet(valid_workflow_text(), "path: ${{ runner.temp }}/dependency-update-audit-reports/")
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert (
        "dependency-update workflow must include audit-report snippet "
        "'path: ${{ runner.temp }}/dependency-update-audit-reports/'"
    ) in errors


def test_forbidden_target_agnostic_update_command_is_reported() -> None:
    broken = add_to_step_run(
        valid_workflow_text(),
        "Check lock ownership authority",
        "make update LOCK_PYTHON_VERSION=3.11",
    )
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert "dependency-update workflow must not include snippet 'make update LOCK_PYTHON_VERSION=3.11'" in errors


def test_forbidden_linux_target_owned_update_command_is_reported() -> None:
    broken = add_to_step_run(
        valid_workflow_text(),
        "Check lock ownership authority",
        "make update-ml-linux-x86_64 LOCK_PYTHON_VERSION=3.11",
    )
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert "dependency-update workflow must not include snippet 'make update-ml-linux-x86_64'" in errors


def test_repo_local_audit_reports_usage_is_reported() -> None:
    broken = add_to_step_run(valid_workflow_text(), "Check for vulnerabilities", "mkdir -p audit-reports").replace(
        "        path: ${{ runner.temp }}/dependency-update-audit-reports/\n",
        "        path: audit-reports/\n",
        1,
    )
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert "dependency-update workflow must not include snippet 'mkdir -p audit-reports'" in errors
    assert "dependency-update workflow must not include snippet 'path: audit-reports/'" in errors


def test_retired_pygments_exception_is_reported() -> None:
    broken = add_to_step_run(
        valid_workflow_text(),
        "Check for vulnerabilities",
        "# CVE-2026-4539 (pygments): No fix available yet - temporary exception",
        "pip-audit --ignore-vuln CVE-2026-4539 -r requirements/security.txt",
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update workflow must not include snippet '--ignore-vuln CVE-2026-4539'" in errors
    assert (
        "dependency-update workflow must not include snippet " "'CVE-2026-4539 (pygments): No fix available yet'"
    ) in errors


def test_missing_audit_targets_block_is_reported_independently_of_pr_body_references() -> None:
    broken = remove_audit_targets_block(valid_workflow_text())
    errors = workflow_contract.validate_dependency_update_workflow(broken)
    assert "dependency-update workflow must define an audit_targets block" in errors
    assert ("dependency-update PR body must reference checked-in contract file " "'requirements/security.txt'") not in errors


def test_dependency_update_rejects_arbitrary_intervening_step() -> None:
    free_disk_marker = "    - name: Free disk space\n"
    broken = valid_workflow_text().replace(
        free_disk_marker,
        "    - name: Seed shell environment\n"
        "      run: |\n"
        "        echo 'exit 0' > /tmp/untrusted-bash-env\n"
        "        echo 'BASH_ENV=/tmp/untrusted-bash-env' >> \"${GITHUB_ENV}\"\n" + free_disk_marker,
        1,
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update steps must match the exact governed sequence" in errors


def test_dependency_update_rejects_disk_cleanup_environment_poisoning() -> None:
    broken = add_to_step_run(
        valid_workflow_text(),
        workflow_contract.FREE_DISK_STEP_NAME,
        "echo 'exit 0' > /tmp/untrusted-bash-env",
        "echo 'BASH_ENV=/tmp/untrusted-bash-env' >> \"${GITHUB_ENV}\"",
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update disk cleanup must use only the exact sanitized commands" in errors


@pytest.mark.parametrize("shell", ("bash", "/bin/true {0}"))
def test_dependency_update_requires_exact_sanitized_update_shell(shell: str) -> None:
    broken = valid_workflow_text().replace(
        "    - name: Update dependencies\n" f"      shell: {workflow_contract.TRUSTED_PREFLIGHT_SHELL}\n",
        "    - name: Update dependencies\n" f"      shell: {shell}\n",
        1,
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update update step must use only the governed environment" in errors


def test_dependency_update_rejects_untrusted_post_update_python() -> None:
    trusted_command = next(
        command
        for command in workflow_contract.REQUIRED_LOCK_AUTHORITY_COMMANDS
        if "scripts/validation/check_lock_ownership.py" in command
    )
    broken = valid_workflow_text().replace(
        f"        {trusted_command}\n",
        "        python3 scripts/validation/check_lock_ownership.py --context ubuntu-x64-generic "
        '--changed-files-file "${changed_files_path}"\n',
        1,
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update lock authority check must use only the exact sanitized commands" in errors


def test_dependency_update_rejects_unpinned_artifact_action() -> None:
    broken = valid_workflow_text().replace(workflow_contract.TRUSTED_UPLOAD_ACTION, "actions/upload-artifact@main", 1)

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update audit upload must use only the pinned action and governed inputs" in errors


def test_dependency_update_rejects_extra_pull_request_action_input() -> None:
    broken = valid_workflow_text().replace(
        "        add-paths: |\n",
        "        sign-commits: true\n        add-paths: |\n",
        1,
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update PR creation must use only the pinned action and governed inputs" in errors


def test_dependency_update_inventory_uses_immutable_event_sha() -> None:
    command = '/usr/bin/git diff --name-only "${baseline_sha}" > "${changed_files_path}"'
    broken = valid_workflow_text().replace(
        f"        {command}\n",
        '        /usr/bin/git diff --name-only HEAD > "${changed_files_path}"\n',
        1,
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update lock authority check must use only the exact sanitized commands" in errors


def test_dependency_update_inventory_includes_all_untracked_files() -> None:
    command = '/usr/bin/git ls-files --others --exclude-standard >> "${changed_files_path}"'
    broken = valid_workflow_text().replace(f"        {command}\n", "", 1)

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update lock authority check must use only the exact sanitized commands" in errors


def test_dependency_update_inventory_commands_capture_staged_and_untracked_files(tmp_path: Path) -> None:
    requirements_dir = tmp_path / "requirements"
    requirements_dir.mkdir()
    generic_lock = requirements_dir / "base.txt"
    target_lock = requirements_dir / "ml-core-darwin-arm64.txt"
    generic_lock.write_text("base\n", encoding="utf-8")
    target_lock.write_text("target\n", encoding="utf-8")

    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "contract@example.invalid"),
        ("git", "config", "user.name", "Contract Test"),
        ("git", "add", "requirements"),
        ("git", "commit", "-qm", "initial"),
    ):
        subprocess.run(command, cwd=tmp_path, check=True)

    target_lock.write_text("staged target change\n", encoding="utf-8")
    subprocess.run(("git", "add", "requirements/ml-core-darwin-arm64.txt"), cwd=tmp_path, check=True)
    (requirements_dir / "unexpected.txt").write_text("untracked\n", encoding="utf-8")

    inventory_commands = ['baseline_sha="$(/usr/bin/git rev-parse HEAD)"']
    collecting_inventory = False
    for command in workflow_contract.REQUIRED_LOCK_AUTHORITY_COMMANDS:
        if command.startswith("changed_files_path="):
            collecting_inventory = True
        if collecting_inventory:
            inventory_commands.append(command)
        if collecting_inventory and command.startswith("LC_ALL=C sort"):
            break
    inventory_commands.append('cat "${changed_files_path}"')
    completed = subprocess.run(
        ("/bin/bash", "-c", "\n".join(inventory_commands)),
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert set(completed.stdout.splitlines()) == {
        "requirements/ml-core-darwin-arm64.txt",
        "requirements/unexpected.txt",
    }


def test_dependency_update_publication_boundary_rejects_workflow_created_commit(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.email", "contract@example.invalid"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.name", "Contract Test"), cwd=tmp_path, check=True)
    tracked_file = tmp_path / "tracked.txt"
    tracked_file.write_text("initial\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt"), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "initial"), cwd=tmp_path, check=True)
    baseline_sha = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    tracked_file.write_text("committed by updater\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt"), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "updater commit"), cwd=tmp_path, check=True)

    boundary_commands = []
    for command in workflow_contract.REQUIRED_LOCK_AUTHORITY_COMMANDS:
        boundary_commands.append(command)
        if command == "fi":
            break
    boundary_commands[1] = f'baseline_sha="{baseline_sha}"'
    completed = subprocess.run(
        ("/bin/bash", "-c", "\n".join(boundary_commands)),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "must not create or amend commits before publication" in completed.stdout


def test_dependency_update_pull_request_paths_are_exactly_governed_locks() -> None:
    governed_add_paths = "        add-paths: |\n" + "".join(
        f"          {path}\n" for path in workflow_contract.REQUIRED_AUDIT_TARGETS
    )
    broken = valid_workflow_text().replace(
        governed_add_paths,
        "        add-paths: requirements/**\n",
        1,
    )

    errors = workflow_contract.validate_dependency_update_workflow(broken)

    assert "dependency-update PR creation must use only the pinned action and governed inputs" in errors
