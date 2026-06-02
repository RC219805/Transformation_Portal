#!/usr/bin/env python3
"""
CI Configuration Validator for Transformation Portal.

Validates GitHub Actions workflow configurations to catch common issues:
- Invalid YAML syntax
- Missing required fields
- Incorrect Python version matrix
- Invalid job dependencies
- Common misconfigurations

Usage:
    python scripts/validate_ci_config.py [--fix] [workflow...]

Options:
    --fix         Auto-fix common issues where possible
    workflow...   Specific workflow files to check (default: all in .github/workflows/)
"""

import argparse
import re
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed (pip install PyYAML)")
    sys.exit(1)


class CIValidator:
    """Validate GitHub Actions workflow configurations."""

    MIN_CHECKOUT_MAJOR = 4
    MYPY_POLICY_WORKFLOWS = frozenset({"build.yml", "ci.yml", "ci-quality-firewall.yml"})
    REQUIRED_FLAKE8_FATAL_CODES = {"E9", "F63", "F7", "F82"}
    ML_NO_COV_BLOCK_PATTERN = re.compile(
        r'if\s+\[\s*"\$\{\{\s*matrix\.test-type\s*\}\}"\s*=\s*"ml"\s*\]\s*;\s*then\s*\n\s*' r'COV_FLAGS="--no-cov"'
    )
    CORE_COV_FLAGS_PATTERN = re.compile(r'else\s*\n\s*COV_FLAGS="(?P<flags>[^"]*)"')
    REQUIRED_CORE_COVERAGE_FLAGS = (
        "--cov=src/transformation_portal",
        "--cov=lux_depth_v3",
        "--cov-report=xml",
        "--cov-report=html",
        "--cov-fail-under",
    )
    BRANCH_COVERAGE_SCRIPT = "scripts/ci/check_per_package_branch_coverage.py"
    BRANCH_COVERAGE_XML = "coverage.xml"
    REQUIRED_BRANCH_COVERAGE_CHECK = f"python {BRANCH_COVERAGE_SCRIPT} {BRANCH_COVERAGE_XML}"
    COLD_ZONE_TOUCHED_FILE_SCRIPT = "scripts/ci/check_cold_zone_touched_files.py"
    REQUIRED_COLD_ZONE_TOUCHED_FILE_CHECK = (
        f"python {COLD_ZONE_TOUCHED_FILE_SCRIPT} {BRANCH_COVERAGE_XML} --compare-ref origin/main"
    )
    REQUIRED_COLD_ZONE_COMPARE_REF_FETCH = "git fetch --no-tags --depth=1 origin main:refs/remotes/origin/main"

    def __init__(self, repo_root: Path, fix_mode: bool = False):
        self.repo_root = repo_root
        self.fix_mode = fix_mode
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.fixes_applied: List[str] = []

    def log_error(self, message: str):
        """Log validation error."""
        self.errors.append(f"✗ {message}")

    def log_warning(self, message: str):
        """Log validation warning."""
        self.warnings.append(f"⚠ {message}")

    def log_fix(self, message: str):
        """Log applied fix."""
        self.fixes_applied.append(f"✓ {message}")

    def validate_yaml_syntax(self, workflow_path: Path) -> Optional[Dict]:
        """Validate YAML syntax and return parsed content."""
        try:
            with open(workflow_path, "r") as f:
                # Use yaml.safe_load with version 1.2 to avoid 'on' being interpreted as boolean
                content = yaml.safe_load(f)

            # Fix 'on' key being interpreted as boolean True
            if content and True in content and "on" not in content:
                content["on"] = content.pop(True)

            return content
        except yaml.YAMLError as e:
            self.log_error(f"{workflow_path.name}: Invalid YAML syntax: {e}")
            return None

    def validate_workflow_structure(self, workflow_path: Path, config: Dict) -> bool:
        """Validate basic workflow structure."""
        valid = True

        # Check required top-level keys
        required_keys = ["name", "on", "jobs"]
        for key in required_keys:
            if key not in config:
                self.log_error(f"{workflow_path.name}: Missing required key '{key}'")
                valid = False

        # Validate jobs
        if "jobs" in config:
            if not isinstance(config["jobs"], dict):
                self.log_error(f"{workflow_path.name}: 'jobs' must be a dictionary")
                valid = False
            elif not config["jobs"]:
                self.log_error(f"{workflow_path.name}: No jobs defined")
                valid = False

        return valid

    def validate_python_matrix(self, workflow_path: Path, config: Dict) -> bool:
        """Validate Python version matrix."""
        valid = True
        supported_versions = {"3.10", "3.11", "3.12"}

        for job_name, job_config in config.get("jobs", {}).items():
            if not isinstance(job_config, dict):
                continue

            strategy = job_config.get("strategy", {})
            if not isinstance(strategy, dict):
                continue

            matrix = strategy.get("matrix", {})
            if not isinstance(matrix, dict):
                continue

            python_versions = matrix.get("python-version", [])
            if not python_versions:
                continue

            # Ensure it's a list
            if not isinstance(python_versions, list):
                python_versions = [python_versions]

            # Check versions
            for version in python_versions:
                version_str = str(version)
                if version_str not in supported_versions:
                    self.log_warning(
                        f"{workflow_path.name}:{job_name}: "
                        f"Python {version_str} not in supported versions {supported_versions}"
                    )

        return valid

    def validate_job_dependencies(self, workflow_path: Path, config: Dict) -> bool:
        """Validate job dependencies (needs clause)."""
        valid = True
        job_names = set(config.get("jobs", {}).keys())

        for job_name, job_config in config.get("jobs", {}).items():
            if not isinstance(job_config, dict):
                continue

            needs = job_config.get("needs", [])
            if isinstance(needs, str):
                needs = [needs]

            for dependency in needs:
                if dependency not in job_names:
                    self.log_error(f"{workflow_path.name}:{job_name}: " f"Depends on non-existent job '{dependency}'")
                    valid = False

        return valid

    def validate_checkout_action(self, workflow_path: Path, config: Dict) -> bool:
        """Validate checkout action versions against policy baselines."""
        valid = True

        for job_name, job_config in config.get("jobs", {}).items():
            if not isinstance(job_config, dict):
                continue

            steps = job_config.get("steps", [])
            if not isinstance(steps, list):
                continue

            for step in steps:
                if not isinstance(step, dict):
                    continue

                uses = step.get("uses", "")
                if "actions/checkout@" in uses:
                    version = uses.split("@")[1] if "@" in uses else ""
                    if re.fullmatch(r"[a-f0-9]{40}", version):
                        # Full-length pin is acceptable and typically stronger than tag pinning.
                        continue

                    match = re.fullmatch(r"v(\d+)(?:\.\d+)?(?:\.\d+)?", version)
                    if match and int(match.group(1)) >= self.MIN_CHECKOUT_MAJOR:
                        continue

                    self.log_warning(
                        f"{workflow_path.name}:{job_name}: "
                        f"Checkout action version '{version}' is below minimum supported v{self.MIN_CHECKOUT_MAJOR}"
                    )

        return valid

    def validate_common_issues(self, workflow_path: Path, config: Dict) -> bool:
        """Check for common configuration issues."""
        valid = True

        for job_name, job_config in config.get("jobs", {}).items():
            if not isinstance(job_config, dict):
                continue

            # Check for matrix without strategy
            if "matrix" in job_config and "strategy" not in job_config:
                self.log_error(f"{workflow_path.name}:{job_name}: " "'matrix' defined without 'strategy' wrapper")
                valid = False

            # Check for runs-on
            if "runs-on" not in job_config:
                self.log_error(f"{workflow_path.name}:{job_name}: " "Missing 'runs-on' specification")
                valid = False

            # Check for empty steps
            steps = job_config.get("steps", [])
            if isinstance(steps, list) and len(steps) == 0:
                self.log_warning(f"{workflow_path.name}:{job_name}: " "Job has no steps")

        return valid

    def validate_flake8_config(self, workflow_path: Path, config: Dict) -> bool:
        """Validate flake8 configuration against minimum fatal-code policy."""
        valid = True

        for job_name, job_config in config.get("jobs", {}).items():
            if not isinstance(job_config, dict):
                continue

            steps = job_config.get("steps", [])
            if not isinstance(steps, list):
                continue

            for step in steps:
                if not isinstance(step, dict):
                    continue

                run_cmd = step.get("run", "")
                if "flake8" in run_cmd:
                    if "--select=" in run_cmd:
                        match = re.search(r"--select=([^\s]+)", run_cmd)
                        if match:
                            selected_codes = {code.strip() for code in match.group(1).split(",") if code.strip()}
                            if not self.REQUIRED_FLAKE8_FATAL_CODES.issubset(selected_codes):
                                required = ",".join(sorted(self.REQUIRED_FLAKE8_FATAL_CODES))
                                self.log_warning(
                                    f"{workflow_path.name}:{job_name}: "
                                    f"flake8 select flags missing required fatal codes '{required}'"
                                )

        return valid

    def validate_build_coverage_contract(self, workflow_path: Path, config: Dict) -> bool:
        """Validate the canonical PR gate coverage artifact contract."""
        if workflow_path.name != "build.yml":
            return True

        valid = True
        jobs = config.get("jobs", {})
        test_job = jobs.get("test") if isinstance(jobs, dict) else None
        if not isinstance(test_job, dict):
            self.log_error("build.yml:test: Missing test job for coverage artifact contract")
            return False

        steps = test_job.get("steps", [])
        if not isinstance(steps, list):
            self.log_error("build.yml:test: Steps must be a list for coverage artifact contract")
            return False

        run_tests_step = self._find_step_by_name(steps, "Run tests")
        if run_tests_step is None:
            self.log_error("build.yml:test: Missing 'Run tests' step for coverage artifact contract")
            valid = False
        else:
            valid = self._validate_run_tests_coverage_contract(run_tests_step.get("run")) and valid

        upload_step = self._find_step_by_name(steps, "Upload coverage reports")
        if upload_step is None:
            self.log_error("build.yml:test: Missing 'Upload coverage reports' step for coverage artifact contract")
            return False

        expected_if = "always() && matrix.test-type == 'core'"
        if upload_step.get("if") != expected_if:
            self.log_error(
                "build.yml:test: Coverage upload step must be guarded by "
                f"{expected_if!r} because only core legs generate coverage artifacts"
            )
            valid = False

        with_config = upload_step.get("with", {})
        upload_paths = self._normalize_upload_paths(with_config.get("path") if isinstance(with_config, dict) else None)
        for required_path in ("coverage.xml", "htmlcov/"):
            if required_path not in upload_paths:
                self.log_error(f"build.yml:test: Coverage upload path must include {required_path!r}")
                valid = False

        return valid

    def validate_mypy_policy_contract(self, workflow_path: Path, config: Dict) -> bool:
        """Validate mypy workflow whitelists against the current policy doc."""
        if workflow_path.name not in self.MYPY_POLICY_WORKFLOWS:
            return True

        valid = True
        expected_paths = self._mypy_policy_doc_paths()
        if not expected_paths:
            self.log_error("docs/ci/TYPE_CHECKING_POLICY.md: Missing actively enforced mypy whitelist paths")
            return False

        jobs = config.get("jobs", {})
        typecheck_job = jobs.get("typecheck") if isinstance(jobs, dict) else None
        if not isinstance(typecheck_job, dict):
            self.log_error(f"{workflow_path.name}:typecheck: Missing typecheck job for mypy policy contract")
            return False

        steps = typecheck_job.get("steps", [])
        if not isinstance(steps, list):
            self.log_error(f"{workflow_path.name}:typecheck: Steps must be a list for mypy policy contract")
            return False

        typecheck_step = self._find_step_by_name(steps, "Type check with mypy (critical modules)")
        if typecheck_step is None:
            self.log_error(f"{workflow_path.name}:typecheck: Missing mypy typecheck step")
            return False

        actual_paths, has_config = self._mypy_step_paths(typecheck_step.get("run"))
        if not has_config:
            self.log_error(f"{workflow_path.name}:typecheck: mypy command must use --config-file=mypy.ini")
            valid = False

        if actual_paths != expected_paths:
            self.log_error(
                f"{workflow_path.name}:typecheck: mypy whitelist must match docs/ci/TYPE_CHECKING_POLICY.md "
                f"(expected {expected_paths}, found {actual_paths})"
            )
            valid = False

        return valid

    def _validate_run_tests_coverage_contract(self, run_script: object) -> bool:
        """Validate coverage flag scoping in the build.yml Run tests shell script."""
        valid = True
        if not isinstance(run_script, str):
            self.log_error("build.yml:test: 'Run tests' step must define a run script")
            return False

        if not self.ML_NO_COV_BLOCK_PATTERN.search(run_script):
            self.log_error(
                "build.yml:test: 'Run tests' must scope '--no-cov' to the ML matrix leg with an explicit "
                "matrix.test-type == 'ml' conditional"
            )
            valid = False

        core_flags_match = self.CORE_COV_FLAGS_PATTERN.search(run_script)
        if core_flags_match is None:
            self.log_error("build.yml:test: Core test leg must define COV_FLAGS in the non-ML branch")
            return False

        core_flags = core_flags_match.group("flags")
        missing_core_flags = [flag for flag in self.REQUIRED_CORE_COVERAGE_FLAGS if flag not in core_flags]
        if missing_core_flags:
            missing = ", ".join(repr(flag) for flag in missing_core_flags)
            self.log_error(f"build.yml:test: Core test leg must retain coverage generation flags: {missing}")
            valid = False

        branch_coverage_commands = self._branch_coverage_checker_commands(run_script)
        if not any(self.BRANCH_COVERAGE_XML in command for command in branch_coverage_commands):
            self.log_error(
                "build.yml:test: Core test leg must retain branch coverage enforcement check "
                f"{self.REQUIRED_BRANCH_COVERAGE_CHECK!r}"
            )
            valid = False

        if any("--dry-run" in command for command in branch_coverage_commands):
            self.log_error(
                "build.yml:test: Core test leg must enforce branch coverage without --dry-run "
                f"({self.BRANCH_COVERAGE_SCRIPT!r} invocations may not include '--dry-run')"
            )
            valid = False

        touched_file_commands = self._script_invocations(run_script, self.COLD_ZONE_TOUCHED_FILE_SCRIPT)
        if not any(
            self.BRANCH_COVERAGE_XML in command and self._command_has_option_value(command, "--compare-ref", "origin/main")
            for command in touched_file_commands
        ):
            self.log_error(
                "build.yml:test: Core test leg must retain cold-zone touched-file coverage evidence check "
                f"{self.REQUIRED_COLD_ZONE_TOUCHED_FILE_CHECK!r}"
            )
            valid = False

        if not self._has_required_command(run_script, self.REQUIRED_COLD_ZONE_COMPARE_REF_FETCH):
            self.log_error(
                "build.yml:test: Core test leg must fetch origin/main before cold-zone touched-file evidence "
                f"{self.REQUIRED_COLD_ZONE_COMPARE_REF_FETCH!r}"
            )
            valid = False

        return valid

    def _mypy_policy_doc_paths(self) -> List[str]:
        """Return the live mypy whitelist from docs/ci/TYPE_CHECKING_POLICY.md."""
        policy_path = self.repo_root / "docs" / "ci" / "TYPE_CHECKING_POLICY.md"
        try:
            text = policy_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []

        try:
            start = text.index("**Enforced as of")
            end = text.index("**N-1 tranche notes", start)
        except ValueError:
            return []

        return re.findall(r"^- `([^`]+)`", text[start:end], flags=re.MULTILINE)

    @staticmethod
    def _mypy_step_paths(run_script: object) -> Tuple[List[str], bool]:
        """Return src paths and config-file usage from a mypy workflow step."""
        if not isinstance(run_script, str):
            return [], False

        logical_script = run_script.replace("\\\n", " ")
        try:
            tokens = shlex.split(logical_script, comments=True, posix=True)
        except ValueError:
            tokens = logical_script.split()

        if "mypy" not in tokens:
            return [], False

        has_inline_config = "--config-file=mypy.ini" in tokens
        has_split_config = any(
            token == "--config-file" and index + 1 < len(tokens) and tokens[index + 1] == "mypy.ini"
            for index, token in enumerate(tokens)
        )
        paths = [token for token in tokens if token.startswith("src/")]
        return paths, has_inline_config or has_split_config

    @classmethod
    def _branch_coverage_checker_commands(cls, run_script: str) -> List[List[str]]:
        """Return shell-tokenized branch coverage checker invocations."""
        return cls._script_invocations(run_script, cls.BRANCH_COVERAGE_SCRIPT)

    @staticmethod
    def _script_invocations(run_script: str, script_path: str) -> List[List[str]]:
        """Return shell-tokenized invocations for a Python script path."""
        logical_script = run_script.replace("\\\n", " ")
        commands: List[List[str]] = []
        for raw_line in logical_script.splitlines():
            if script_path not in raw_line:
                continue
            try:
                tokens = shlex.split(raw_line, comments=True, posix=True)
            except ValueError:
                tokens = raw_line.split()
            if script_path not in tokens:
                continue
            script_index = tokens.index(script_path)
            commands.append(tokens[script_index:])
        return commands

    @staticmethod
    def _has_required_command(run_script: str, required_command: str) -> bool:
        """Return True when a shell script contains the required command tokens."""
        expected_tokens = shlex.split(required_command, comments=True, posix=True)
        logical_script = run_script.replace("\\\n", " ")
        for raw_line in logical_script.splitlines():
            try:
                tokens = shlex.split(raw_line, comments=True, posix=True)
            except ValueError:
                tokens = raw_line.split()
            if len(tokens) >= len(expected_tokens) and tokens[: len(expected_tokens)] == expected_tokens:
                return True
        return False

    @staticmethod
    def _command_has_option_value(command: List[str], option: str, value: str) -> bool:
        """Return True when command contains ``--option value`` or ``--option=value``."""
        for index, token in enumerate(command):
            if token == option and index + 1 < len(command) and command[index + 1] == value:
                return True
            if token == f"{option}={value}":
                return True
        return False

    @staticmethod
    def _find_step_by_name(steps: List[Dict], name: str) -> Optional[Dict]:
        """Return the first workflow step with the supplied display name."""
        for step in steps:
            if isinstance(step, dict) and step.get("name") == name:
                return step
        return None

    @staticmethod
    def _normalize_upload_paths(path_config: object) -> Set[str]:
        """Return normalized artifact upload paths from workflow action config."""
        if isinstance(path_config, str):
            return {line.strip() for line in path_config.splitlines() if line.strip()}
        if isinstance(path_config, list):
            return {str(item).strip() for item in path_config if str(item).strip()}
        return set()

    def validate_workflow(self, workflow_path: Path) -> bool:
        """Validate a single workflow file."""
        print(f"\n→ Validating {workflow_path.name}...")

        # Parse YAML
        config = self.validate_yaml_syntax(workflow_path)
        if config is None:
            return False

        # Run all validations
        validations = [
            self.validate_workflow_structure(workflow_path, config),
            self.validate_python_matrix(workflow_path, config),
            self.validate_job_dependencies(workflow_path, config),
            self.validate_checkout_action(workflow_path, config),
            self.validate_common_issues(workflow_path, config),
            self.validate_flake8_config(workflow_path, config),
            self.validate_build_coverage_contract(workflow_path, config),
            self.validate_mypy_policy_contract(workflow_path, config),
        ]

        return all(validations)


def main():
    parser = argparse.ArgumentParser(
        description="Validate GitHub Actions workflow configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("workflows", nargs="*", help="Specific workflow files to validate (default: all)")
    parser.add_argument("--fix", action="store_true", help="Auto-fix common issues where possible")

    args = parser.parse_args()

    # Get repository root
    try:
        import subprocess

        result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
        repo_root = Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        repo_root = Path.cwd()

    workflows_dir = repo_root / ".github" / "workflows"

    if not workflows_dir.exists():
        print(f"Error: Workflows directory not found: {workflows_dir}")
        sys.exit(1)

    # Get workflow files
    if args.workflows:
        workflow_files = [Path(w) for w in args.workflows]
    else:
        workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))

    if not workflow_files:
        print("No workflow files found")
        sys.exit(1)

    print("╔════════════════════════════════════════════╗")
    print("║  CI Configuration Validator                ║")
    print("╚════════════════════════════════════════════╝")
    print(f"\nValidating {len(workflow_files)} workflow(s)...")

    # Validate each workflow
    validator = CIValidator(repo_root, fix_mode=args.fix)
    all_valid = True

    for workflow in workflow_files:
        if not workflow.exists():
            print(f"\n✗ Workflow not found: {workflow}")
            all_valid = False
            continue

        valid = validator.validate_workflow(workflow)
        if not valid:
            all_valid = False

    # Print summary
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if validator.errors:
        print(f"\n\033[0;31mErrors ({len(validator.errors)}):\033[0m")
        for error in validator.errors:
            print(f"  {error}")

    if validator.warnings:
        print(f"\n\033[1;33mWarnings ({len(validator.warnings)}):\033[0m")
        for warning in validator.warnings:
            print(f"  {warning}")

    if validator.fixes_applied:
        print(f"\n\033[0;32mFixes Applied ({len(validator.fixes_applied)}):\033[0m")
        for fix in validator.fixes_applied:
            print(f"  {fix}")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if all_valid and not validator.errors:
        print("\033[0;32m✓ All workflows valid!\033[0m")
        sys.exit(0)
    else:
        print("\033[0;31m✗ Validation failed\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
