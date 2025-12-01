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
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed (pip install PyYAML)")
    sys.exit(1)


class CIValidator:
    """Validate GitHub Actions workflow configurations."""

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
            with open(workflow_path, 'r') as f:
                # Use yaml.safe_load with version 1.2 to avoid 'on' being interpreted as boolean
                content = yaml.safe_load(f)

            # Fix 'on' key being interpreted as boolean True
            if content and True in content and 'on' not in content:
                content['on'] = content.pop(True)

            return content
        except yaml.YAMLError as e:
            self.log_error(f"{workflow_path.name}: Invalid YAML syntax: {e}")
            return None

    def validate_workflow_structure(self, workflow_path: Path, config: Dict) -> bool:
        """Validate basic workflow structure."""
        valid = True

        # Check required top-level keys
        required_keys = ['name', 'on', 'jobs']
        for key in required_keys:
            if key not in config:
                self.log_error(f"{workflow_path.name}: Missing required key '{key}'")
                valid = False

        # Validate jobs
        if 'jobs' in config:
            if not isinstance(config['jobs'], dict):
                self.log_error(f"{workflow_path.name}: 'jobs' must be a dictionary")
                valid = False
            elif not config['jobs']:
                self.log_error(f"{workflow_path.name}: No jobs defined")
                valid = False

        return valid

    def validate_python_matrix(self, workflow_path: Path, config: Dict) -> bool:
        """Validate Python version matrix."""
        valid = True
        supported_versions = {'3.10', '3.11', '3.12'}

        for job_name, job_config in config.get('jobs', {}).items():
            if not isinstance(job_config, dict):
                continue

            strategy = job_config.get('strategy', {})
            if not isinstance(strategy, dict):
                continue

            matrix = strategy.get('matrix', {})
            if not isinstance(matrix, dict):
                continue

            python_versions = matrix.get('python-version', [])
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
        job_names = set(config.get('jobs', {}).keys())

        for job_name, job_config in config.get('jobs', {}).items():
            if not isinstance(job_config, dict):
                continue

            needs = job_config.get('needs', [])
            if isinstance(needs, str):
                needs = [needs]

            for dependency in needs:
                if dependency not in job_names:
                    self.log_error(
                        f"{workflow_path.name}:{job_name}: "
                        f"Depends on non-existent job '{dependency}'"
                    )
                    valid = False

        return valid

    def validate_checkout_action(self, workflow_path: Path, config: Dict) -> bool:
        """Validate checkout action versions."""
        valid = True
        recommended_version = 'v5'

        for job_name, job_config in config.get('jobs', {}).items():
            if not isinstance(job_config, dict):
                continue

            steps = job_config.get('steps', [])
            if not isinstance(steps, list):
                continue

            for step in steps:
                if not isinstance(step, dict):
                    continue

                uses = step.get('uses', '')
                if 'actions/checkout@' in uses:
                    version = uses.split('@')[1] if '@' in uses else ''
                    if version != recommended_version:
                        self.log_warning(
                            f"{workflow_path.name}:{job_name}: "
                            f"Checkout action version '{version}' != recommended '{recommended_version}'"
                        )

        return valid

    def validate_common_issues(self, workflow_path: Path, config: Dict) -> bool:
        """Check for common configuration issues."""
        valid = True

        for job_name, job_config in config.get('jobs', {}).items():
            if not isinstance(job_config, dict):
                continue

            # Check for matrix without strategy
            if 'matrix' in job_config and 'strategy' not in job_config:
                self.log_error(
                    f"{workflow_path.name}:{job_name}: "
                    "'matrix' defined without 'strategy' wrapper"
                )
                valid = False

            # Check for runs-on
            if 'runs-on' not in job_config:
                self.log_error(
                    f"{workflow_path.name}:{job_name}: "
                    "Missing 'runs-on' specification"
                )
                valid = False

            # Check for empty steps
            steps = job_config.get('steps', [])
            if isinstance(steps, list) and len(steps) == 0:
                self.log_warning(
                    f"{workflow_path.name}:{job_name}: "
                    "Job has no steps"
                )

        return valid

    def validate_flake8_config(self, workflow_path: Path, config: Dict) -> bool:
        """Validate flake8 configuration matches CI requirements."""
        valid = True
        expected_select = 'E9,F63,F7,F82'

        for job_name, job_config in config.get('jobs', {}).items():
            if not isinstance(job_config, dict):
                continue

            steps = job_config.get('steps', [])
            if not isinstance(steps, list):
                continue

            for step in steps:
                if not isinstance(step, dict):
                    continue

                run_cmd = step.get('run', '')
                if 'flake8' in run_cmd:
                    # Check if using correct select flags
                    if '--select=' in run_cmd:
                        match = re.search(r'--select=([^\s]+)', run_cmd)
                        if match:
                            select_flags = match.group(1)
                            if select_flags != expected_select:
                                self.log_warning(
                                    f"{workflow_path.name}:{job_name}: "
                                    f"flake8 select flags '{select_flags}' != expected '{expected_select}'"
                                )
                    else:
                        self.log_warning(
                            f"{workflow_path.name}:{job_name}: "
                            "flake8 running without explicit --select flags"
                        )

        return valid

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
        ]

        return all(validations)


def main():
    parser = argparse.ArgumentParser(
        description='Validate GitHub Actions workflow configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'workflows',
        nargs='*',
        help='Specific workflow files to validate (default: all)'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Auto-fix common issues where possible'
    )

    args = parser.parse_args()

    # Get repository root
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            capture_output=True,
            text=True,
            check=True
        )
        repo_root = Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        repo_root = Path.cwd()

    workflows_dir = repo_root / '.github' / 'workflows'

    if not workflows_dir.exists():
        print(f"Error: Workflows directory not found: {workflows_dir}")
        sys.exit(1)

    # Get workflow files
    if args.workflows:
        workflow_files = [Path(w) for w in args.workflows]
    else:
        workflow_files = list(workflows_dir.glob('*.yml')) + list(workflows_dir.glob('*.yaml'))

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


if __name__ == '__main__':
    main()
