#!/usr/bin/env python3
"""
Parse and validate GitHub Actions workflow files for common bugs.

This script identifies:
- YAML syntax errors
- Missing step IDs when outputs are referenced
- Unclosed conditionals in shell scripts
- Invalid job dependencies
- Duplicate YAML mapping keys (including job names)
- Invalid GitHub Actions syntax
- Deprecated OpenAI model usage
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# Precompiled regex patterns for performance
_NEWLINE_SEMICOLON = re.compile(r"[\n;]")
_COMMENT_PATTERN = re.compile(r"#.*$")
_IF_PATTERN = re.compile(r"\bif\s+")
_ELIF_PATTERN = re.compile(r"\belif\s+")
_FI_PATTERN = re.compile(r"(^|\s)fi(\s|$)")
_STEP_OUTPUT_REF = re.compile(r"\$\{\{\s*steps\.([a-zA-Z0-9_-]+)\.outputs")
_MODEL_PATTERN1 = re.compile(r'"model":\s*"([^"]+)"')
_MODEL_PATTERN2 = re.compile(r'\\"model\\":\s*\\"([^"\\]+)\\"')

# Constants for magic strings
_IF_SEARCH_PATTERN = "if [ "
_DEVICE_SEARCH_PATTERN = "device:"
_GITHUB_COMMAND_ESCAPE = {
    "%": "%25",
    "\r": "%0D",
    "\n": "%0A",
}
_GITHUB_PROPERTY_ESCAPE = {
    **_GITHUB_COMMAND_ESCAPE,
    ":": "%3A",
    ",": "%2C",
}


class _DuplicateKeySafeLoader(yaml.SafeLoader):  # pylint: disable=too-many-ancestors
    """YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: Any, deep: bool = False) -> Dict[Any, Any]:
    mapping: Dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_DuplicateKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _yaml_error_line_number(error: yaml.YAMLError) -> Optional[int]:
    """Extract a 1-based line number from a PyYAML exception when available."""
    for mark_attr in ("problem_mark", "context_mark"):
        mark = getattr(error, mark_attr, None)
        if mark is not None and hasattr(mark, "line"):
            return mark.line + 1
    return None


def _escape_github_command_value(value: str) -> str:
    """Escape workflow command data values per GitHub Actions rules."""
    escaped = value
    for original, replacement in _GITHUB_COMMAND_ESCAPE.items():
        escaped = escaped.replace(original, replacement)
    return escaped


def _escape_github_command_property(value: str) -> str:
    """Escape workflow command property values per GitHub Actions rules."""
    escaped = value
    for original, replacement in _GITHUB_PROPERTY_ESCAPE.items():
        escaped = escaped.replace(original, replacement)
    return escaped


class WorkflowBug:
    """Represents a bug found in a workflow file."""

    def __init__(
        self,
        file_path: str,
        line_number: Optional[int],
        severity: str,
        message: str,
        context: Optional[str] = None,
    ):
        self.file_path = file_path
        self.line_number = line_number
        self.severity = severity  # 'error', 'warning', 'info'
        self.message = message
        self.context = context

    def __str__(self) -> str:
        location = f"{self.file_path}"
        if self.line_number:
            location += f":{self.line_number}"
        return f"[{self.severity.upper()}] {location} - {self.message}"


class WorkflowParser:
    """Parse and validate GitHub Actions workflows."""

    def __init__(self, workflow_dir: Path):
        self.workflow_dir = workflow_dir
        self.bugs: List[WorkflowBug] = []

    def parse_all_workflows(self) -> List[WorkflowBug]:
        """Parse all workflow files in the directory."""
        yml_files = list(self.workflow_dir.glob("*.yml"))
        yaml_files = list(self.workflow_dir.glob("*.yaml"))
        workflow_files = yml_files + yaml_files

        for workflow_file in workflow_files:
            self._parse_workflow(workflow_file)

        return self.bugs

    def _parse_workflow(self, workflow_file: Path) -> None:
        """Parse a single workflow file."""
        try:
            content = workflow_file.read_text(encoding="utf-8")
            lines = content.splitlines()

            # Try to parse YAML
            try:
                workflow = yaml.load(content, Loader=_DuplicateKeySafeLoader)
            except yaml.YAMLError as e:
                line_num = _yaml_error_line_number(e)
                self.bugs.append(
                    WorkflowBug(
                        str(workflow_file),
                        line_num,
                        "error",
                        f"YAML syntax error: {e}",
                    )
                )
                return

            # Validate workflow structure
            self._validate_workflow_structure(workflow_file, workflow, lines)
            self._check_step_references(workflow_file, workflow, lines)
            self._check_shell_scripts(workflow_file, workflow, lines)
            self._check_job_dependencies(workflow_file, workflow)
            self._check_matrix_usage(workflow_file, workflow, lines)
            self._check_openai_models(workflow_file, workflow, lines)

        except Exception as e:
            error_msg = f"Failed to parse file: {e}"
            self.bugs.append(WorkflowBug(str(workflow_file), None, "error", error_msg))

    def _validate_workflow_structure(self, workflow_file: Path, workflow: Dict[str, Any], lines: List[str]) -> None:
        """Validate basic workflow structure."""
        if not workflow:
            self.bugs.append(
                WorkflowBug(
                    str(workflow_file),
                    None,
                    "error",
                    "Empty workflow file",
                )
            )
            return

        if not isinstance(workflow, dict):
            self.bugs.append(
                WorkflowBug(
                    str(workflow_file),
                    None,
                    "error",
                    "Workflow root must be a mapping",
                )
            )
            return

        # Check for required fields
        # Note: YAML parsers interpret 'on:' as boolean True
        on_check = "on" in workflow
        true_check = any(key is True for key in workflow.keys())
        has_on_trigger = on_check or true_check
        if not has_on_trigger:
            self.bugs.append(
                WorkflowBug(
                    str(workflow_file),
                    None,
                    "error",
                    "Missing 'on' trigger definition",
                )
            )

        if "jobs" not in workflow:
            missing_jobs_msg = "Missing 'jobs' section"
            self.bugs.append(WorkflowBug(str(workflow_file), None, "error", missing_jobs_msg))

    def _check_step_references(self, workflow_file: Path, workflow: Dict[str, Any], lines: List[str]) -> None:
        """Check for missing step IDs when outputs are referenced."""
        if "jobs" not in workflow:
            return

        for job_name, job_config in workflow.get("jobs", {}).items():
            if not isinstance(job_config, dict):
                continue

            steps = job_config.get("steps", [])
            if not steps:
                continue

            # Build a map of step IDs
            step_ids = set()
            for idx, step in enumerate(steps):
                if not isinstance(step, dict):
                    continue
                if "id" in step:
                    step_ids.add(step["id"])

            # Check for references to step outputs
            for idx, step in enumerate(steps):
                if not isinstance(step, dict):
                    continue

                # Convert step to string to search for references
                step_str: str = yaml.dump(step)

                # Find step output references
                references = _STEP_OUTPUT_REF.findall(step_str)

                for ref_id in references:
                    if ref_id not in step_ids:
                        line_num = self._find_line_number(lines, f"steps.{ref_id}.outputs")
                        step_ref_msg = (
                            f"Step output referenced "
                            f"'steps.{ref_id}.outputs' but step id "
                            f"'{ref_id}' not found in job '{job_name}'"
                        )
                        self.bugs.append(
                            WorkflowBug(
                                str(workflow_file),
                                line_num,
                                "error",
                                step_ref_msg,
                            )
                        )

    def _check_shell_scripts(self, workflow_file: Path, workflow: Dict[str, Any], lines: List[str]) -> None:
        """Check for common shell script bugs in run commands."""
        if "jobs" not in workflow:
            return

        for job_name, job_config in workflow.get("jobs", {}).items():
            if not isinstance(job_config, dict):
                continue

            for idx, step in enumerate(job_config.get("steps", [])):
                if not isinstance(step, dict):
                    continue

                run_script = step.get("run")
                if not run_script:
                    continue

                # Check for unclosed conditionals
                self._check_conditionals(workflow_file, job_name, run_script, lines)

    def _check_conditionals(self, workflow_file: Path, job_name: str, script: str, lines: List[str]) -> None:
        """Check for unclosed if/fi statements."""
        if_count = 0
        fi_count = 0

        # Split by newlines and semicolons
        statements = _NEWLINE_SEMICOLON.split(script)

        for statement in statements:
            # Remove comments
            statement = _COMMENT_PATTERN.sub("", statement).strip()

            # Count if statements (excluding elif)
            if _IF_PATTERN.search(statement) and not _ELIF_PATTERN.search(statement):
                if_count += 1

            # Count fi statements - must be at start or after whitespace,
            # and must be end of command
            if _FI_PATTERN.search(statement):
                fi_count += 1

        if if_count != fi_count:
            primary_search = self._find_line_number(lines, _IF_SEARCH_PATTERN)
            fallback_search = self._find_line_number(lines, "if[")
            line_num = primary_search or fallback_search
            conditional_msg = (
                f"Unclosed conditional in job '{job_name}': "
                f"found {if_count} 'if' statements but "
                f"{fi_count} 'fi' statements"
            )
            self.bugs.append(
                WorkflowBug(
                    str(workflow_file),
                    line_num,
                    "error",
                    conditional_msg,
                )
            )

    def _check_job_dependencies(self, workflow_file: Path, workflow: Dict[str, Any]) -> None:
        """Check for invalid job dependencies."""
        if "jobs" not in workflow:
            return

        job_names = set(workflow["jobs"].keys())

        for job_name, job_config in workflow.get("jobs", {}).items():
            if not isinstance(job_config, dict):
                continue

            needs: Union[str, List[str]] = job_config.get("needs", [])
            needs_list = [needs] if isinstance(needs, str) else needs

            for needed_job in needs_list:
                if needed_job not in job_names:
                    dependency_msg = f"Job '{job_name}' depends on non-existent job " f"'{needed_job}'"
                    self.bugs.append(
                        WorkflowBug(
                            str(workflow_file),
                            None,
                            "error",
                            dependency_msg,
                        )
                    )

    def _check_matrix_usage(self, workflow_file: Path, workflow: Dict[str, Any], lines: List[str]) -> None:
        """Check for inefficient or incorrect matrix usage."""
        if "jobs" not in workflow:
            return

        for job_name, job_config in workflow.get("jobs", {}).items():
            if not isinstance(job_config, dict):
                continue

            strategy = job_config.get("strategy", {})
            if not isinstance(strategy, dict):
                continue

            matrix = strategy.get("matrix", {})
            if not matrix:
                continue

            # Check for exclusions
            exclusions: List[Dict[str, Any]] = matrix.get("exclude", [])

            # Check for task/device matrix combinations that don't make sense
            if "task" in matrix and "device" in matrix:
                tasks = matrix.get("task", [])
                devices = matrix.get("device", [])

                if isinstance(tasks, list) and isinstance(devices, list):
                    # Check if lint+gpu is excluded
                    lint_gpu_excluded = (
                        any(
                            isinstance(exc, dict) and exc.get("task") == "lint" and exc.get("device") == "gpu"
                            for exc in exclusions
                        )
                        if exclusions
                        else False
                    )

                    if "lint" in tasks and len(devices) > 1 and not lint_gpu_excluded:
                        line_num = self._find_line_number(lines, _DEVICE_SEARCH_PATTERN)
                        device_list = ", ".join(devices)
                        matrix_msg = (
                            f"Job '{job_name}' has device matrix "
                            f"[{device_list}] but includes 'lint' task "
                            "which doesn't require multiple devices"
                        )
                        self.bugs.append(
                            WorkflowBug(
                                str(workflow_file),
                                line_num,
                                "warning",
                                matrix_msg,
                            )
                        )

    def _find_line_number(self, lines: List[str], search_text: str) -> Optional[int]:
        """Find the line number containing the search text."""
        for idx, line in enumerate(lines, 1):
            if search_text in line:
                return idx
        return None

    def _check_openai_models(self, workflow_file: Path, workflow: Dict[str, Any], lines: List[str]) -> None:
        """Check for invalid or deprecated OpenAI model names."""
        valid_models = {
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        }

        # Valid prefixes for date-stamped versions
        # (e.g., gpt-4-turbo-YYYY-MM-DD)
        valid_prefixes = {"gpt-4-turbo-", "gpt-4o-", "gpt-3.5-turbo-"}

        if "jobs" not in workflow:
            return

        for job_name, job_config in workflow.get("jobs", {}).items():
            if not isinstance(job_config, dict):
                continue

            for idx, step in enumerate(job_config.get("steps", [])):
                if not isinstance(step, dict):
                    continue

                run_script = step.get("run")
                if not run_script:
                    continue

                # Search for OpenAI model references
                model_matches = _MODEL_PATTERN1.findall(run_script)
                model_matches += _MODEL_PATTERN2.findall(run_script)

                for model_match in model_matches:
                    # Check if it looks like a GPT model but isn't valid
                    if model_match.startswith("gpt-") and model_match not in valid_models:
                        is_versioned = any(model_match.startswith(vp) for vp in valid_prefixes)

                        if not is_versioned:
                            line_num = self._find_line_number(lines, model_match)
                            model_error_msg = f"Potentially invalid OpenAI model " f"name '{model_match}' in job '{job_name}'"
                            self.bugs.append(
                                WorkflowBug(
                                    str(workflow_file),
                                    line_num,
                                    "warning",
                                    model_error_msg,
                                )
                            )


def render_github_annotations(bugs: List[WorkflowBug]) -> None:
    """Emit GitHub Actions workflow commands to annotate the PR."""
    for bug in bugs:
        # Map severity to GitHub annotation levels
        # info -> notice, warning -> warning, error -> error
        level = "notice" if bug.severity == "info" else bug.severity

        annotation_msg = (
            f"::{level} file={_escape_github_command_property(bug.file_path)},"
            f"line={bug.line_number or 1},"
            f"title={_escape_github_command_property('Workflow Issue')}::"
            f"{_escape_github_command_value(bug.message)}"
        )
        print(annotation_msg)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GitHub Actions Workflow Validator")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root to search for .github/workflows",
    )
    parser.add_argument(
        "--github-actions",
        action="store_true",
        help="Emit GitHub Actions workflow commands.",
    )
    return parser.parse_args(argv)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Locate workflow directory relative to root
    workflow_dir = args.root / ".github" / "workflows"

    if not workflow_dir.exists():
        # Fallback: check if we are already inside .github/workflows
        # or close to it
        if Path(".github/workflows").exists():
            workflow_dir = Path(".github/workflows")
        else:
            print(f"Info: Workflow directory not found at {workflow_dir}")
            sys.exit(0)

    parser = WorkflowParser(workflow_dir)
    bugs = parser.parse_all_workflows()

    # Always print human-readable summary
    if not bugs:
        print("\n✅ No bugs found in workflow files!")
        return 0

    # Sort bugs by severity
    severity_order = {"error": 0, "warning": 1, "info": 2}
    bugs.sort(
        key=lambda b: (
            severity_order.get(b.severity, 3),
            b.file_path,
            b.line_number or 0,
        )
    )

    print(f"\n{'=' * 80}")
    print(f"Found {len(bugs)} issue(s) in workflow files:")
    print(f"{'=' * 80}\n")

    for bug in bugs:
        print(bug)
        if bug.context:
            print(f"  Context: {bug.context}")
        print()

    # Emit CI Annotations if requested
    if args.github_actions:
        render_github_annotations(bugs)

    # Summary
    error_count = sum(1 for b in bugs if b.severity == "error")

    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
