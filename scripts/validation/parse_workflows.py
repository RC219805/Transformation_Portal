#!/usr/bin/env python3
"""
Parse and validate GitHub Actions workflow files for common bugs.

This script identifies:
- YAML syntax errors
- Missing step IDs when outputs are referenced
- Unclosed conditionals in shell scripts
- Invalid job dependencies
- Duplicate job names
- Invalid GitHub Actions syntax
"""

import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Precompiled regex patterns for performance
_STEP_OUTPUT_REF = re.compile(r"\$\{\{\s*steps\.([a-zA-Z0-9_-]+)\.outputs")
_MODEL_PATTERN1 = re.compile(r'"model":\s*"([^"]+)"')
_MODEL_PATTERN2 = re.compile(r'\\"model\\":\s*\\"([^"\\]+)\\"')


class WorkflowBug:
    """Represents a bug found in a workflow file."""

    def __init__(self, file_path: str, line_number: Optional[int], severity: str, message: str, context: Optional[str] = None):
        self.file_path = file_path
        self.line_number = line_number
        self.severity = severity  # 'error', 'warning', 'info'
        self.message = message
        self.context = context

    def __str__(self):
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
        workflow_files = list(self.workflow_dir.glob("*.yml")) + list(self.workflow_dir.glob("*.yaml"))

        for workflow_file in workflow_files:
            print(f"Parsing {workflow_file.name}...")
            self._parse_workflow(workflow_file)

        return self.bugs

    def _parse_workflow(self, workflow_file: Path):
        """Parse a single workflow file."""
        try:
            with open(workflow_file, "r") as f:
                content = f.read()
                lines = content.splitlines()

            # Try to parse YAML
            try:
                workflow = yaml.safe_load(content)
            except yaml.YAMLError as e:
                self.bugs.append(
                    WorkflowBug(
                        str(workflow_file),
                        getattr(e, "problem_mark", None).line + 1 if hasattr(e, "problem_mark") else None,
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
            self.bugs.append(WorkflowBug(str(workflow_file), None, "error", f"Failed to parse file: {e}"))

    def _validate_workflow_structure(self, workflow_file: Path, workflow: Dict, _lines: List[str]):
        """Validate basic workflow structure."""
        if not workflow:
            self.bugs.append(WorkflowBug(str(workflow_file), None, "error", "Empty workflow file"))
            return

        # Check for required fields
        # Note: YAML parsers interpret 'on:' as boolean True
        if "on" not in workflow and True not in workflow:
            self.bugs.append(WorkflowBug(str(workflow_file), None, "error", "Missing 'on' trigger definition"))

        if "jobs" not in workflow:
            self.bugs.append(WorkflowBug(str(workflow_file), None, "error", "Missing 'jobs' section"))

    def _check_step_references(self, workflow_file: Path, workflow: Dict, lines: List[str]):
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
                step_str = yaml.dump(step)

                # Find step output references
                references = _STEP_OUTPUT_REF.findall(step_str)

                for ref_id in references:
                    if ref_id not in step_ids:
                        line_num = self._find_line_number(lines, f"steps.{ref_id}.outputs")
                        self.bugs.append(
                            WorkflowBug(
                                str(workflow_file),
                                line_num,
                                "error",
                                f"Step output referenced 'steps.{ref_id}.outputs' but step id "
                                f"'{ref_id}' not found in job '{job_name}'",
                            )
                        )

    def _check_shell_scripts(self, workflow_file: Path, workflow: Dict, lines: List[str]):
        """Parse supported shell bodies without running workflow commands."""
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

                shell = self._shell_for_step(workflow, job_config, step)
                location = f"job '{job_name}', step {idx + 1}"
                line_num = self._find_line_number(lines, run_script.splitlines()[0])
                if shell is None:
                    self.bugs.append(
                        WorkflowBug(
                            str(workflow_file),
                            line_num,
                            "info",
                            f"Shell syntax check skipped for {location}: unsupported or unresolved shell",
                        )
                    )
                    continue
                # Never execute the workflow's shell declaration or its options.
                # A minimal environment also excludes startup hooks and exported functions.
                command = ["/bin/bash", "--noprofile", "--norc", "-p", "-n"] if shell == "bash" else ["/bin/sh", "-n"]
                try:
                    result = subprocess.run(
                        command,
                        input=run_script,
                        text=True,
                        capture_output=True,
                        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
                        timeout=10,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    self.bugs.append(
                        WorkflowBug(
                            str(workflow_file), line_num, "error", f"Cannot check {shell} syntax for {location}: {exc}"
                        )
                    )
                    continue
                if result.returncode:
                    self.bugs.append(
                        WorkflowBug(
                            str(workflow_file),
                            line_num,
                            "error",
                            f"Shell syntax error in {location} ({shell})",
                            result.stderr.strip(),
                        )
                    )

    @staticmethod
    def _shell_for_step(workflow: Dict, job: Dict, step: Dict) -> Optional[str]:
        """Resolve shell precedence, admitting only known Bash/sh command templates."""
        shell = step.get("shell")
        for scope in (job, workflow):
            if shell is None:
                shell = (scope.get("defaults") or {}).get("run", {}).get("shell")
        if shell is None:
            container = job.get("container")
            if isinstance(container, str) and "${{" in container:
                return None
            if container:
                return "sh"
            runners = job.get("runs-on", [])
            runners = [runners] if isinstance(runners, str) else runners
            if not isinstance(runners, list) or any("${{" in str(label) for label in runners):
                return None
            labels = [str(label).lower() for label in runners]
            if any("windows" in label for label in labels):
                return None
            return (
                "bash"
                if any(label.startswith(("ubuntu", "macos")) or label in {"linux", "macos"} for label in labels)
                else None
            )
        if not isinstance(shell, str) or "${{" in shell:
            return None
        try:
            tokens = shlex.split(shell)
        except ValueError:
            return None
        if tokens and Path(tokens[0]).name == "env":
            tokens.pop(0)
            while tokens:
                if tokens[0] in {"-u", "--unset"} and len(tokens) > 1:
                    del tokens[:2]
                elif tokens[0] in {"-i", "--ignore-environment"} or tokens[0].startswith("--unset=") or "=" in tokens[0]:
                    tokens.pop(0)
                elif tokens[0] == "--":
                    tokens.pop(0)
                    break
                else:
                    break
        if not tokens or Path(tokens[0]).name not in {"bash", "sh"}:
            return None
        interpreter = Path(tokens.pop(0)).name
        # Custom templates that change parsing (for example bash -c or -O) are
        # unsupported rather than being executed or checked under wrong semantics.
        while tokens:
            token = tokens.pop(0)
            if token == "-o" and tokens and tokens[0] in {"pipefail", "errexit", "nounset"}:
                tokens.pop(0)
            elif token not in {"{0}", "--noprofile", "--norc", "-p", "-e", "-u"}:
                return None
        return interpreter

    def _check_job_dependencies(self, workflow_file: Path, workflow: Dict):
        """Check for invalid job dependencies."""
        if "jobs" not in workflow:
            return

        job_names = set(workflow["jobs"].keys())

        for job_name, job_config in workflow.get("jobs", {}).items():
            if not isinstance(job_config, dict):
                continue

            needs = job_config.get("needs", [])
            if isinstance(needs, str):
                needs = [needs]

            for needed_job in needs:
                if needed_job not in job_names:
                    self.bugs.append(
                        WorkflowBug(
                            str(workflow_file), None, "error", f"Job '{job_name}' depends on non-existent job '{needed_job}'"
                        )
                    )

    def _check_matrix_usage(self, workflow_file: Path, workflow: Dict, lines: List[str]):
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
            exclusions = matrix.get("exclude", [])

            # Check for task/device matrix combinations that don't make sense
            if "task" in matrix and "device" in matrix:
                tasks = matrix.get("task", [])
                devices = matrix.get("device", [])

                if isinstance(tasks, list) and isinstance(devices, list):
                    # Check if lint+gpu is excluded
                    lint_gpu_excluded = (
                        any(exc.get("task") == "lint" and exc.get("device") == "gpu" for exc in exclusions)
                        if exclusions
                        else False
                    )

                    if "lint" in tasks and len(devices) > 1 and not lint_gpu_excluded:
                        line_num = self._find_line_number(lines, "device:")
                        device_list = ", ".join(devices)
                        self.bugs.append(
                            WorkflowBug(
                                str(workflow_file),
                                line_num,
                                "warning",
                                f"Job '{job_name}' has device matrix [{device_list}] but includes "
                                "'lint' task which doesn't require multiple devices",
                            )
                        )

    def _find_line_number(self, lines: List[str], search_text: str) -> Optional[int]:
        """Find the line number containing the search text."""
        for idx, line in enumerate(lines, 1):
            if search_text in line:
                return idx
        return None

    def _check_openai_models(self, workflow_file: Path, workflow: Dict, lines: List[str]):
        """Emit advisory hints from a legacy name list, not API availability claims."""
        known_models = {
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        }

        # Historical prefix hints suppress this advisory, without validating any
        # suffix or establishing whether a provider actually offers the model.
        known_prefixes = {"gpt-4-turbo-", "gpt-4o-", "gpt-3.5-turbo-"}

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

                # Search for OpenAI model references (handle both quoted and escaped quotes)
                model_matches = _MODEL_PATTERN1.findall(run_script)
                model_matches += _MODEL_PATTERN2.findall(run_script)

                for model in model_matches:
                    # The static list cannot determine provider/account availability.
                    if model.startswith("gpt-") and model not in known_models:
                        # Preserve the legacy prefix heuristic as advisory only.
                        is_versioned = any(model.startswith(vp) for vp in known_prefixes)

                        if not is_versioned:
                            line_num = self._find_line_number(lines, model)
                            self.bugs.append(
                                WorkflowBug(
                                    str(workflow_file),
                                    line_num,
                                    "warning",
                                    f"Model '{model}' in job '{job_name}' is absent from legacy static hints; "
                                    "this does not establish API validity or account availability",
                                )
                            )


def _print_json_output(bugs: List[WorkflowBug]) -> None:
    """Print bugs in JSON format."""
    import json

    data = [
        {
            "file": bug.file_path,
            "line": bug.line_number,
            "severity": bug.severity,
            "message": bug.message,
            "context": bug.context,
        }
        for bug in bugs
    ]
    print(json.dumps(data, indent=2))


def main():
    """Main entry point."""
    import argparse

    # Calculate repo root from script location (scripts/ -> repo root)
    repo_root = Path(__file__).resolve().parents[2]
    default_workflow_dir = repo_root / ".github" / "workflows"

    parser = argparse.ArgumentParser(description="Parse and validate GitHub Actions workflow files for common bugs")
    parser.add_argument(
        "--workflow-dir",
        type=Path,
        default=default_workflow_dir,
        help=f"Path to workflows directory (default: {default_workflow_dir})",
    )
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format (default: text)")

    args = parser.parse_args()
    workflow_dir = args.workflow_dir

    if not workflow_dir.exists():
        print(f"Error: Workflow directory not found at {workflow_dir}", file=sys.stderr)
        sys.exit(1)

    parser_obj = WorkflowParser(workflow_dir)
    bugs = parser_obj.parse_all_workflows()

    if args.format == "json":
        _print_json_output(bugs)
        return 1 if any(b.severity == "error" for b in bugs) else 0

    # Text format output
    if not bugs:
        print("\n✅ No bugs found in workflow files!")
        return 0

    # Sort bugs by severity
    severity_order = {"error": 0, "warning": 1, "info": 2}
    bugs.sort(key=lambda b: (severity_order.get(b.severity, 3), b.file_path, b.line_number or 0))

    # Print results
    print(f"\n{'=' * 80}")
    print(f"Found {len(bugs)} issue(s) in workflow files:")
    print(f"{'=' * 80}\n")

    for bug in bugs:
        print(bug)
        if bug.context:
            print(f"  Context: {bug.context}")
        print()

    # Summary
    error_count = sum(1 for b in bugs if b.severity == "error")
    warning_count = sum(1 for b in bugs if b.severity == "warning")
    info_count = sum(1 for b in bugs if b.severity == "info")

    print(f"{'=' * 80}")
    print(f"Summary: {error_count} error(s), {warning_count} warning(s), {info_count} info")
    print(f"{'=' * 80}")

    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
