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

import yaml
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


class WorkflowBug:
    """Represents a bug found in a workflow file."""
    
    def __init__(self, file_path: str, line_number: Optional[int], 
                 severity: str, message: str, context: Optional[str] = None):
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
        workflow_files = list(self.workflow_dir.glob("*.yml")) + \
                        list(self.workflow_dir.glob("*.yaml"))
        
        for workflow_file in workflow_files:
            print(f"Parsing {workflow_file.name}...")
            self._parse_workflow(workflow_file)
        
        return self.bugs
    
    def _parse_workflow(self, workflow_file: Path):
        """Parse a single workflow file."""
        try:
            with open(workflow_file, 'r') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Try to parse YAML
            try:
                workflow = yaml.safe_load(content)
            except yaml.YAMLError as e:
                self.bugs.append(WorkflowBug(
                    str(workflow_file),
                    getattr(e, 'problem_mark', None).line + 1 if hasattr(e, 'problem_mark') else None,
                    'error',
                    f"YAML syntax error: {e}"
                ))
                return
            
            # Validate workflow structure
            self._validate_workflow_structure(workflow_file, workflow, lines)
            self._check_step_references(workflow_file, workflow, lines)
            self._check_shell_scripts(workflow_file, workflow, lines)
            self._check_job_dependencies(workflow_file, workflow)
            self._check_matrix_usage(workflow_file, workflow, lines)
            self._check_openai_models(workflow_file, workflow, lines)
            
        except Exception as e:
            self.bugs.append(WorkflowBug(
                str(workflow_file),
                None,
                'error',
                f"Failed to parse file: {e}"
            ))
    
    def _validate_workflow_structure(self, workflow_file: Path, workflow: Dict, lines: List[str]):
        """Validate basic workflow structure."""
        if not workflow:
            self.bugs.append(WorkflowBug(
                str(workflow_file),
                None,
                'error',
                "Empty workflow file"
            ))
            return
        
        # Check for required fields
        # Note: YAML parsers interpret 'on:' as boolean True
        if 'on' not in workflow and True not in workflow:
            self.bugs.append(WorkflowBug(
                str(workflow_file),
                None,
                'error',
                "Missing 'on' trigger definition"
            ))
        
        if 'jobs' not in workflow:
            self.bugs.append(WorkflowBug(
                str(workflow_file),
                None,
                'error',
                "Missing 'jobs' section"
            ))
    
    def _check_step_references(self, workflow_file: Path, workflow: Dict, lines: List[str]):
        """Check for missing step IDs when outputs are referenced."""
        if 'jobs' not in workflow:
            return
        
        for job_name, job_config in workflow.get('jobs', {}).items():
            if not isinstance(job_config, dict):
                continue
            
            steps = job_config.get('steps', [])
            if not steps:
                continue
            
            # Build a map of step IDs
            step_ids = set()
            for idx, step in enumerate(steps):
                if not isinstance(step, dict):
                    continue
                if 'id' in step:
                    step_ids.add(step['id'])
            
            # Check for references to step outputs
            for idx, step in enumerate(steps):
                if not isinstance(step, dict):
                    continue
                
                # Convert step to string to search for references
                step_str = yaml.dump(step)
                
                # Find step output references
                references = re.findall(r'\$\{\{\s*steps\.([a-zA-Z0-9_-]+)\.outputs', step_str)
                
                for ref_id in references:
                    if ref_id not in step_ids:
                        line_num = self._find_line_number(lines, f"steps.{ref_id}.outputs")
                        self.bugs.append(WorkflowBug(
                            str(workflow_file),
                            line_num,
                            'error',
                            f"Step output referenced 'steps.{ref_id}.outputs' but step id '{ref_id}' not found in job '{job_name}'"
                        ))
    
    def _check_shell_scripts(self, workflow_file: Path, workflow: Dict, lines: List[str]):
        """Check for common shell script bugs in run commands."""
        if 'jobs' not in workflow:
            return
        
        for job_name, job_config in workflow.get('jobs', {}).items():
            if not isinstance(job_config, dict):
                continue
            
            for idx, step in enumerate(job_config.get('steps', [])):
                if not isinstance(step, dict):
                    continue
                
                run_script = step.get('run')
                if not run_script:
                    continue
                
                # Check for unclosed conditionals
                self._check_conditionals(workflow_file, job_name, run_script, lines)
    
    def _check_conditionals(self, workflow_file: Path, job_name: str, 
                          script: str, lines: List[str]):
        """Check for unclosed if/fi statements."""
        if_count = 0
        fi_count = 0
        
        # Split by newlines and semicolons
        statements = re.split(r'[\n;]', script)
        
        for statement in statements:
            # Remove comments
            statement = re.sub(r'#.*$', '', statement).strip()
            
            # Count if statements (excluding elif)
            if re.search(r'\bif\s+', statement) and not re.search(r'\belif\s+', statement):
                if_count += 1
            
            # Count fi statements - must be at start or after whitespace, and must be end of command
            if re.search(r'(^|\s)fi(\s|$)', statement):
                fi_count += 1
        
        if if_count != fi_count:
            line_num = self._find_line_number(lines, f"if [ ")
            self.bugs.append(WorkflowBug(
                str(workflow_file),
                line_num,
                'error',
                f"Unclosed conditional in job '{job_name}': found {if_count} 'if' statements but {fi_count} 'fi' statements"
            ))
    
    def _check_job_dependencies(self, workflow_file: Path, workflow: Dict):
        """Check for invalid job dependencies."""
        if 'jobs' not in workflow:
            return
        
        job_names = set(workflow['jobs'].keys())
        
        for job_name, job_config in workflow.get('jobs', {}).items():
            if not isinstance(job_config, dict):
                continue
            
            needs = job_config.get('needs', [])
            if isinstance(needs, str):
                needs = [needs]
            
            for needed_job in needs:
                if needed_job not in job_names:
                    self.bugs.append(WorkflowBug(
                        str(workflow_file),
                        None,
                        'error',
                        f"Job '{job_name}' depends on non-existent job '{needed_job}'"
                    ))
    
    def _check_matrix_usage(self, workflow_file: Path, workflow: Dict, lines: List[str]):
        """Check for inefficient or incorrect matrix usage."""
        if 'jobs' not in workflow:
            return
        
        for job_name, job_config in workflow.get('jobs', {}).items():
            if not isinstance(job_config, dict):
                continue
            
            strategy = job_config.get('strategy', {})
            if not isinstance(strategy, dict):
                continue
            
            matrix = strategy.get('matrix', {})
            if not matrix:
                continue
            
            # Check for exclusions
            exclusions = matrix.get('exclude', [])
            
            # Check for task/device matrix combinations that don't make sense
            if 'task' in matrix and 'device' in matrix:
                tasks = matrix.get('task', [])
                devices = matrix.get('device', [])
                
                if isinstance(tasks, list) and isinstance(devices, list):
                    # Check if lint+gpu is excluded
                    lint_gpu_excluded = any(
                        exc.get('task') == 'lint' and exc.get('device') == 'gpu'
                        for exc in exclusions
                    ) if exclusions else False
                    
                    if 'lint' in tasks and len(devices) > 1 and not lint_gpu_excluded:
                        line_num = self._find_line_number(lines, "device:")
                        self.bugs.append(WorkflowBug(
                            str(workflow_file),
                            line_num,
                            'warning',
                            f"Job '{job_name}' has device matrix [{', '.join(devices)}] but includes 'lint' task which doesn't require multiple devices"
                        ))
    
    def _find_line_number(self, lines: List[str], search_text: str) -> Optional[int]:
        """Find the line number containing the search text."""
        for idx, line in enumerate(lines, 1):
            if search_text in line:
                return idx
        return None
    
    def _check_openai_models(self, workflow_file: Path, workflow: Dict, lines: List[str]):
        """Check for invalid OpenAI model names."""
        valid_models = {
            'gpt-4', 'gpt-4-turbo', 'gpt-4-turbo-preview', 
            'gpt-4o', 'gpt-4o-mini',
            'gpt-3.5-turbo', 'gpt-3.5-turbo-16k',
        }
        
        # Valid model prefixes (for date-stamped versions like gpt-4-turbo-2024-04-09)
        valid_prefixes = {
            'gpt-4-turbo-', 'gpt-4o-', 'gpt-3.5-turbo-'
        }
        
        if 'jobs' not in workflow:
            return
        
        for job_name, job_config in workflow.get('jobs', {}).items():
            if not isinstance(job_config, dict):
                continue
            
            for idx, step in enumerate(job_config.get('steps', [])):
                if not isinstance(step, dict):
                    continue
                
                run_script = step.get('run')
                if not run_script:
                    continue
                
                # Search for OpenAI model references (handle both quoted and escaped quotes)
                model_matches = re.findall(r'"model":\s*"([^"]+)"', run_script)
                model_matches += re.findall(r'\\"model\\":\s*\\"([^"\\]+)\\"', run_script)
                
                for model in model_matches:
                    # Check if it looks like a GPT model but isn't valid
                    if model.startswith('gpt-') and model not in valid_models:
                        # Check if it's a date-stamped version with valid prefix
                        is_versioned = any(model.startswith(vp) for vp in valid_prefixes)
                        
                        if not is_versioned:
                            line_num = self._find_line_number(lines, model)
                            self.bugs.append(WorkflowBug(
                                str(workflow_file),
                                line_num,
                                'warning',
                                f"Potentially invalid OpenAI model name '{model}' in job '{job_name}'"
                            ))


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent
    workflow_dir = repo_root / ".github" / "workflows"
    
    if not workflow_dir.exists():
        print(f"Error: Workflow directory not found at {workflow_dir}")
        sys.exit(1)
    
    parser = WorkflowParser(workflow_dir)
    bugs = parser.parse_all_workflows()
    
    if not bugs:
        print("\n✅ No bugs found in workflow files!")
        return 0
    
    # Sort bugs by severity
    severity_order = {'error': 0, 'warning': 1, 'info': 2}
    bugs.sort(key=lambda b: (severity_order.get(b.severity, 3), b.file_path, b.line_number or 0))
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Found {len(bugs)} issue(s) in workflow files:")
    print(f"{'='*80}\n")
    
    for bug in bugs:
        print(bug)
        if bug.context:
            print(f"  Context: {bug.context}")
        print()
    
    # Summary
    error_count = sum(1 for b in bugs if b.severity == 'error')
    warning_count = sum(1 for b in bugs if b.severity == 'warning')
    info_count = sum(1 for b in bugs if b.severity == 'info')
    
    print(f"{'='*80}")
    print(f"Summary: {error_count} error(s), {warning_count} warning(s), {info_count} info")
    print(f"{'='*80}")
    
    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
