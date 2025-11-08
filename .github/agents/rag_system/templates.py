"""
Canonical Prompt Templates for Transformation Portal Specialist

Provides structured templates for common workflows:
- Feature implementation
- Bug triage
- CI changes
"""

import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class FileModification:
    """Represents a file modification with patch."""

    path: str
    patch: str
    description: Optional[str] = None


@dataclass
class CodeModificationResponse:
    """
    Structured response schema for code modification requests.

    This schema ensures machine-parseable responses for CI validation.
    """

    summary: str
    files: List[FileModification]
    tests: List[str]
    explanation: str
    confidence: float = 0.0
    citations: Optional[List[Dict]] = None

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = {
            'summary': self.summary,
            'files': [
                {'path': f.path, 'patch': f.patch, 'description': f.description}
                for f in self.files
            ],
            'tests': self.tests,
            'explanation': self.explanation,
            'confidence': self.confidence,
            'citations': self.citations or [],
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'CodeModificationResponse':
        """Create from JSON string."""
        data = json.loads(json_str)
        files = [
            FileModification(
                path=f['path'],
                patch=f['patch'],
                description=f.get('description')
            )
            for f in data['files']
        ]
        return cls(
            summary=data['summary'],
            files=files,
            tests=data['tests'],
            explanation=data['explanation'],
            confidence=data.get('confidence', 0.0),
            citations=data.get('citations'),
        )


class PromptTemplates:
    """Canonical prompt templates for common workflows."""

    @staticmethod
    def feature_implementation(
        feature_description: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Template for feature implementation workflow.

        Workflow: Requirements → Files to modify → Tests to add → PR body
        """
        template = f"""# Feature Implementation Request

## Feature Description
{feature_description}

## Context
{context or 'No additional context provided.'}

## Required Analysis

Please analyze and provide:

### 1. Requirements Clarification
- Core functionality requirements
- Edge cases to consider
- Performance implications
- Dependencies (new packages, ML models, etc.)

### 2. Files to Modify
For each file:
- File path
- Specific changes needed (functions/classes to add/modify)
- Reason for the change

### 3. Tests to Add
- Test file paths
- Test scenarios to cover
- Edge cases to test

### 4. Implementation Plan
- Step-by-step implementation order
- Integration points with existing pipelines
- Configuration changes needed

### 5. PR Description Template
Generate a PR description including:
- Feature summary
- Technical changes made
- Testing performed
- Performance impact

## Response Format
Provide response in JSON format following CodeModificationResponse schema:
```json
{{
  "summary": "Brief feature summary",
  "files": [
    {{
      "path": "path/to/file.py",
      "patch": "unified diff format or description of changes",
      "description": "Why this change is needed"
    }}
  ],
  "tests": ["tests/test_feature.py", "tests/integration/test_pipeline.py"],
  "explanation": "Detailed explanation of implementation approach",
  "confidence": 0.85,
  "citations": [
    {{
      "file_path": "existing_file.py",
      "snippet": "relevant code example",
      "relevance": "shows similar pattern"
    }}
  ]
}}
```

## Examples from Repository
[RAG system will inject relevant examples here]
"""
        return template

    @staticmethod
    def bug_triage(
        error_log: str,
        reproduction_steps: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> str:
        """
        Template for bug triage workflow.

        Workflow: Error log → Probable cause → Minimal repro → Fix steps
        """
        template = f"""# Bug Triage Request

## Error Log
```
{error_log}
```

## Reproduction Steps
{reproduction_steps or 'Not provided'}

## Environment
{environment or 'Not specified'}

## Required Analysis

Please analyze and provide:

### 1. Error Classification
- Error type (import error, runtime error, logic error, etc.)
- Severity (critical, high, medium, low)
- Affected components/pipelines

### 2. Root Cause Analysis
- Probable cause of the error
- Which file(s) contain the problematic code
- Why the error occurs (missing dependency, logic flaw, etc.)

### 3. Minimal Reproduction
- Minimal code to reproduce the issue
- Required inputs/configuration
- Expected vs actual behavior

### 4. Fix Strategy
- Recommended fix approach
- Files to modify
- Potential side effects
- Alternative approaches if applicable

### 5. Testing Strategy
- How to test the fix
- Regression tests needed
- Edge cases to verify

## Response Format
Provide response in JSON format:
```json
{{
  "summary": "Brief description of the bug and fix",
  "files": [
    {{
      "path": "path/to/buggy_file.py",
      "patch": "@@ -10,5 +10,5 @@\\n-old code\\n+new code",
      "description": "Fix explanation"
    }}
  ],
  "tests": ["tests/test_regression_bug123.py"],
  "explanation": "Root cause and fix rationale",
  "confidence": 0.90,
  "citations": [
    {{
      "file_path": "related_code.py",
      "snippet": "similar error handling pattern",
      "relevance": "shows correct approach"
    }}
  ]
}}
```

## Similar Issues in Repository
[RAG system will inject similar past issues/fixes here]
"""
        return template

    @staticmethod
    def ci_change(
        workflow_name: str,
        change_description: str,
        reason: Optional[str] = None,
    ) -> str:
        """
        Template for CI/workflow changes.

        Workflow: Workflow name → Job steps → Test coverage → Required secrets
        """
        template = f"""# CI Workflow Change Request

## Workflow Name
{workflow_name}

## Requested Change
{change_description}

## Reason
{reason or 'Not provided'}

## Required Analysis

Please analyze and provide:

### 1. Current Workflow Analysis
- Current workflow file: `.github/workflows/{workflow_name}`
- Existing jobs and steps
- Current triggers and conditions

### 2. Proposed Changes
- Specific changes to workflow YAML
- New jobs or steps to add
- Modified triggers or conditions

### 3. Testing Strategy
- How to test workflow changes (workflow_dispatch, PR testing)
- Expected outcomes
- Failure scenarios to handle

### 4. Required Secrets/Variables
- New GitHub secrets needed
- Repository variables to configure
- Environment-specific settings

### 5. Impact Assessment
- Build time impact
- Cost implications (GitHub Actions minutes)
- Effects on PR checks

## Response Format
Provide response in JSON format:
```json
{{
  "summary": "Brief description of workflow change",
  "files": [
    {{
      "path": ".github/workflows/{workflow_name}",
      "patch": "YAML diff showing changes",
      "description": "What this change does"
    }}
  ],
  "tests": [
    "Manual test: trigger workflow with workflow_dispatch",
    "Verify: check Actions tab for successful run"
  ],
  "explanation": "Why this change is needed and how it works",
  "confidence": 0.80,
  "citations": [
    {{
      "file_path": ".github/workflows/existing_workflow.yml",
      "snippet": "similar workflow pattern",
      "relevance": "example of correct syntax"
    }}
  ]
}}
```

## Existing Workflows
[RAG system will inject relevant workflow examples here]
"""
        return template

    @staticmethod
    def add_few_shot_examples(
        template: str,
        examples: List[Dict],
    ) -> str:
        """
        Add few-shot examples to a template.

        Args:
            template: Base template string
            examples: List of example dicts with 'input' and 'output' keys

        Returns:
            Template with examples appended
        """
        if not examples:
            return template

        examples_section = "\n\n## Few-Shot Examples\n\n"

        for i, example in enumerate(examples, 1):
            examples_section += f"### Example {i}\n\n"
            examples_section += f"**Input:**\n```\n{example.get('input', 'N/A')}\n```\n\n"
            examples_section += f"**Output:**\n```json\n{example.get('output', 'N/A')}\n```\n\n"

        return template + examples_section


class FewShotExamples:
    """Few-shot examples from the repository."""

    @staticmethod
    def get_feature_examples() -> List[Dict]:
        """Get feature implementation examples."""
        return [
            {
                'input': 'Add depth-based atmospheric haze effect to the depth pipeline',
                'output': json.dumps({
                    'summary': 'Add atmospheric haze effect based on depth information',
                    'files': [
                        {
                            'path': 'depth_pipeline/processors/atmospheric.py',
                            'patch': 'Add haze_intensity parameter and depth-based blending',
                            'description': 'New processor for atmospheric effects'
                        },
                        {
                            'path': 'config/presets/exterior.yaml',
                            'patch': 'Add haze_intensity: 0.3 to preset',
                            'description': 'Configure default haze for exteriors'
                        }
                    ],
                    'tests': [
                        'tests/test_atmospheric_processor.py',
                        'tests/integration/test_depth_pipeline.py'
                    ],
                    'explanation': (
                        """Atmospheric haze is implemented by blending a fog color proportional to depth distance. Uses depth maps to determine haze intensity per-pixel."""
                    ),
                    'confidence': 0.85,
                }, indent=2)
            },
            {
                'input': 'Add new LUT preset for warm sunset aesthetic',
                'output': json.dumps({
                    'summary': 'Add California Golden Hour LUT preset',
                    'files': [
                        {
                            'path': 'luxury_video_master_grader.py',
                            'patch': 'Add sunset_estate preset with California_Golden_Hour.cube',
                            'description': 'New preset in PRESETS dictionary'
                        }
                    ],
                    'tests': [
                        'tests/test_luxury_video_master_grader.py::test_preset_exists'
                    ],
                    'explanation': 'Adding a new preset is straightforward: define PresetConfig '
                                   'with LUT path, exposure, contrast, saturation values. '
                                   'LUT file should exist in assets/luts/location_aesthetic/',
                    'confidence': 0.95,
                }, indent=2)
            }
        ]

    @staticmethod
    def get_bug_triage_examples() -> List[Dict]:
        """Get bug triage examples."""
        return [
            {
                'input': 'ImportError: No module named "tifffile"',
                'output': json.dumps({
                    'summary': 'Missing optional dependency: tifffile',
                    'files': [
                        {
                            'path': 'luxury_tiff_batch_processor.py',
                            'patch': 'Wrap tifffile import in try/except with fallback to Pillow',
                            'description': 'Make tifffile optional with graceful fallback'
                        }
                    ],
                    'tests': [
                        'tests/test_luxury_tiff_processor.py::test_works_without_tifffile'
                    ],
                    'explanation': 'tifffile is an optional dependency for 16-bit TIFF support. '
                                   'When not available, should fall back to Pillow with a warning. '
                                   'This follows the pattern in other scripts.',
                    'confidence': 0.90,
                }, indent=2)
            }
        ]

    @staticmethod
    def get_ci_change_examples() -> List[Dict]:
        """Get CI workflow change examples."""
        return [
            {
                'input': 'Add Python 3.12 to test matrix',
                'output': json.dumps({
                    'summary': 'Add Python 3.12 to CI test matrix',
                    'files': [
                        {
                            'path': '.github/workflows/build.yml',
                            'patch': 'Add "3.12" to python-version matrix',
                            'description': 'Extend test coverage to Python 3.12'
                        }
                    ],
                    'tests': [
                        'Manual: Push to branch and verify Actions tab shows Python 3.12 job',
                        'Verify: All tests pass on Python 3.12'
                    ],
                    'explanation': 'Python 3.12 is now stable. Adding to matrix ensures '
                                   'compatibility. No code changes needed if already compatible.',
                    'confidence': 0.95,
                }, indent=2)
            }
        ]


def validate_response_schema(response: str) -> bool:
    """
    Validate that a response conforms to CodeModificationResponse schema.

    Args:
        response: JSON string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        data = json.loads(response)

        # Check required fields
        required_fields = ['summary', 'files', 'tests', 'explanation']
        for field in required_fields:
            if field not in data:
                print(f"Missing required field: {field}")
                return False

        # Validate files structure
        if not isinstance(data['files'], list):
            print("'files' must be a list")
            return False

        for file_mod in data['files']:
            if 'path' not in file_mod or 'patch' not in file_mod:
                print("Each file must have 'path' and 'patch'")
                return False

        # Validate tests
        if not isinstance(data['tests'], list):
            print("'tests' must be a list")
            return False

        # Optional: validate confidence range
        if 'confidence' in data:
            conf = data['confidence']
            if not (0.0 <= conf <= 1.0):
                print(f"'confidence' must be between 0.0 and 1.0, got {conf}")
                return False

        return True

    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"Validation error: {e}")
        return False


def main():
    """CLI for template generation and validation."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate prompt templates')
    parser.add_argument('--type', choices=['feature', 'bug', 'ci'], required=True,
                        help='Template type')
    parser.add_argument('--description', required=True, help='Description/error log')
    parser.add_argument('--context', help='Additional context')
    parser.add_argument('--with-examples', action='store_true',
                        help='Include few-shot examples')
    parser.add_argument('--validate', help='Validate a JSON response file')

    args = parser.parse_args()

    if args.validate:
        with open(args.validate) as f:
            response = f.read()

        if validate_response_schema(response):
            print("✓ Response is valid")
            # Pretty print
            data = json.loads(response)
            print("\nParsed response:")
            print(json.dumps(data, indent=2))
        else:
            print("✗ Response is invalid")
            sys.exit(1)
        return

    # Generate template
    if args.type == 'feature':
        template = PromptTemplates.feature_implementation(
            args.description,
            context=args.context,
        )
        if args.with_examples:
            examples = FewShotExamples.get_feature_examples()
            template = PromptTemplates.add_few_shot_examples(template, examples)

    elif args.type == 'bug':
        template = PromptTemplates.bug_triage(
            args.description,
            environment=args.context,
        )
        if args.with_examples:
            examples = FewShotExamples.get_bug_triage_examples()
            template = PromptTemplates.add_few_shot_examples(template, examples)

    elif args.type == 'ci':
        # Parse workflow name from description
        workflow_name = args.description.split()[0] if args.description else 'workflow.yml'
        template = PromptTemplates.ci_change(
            workflow_name,
            args.description,
            reason=args.context,
        )
        if args.with_examples:
            examples = FewShotExamples.get_ci_change_examples()
            template = PromptTemplates.add_few_shot_examples(template, examples)

    print(template)


if __name__ == '__main__':
    main()
