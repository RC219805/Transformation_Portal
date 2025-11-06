# Feature Implementation Request

## Feature Description
Add HDR tone mapping with custom transfer function

## Context
Existing tone mapping in tonemapper_agx_filmic.py

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
{
  "summary": "Brief feature summary",
  "files": [
    {
      "path": "path/to/file.py",
      "patch": "unified diff format or description of changes",
      "description": "Why this change is needed"
    }
  ],
  "tests": ["tests/test_feature.py", "tests/integration/test_pipeline.py"],
  "explanation": "Detailed explanation of implementation approach",
  "confidence": 0.85,
  "citations": [
    {
      "file_path": "existing_file.py",
      "snippet": "relevant code example",
      "relevance": "shows similar pattern"
    }
  ]
}
```

## Examples from Repository
[RAG system will inject relevant examples here]
