# RAG System Template Index

**Template Collection Version**: 1.0  
**Created**: 2025-11-06  
**Purpose**: Comprehensive prompt templates for Transformation Portal development workflows

---

## Quick Template Selector

**Choose a template based on your task**:

| What are you doing? | Use this template |
|---------------------|-------------------|
| Adding a new depth effect or processor | [Feature Implementation](#feature-implementation) |
| Fixing an ImportError or runtime error | [Bug Triage](#bug-triage) |
| Creating a new YAML config or preset | [Pipeline Configuration](#pipeline-configuration) |
| Writing tests for new code | [Testing](#testing) |
| Documenting a new feature or API | [Documentation](#documentation) |
| Making code faster or use less memory | [Performance Optimization](#performance-optimization) |

---

## Template Summaries

### Feature Implementation
**File**: `feature_implementation.md` (20KB)

**Workflow**:
```
Requirements → Implementation Plan → Testing → Documentation → Validation
```

**Key Sections**:
1. **Requirements Analysis** - Core functionality, edge cases, dependencies
2. **Implementation Plan** - Step-by-step with code examples
3. **Testing Strategy** - Unit, integration, property-based, performance tests
4. **Documentation** - README updates, docstrings, examples
5. **Validation Checklist** - Code quality, testing, performance, metadata
6. **Few-Shot Examples** - Real repository patterns

**Output Format**: JSON (CodeModificationResponse schema)

**Best for**:
- New depth pipeline processors
- Material enhancement features
- LUT presets and color grading
- CLI tool extensions

---

### Bug Triage
**File**: `bug_triage.md` (18KB)

**Workflow**:
```
Error Classification → Root Cause → Fix Strategy → Testing → Validation
```

**Key Sections**:
1. **Error Classification** - Type, severity, affected components
2. **Root Cause Analysis** - Hypothesis, evidence, probable files
3. **Fix Strategy** - Multiple approaches with pros/cons
4. **Fix Implementation** - Code changes with diffs
5. **Testing Strategy** - Regression tests, edge cases
6. **Common Bug Patterns** - ImportError, FFmpeg, OOM, metadata

**Output Format**: JSON (CodeModificationResponse schema)

**Best for**:
- Missing dependency errors
- FFmpeg command failures
- Memory errors (OOM)
- Platform-specific issues
- Metadata preservation bugs

---

### Pipeline Configuration
**File**: `pipeline_configuration.md` (18KB)

**Workflow**:
```
Config Overview → Parameters → Validation → Testing → Deployment
```

**Key Sections**:
1. **Configuration Templates** - YAML structure for different pipelines
2. **Parameter Reference** - Tables with ranges, defaults, descriptions
3. **Use Case Presets** - Interior, exterior, aerial, product configurations
4. **Validation** - File existence, parameter ranges, logical consistency
5. **Testing** - Manual and automated validation workflows
6. **Performance Tips** - Speed vs quality tradeoffs

**Output Format**: YAML configuration files

**Best for**:
- Depth pipeline presets
- Video grading configurations
- Batch processor settings
- Custom workflow configs

---

### Testing
**File**: `testing.md` (26KB)

**Workflow**:
```
Test Plan → Unit Tests → Integration → Benchmarks → CI Setup
```

**Key Sections**:
1. **Test Structure** - File organization, pytest markers
2. **Unit Tests** - Class-based test suites, parametrized tests
3. **Integration Tests** - Full pipeline workflows
4. **Property-Based Tests** - Hypothesis for mathematical properties
5. **Performance Benchmarks** - Time and memory profiling
6. **Mock-Based Tests** - Isolating heavy dependencies
7. **Repository Patterns** - Metadata, FFmpeg, optional dependencies

**Output Format**: pytest test files

**Best for**:
- New feature test coverage
- Regression test suites
- Performance benchmarking
- CI/CD test configuration

---

### Documentation
**File**: `documentation.md` (15KB)

**Workflow**:
```
Doc Type Selection → Structure → Examples → API Docs → Validation
```

**Key Sections**:
1. **Feature Documentation** - Structure for README entries
2. **API Documentation** - NumPy docstring format
3. **Usage Guides** - Step-by-step how-to guides
4. **Configuration Docs** - Parameter reference format
5. **Troubleshooting** - Common issues and solutions
6. **Documentation Checklist** - Quality standards

**Output Format**: Markdown files, docstrings

**Best for**:
- Feature announcements
- API reference documentation
- User guides and tutorials
- Configuration guides
- Troubleshooting FAQs

---

### Performance Optimization
**File**: `performance_optimization.md` (19KB)

**Workflow**:
```
Profiling → Bottleneck ID → Optimization → Benchmarking → Validation
```

**Key Sections**:
1. **Profiling** - cProfile, memory_profiler, GPU profiling
2. **Optimization Strategies** - Caching, batching, GPU, vectorization
3. **Benchmarking** - Structured performance testing
4. **Validation** - Before/after comparison
5. **Repository Optimizations** - Depth pipeline, video processing specific

**Output Format**: Benchmark reports, optimized code

**Best for**:
- Reducing processing time
- Memory optimization
- GPU/CoreML acceleration
- Batch throughput improvement

---

## Template Usage Patterns

### 1. Feature Development Workflow

```
1. Feature Implementation Template
   ↓ (generate implementation plan)
2. Testing Template
   ↓ (write test suite)
3. Documentation Template
   ↓ (document feature)
4. Performance Optimization Template (if needed)
   ↓ (profile and optimize)
5. Validation
```

### 2. Bug Fix Workflow

```
1. Bug Triage Template
   ↓ (diagnose and fix)
2. Testing Template (regression tests)
   ↓ (prevent recurrence)
3. Documentation Template (troubleshooting)
   ↓ (help others)
```

### 3. Configuration Creation Workflow

```
1. Pipeline Configuration Template
   ↓ (create YAML config)
2. Testing Template (config validation)
   ↓ (test configuration)
3. Documentation Template (parameter guide)
   ↓ (document usage)
```

---

## Integration with RAG System

### Python API

```python
from pathlib import Path

# Load markdown template
template_dir = Path('.github/agents/rag_system/templates/')
template_path = template_dir / 'feature_implementation.md'

with open(template_path) as f:
    template_content = f.read()

# Or use Python templates for dynamic generation
from rag_system.templates import PromptTemplates

template = PromptTemplates.feature_implementation(
    feature_description="Add depth-based fog effect",
    context="For exterior architectural renders"
)
```

### CLI Usage

```bash
# Generate template
python .github/agents/rag_system/templates.py \
    --type feature \
    --description "Add atmospheric haze effect" \
    --context "Depth pipeline processor" \
    --with-examples

# Validate JSON response
python .github/agents/rag_system/templates.py \
    --validate response.json
```

---

## Template Quality Standards

### All Templates Must Include

- [ ] Clear workflow diagram or sequence
- [ ] Code examples from repository
- [ ] Validation checklist
- [ ] Output format specification
- [ ] Few-shot examples (where applicable)

### Code Examples Must

- [ ] Be syntactically correct Python
- [ ] Follow repository style (PEP 8, max line 127)
- [ ] Include docstrings for functions/classes
- [ ] Show expected output or behavior
- [ ] Include performance notes

### Documentation Must

- [ ] Use consistent terminology
- [ ] Link to related templates/docs
- [ ] Include troubleshooting section
- [ ] Specify template version and date

---

## Template Maintenance

### When to Update Templates

- **Monthly**: Review examples for outdated patterns
- **On major release**: Update version references and API changes
- **On new feature**: Add relevant few-shot examples
- **On bug fix**: Add to common bug patterns section

### Update Process

1. Identify section to update
2. Test updated examples
3. Update template version number
4. Update changelog in template
5. Update this index document
6. Commit with descriptive message

### Version Format

Templates use semantic versioning:
- **Major** (1.x.x): Structure changes, new sections
- **Minor** (x.1.x): New examples, expanded content
- **Patch** (x.x.1): Fixes, clarifications

---

## Template File Sizes

| Template | Size | Lines | Code Examples |
|----------|------|-------|---------------|
| feature_implementation.md | 20KB | 850+ | 15+ |
| bug_triage.md | 18KB | 750+ | 12+ |
| pipeline_configuration.md | 18KB | 700+ | 10+ |
| testing.md | 26KB | 1100+ | 20+ |
| documentation.md | 15KB | 650+ | 10+ |
| performance_optimization.md | 19KB | 800+ | 15+ |
| **Total** | **116KB** | **4850+** | **82+** |

---

## Repository-Specific Patterns

These patterns appear across multiple templates:

### 1. Depth-Based Processing
```python
# Normalize depth maps to [0, 1]
depth_map = (depth - depth.min()) / (depth.max() - depth.min())

# Zone-based processing
foreground = depth_map < 0.3
midground = (depth_map >= 0.3) & (depth_map < 0.7)
background = depth_map >= 0.7
```

### 2. Metadata Preservation
```python
# Always preserve PIL Image.info
original_info = image.info.copy()
result = process(image)
result.info = original_info
```

### 3. Optional Dependencies
```python
# Graceful fallback
try:
    import optional_package
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
    warnings.warn("Package not available")
```

### 4. Performance Caching
```python
# LRU cache for expensive operations
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_operation(cache_key: str):
    return compute_result(cache_key)
```

### 5. Testing with Fixtures
```python
# Pytest fixtures for common test data
@pytest.fixture
def sample_image(tmp_path):
    image = Image.new('RGB', (512, 512))
    path = tmp_path / "test.jpg"
    image.save(path)
    return path
```

---

## Response Schema

All code modification templates use this JSON schema:

```json
{
  "summary": "Brief one-line description",
  "files": [
    {
      "path": "relative/path/to/file.py",
      "patch": "unified diff or description",
      "description": "Why this change is needed"
    }
  ],
  "tests": ["tests/test_module.py"],
  "explanation": "Detailed rationale with tradeoffs",
  "confidence": 0.85,
  "citations": [
    {
      "file_path": "existing_code.py",
      "snippet": "relevant code",
      "relevance": "shows similar pattern"
    }
  ]
}
```

**Validation**: Use `templates.py --validate response.json`

---

## Resources

### Template Files
- [Feature Implementation](feature_implementation.md)
- [Bug Triage](bug_triage.md)
- [Pipeline Configuration](pipeline_configuration.md)
- [Testing](testing.md)
- [Documentation](documentation.md)
- [Performance Optimization](performance_optimization.md)

### Related Documentation
- [RAG System README](../README.md)
- [Python Templates](../templates.py)
- [Knowledge Engine](../knowledge_engine.py)
- [Citation Generator](../citation.py)

### Repository Documentation
- [Main README](../../../../README.md)
- [Depth Pipeline Guide](../../../../docs/depth_pipeline/DEPTH_PIPELINE_README.md)
- [Architecture](../../../../docs/ARCHITECTURE.md)
- [Performance Guide](../../../../docs/PERFORMANCE_OPTIMIZATION.md)

---

**Template Collection Statistics**:
- **Total Templates**: 6
- **Total Size**: 116KB
- **Code Examples**: 82+
- **Coverage**: Feature dev, debugging, config, testing, docs, performance
- **Version**: 1.0.0
- **Last Updated**: 2025-11-06

**Maintained by**: Transformation Portal RAG System
