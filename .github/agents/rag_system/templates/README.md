# RAG System Prompt Templates

This directory contains comprehensive prompt templates for the Transformation Portal RAG system. These templates provide structured workflows for common development tasks, ensuring consistency and completeness.

## Available Templates

### 1. Feature Implementation (`feature_implementation.md`)
**Use for**: Adding new features to pipelines, processors, or utilities

**Includes**:
- Requirements analysis framework
- Implementation plan structure
- Testing strategy (unit, integration, property-based)
- Documentation requirements
- Performance benchmarking approach
- Few-shot examples from repository

**When to use**:
- Adding depth-based effects to pipelines
- Implementing new material enhancement techniques
- Creating new LUT presets or color grading workflows
- Extending batch processing capabilities

---

### 2. Bug Triage (`bug_triage.md`)
**Use for**: Debugging errors, import issues, FFmpeg problems, runtime failures

**Includes**:
- Error classification system
- Root cause analysis framework
- Fix strategy comparison (multiple approaches)
- Regression test creation
- Common bug patterns and solutions

**When to use**:
- ImportError for missing dependencies
- FFmpeg command failures
- Out of memory errors
- Metadata preservation issues
- Platform-specific errors (CoreML, CUDA)

---

### 3. Pipeline Configuration (`pipeline_configuration.md`)
**Use for**: Creating YAML configs for depth pipeline, video grader, batch processors

**Includes**:
- Complete parameter reference tables
- Configuration validation checklist
- Performance vs quality tradeoffs
- Use case-specific presets (interior, exterior, aerial, product)
- Testing and validation workflow

**When to use**:
- Creating custom depth pipeline presets
- Adding video grading presets
- Configuring batch processing workflows
- Optimizing for specific content types

---

### 4. Testing (`testing.md`)
**Use for**: Writing comprehensive test suites (unit, integration, performance)

**Includes**:
- Unit test structure and patterns
- Integration test workflows
- Property-based testing with Hypothesis
- Performance benchmarking framework
- Mock-based testing for heavy dependencies
- CI test configuration

**When to use**:
- Testing new feature implementations
- Adding regression tests for bug fixes
- Benchmarking pipeline performance
- Testing metadata preservation
- Validating FFmpeg integration

---

### 5. Documentation (`documentation.md`)
**Use for**: Documenting features, APIs, configurations, and workflows

**Includes**:
- Feature documentation structure
- NumPy-style docstring format
- Usage guide templates
- Configuration reference format
- Troubleshooting guide structure
- API reference patterns

**When to use**:
- Documenting new pipeline features
- Writing user guides for workflows
- Creating API documentation
- Documenting configuration presets
- Writing troubleshooting guides

---

### 6. Performance Optimization (`performance_optimization.md`)
**Use for**: Profiling and optimizing processing pipelines

**Includes**:
- Profiling tools and techniques (cProfile, memory_profiler, PyTorch profiler)
- Optimization strategies (caching, batching, GPU acceleration, vectorization)
- Benchmarking framework
- Before/after comparison methodology
- Repository-specific optimizations

**When to use**:
- Optimizing depth estimation speed
- Reducing memory usage for large images
- Improving batch processing throughput
- Implementing GPU/CoreML acceleration
- Profiling video processing workflows

---

## Template Usage

### Basic Usage

Each template follows a structured format with:
1. **Overview** - Purpose and when to use
2. **Workflow Steps** - Sequential process to follow
3. **Code Examples** - Real patterns from the repository
4. **Validation Checklist** - Ensure completeness
5. **Few-Shot Examples** - Actual repository examples

### Integration with RAG System

Templates are designed to work with the RAG system's retrieval and reranking:

```python
from rag_system.templates import PromptTemplates
from rag_system.retriever import HybridRetriever
from rag_system.citation import CitationGenerator

# 1. Generate base template
template = PromptTemplates.feature_implementation(
    feature_description="Add depth-based atmospheric haze",
    context="For exterior architectural renders"
)

# 2. Retrieve relevant repository examples
retriever = HybridRetriever()
examples = retriever.retrieve(
    query="atmospheric effects depth processing",
    top_k=5
)

# 3. Add few-shot examples to template
template_with_examples = PromptTemplates.add_few_shot_examples(
    template, examples
)

# 4. Generate citations
citations = CitationGenerator.generate_citations(examples)
```

### Customization

Templates include placeholders in `{CURLY_BRACES}` for user-specific details:
- `{FEATURE_NAME}` - Name of the feature being implemented
- `{MODULE_NAME}` - Python module/file name
- `{CONFIG_NAME}` - Configuration file name
- `{USE_CASE}` - Intended use case or scenario

Replace these with actual values when using templates.

---

## Template Standards

### Code Examples
- ✅ Use actual patterns from the repository
- ✅ Include both "before" and "after" examples
- ✅ Show expected output or results
- ✅ Document performance characteristics

### Testing
- ✅ Include unit, integration, and property-based tests
- ✅ Cover edge cases and error conditions
- ✅ Use pytest markers (fast, slow, integration)
- ✅ Mock heavy dependencies for CI

### Documentation
- ✅ Follow NumPy docstring format
- ✅ Include usage examples with realistic parameters
- ✅ Document performance characteristics
- ✅ Link to related documentation

### Performance
- ✅ Profile before optimizing
- ✅ Document baseline and optimized performance
- ✅ Include memory usage measurements
- ✅ Test on representative data sizes

---

## Repository-Specific Patterns

### Depth Pipeline
```python
# Pattern: Zone-based processing with depth maps
depth_map = normalize_depth(raw_depth)
foreground_mask = depth_map < 0.3
background_mask = depth_map >= 0.7

result[foreground_mask] = process_foreground(image[foreground_mask])
result[background_mask] = process_background(image[background_mask])
```

### Metadata Preservation
```python
# Pattern: Preserve PIL Image.info dict
original_info = image.info.copy()
result = process_image(image)
result.info = original_info
```

### Optional Dependencies
```python
# Pattern: Graceful fallback for optional packages
try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
    warnings.warn("tifffile not available, using Pillow")
```

### LRU Caching
```python
# Pattern: Cache expensive computations
from functools import lru_cache

@lru_cache(maxsize=128)
def estimate_depth(image_hash: str) -> np.ndarray:
    image = load_from_hash(image_hash)
    return depth_model.estimate(image)
```

---

## Version Control

**Template Version**: 1.0  
**Last Updated**: 2025-11-06  
**Maintained By**: Transformation Portal RAG System

### Changelog

**v1.0.0** (2025-11-06)
- Initial template collection
- Feature implementation template
- Bug triage template
- Pipeline configuration template
- Testing template
- Documentation template
- Performance optimization template

---

## Contributing

When updating templates:

1. **Maintain structure** - Don't change placeholder format `{VARIABLE}`
2. **Update examples** - Use current repository patterns
3. **Test templates** - Verify placeholders are comprehensive
4. **Document changes** - Update this README and template version
5. **Review patterns** - Ensure alignment with repository standards

---

## Related Documentation

- [RAG System Architecture](../README.md)
- [Knowledge Engine](../knowledge_engine.py)
- [Citation Generator](../citation.py)
- [Prompt Templates (Python)](../templates.py)

---

## Quick Reference

| Template | Primary Use | Key Sections |
|----------|-------------|--------------|
| Feature Implementation | New pipeline features | Requirements, Implementation Plan, Testing, Few-Shot Examples |
| Bug Triage | Debugging errors | Error Classification, Root Cause, Fix Strategy, Regression Tests |
| Pipeline Configuration | YAML configs | Parameter Reference, Use Cases, Validation, Performance |
| Testing | Test suites | Unit Tests, Integration Tests, Benchmarks, CI Configuration |
| Documentation | Feature docs | API Reference, Usage Examples, Troubleshooting, Performance |
| Performance Optimization | Speed/memory tuning | Profiling, Optimization Strategies, Benchmarking, Validation |

---

**Need help?** Refer to the main [RAG System README](../README.md) or examine the [templates.py](../templates.py) Python implementation.
