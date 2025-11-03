# Transformation Portal Custom Agent Guide

## Overview

The **Transformation Portal Specialist** is a custom GitHub Copilot agent specifically designed for this repository. It has deep expertise in luxury real estate rendering, architectural visualization, and professional image/video processing workflows.

## What Makes This Agent Special?

Unlike general-purpose AI assistants, the Transformation Portal Specialist has:

### 1. **Domain-Specific Knowledge**
- **Image Processing Pipelines**: Depth-aware processing, Material Response, batch workflows
- **AI/ML Integration**: Stable Diffusion XL, ControlNet, Depth Anything V2, Real-ESRGAN
- **Professional Color Science**: LUTs, HDR tone mapping, ACES ODT, color space transforms
- **Hardware Optimization**: Apple Silicon (CoreML), CUDA, MPS acceleration strategies

### 2. **Repository Architecture Understanding**
The agent knows the modular structure:
```
depth_pipeline/              # Depth Anything V2 integration
lux_render_pipeline.py      # AI-powered refinement
material_response.py        # Surface enhancement
luxury_video_master_grader.py  # Professional color grading
config/                     # YAML presets
```

### 3. **Best Practices & Standards**
- **Code Quality**: Pytest, hypothesis, mocking strategies, linting standards
- **Performance**: Lazy loading, LRU caching, batch optimization, profiling
- **Metadata**: IPTC/XMP/GPS preservation, 16-bit precision, color space metadata
- **FFmpeg**: Filter graphs, HDR handling, metadata preservation

### 4. **Troubleshooting Expertise**
- Import errors and dependency issues
- FFmpeg filter graph problems
- GPU/MPS/CoreML acceleration issues
- Memory optimization for large batches
- HDR video processing challenges

## When to Use the Custom Agent

### Perfect For:
✅ **Pipeline Development**
- Implementing new depth-aware effects
- Adding Material Response enhancements
- Creating video processing workflows
- Optimizing batch processing performance

✅ **AI/ML Integration**
- Working with diffusion models
- Implementing ControlNet features
- Optimizing model inference speed
- Managing GPU/CoreML acceleration

✅ **Professional Color Grading**
- Creating LUT-based presets
- Implementing tone mapping operators
- HDR workflow development
- Color space transformations

✅ **Performance Optimization**
- Profiling memory usage
- Implementing caching strategies
- Batch processing optimization
- Hardware-specific acceleration

✅ **Testing & Quality**
- Writing comprehensive tests
- Mocking heavy dependencies
- Property-based testing
- CI/CD optimization

### Use General Copilot For:
- Basic Python syntax questions
- Generic file operations
- Simple utility functions
- Non-domain-specific tasks

## How to Use the Agent

### Basic Usage Pattern

In GitHub Copilot Chat, prefix your prompt with the agent name:

```
@transformation-portal-specialist [your request]
```

### Example Requests

#### 1. **Implementing Features**
```
@transformation-portal-specialist Add a depth-based atmospheric haze effect 
to the ArchitecturalDepthPipeline that increases with distance from camera
```

**What you'll get**:
- Context about the pipeline architecture
- Complete implementation with proper integration points
- Configuration YAML updates
- Test cases
- Performance considerations

#### 2. **Optimizing Performance**
```
@transformation-portal-specialist The batch processor is using 18GB RAM for 
4K images. How can I reduce memory usage while maintaining quality?
```

**What you'll get**:
- Analysis of current memory patterns
- Specific optimization strategies
- Code examples with profiling
- Trade-off discussions
- Testing approach

#### 3. **Troubleshooting**
```
@transformation-portal-specialist Getting "RuntimeError: CUDA out of memory" 
when processing more than 5 images in batch
```

**What you'll get**:
- Root cause analysis
- Multiple solution approaches
- Code fixes with explanations
- Prevention strategies
- Alternative approaches (CPU/MPS fallback)

#### 4. **Creating Presets**
```
@transformation-portal-specialist Create a new video grading preset for 
beachfront properties with warm sunset tones
```

**What you'll get**:
- Preset configuration code
- LUT selection reasoning
- Parameter explanations
- Integration steps
- Usage examples

#### 5. **Testing Complex Features**
```
@transformation-portal-specialist Write comprehensive tests for the new 
zone-based tone mapping feature including edge cases
```

**What you'll get**:
- Unit tests with fixtures
- Property-based tests using hypothesis
- Mocking strategies for ML models
- Edge case coverage
- Performance benchmarks

## Real-World Workflow Examples

### Example 1: Adding New Pipeline Feature

**Step 1**: Ask for implementation
```
@transformation-portal-specialist I want to add depth-based bokeh simulation 
to blur background more than foreground in architectural renders
```

**Step 2**: Review the response
- Implementation in depth_pipeline/processors/
- Configuration additions
- Integration with main pipeline
- Test cases

**Step 3**: Iterate with follow-ups
```
@transformation-portal-specialist Can you add a parameter to control the 
transition zone between sharp and blurred areas?
```

**Step 4**: Ask for optimization
```
@transformation-portal-specialist Profile this bokeh implementation and 
optimize if it takes more than 50ms per image
```

### Example 2: Debugging Production Issue

**Step 1**: Describe the problem
```
@transformation-portal-specialist Video processing is failing with 
"Invalid pixel format" when processing HDR content from iPhone 15 Pro Max
```

**Step 2**: Get diagnostic commands
```
@transformation-portal-specialist Show me how to inspect the video metadata 
with ffprobe to diagnose this issue
```

**Step 3**: Implement the fix
```
@transformation-portal-specialist Based on the metadata showing HLG color 
transfer, what's the correct filter graph to handle this?
```

**Step 4**: Add test coverage
```
@transformation-portal-specialist Add tests to prevent this HDR detection 
issue in the future
```

### Example 3: Performance Optimization

**Step 1**: Request profiling
```
@transformation-portal-specialist Profile the Material Response batch 
processor and identify bottlenecks
```

**Step 2**: Implement optimizations
```
@transformation-portal-specialist Implement the top 3 optimizations you 
identified that don't sacrifice quality
```

**Step 3**: Verify improvements
```
@transformation-portal-specialist Create benchmark tests to measure the 
performance improvement
```

**Step 4**: Document changes
```
@transformation-portal-specialist Update the performance documentation with 
the new throughput numbers
```

## Agent Communication Style

The agent follows a structured response pattern:

1. **Context**: "I understand you're working with [component]..."
2. **Analysis**: "The issue is caused by / The approach would be..."
3. **Implementation**: [Code examples with explanations]
4. **Integration**: "Here's how to integrate this..."
5. **Testing**: "Test this with..."
6. **Performance**: "Expected throughput: X images/hour"
7. **Documentation**: "Update these docs..."

This ensures comprehensive, actionable responses.

## Best Practices for Agent Interaction

### DO:
✅ Be specific about which pipeline or component you're working with
✅ Provide error messages, logs, or code snippets for context
✅ Ask for complete solutions including tests and documentation
✅ Request performance benchmarks and optimization strategies
✅ Ask follow-up questions to refine the solution
✅ Mention hardware constraints (GPU, memory, CPU)

### DON'T:
❌ Ask extremely vague questions like "make it better"
❌ Omit important context (error messages, configurations)
❌ Request changes without considering existing code
❌ Ignore testing and documentation needs
❌ Skip performance considerations

### Example: Good vs. Bad Prompts

**❌ Bad Prompt:**
```
The depth thing isn't working
```

**✅ Good Prompt:**
```
@transformation-portal-specialist The ArchitecturalDepthPipeline is throwing 
"RuntimeError: Expected tensor for argument #1 'input' to have size 518x518" 
when processing 4K images. How should I handle variable image sizes?
```

**❌ Bad Prompt:**
```
Add tests
```

**✅ Good Prompt:**
```
@transformation-portal-specialist Write comprehensive tests for the new 
AtmosphericEffects processor, including unit tests for the fog calculation 
and integration tests with the full pipeline. Mock the depth model to avoid 
CI timeouts.
```

## Advanced Usage

### Chaining Agent Interactions

For complex tasks, break them into steps:

```
1. @transformation-portal-specialist Design the architecture for a new 
   real-time preview pipeline that shows depth effects before full processing

2. @transformation-portal-specialist Implement the core preview rendering 
   with 256px resolution for speed

3. @transformation-portal-specialist Add a WebSocket server to stream 
   previews to a browser interface

4. @transformation-portal-specialist Create tests for the preview pipeline 
   including latency measurements

5. @transformation-portal-specialist Write user documentation for the 
   preview feature
```

### Combining with Code Review

Use the agent to review code before committing:

```
@transformation-portal-specialist Review this implementation of zone-based 
tone mapping. Check for performance issues, correctness, and alignment 
with repository standards.

[paste code]
```

### Learning from the Agent

Use it as a teacher:

```
@transformation-portal-specialist Explain how the Material Response 
technology works, including the physics behind surface enhancement and 
how it differs from simple sharpening
```

## Agent Limitations

The agent acknowledges when:
- GPU resources aren't available for testing
- Changes might impact production workflows (suggests careful testing)
- Real-world profiling data would be beneficial
- FFmpeg version-specific features might vary
- Large ML models can't be tested in CI (suggests mocking)

It will guide you toward appropriate testing and validation strategies.

## Measuring Agent Effectiveness

Track these metrics to evaluate the agent:
- **Accuracy**: Does it provide correct, working code?
- **Completeness**: Does it include tests, docs, and error handling?
- **Context**: Does it understand repository patterns and standards?
- **Efficiency**: Does it save time compared to manual implementation?
- **Learning**: Does it help you understand the codebase better?

## Improving the Agent

The agent learns from the repository's evolving patterns. Update it when:
- New pipelines or major features are added
- Coding standards change
- New performance optimization patterns emerge
- Common issues or FAQs are identified
- Dependencies or tools are updated

To update: Edit `.github/agents/transformation-portal-specialist.md`

## Integration with Development Workflow

### Development Cycle with Agent

```
1. Design Phase
   └─ @agent: "Design architecture for [feature]"

2. Implementation Phase  
   └─ @agent: "Implement [component] with tests"

3. Optimization Phase
   └─ @agent: "Profile and optimize [code]"

4. Review Phase
   └─ @agent: "Review this implementation"

5. Documentation Phase
   └─ @agent: "Document [feature] with examples"
```

### CI/CD Integration

The agent understands CI/CD constraints:
- Tests must run in < 5 minutes
- Mock heavy dependencies (ML models, FFmpeg for unit tests)
- Python 3.10/3.11/3.12 compatibility
- Linting with flake8 and pylint
- Code coverage expectations

## Resources

- **Agent File**: `.github/agents/transformation-portal-specialist.md`
- **Agent README**: `.github/agents/README.md`
- **Repository Docs**: `docs/`
- **Copilot Instructions**: `.github/copilot-instructions.md`

## Support

If the agent provides incorrect or unhelpful responses:
1. Rephrase your question with more context
2. Break complex requests into smaller steps
3. Ask for alternative approaches
4. Check if you're using the right agent for the task
5. Report persistent issues for agent improvement

---

**Remember**: The Transformation Portal Specialist is designed to be your expert partner in building world-class image and video processing pipelines. Use it actively, iterate on responses, and provide feedback to make it even better!
