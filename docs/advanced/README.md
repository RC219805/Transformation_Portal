# Advanced Workflows

**⚠️ This directory contains advanced features for power users.**

These workflows are **NOT part of the Golden Path** and require deeper understanding of the system.

---

## When to Use Advanced Workflows

Use these workflows **only if**:
- Golden Path doesn't meet your specific requirements
- You understand the trade-offs and complexity
- You're willing to handle edge cases yourself
- You need capabilities beyond standard presets

**Otherwise**: Use the [Golden Path](../../QUICKSTART.md) (`lux_depth_v2`).

---

## Available Advanced Workflows

### 🔄 Async/Streaming Pipeline
**Use when**: Processing 1000+ images, need 3-5x throughput improvement  
**Documentation**: [ASYNC_PIPELINE.md](ASYNC_PIPELINE.md)  
**Location**: `src/transformation_portal/streaming/`  
**Status**: Stable, production-ready

### 🧠 Context-Aware Rendering
**Use when**: Need document-driven architectural intelligence  
**Documentation**: [CONTEXT_AWARE_RENDERING.md](CONTEXT_AWARE_RENDERING.md)  
**Location**: `src/transformation_portal/context_aware_rendering/`  
**Status**: Stable, specialized use case

### 🎨 Material Response (Advanced)
**Use when**: Custom material enhancement beyond 8 standard types  
**Documentation**: [MATERIAL_RESPONSE.md](MATERIAL_RESPONSE.md)  
**Location**: `material_response.py`  
**Status**: Stable, expert-level

### 🎬 Video Processing
**Use when**: Processing video files (not images)  
**Documentation**: [VIDEO_PROCESSING.md](VIDEO_PROCESSING.md)  
**Tool**: `luxury_video_master_grader.py`  
**Status**: Stable, separate domain

---

## Stability Guarantees

Advanced workflows have **different stability guarantees** than the Golden Path:

| Workflow | Stability | Breaking Changes | Support Level |
|----------|-----------|------------------|---------------|
| Golden Path (`lux_depth_v2`) | ✅ Feature-frozen | No | Full |
| Async Pipeline | ✅ Stable | Rare | Community |
| Context-Aware | ✅ Stable | Rare | Community |
| Material Response | ✅ Stable | Rare | Community |
| Video Processing | ✅ Stable | Rare | Community |

---

## Migration Path

If you're using an advanced workflow and want to simplify:

1. **Evaluate if Golden Path meets your needs** (it probably does)
2. **Test with Golden Path** on representative samples
3. **Compare quality/performance** against your current workflow
4. **Migrate incrementally** if suitable

**Remember**: Complexity is a liability. Use the simplest tool that works.

---

## Contributing to Advanced Features

Advanced features follow **standard development practices** (not feature-frozen like Golden Path):

✅ **Allowed**:
- New features (with justification)
- Breaking changes (with migration guide)
- Experimental integrations (clearly labeled)
- Performance optimizations

⚠️ **Requirements**:
- Comprehensive tests
- Documentation
- Migration guides for breaking changes
- Clear use case justification

---

## Related Documentation

- **[Golden Path Quick Start](../../QUICKSTART.md)** - Primary workflow
- **[Research Features](../research/)** - Experimental/unstable features
- **[Architecture](../architecture/)** - System design
- **[CONTRIBUTING.md](../../CONTRIBUTING.md)** - Development guidelines

---

*If you're not sure whether you need an advanced workflow, you probably don't. Start with the [Golden Path](../../QUICKSTART.md).*
