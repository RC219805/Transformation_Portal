# Custom Agent Quick Reference

## Using the Agent

```
@transformation-portal-specialist [your request]
```

## Common Use Cases

### 🎨 Pipeline Development
```
@transformation-portal-specialist Add a depth-based atmospheric haze effect 
to the ArchitecturalDepthPipeline that increases with distance
```

### ⚡ Performance Optimization
```
@transformation-portal-specialist The batch processor uses 18GB RAM for 4K 
images. Optimize memory usage while maintaining quality
```

### 🐛 Troubleshooting
```
@transformation-portal-specialist Getting "CUDA out of memory" when 
processing more than 5 images. What's the best solution?
```

### 🎬 Video Processing
```
@transformation-portal-specialist Create a new video grading preset for 
beachfront properties with warm sunset tones
```

### 🧪 Testing
```
@transformation-portal-specialist Write comprehensive tests for the new 
zone-based tone mapping feature including edge cases
```

### 🎯 Material Response
```
@transformation-portal-specialist Implement a new material detection algorithm 
for identifying glass surfaces in architectural renders
```

### 🔧 FFmpeg Workflows
```
@transformation-portal-specialist Build an FFmpeg filter graph for HDR 
(PQ) to SDR conversion with proper tone mapping
```

## Agent Expertise Areas

| Area | What It Knows |
|------|---------------|
| **Depth Pipeline** | Depth Anything V2, CoreML optimization, zone-based processing |
| **Lux Render** | SDXL, ControlNet, Real-ESRGAN, AI enhancement workflows |
| **Material Response** | Surface detection, physics-based enhancement, material-specific processing |
| **Video Grading** | FFmpeg filter graphs, HDR/SDR, LUT application, tone mapping |
| **Performance** | Profiling, caching, batch optimization, GPU/CoreML acceleration |
| **Testing** | Pytest, hypothesis, mocking strategies, CI/CD optimization |
| **Color Science** | ACES ODT, LUTs, color spaces, metadata preservation |

## Quick Tips

### ✅ DO:
- Be specific about which pipeline or component
- Provide error messages and context
- Ask for complete solutions (code + tests + docs)
- Request performance benchmarks
- Ask follow-up questions to refine

### ❌ DON'T:
- Ask vague questions without context
- Omit error messages or logs
- Skip testing requirements
- Ignore performance considerations

## Example Workflows

### Adding a New Feature
1. "Design architecture for [feature]"
2. "Implement [component] with tests"
3. "Profile and optimize [code]"
4. "Review this implementation"
5. "Document [feature] with examples"

### Debugging an Issue
1. "I'm getting [error]. What's the cause?"
2. "Show diagnostic commands to investigate"
3. "Implement the fix with proper error handling"
4. "Add tests to prevent this in the future"

### Optimizing Performance
1. "Profile [pipeline] and identify bottlenecks"
2. "Implement the top 3 optimizations"
3. "Create benchmark tests to verify improvement"
4. "Update performance documentation"

## More Information

- **Full Guide**: [docs/CUSTOM_AGENT_GUIDE.md](CUSTOM_AGENT_GUIDE.md)
- **Agent README**: [.github/agents/README.md](../.github/agents/README.md)
- **Implementation Summary**: [CUSTOM_AGENT_SUMMARY.md](../CUSTOM_AGENT_SUMMARY.md)

---

**Quick Start**: Just type `@transformation-portal-specialist` followed by what you need help with!
