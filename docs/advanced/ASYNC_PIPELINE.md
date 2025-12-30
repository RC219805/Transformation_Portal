# Async/Streaming Pipeline Architecture

**⚠️ ADVANCED FEATURE** - Use only if Golden Path doesn't meet your needs.

---

## Quick Decision

**Use Async Pipeline if**:
- Processing 1000+ images
- Need 3-5x throughput improvement
- Have infrastructure to manage streaming
- Willing to handle complexity

**Use Golden Path if**:
- Processing < 1000 images
- Standard batch processing is sufficient
- Want simplicity and predictability

---

## Overview

The async/streaming pipeline provides high-throughput image processing through:
- **Async I/O**: Non-blocking file operations
- **Pipeline Parallelism**: Concurrent processing stages
- **Streaming**: Memory-efficient large batch handling
- **Resource Management**: Smart GPU/CPU allocation

---

## Performance Characteristics

| Metric | Synchronous (Golden Path) | Async Pipeline |
|--------|---------------------------|----------------|
| Throughput | 127-400 images/hour | 400-1200 images/hour |
| Latency | Low (immediate start) | Higher (batching overhead) |
| Memory | Low (sequential) | Medium (buffering) |
| Complexity | Low | High |

**Trade-off**: Higher throughput but more complex failure modes.

---

## Architecture

```
Input Queue → Loader → Depth Estimator → Material Processor → Grader → Output Queue
     ↓           ↓            ↓                  ↓              ↓           ↓
  (async)    (async)      (GPU pool)        (GPU pool)     (async)    (async)
```

**Key components**:
- `src/transformation_portal/streaming/` - Core implementation
- `src/transformation_portal/streaming/pipeline.py` - Pipeline orchestrator
- `src/transformation_portal/streaming/queue.py` - Async queue management
- `src/transformation_portal/streaming/worker.py` - Worker pool

---

## Usage

### Basic Usage

```bash
# Use async mode
lux-depth-v2 \
  --input-dir large_batch/ \
  --output-dir processed/ \
  --preset interior_luxury \
  --async \
  --workers 4
```

### Advanced Configuration

```python
from transformation_portal.streaming import AsyncPipeline

pipeline = AsyncPipeline(
    workers=4,
    batch_size=16,
    queue_size=64,
    gpu_pool_size=2
)

await pipeline.process_directory(
    input_dir="renders/",
    output_dir="processed/",
    preset="interior_luxury"
)
```

---

## When It Goes Wrong

**Symptom**: OOM (Out of Memory) errors
**Fix**: Reduce `batch_size` and `queue_size`

**Symptom**: Low throughput (worse than sync)
**Fix**: Increase `workers`, check GPU utilization

**Symptom**: Inconsistent outputs
**Fix**: Check for race conditions, use synchronous mode

**Symptom**: Hung pipeline (no progress)
**Fix**: Enable debug logging, check for deadlocks

---

## Migration from Golden Path

1. **Validate with subset** (100 images)
2. **Compare outputs** (quality check)
3. **Benchmark throughput** (async vs sync)
4. **Monitor resources** (memory, GPU)
5. **Gradual rollout** (increase batch size)

**Rollback plan**: Golden Path is always available.

---

## Maintenance

**Stability**: ✅ Stable (no breaking changes expected)
**Support**: Community-supported (not feature-frozen like Golden Path)
**Testing**: Moderate coverage (integration tests required)

---

## Related Documentation

- **[Golden Path](../../QUICKSTART.md)** - Primary workflow
- **[Advanced README](README.md)** - Advanced workflows overview
- **[Architecture](../architecture/)** - System design

---

## Full Technical Details

For implementation details, see:
- `src/transformation_portal/streaming/README.md`
- `docs/pipeline/async_pipeline_architecture.md` (if exists)

---

*Remember: Complexity is a liability. Only use async pipeline if you've exhausted simpler options.*
