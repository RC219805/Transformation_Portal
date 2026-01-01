# Workflow Selector Guide

**Purpose**: Help users choose the right workflow for their specific use case. Prevents accidental use of experimental features in production.

**Last Updated**: 2026-01-01

---

## Quick Decision Tree

```
START
  │
  ├─ Processing images (not video)?
  │  │
  │  ├─ YES → How many images?
  │  │  │
  │  │  ├─ < 100 images → Golden Path (lux_depth_v2)
  │  │  ├─ 100-1000 images → Golden Path (lux_depth_v2) with batch mode
  │  │  └─ > 1000 images → Consider Async Pipeline (3-5x faster)
  │  │
  │  └─ NO → Video Processing Workflow
  │
  ├─ Need document-driven intelligence?
  │  │
  │  └─ YES → Context-Aware Rendering
  │
  ├─ Research or experimentation?
  │  │
  │  └─ YES → lux_depth_v3 (with --experimental-ok flag)
  │
  └─ DEFAULT → Golden Path (lux_depth_v2)
```

---

## The Golden Path (Recommended for 95% of Use Cases)

**Use When:**
- Processing architectural renders or images
- Need reliable, predictable behavior
- Security-hardened processing required
- Deploying to production
- Client deliverables

**Command:**
```bash
lux-depth-v2 --input-dir renders/ --output-dir output/ --preset interior_luxury
```

**Presets:**
- `--list-presets` - See all available presets
- `--list-stable` - See production-ready presets only
- `--describe-preset <name>` - Detailed preset information

**Quality Tiers:**
- `standard` (⚡) - Fast, 200-300 img/hr
- `max` (⭐) - Balanced quality, 100-200 img/hr
- `apex` (💎) - Maximum quality, 50-100 img/hr

**Documentation:**
- [QUICKSTART.md](../QUICKSTART.md) - Get started in 2 minutes
- [Phase 2 User Guide](PHASE2_USER_GUIDE.md) - Complete walkthrough
- [Quick Reference](QUICK_REFERENCE_PHASE2.md) - CLI cheat sheet

---

## Advanced Workflows

### When to Use Advanced Workflows

**⚠️ Important**: These are NOT replacements for the Golden Path. Use them only when you have specific needs that the Golden Path doesn't address.

### 1. Async Pipeline (High Throughput)

**Use When:**
- Processing 1000+ images
- Need 3-5x throughput improvement
- Have sufficient system resources

**Don't Use If:**
- &lt;100 images (overhead not worth it)
- Constrained memory (&lt;16GB RAM)

**Command:**
```bash
# Via Golden Path with parallel workers
lux-depth-v2 --input-dir renders/ --output-dir output/ \
  --parallel-workers 4 --preset interior_luxury
```

**Documentation:**
- [Async Pipeline Architecture](advanced/ASYNC_PIPELINE.md)

---

### 2. Context-Aware Rendering

**Use When:**
- Need document-driven architectural intelligence
- Scene-aware processing based on project context
- Multi-room project with context preservation

**Don't Use If:**
- Standard preset-based processing is sufficient
- No project documentation available

**Command:**
```bash
# Requires project context file
lux-depth-v2 --input-dir renders/ --output-dir output/ \
  --context-file project.yaml --preset interior_luxury
```

**Documentation:**
- [Context-Aware Rendering](advanced/CONTEXT_AWARE_RENDERING.md)
- [Context System Guide](CONTEXT_SYSTEM_COMPLETE.md)

---

### 3. Video Processing

**Use When:**
- Processing video files (not images)
- Need ProRes 422 HQ output
- HDR tone mapping required

**Don't Use If:**
- Processing still images (use Golden Path instead)

**Command:**
```bash
python luxury_video_master_grader.py \
  --input video.mp4 \
  --output output.mov \
  --preset signature_estate
```

**Documentation:**
- [Video Processing Guide](advanced/VIDEO_PROCESSING.md)
- See README section on video processing

---

### 4. Material Response (Custom Enhancement)

**Use When:**
- Need specialized material enhancement
- Custom surface-aware rendering
- Physics-based material processing

**Don't Use If:**
- Standard presets provide sufficient material handling

**Command:**
```bash
# Material Response is built into Golden Path
lux-depth-v2 --input-dir renders/ --output-dir output/ \
  --materials-v2 --preset interior_luxury_apex_quality
```

**Documentation:**
- [Material Response Guide](advanced/MATERIAL_RESPONSE.md)

---

## Experimental Workflows (⚠️ NOT Production-Ready)

### lux_depth_v3 + DA3

**Use When:**
- Research on custom depth models
- Proof-of-concept for new algorithms
- Evaluation of DA3 integration

**Don't Use If:**
- Processing for client deliverables
- Need stable, predictable output
- Production deployment

**Command:**
```bash
# Requires explicit experimental flag
lux-depth-v3 --input image.jpg --output-dir output/ \
  --experimental-ok --depth-backend da3
```

**Documentation:**
- [DA3 Integration Guide](DA3_INTEGRATION.md)
- [lux_depth_v3 README](../lux_depth_v3/README.md)

**⚠️ Warning**: Output quality and performance are NOT guaranteed. Use only for research and testing.

---

## Preset Selection Guide

### By Use Case

| Use Case | Recommended Preset | Quality Tier |
|----------|-------------------|--------------|
| Quick preview | `ci_baseline` | Standard ⚡ |
| Client preview | `production_standard` | Standard ⚡ |
| Interior renders | `interior_luxury` | Max ⭐ |
| Exterior renders | `exterior_showcase` | Max ⭐ |
| Hero shots | `interior_luxury_apex_quality` | Apex 💎 |
| Portfolio work | `interior_luxury_max_quality` | Max ⭐ |
| Archival | `archival_quality` | Apex 💎 |
| Pool/water | `exterior_pool_apex_quality` | Apex 💎 |

### By Quality Requirements

**Standard Quality (⚡):**
- Throughput: 200-400 img/hr
- Memory: 2-4 GB
- Use for: Previews, iterations, testing

**Max Quality (⭐):**
- Throughput: 100-200 img/hr
- Memory: 3-6 GB
- Use for: Client deliverables, portfolio

**Apex Quality (💎):**
- Throughput: 50-100 img/hr
- Memory: 6-10 GB
- Use for: Hero shots, awards, archival

---

## Troubleshooting by Symptom

### Slow Throughput

**Symptoms:**
- Processing much slower than expected
- System feels sluggish

**Solutions:**
1. Check quality tier - use lower tier for faster processing
2. Reduce `--upscale` factor (4→2)
3. Enable Phase 2 optimizations: `--phase2-optimizations`
4. Use CPU if GPU is unavailable: `--device cpu`
5. For large batches, use parallel workers: `--parallel-workers 4`

**See Also:**
- [Performance Optimization](PERFORMANCE_OPTIMIZATION.md)

---

### Artifacts Look Over-Sharpened

**Symptoms:**
- Harsh edges
- Over-enhanced details
- Unnatural look

**Solutions:**
1. Reduce clarity in preset (use lower tier)
2. Try different preset: `architectural` instead of `interior_luxury`
3. Disable materials v2: remove `--materials-v2` flag
4. Use `archival_quality` for minimal enhancement

**See Also:**
- Preset descriptions: `--describe-preset <name>`

---

### Segmentation Errors

**Symptoms:**
- Material detection failures
- Wrong material classifications

**Solutions:**
1. Check segmentation backend: `--seg-backend auto`
2. Adjust confidence threshold: `--confidence-threshold 0.5`
3. Try heuristic backend: `--seg-backend heuristic`
4. Disable materials v2 if not needed

**See Also:**
- [Material Segmentation Guide](advanced/MATERIAL_RESPONSE.md)

---

### Service Returns 5xx

**Symptoms:**
- HTTP 500 errors from service
- Processing fails in service mode

**Solutions:**
1. Check service logs for error details
2. Verify input file format is supported
3. Check file size limits (default 100MB)
4. Ensure sufficient memory for service
5. Verify models are loaded: `GET /ready`

**See Also:**
- [Service Mode Documentation](../lux_depth_v2/README.md)
- [Security Guide](../lux_depth_v2/SECURITY.md)

---

### Out of Memory

**Symptoms:**
- OOM errors
- System freezing
- Crashes during processing

**Solutions:**
1. Reduce upscale factor: `--upscale 2`
2. Use lower quality tier
3. Process smaller batches
4. Use streaming upscale: `--streaming-upscale`
5. Set memory budget: `--memory-budget 4.0` (GB)

**See Also:**
- [Resource Requirements](../README.md#performance-characteristics)

---

## Service Deployment Guide

### CPU Deployment (Standard)

```bash
# Start service (CPU)
docker-compose up -d lux-depth-v2-service

# Verify health
curl http://localhost:8088/health

# Verify readiness (models loaded)
curl http://localhost:8088/ready

# Process an image
curl -X POST http://localhost:8088/v2/process \
  -F "image=@input.jpg" \
  -F "preset=interior_luxury"
```

### GPU Deployment (Accelerated)

```bash
# Start service (GPU)
docker-compose up -d lux-depth-v2-gpu

# Verify GPU is available
docker exec lux-depth-v2-gpu python -c "import torch; print(torch.cuda.is_available())"

# Process with GPU acceleration (3-5x faster)
curl -X POST http://localhost:8089/v2/process \
  -F "image=@input.jpg" \
  -F "preset=exterior_showcase"
```

**See Also:**
- [Production Deployment Guide](deployment/DEPLOYMENT_GUIDE.md)

---

## Quick Reference Card

### Most Common Commands

```bash
# List available presets
lux-depth-v2 --list-presets

# Get preset details
lux-depth-v2 --describe-preset interior_luxury

# Process single image
lux-depth-v2 --input image.jpg --output-dir output/ --preset interior_luxury

# Process directory (batch)
lux-depth-v2 --input-dir renders/ --output-dir output/ --preset interior_luxury

# High-quality batch with parallel processing
lux-depth-v2 --input-dir renders/ --output-dir output/ \
  --preset interior_luxury_apex_quality \
  --parallel-workers 4 \
  --phase2-optimizations

# Service mode
lux-depth-v2 --service --output-dir /data/output --port 8088
```

---

## When in Doubt

1. **Start with Golden Path**: `lux-depth-v2 --input-dir renders/ --output-dir output/ --preset interior_luxury`
2. **Check preset options**: `lux-depth-v2 --list-presets`
3. **Read preset details**: `lux-depth-v2 --describe-preset <name>`
4. **Refer to QUICKSTART**: [QUICKSTART.md](../QUICKSTART.md)
5. **Ask for help**: Open an issue on GitHub

---

## Additional Resources

- [README.md](../README.md) - Full feature overview
- [QUICKSTART.md](../QUICKSTART.md) - Get started in 2 minutes
- [DECISION_GUIDE.md](DECISION_GUIDE.md) - Workflow decision tree
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [AGENT_REPO_MAP.md](AGENT_REPO_MAP.md) - Repository structure
- [Phase 2 User Guide](PHASE2_USER_GUIDE.md) - Advanced features

---

**End of Workflow Selector Guide**
