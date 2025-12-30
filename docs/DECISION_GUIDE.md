# Decision Guide - Transformation Portal

**Quick navigation to the right tool for your use case**

---

## 🎯 Primary Decision Tree

### "I want to process architectural renders for production"

**→ Use `lux-depth-v2` (Golden Path)**

```bash
pip install -e .
lux-depth-v2 --input-dir renders/ --output-dir output/ --preset interior_luxury
```

**Why**: Production-validated, security-hardened, 127-400 images/hour throughput

**Documentation**: [docs/PHASE2_USER_GUIDE.md](PHASE2_USER_GUIDE.md)

---

### "I need a web API for batch processing"

**→ Use `lux-depth-v2-service` (Docker deployment)**

```bash
docker-compose up -d lux-depth-v2-service
curl -X POST http://localhost:8088/v2/process -F "image=@input.jpg" -F "preset=interior_luxury"
```

**Why**: FastAPI service with rate limiting, health checks, Prometheus metrics

**Documentation**: [docs/PHASE2_DEPLOYMENT_GUIDE.md](PHASE2_DEPLOYMENT_GUIDE.md)

---

### "I need GPU acceleration for high throughput"

**→ Use `lux-depth-v2-gpu` (CUDA deployment)**

```bash
docker-compose up -d lux-depth-v2-gpu
# 3-5x faster processing
```

**Requires**: NVIDIA GPU, CUDA-capable Docker runtime

**Documentation**: [lux_depth_v2/README.md](../lux_depth_v2/README.md)

---

### "I need edge refinement for architectural details"

**→ Enable edge refinement in lux-depth-v2**

```bash
lux-depth-v2 --input-dir renders/ --output-dir output/ --edge-refinement --preset interior_luxury
```

**When to use**: High-resolution architectural renders where fine structural edges matter

**Documentation**: Coming soon - [Edge Refinement Validation](../lux_depth_v2/docs/EDGE_REFINEMENT_VALIDATION.md)

---

### "I need custom material enhancement profiles"

**→ Use Materials V3 system (Advanced)**

```bash
python -m lux_depth_v2.materials_v3 --input input.jpg --output output.jpg --material wood --strength 0.8
```

**Why**: Physics-based surface enhancement with custom profiles

**Documentation**: [docs/MATERIALS_V2_GUIDE.md](MATERIALS_V2_GUIDE.md)

---

### "I need to process video (not images)"

**→ Use `luxury_video_master_grader.py` (Separate domain)**

```bash
python luxury_video_master_grader.py --input video.mp4 --output graded.mov --preset signature_estate
```

**Why**: FFmpeg-based video processing with LUT grading and HDR support

**Documentation**: See README section on video processing

---

### "I need context-aware rendering with RAG intelligence"

**→ Use Context-Aware Rendering system (Advanced)**

```bash
python -m src.transformation_portal.context_aware_rendering --input renders/ --context luxury_penthouse
```

**Why**: Document-driven architectural intelligence for scene-aware processing

**Documentation**: [docs/CONTEXT_SYSTEM_COMPLETE.md](CONTEXT_SYSTEM_COMPLETE.md)

---

### "I need high-throughput batch processing (1000+ images)"

**→ Use Async Pipeline infrastructure (Performance layer)**

```bash
python examples/async_pipeline_batch.py --input-dir large_batch/ --workers 8
```

**Why**: 3-5x throughput improvement via async queue orchestration

**Documentation**: [docs/pipeline/async_pipeline_architecture.md](pipeline/async_pipeline_architecture.md)

---

### "I want to train custom neural network models"

**→ Use Training infrastructure (Research)**

⚠️ **Advanced Feature** - Not part of default production workflow

```bash
./scripts/train_with_750picacho.sh
# Requires: GPU, 10GB+ disk, 2-3 hours
```

**Why**: Custom model adaptation for specific property types or datasets

**Documentation**: [docs/training/TRAINING_GUIDE.md](training/TRAINING_GUIDE.md)

---

## 🔍 Feature Comparison Matrix

| Feature | lux-depth-v2 (Golden Path) | Async Pipeline | Context-Aware | Training |
|---------|---------------------------|----------------|---------------|----------|
| **Production Ready** | ✅ Yes | ✅ Yes | 🟡 Advanced | ❌ Research |
| **Security Hardened** | ✅ CVE mitigated | ✅ Yes | ⚠️ Review needed | N/A |
| **GPU Accelerated** | ✅ CUDA | ✅ CUDA/MPS | ✅ CUDA | ✅ Required |
| **Docker Deployment** | ✅ Multi-stage | 🟡 Manual | 🟡 Manual | ❌ Local only |
| **Monitoring** | ✅ Prometheus | 🟡 Custom | ❌ None | ❌ None |
| **Throughput** | 127-400 img/hr | 400-1200 img/hr | 100-200 img/hr | N/A |
| **Learning Curve** | ⭐ Easy | ⭐⭐ Moderate | ⭐⭐⭐ Advanced | ⭐⭐⭐⭐ Expert |

---

## 🚦 When NOT to Use

### Don't use lux-depth-v2 if:
- ❌ You need real-time video processing → Use `luxury_video_master_grader.py`
- ❌ You need <50ms latency → Consider optimized inference pipeline
- ❌ You need custom neural architectures → Use training infrastructure

### Don't use Async Pipeline if:
- ❌ You're processing <100 images → Overhead not worth it, use lux-depth-v2
- ❌ You need simple deployment → Use Docker stack instead
- ❌ You need guaranteed processing order → Use sequential pipeline

### Don't use Training infrastructure if:
- ❌ You just want to process images → Use pre-trained models in lux-depth-v2
- ❌ You don't have GPU → Training will take 12-18 hours on CPU
- ❌ You need results today → Training takes 2-3 hours minimum

---

## 🎯 Quick Decision Flowchart

```
Start
  │
  ├─ Processing images? ───────────────────────┐
  │                                             │
  ├─ Processing video? ─────────────────────┐  │
  │                                          │  │
  └─ Training models? ───────────────────┐  │  │
                                         │  │  │
                                         │  │  └─→ <100 images? ─────┐
                                         │  │                        │
                                         │  │     ≥100 images? ──────┼─→ lux-depth-v2 (Golden Path)
                                         │  │                        │
                                         │  │                        └─→ ≥1000 images? → Async Pipeline
                                         │  │
                                         │  └─→ luxury_video_master_grader.py
                                         │
                                         └─→ docs/training/TRAINING_GUIDE.md
```

---

## 📚 Additional Resources

**Golden Path Documentation**: [docs/GOLDEN_PATH_INDEX.md](GOLDEN_PATH_INDEX.md)
**Feature Freeze Policy**: [docs/FEATURE_FREEZE_POLICY.md](FEATURE_FREEZE_POLICY.md)
**Strategic Assessment**: [docs/STRATEGIC_ARCHITECTURE_ASSESSMENT_2025-12-20.md](STRATEGIC_ARCHITECTURE_ASSESSMENT_2025-12-20.md)

---

**Last Updated**: December 20, 2025
**Maintained by**: Transformation Portal Architect
