# ⚡ Transformation Portal - Quick Start

**Get processing images in under 5 minutes.**

---

## 🎯 The One Workflow You Need

Unless you have a **specific reason** to use something else, follow this:

```bash
# 1. Install (one time)
pip install -e .

# 2. Process images
lux-depth-v2 --input-dir renders/ --output-dir output/ --preset interior_luxury

# 3. Done.
```

**That's it.** Your images are processed with:
- ✅ Security-hardened pipeline (CVE-2024-27763 mitigated)
- ✅ Production-validated quality (1,348 tests passing)
- ✅ 16-bit precision maintained
- ✅ 127-400 images/hour throughput

---

## 📊 Common Scenarios

### Scenario 1: "I have architectural renders to process"

```bash
lux-depth-v2 \
  --input-dir my_renders/ \
  --output-dir processed/ \
  --preset interior_luxury
```

**Presets available**:
- `interior_luxury` - Indoor architectural scenes (default)
- `exterior_showcase` - Outdoor building photography
- `product_closeup` - Product/detail shots
- `balanced` - General-purpose processing

### Scenario 2: "I need a web API for batch processing"

```bash
# Start service (Docker)
docker-compose up -d lux-depth-v2-service

# Send images via HTTP
curl -X POST http://localhost:8088/process \
  -F "file=@render.jpg" \
  -F "preset=interior_luxury"
```

**Service features**:
- 🔒 Rate limiting (10-100 req/min)
- 🔒 Input validation (path traversal protection)
- 📊 Prometheus metrics
- 🏥 Health checks

### Scenario 3: "I need GPU acceleration"

```bash
# Same command, just add --device cuda
lux-depth-v2 \
  --input-dir renders/ \
  --output-dir output/ \
  --preset interior_luxury \
  --device cuda
```

**Performance gain**: 3-5x faster with CUDA support.

---

## 🚫 When NOT to Use the Golden Path

**Use advanced workflows if**:

❌ You need **custom material enhancement** beyond 8 standard types
→ See [docs/advanced/MATERIAL_RESPONSE.md](docs/advanced/MATERIAL_RESPONSE.md)

❌ You're processing **1000+ images** and need async pipeline
→ See [docs/advanced/ASYNC_PIPELINE.md](docs/advanced/ASYNC_PIPELINE.md)

❌ You need **document-driven architectural intelligence**
→ See [docs/advanced/CONTEXT_AWARE_RENDERING.md](docs/advanced/CONTEXT_AWARE_RENDERING.md)

❌ You're **training custom models** on your dataset
→ See [docs/research/TRAINING_GUIDE.md](docs/research/TRAINING_GUIDE.md)

❌ You're processing **video files** (not images)
→ Use `luxury_video_master_grader.py` instead

**Otherwise**: Use the Golden Path. It handles 95% of use cases.

---

## 📚 Next Steps

### If It Worked ✅

**Understand what just happened**:
- [docs/PHASE2_USER_GUIDE.md](docs/PHASE2_USER_GUIDE.md) - Complete walkthrough
- [docs/QUICK_REFERENCE_PHASE2.md](docs/QUICK_REFERENCE_PHASE2.md) - One-page CLI reference

**Optimize your workflow**:
- [docs/QUALITY_TIERS.md](docs/QUALITY_TIERS.md) - Preset selection guide
- [docs/PHASE2_PERFORMANCE.md](docs/PHASE2_PERFORMANCE.md) - Performance tuning

**Deploy to production**:
- [deployment/README.md](deployment/README.md) - Docker deployment guide
- [lux_depth_v2/SECURITY.md](lux_depth_v2/SECURITY.md) - Security best practices

### If It Didn't Work ❌

**Common issues**:

**"Command not found: lux-depth-v2"**
```bash
# Reinstall with entry points
pip install -e .
```

**"CUDA out of memory"**
```bash
# Use CPU instead
lux-depth-v2 --input-dir renders/ --output-dir output/ --device cpu
```

**"Permission denied"**
```bash
# Check file permissions
ls -la renders/
chmod 644 renders/*.jpg
```

**Still stuck?**
- Check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- Open GitHub issue with error logs

---

## 🎓 Philosophy

**Why so opinionated?**

This quick start is **deliberately minimal** because:

1. **Decision fatigue is real** - Too many options paralyze users
2. **95% of users need the same thing** - Depth processing + quality enhancement
3. **Defaults should be excellent** - Not configurable for the sake of configurability
4. **Advanced features should be invisible** - Until you need them

The Golden Path (`lux_depth_v2`) is:
- **Feature-frozen** (predictable, stable)
- **Production-validated** (127-400 images/hour)
- **Security-hardened** (CVE mitigated, input validated)
- **Well-tested** (1,348 tests passing)

Everything else is either:
- **Advanced** (for power users with specific needs)
- **Research** (experimental, not production-ready)
- **Deprecated** (legacy, being phased out)

---

## 🚀 Ready?

```bash
pip install -e .
lux-depth-v2 --input-dir renders/ --output-dir output/ --preset interior_luxury
```

**Done in 2 lines.** Now go process some images.

---

## 📖 Full Documentation

- **[README.md](README.md)** - Complete project overview
- **[docs/PHASE2_USER_GUIDE.md](docs/PHASE2_USER_GUIDE.md)** - Detailed user guide
- **[docs/advanced/](docs/advanced/)** - Advanced workflows
- **[docs/research/](docs/research/)** - Experimental features

---

*Last updated: December 23, 2025*
