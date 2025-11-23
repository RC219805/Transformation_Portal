# Training Decision Tree - Choose Your Path

**Transformation Portal Model Training**

---

## 🤔 Which Training Method Should I Use?

Follow this decision tree to choose the best training approach for your needs:

```
START HERE
    ├─ Do you have 750 Picacho BIM data in the repository?
    │   │
    │   ├─ YES ──→ Do you have GPU (CUDA or Apple Silicon MPS)?
    │   │           │
    │   │           ├─ YES ──→ Do you want BEST QUALITY (103-107/100)?
    │   │           │           │
    │   │           │           ├─ YES ──→ ✅ USE: train_with_750picacho.sh
    │   │           │           │          Time: 2.5-3.5 hours
    │   │           │           │          Quality: 103-107/100 ⭐ RECOMMENDED
    │   │           │           │
    │   │           │           └─ NO ──→ Do you need QUICK BASELINE (100-103/100)?
    │   │           │                      │
    │   │           │                      └─ YES ──→ ✅ USE: quickstart_training.sh
    │   │           │                                 Time: 2-3 hours
    │   │           │                                 Quality: 100-103/100
    │   │           │
    │   │           └─ NO (CPU only) ──→ Do you have 12-18 hours to wait?
    │   │                                 │
    │   │                                 ├─ YES ──→ ✅ USE: train_with_750picacho.sh
    │   │                                 │          Time: 12-18 hours (slow but best quality)
    │   │                                 │          Quality: 103-107/100
    │   │                                 │
    │   │                                 └─ NO ──→ ✅ USE: quickstart_training.sh
    │   │                                            Time: 8-12 hours (faster)
    │   │                                            Quality: 100-103/100
    │   │
    │   └─ NO ──→ Do you have GPU?
    │               │
    │               ├─ YES ──→ ✅ USE: quickstart_training.sh
    │               │          Time: 2-3 hours
    │               │          Quality: 100-103/100 (synthetic data)
    │               │
    │               └─ NO ──→ Do you want to wait 8-12 hours?
    │                          │
    │                          ├─ YES ──→ ✅ USE: quickstart_training.sh
    │                          │          Time: 8-12 hours
    │                          │          Quality: 100-103/100
    │                          │
    │                          └─ NO ──→ ✅ USE: quick_train_demo.py
    │                                     Time: 5-10 minutes
    │                                     Quality: Demo only (validation)
    │                                     ⚠️ NOT for production!
```

---

## 📊 Quick Comparison Table

| Scenario | Recommended Method | Command | Time (GPU) | Time (CPU) | Quality |
|----------|-------------------|---------|------------|------------|---------|
| **Best Quality + Have 750 Picacho data** | 750 Picacho Training | `./scripts/train_with_750picacho.sh` | 2.5-3.5h | 12-18h | 103-107/100 ⭐ |
| **Quick Baseline + No special data** | Synthetic Training | `./scripts/quickstart_training.sh` | 2-3h | 8-12h | 100-103/100 |
| **Custom Dataset** | Manual Training | `python src/enhancements/train_hyper_reality.py --data-dir my_data/` | Varies | Varies | Varies |
| **Just Testing Infrastructure** | Quick Demo | `python scripts/quick_train_demo.py` | 5-10min | 5-10min | Demo only |

---

## 🎯 By Use Case

### Production / Client Work
→ **Use:** `train_with_750picacho.sh`  
→ **Why:** Best quality (103-107/100), real architectural data, material realism  
→ **Time:** Worth waiting 2.5-3.5 hours for production-ready results

### Experimentation / Research
→ **Use:** `quickstart_training.sh`  
→ **Why:** Good baseline (100-103/100), fast iterations, consistent results  
→ **Time:** 2-3 hours allows multiple experiments

### Custom Client Projects
→ **Use:** Manual training with your data  
→ **Why:** Train on your specific style and requirements  
→ **How:** Organize data as `high_quality/` and `low_quality/` directories

### Infrastructure Testing
→ **Use:** `quick_train_demo.py`  
→ **Why:** Validate setup before long training runs  
→ **Time:** 5-10 minutes, not production-ready

---

## 🖥️ By Hardware

### Apple Silicon (M1/M2/M3/M4)
```bash
# Fast training with MPS acceleration
export PYTORCH_ENABLE_MPS_FALLBACK=1
./scripts/train_with_750picacho.sh --batch-size 4

# Expected: 2.5-3.5 hours, 103-107/100 quality
```

### NVIDIA GPU (CUDA)
```bash
# Fast training with CUDA acceleration
export CUDA_VISIBLE_DEVICES=0
./scripts/train_with_750picacho.sh --batch-size 4

# Expected: 3-4 hours, 103-107/100 quality
```

### CPU Only (No GPU)
```bash
# Recommended: Use synthetic data (faster than real data)
./scripts/quickstart_training.sh --batch-size 2

# Expected: 8-12 hours, 100-103/100 quality
# OR for best quality (slower): train_with_750picacho.sh
```

### Limited Memory (8GB RAM)
```bash
# Reduce batch size to avoid out of memory
./scripts/train_with_750picacho.sh --batch-size 2

# Or use synthetic data with smaller images
./scripts/quickstart_training.sh --batch-size 2
```

### High-End Workstation (32GB+ RAM, GPU)
```bash
# Maximize throughput with larger batch size
./scripts/train_with_750picacho.sh --batch-size 8 --epochs 100

# Expected: Faster convergence, potentially better quality
```

---

## ⏱️ By Available Time

### Have 3-4 hours + GPU
✅ **Use:** `train_with_750picacho.sh`  
→ Best quality, real data, production-ready

### Have 2-3 hours + GPU
✅ **Use:** `quickstart_training.sh`  
→ Good baseline, experiments

### Have 12-18 hours (overnight) + CPU
✅ **Use:** `train_with_750picacho.sh`  
→ Best quality, run overnight

### Have 8-12 hours (overnight) + CPU
✅ **Use:** `quickstart_training.sh`  
→ Good baseline, faster than real data

### Have 10 minutes (testing only)
✅ **Use:** `quick_train_demo.py`  
→ Validate infrastructure

---

## 🎓 By Skill Level

### Beginner
```bash
# Just run this one command:
./scripts/train_with_750picacho.sh

# That's it! Everything is automated.
```

### Intermediate
```bash
# Customize training parameters:
./scripts/train_with_750picacho.sh --epochs 100 --batch-size 8

# Or choose different data:
./scripts/quickstart_training.sh --num-pairs 2000
```

### Advanced
```bash
# Full control with manual training:
python src/enhancements/train_hyper_reality.py \
    --data-dir /custom/data \
    --epochs 100 \
    --batch-size 8 \
    --lr 1e-4 \
    --checkpoint-dir weights/custom \
    --resume-from weights/custom/checkpoint_epoch_50.pth
```

---

## 🚦 Quick Decision Matrix

Answer these questions to find your path:

| Question | Answer | Recommendation |
|----------|--------|----------------|
| **Do you have GPU?** | Yes | Use 750 Picacho (best) or Synthetic (faster) |
| | No | Use Synthetic (recommended for CPU) |
| **Is this for production?** | Yes | Use 750 Picacho (best quality) |
| | No | Use Synthetic (good baseline) |
| **Do you have 750 Picacho data?** | Yes | Use train_with_750picacho.sh ⭐ |
| | No | Use quickstart_training.sh |
| **How much time do you have?** | 3+ hours with GPU | Use train_with_750picacho.sh |
| | 2-3 hours with GPU | Use quickstart_training.sh |
| | 12+ hours with CPU | Use train_with_750picacho.sh |
| | <2 hours | Use quick_train_demo.py (testing only) |
| **What quality do you need?** | Best (103-107/100) | Use train_with_750picacho.sh |
| | Good (100-103/100) | Use quickstart_training.sh |
| | Testing only | Use quick_train_demo.py |

---

## 🎯 Most Common Scenarios

### Scenario 1: "I want the best results for production work"
```bash
# Check if you have the data
ls projects/750_picacho_lane/Final_Production_UltraQuality/

# If yes:
./scripts/train_with_750picacho.sh

# Wait 2.5-3.5 hours (GPU) or overnight (CPU)
# Result: 103-107/100 quality ⭐
```

### Scenario 2: "I want to experiment quickly"
```bash
# Fast baseline training with synthetic data
./scripts/quickstart_training.sh

# Wait 2-3 hours (GPU) or 8-12 hours (CPU)
# Result: 100-103/100 quality
```

### Scenario 3: "I have my own images to train on"
```bash
# Organize your data:
# my_data/high_quality/image_001.jpg
# my_data/low_quality/image_001.jpg
# (matching filenames)

# Train:
python src/enhancements/train_hyper_reality.py \
    --data-dir my_data/ \
    --epochs 50 \
    --batch-size 4

# Result: Custom quality based on your data
```

### Scenario 4: "I'm not sure if my setup works"
```bash
# Run pre-flight check
python scripts/check_training_ready.py

# If all green, run quick demo
python scripts/quick_train_demo.py

# Takes 5-10 minutes, validates everything works
```

---

## 📋 Pre-Training Checklist

Before starting training, use this checklist:

```bash
# 1. Run pre-flight check
python scripts/check_training_ready.py

# 2. Verify GPU (optional but recommended)
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"

# 3. Check disk space
df -h .

# 4. Choose your training method (see decision tree above)

# 5. Start training
./scripts/train_with_750picacho.sh  # or quickstart_training.sh
```

---

## 🆘 Still Not Sure?

### Default Recommendation (Safe Choice)
```bash
# This works for 90% of users:
./scripts/train_with_750picacho.sh
```

**Why?**
- ✅ Best quality (103-107/100)
- ✅ Uses real architectural data
- ✅ Production-ready results
- ✅ Fully automated
- ✅ Well-tested

**Time:** 2.5-3.5 hours (GPU) or 12-18 hours (CPU)

### Alternative (Faster)
```bash
# If you want faster results:
./scripts/quickstart_training.sh
```

**Why?**
- ✅ Good quality (100-103/100)
- ✅ Faster than real data
- ✅ Works without special data
- ✅ Great for experiments

**Time:** 2-3 hours (GPU) or 8-12 hours (CPU)

---

## 📚 Documentation

After choosing your method, refer to:

- **HOW_TO_TRAIN.md** - Complete implementation guide
- **TRAINING_QUICK_REFERENCE.md** - Quick reference card
- **TRAINING_EXECUTION_GUIDE.md** - Detailed execution guide
- **scripts/README_TRAINING.md** - Training scripts documentation

---

## 🎯 Final Recommendation

**For most users, start here:**

```bash
# Step 1: Check if ready
python scripts/check_training_ready.py

# Step 2: If all checks pass, run training
./scripts/train_with_750picacho.sh

# Step 3: Wait 2.5-3.5 hours (GPU) or overnight (CPU)

# Step 4: Test results
python src/enhancements/hyper_reality_enhancement.py test.jpg -o output.jpg
```

**That's it!** 🚀

---

**Last Updated:** 2025-11-19  
**Need Help?** Check HOW_TO_TRAIN.md or run `./scripts/train_with_750picacho.sh --help`
