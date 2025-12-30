# Research & Experimental Features

**⚠️ WARNING: Features in this directory are EXPERIMENTAL and NOT production-ready.**

---

## What This Means

Features documented here are:
- ❌ **NOT stable** - APIs may change without notice
- ❌ **NOT tested** - May have incomplete test coverage
- ❌ **NOT supported** - Community-driven, best-effort support
- ❌ **NOT recommended** for production use

**Use at your own risk.**

---

## When to Use Research Features

Use these features **only if**:
- You're conducting research or experimentation
- You understand the code and can fix issues yourself
- You're willing to track upstream changes
- You have a fallback plan if it breaks

**Otherwise**: Use the [Golden Path](../../QUICKSTART.md) or [Advanced Workflows](../advanced/).

---

## Available Research Features

### 🧪 Model Training Infrastructure
**Use when**: Custom dataset adaptation, research on depth/material models
**Documentation**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
**Location**: `src/training/`, `examples/training/`
**Requirements**: GPU, 10GB+ disk, 2-3 hours training time
**Status**: ⚠️ Experimental

### 🔬 Custom Neural Architectures
**Use when**: Experimenting with novel model architectures
**Documentation**: [CUSTOM_ARCHITECTURES.md](CUSTOM_ARCHITECTURES.md)
**Location**: `src/models/experimental/`
**Status**: ⚠️ Experimental, research-only

### 🎯 Hyperparameter Auto-Tuning
**Use when**: Optimizing processing parameters for specific datasets
**Documentation**: [AUTOTUNE_INTEGRATION_GUIDE.md](../AUTOTUNE_INTEGRATION_GUIDE.md)
**Location**: `src/autotune/`
**Status**: ⚠️ Experimental, high complexity

---

## Experimental Directory Structure

```
experimental/
├── __init__.py          # Import guard (prevents production imports)
├── README.md            # This file
├── models/              # Novel architectures
├── training/            # Custom training pipelines
├── autotune/            # Parameter optimization
└── prototypes/          # Early-stage experiments
```

---

## Import Guards

The `experimental/` directory has **import guards** to prevent accidental use in production:

```python
# experimental/__init__.py
import warnings

warnings.warn(
    "Importing experimental features. NOT production-ready!",
    category=ExperimentalWarning,
    stacklevel=2
)
```

**CI enforcement**: Production code (`lux_depth_v2/`, `src/`) cannot import from `experimental/`.

---

## Graduation Path

For a research feature to become **production-ready**:

1. **Stability**: 6+ months without breaking changes
2. **Testing**: 90%+ code coverage, comprehensive tests
3. **Documentation**: User guide, API docs, examples
4. **Performance**: Validated on representative datasets
5. **Security**: Vulnerability scan, input validation
6. **Community**: 3+ external users successfully using it

Once graduated, feature moves to `src/` (standard) or `docs/advanced/` (power users).

---

## Stability Lifecycle

```
Research (experimental/)
  → Advanced (docs/advanced/, community-supported)
  → Production (lux_depth_v2/, feature-frozen)
  → Deprecated (archive/)
```

**Current policy**: Most features should stabilize at "Advanced" level, not "Production" (to avoid Golden Path bloat).

---

## Contributing Research Features

Experimental features have **minimal governance**:

✅ **Allowed**:
- Anything (within legal/ethical bounds)
- Breaking changes anytime
- Incomplete documentation
- Novel/risky approaches

⚠️ **Required**:
- Clear "EXPERIMENTAL" labeling
- Basic safety checks (no security holes)
- Isolation from production code

📋 **Recommended**:
- Tests (even if incomplete)
- Inline documentation
- Example usage
- Known limitations documented

---

## Risk Assessment

Before using a research feature, ask:

1. **What happens if it breaks?** (Have a fallback)
2. **Can I fix it myself?** (Have source code access)
3. **Is it worth the complexity?** (Simpler alternatives exist?)
4. **Do I understand the trade-offs?** (Read the docs)

If you answered "no" to any question, **don't use it**.

---

## Related Documentation

- **[Golden Path](../../QUICKSTART.md)** - Production workflow
- **[Advanced Features](../advanced/)** - Stable power-user features
- **[Architecture](../architecture/)** - System design
- **[CONTRIBUTING.md](../../CONTRIBUTING.md)** - Development guidelines

---

*Experimental features exist to enable innovation. But innovation without discipline creates chaos. Use responsibly.*
