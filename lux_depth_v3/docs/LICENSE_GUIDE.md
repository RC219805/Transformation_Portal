# DA3 License Compliance Guide

## Overview

Depth Anything 3 (DA3) models are released under two different licenses depending on the model variant. This guide explains the licensing implications and helps you choose the right model for your use case.

## License Types

### Apache-2.0 (Commercial-Friendly) ✅

**Permits:**
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ✅ Patent use

**Models:**
- `DA3-BASE` (0.12B params)
- `DA3-SMALL` (0.08B params)
- `DA3METRIC-LARGE` (0.35B params) - **Recommended for commercial**
- `DA3MONO-LARGE` (0.35B params)

**License URL:** https://www.apache.org/licenses/LICENSE-2.0

### CC-BY-NC-4.0 (Non-Commercial Only) ⚠️

**Permits:**
- ✅ Non-commercial use
- ✅ Modification
- ✅ Distribution (with attribution)
- ❌ Commercial use **NOT ALLOWED**

**Models:**
- `DA3NESTED-GIANT-LARGE-1.1` (1.40B params)
- `DA3NESTED-GIANT-LARGE` (1.40B params, deprecated)
- `DA3-GIANT-1.1` (1.15B params)
- `DA3-GIANT` (1.15B params, deprecated)
- `DA3-LARGE-1.1` (0.35B params)
- `DA3-LARGE` (0.35B params, deprecated)

**License URL:** https://creativecommons.org/licenses/by-nc/4.0/

## License Comparison Table

| Model | Params | License | Commercial | Metric Depth | Gaussian Splatting |
|-------|--------|---------|------------|--------------|-------------------|
| **DA3NESTED-GIANT-LARGE-1.1** | 1.40B | CC-BY-NC-4.0 | ❌ No | ✅ Yes | ✅ Yes |
| **DA3-GIANT-1.1** | 1.15B | CC-BY-NC-4.0 | ❌ No | ❌ No | ✅ Yes |
| **DA3-LARGE-1.1** | 0.35B | CC-BY-NC-4.0 | ❌ No | ❌ No | ❌ No |
| **DA3METRIC-LARGE** | 0.35B | Apache-2.0 | ✅ Yes | ✅ Yes | ❌ No |
| **DA3-BASE** | 0.12B | Apache-2.0 | ✅ Yes | ❌ No | ❌ No |
| **DA3-SMALL** | 0.08B | Apache-2.0 | ✅ Yes | ❌ No | ❌ No |
| **DA3MONO-LARGE** | 0.35B | Apache-2.0 | ✅ Yes | ❌ No | ❌ No |

## What is "Commercial Use"?

### Definitely Commercial ❌ (NC models not allowed)

- Selling depth maps or processed images
- Providing depth estimation as a paid service
- Using in products sold to customers
- Revenue-generating applications
- Internal business use (e.g., real estate processing for sale)
- Advertising/marketing materials for products

### Likely Commercial ⚠️ (Consult legal counsel)

- Free service with ads
- Freemium model (free + paid tiers)
- Internal enterprise tools
- Educational platforms with tuition fees

### Non-Commercial ✅ (NC models allowed)

- Academic research (published papers)
- Personal projects (no revenue)
- Open-source tools (no commercial deployment)
- Educational tutorials (non-profit)
- Art installations (non-profit)

## Commercial Alternatives

For each NC-licensed model, we recommend a commercial-friendly alternative:

| NC Model | → | Commercial Alternative | Trade-off |
|----------|---|----------------------|-----------|
| DA3NESTED-GIANT-LARGE-1.1 | → | **DA3METRIC-LARGE** | Lose GS, keep metric depth |
| DA3-GIANT-1.1 | → | **DA3-BASE** | Smaller model, no GS |
| DA3-LARGE-1.1 | → | **DA3-BASE** | Smaller model |

## Model Selection Flowchart

```
┌─────────────────────────────────┐
│   Is this commercial use?       │
└─────────────┬───────────────────┘
              │
              ├─ Yes ─→ Use Apache-2.0 models ✅
              │         (DA3METRIC-LARGE, DA3-BASE, etc.)
              │
              └─ No ──→ Research/personal?
                        │
                        ├─ Yes ─→ Any model allowed ✅
                        │         (Recommended: DA3NESTED-GIANT-LARGE-1.1)
                        │
                        └─ Unsure ─→ Use Apache-2.0 models ✅
                                     (DA3METRIC-LARGE for best quality)
```

## Python API - License Validation

### Automatic License Warnings

```python
from lux_depth_v3.config import ModelVariant, DA3Config
from lux_depth_v3.inference import DA3InferenceEngine

# Non-commercial use (no warning)
config = DA3Config(model_variant=ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1)
engine = DA3InferenceEngine(config, commercial_use=False)

# Commercial use with NC model (issues warning)
engine = DA3InferenceEngine(config, commercial_use=True)
# ⚠️  LICENSE WARNING: DA3NESTED-GIANT-LARGE-1.1
# License: CC-BY-NC-4.0 (Non-Commercial)
# Commercial use is NOT permitted.
# For commercial applications, use:
#   → DA3METRIC-LARGE (Apache-2.0)

# Strict mode (raises error instead of warning)
engine = DA3InferenceEngine(
    config,
    commercial_use=True,
    validate_license_strict=True
)
# RuntimeError: Model DA3NESTED-GIANT-LARGE-1.1 (CC-BY-NC-4.0) 
#               cannot be used for commercial purposes.
```

### Manual License Checking

```python
from lux_depth_v3.license import LicenseValidator

validator = LicenseValidator()

# Check commercial permissions
variant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1
is_allowed = validator.check_commercial_use(
    variant,
    commercial_use=True,
    warn=False
)
# Returns False for NC-licensed models

# Get license information
info = validator.get_license_info(variant)
print(info)
# {
#     'model': 'DA3NESTED-GIANT-LARGE-1.1',
#     'license': 'CC-BY-NC-4.0',
#     'commercial_allowed': False,
#     'license_url': 'https://creativecommons.org/licenses/by-nc/4.0/',
#     'alternative': 'DA3METRIC-LARGE'
# }

# Get commercial alternative
commercial = ModelVariant.get_commercial_alternative(variant)
print(commercial.info.display_name)
# 'DA3METRIC-LARGE'
```

## CLI - License Validation

### Show License Information

```bash
# View license details for a model
lux-depth-v3 api-process image.jpg -o output \
    -m nested-giant-large-v1.1 \
    --show-license

# Output:
# 📄 License Information
# ======================================================================
# Model: DA3NESTED-GIANT-LARGE-1.1
# License: CC-BY-NC-4.0
# Commercial Use: ❌ Not Allowed
# License URL: https://creativecommons.org/licenses/by-nc/4.0/
# 
# Commercial Alternative: DA3METRIC-LARGE
# ======================================================================
```

### Declare Commercial Use

```bash
# Non-commercial use (default, no flag needed)
lux-depth-v3 api-process images/ -o output -m nested-giant-large-v1.1

# Commercial use with NC model (triggers warning)
lux-depth-v3 api-process images/ -o output \
    -m nested-giant-large-v1.1 \
    --commercial

# Commercial use with Apache model (no warning)
lux-depth-v3 api-process images/ -o output \
    -m metric-large \
    --commercial

# Strict mode (error on violation)
lux-depth-v3 api-process images/ -o output \
    -m nested-giant-large-v1.1 \
    --commercial \
    --strict-license
# ERROR: Model DA3NESTED-GIANT-LARGE-1.1 (CC-BY-NC-4.0) 
#        cannot be used for commercial purposes.
```

## Recommended Models by Use Case

### Research & Academia (Non-Commercial)

**Best:** `DA3NESTED-GIANT-LARGE-1.1`
- Highest accuracy
- All features (metric depth, GS, pose estimation)
- No commercial restrictions

```bash
lux-depth-v3 api-process images/ -o output -m nested-giant-large-v1.1
```

### Commercial Real Estate Rendering

**Best:** `DA3METRIC-LARGE`
- Commercial-friendly (Apache-2.0)
- Metric depth output
- Sky segmentation
- Good accuracy (0.35B params)

```bash
lux-depth-v3 api-process renders/ -o output \
    -m metric-large \
    --commercial
```

### Commercial App/Service (High Performance)

**Best:** `DA3-BASE`
- Commercial-friendly
- Balanced performance (0.12B params)
- Fast inference
- Pose estimation support

```bash
lux-depth-v3 api-process user_images/ -o output \
    -m base \
    --commercial
```

### Commercial App/Service (Mobile/Edge)

**Best:** `DA3-SMALL`
- Commercial-friendly
- Smallest model (0.08B params)
- Fast on CPU
- Suitable for edge deployment

```bash
lux-depth-v3 api-process images/ -o output \
    -m small \
    --commercial \
    --device cpu
```

## Legal Considerations

### ⚠️ Important Notice

This guide provides general information about license types. **It is not legal advice.** For commercial deployments:

1. **Consult Legal Counsel:** Always consult with a lawyer familiar with open-source licensing
2. **Review Full License Text:** Read the complete license terms at the URLs provided
3. **Document Your Use Case:** Maintain records of how models are used
4. **Plan for Audits:** Be prepared to demonstrate license compliance

### Attribution Requirements

Both licenses require attribution:

**CC-BY-NC-4.0:**
- Cite the original paper
- Link to license
- Indicate if modifications were made

**Apache-2.0:**
- Include copyright notice
- Include license text
- Note modifications (if any)

**Example Attribution:**
```
Depth estimation powered by Depth Anything 3
Model: DA3METRIC-LARGE (Apache-2.0)
https://github.com/DepthAnything/Depth-Anything-V3
```

## FAQ

### Q: Can I use NC models in a company for internal research?

**A:** Possibly. "Internal research" may or may not be commercial depending on context. If the research directly supports a commercial product, it's likely commercial use. **Consult legal counsel.**

### Q: Can I use NC models to create training data for a commercial model?

**A:** No. Training data generation for commercial models is likely commercial use.

### Q: Can I publish a paper using NC models and later commercialize it?

**A:** Research publication is non-commercial. Commercialization requires switching to Apache-licensed models.

### Q: What if I only use models locally and never distribute?

**A:** License still applies. Commercial use is about the purpose, not distribution.

### Q: Can I modify NC models and use commercially?

**A:** No. Derivative works inherit the NC restriction.

## Support

For licensing questions:
- **Technical:** GitHub Issues
- **Legal:** Consult your legal counsel
- **License Text:** See links in "License Types" section

---

**Last Updated:** December 19, 2024  
**Disclaimer:** This is not legal advice. Consult a lawyer for commercial deployments.
