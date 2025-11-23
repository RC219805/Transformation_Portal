# BasicSR-TP: Security-Hardened BasicSR Components

## Overview

This is a **minimal, security-hardened vendored copy** of components from [BasicSR v1.4.2](https://github.com/XPixelGroup/BasicSR), created to mitigate CVE-2024-27763.

**Version:** 1.4.2-tp1 (Transformation Portal security patch 1)

## Security Advisory: CVE-2024-27763

### Vulnerability Summary
- **Package:** BasicSR ≤ 1.4.2
- **Issue:** Command injection via unsanitized SLURM_NODELIST environment variable
- **Location:** `basicsr/utils/dist_util.py` line 44
- **CVSS Score:** 5.3 (Medium)
- **Impact:** Local privilege escalation in SLURM environments

### Vulnerable Code (Original BasicSR)
```python
# basicsr/utils/dist_util.py:44
node_list = os.environ['SLURM_NODELIST']
addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
```

If an attacker controls `SLURM_NODELIST`, they can inject shell commands:
```bash
SLURM_NODELIST="foo; whoami #" python train.py
```

### Our Mitigation
This vendored package **completely removes** all SLURM distributed utilities. Only the RRDBNet architecture is included, which is required for Real-ESRGAN upscaling.

**What's Included:**
- ✅ `RRDBNet` - ESRGAN super-resolution architecture
- ✅ `ResidualDenseBlock` - Building block for RRDBNet
- ✅ `RRDB` - Residual in Residual Dense Block
- ✅ Helper functions: `default_init_weights`, `make_layer`, `pixel_unshuffle`

**What's Excluded (removed for security):**
- ❌ `dist_util.py` - SLURM distributed training (vulnerable)
- ❌ Training infrastructure
- ❌ Data loaders and augmentation
- ❌ Metrics and losses
- ❌ Registry system (simplified)
- ❌ All other architectures

## Usage

### Drop-in Replacement
Replace imports in your code:

```python
# OLD (vulnerable):
from basicsr.archs.rrdbnet_arch import RRDBNet

# NEW (secure):
from basicsr_tp.archs.rrdbnet_arch import RRDBNet
```

### Example with Real-ESRGAN
```python
from basicsr_tp.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Create model
model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32
)

# Use with Real-ESRGAN
upsampler = RealESRGANer(
    scale=4,
    model_path='RealESRGAN_x4plus.pth',
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False
)
```

## Installation

This package is vendored within the Transformation Portal repository. No separate installation needed.

If you want to use it standalone:
1. Copy the `basicsr_tp/` directory to your project
2. Install dependencies: `pip install torch`
3. Import: `from basicsr_tp import RRDBNet`

## Maintenance

### Updating from Upstream
If BasicSR releases a security patch (> 1.4.2):

```bash
# Check for new releases
gh release list --repo XPixelGroup/BasicSR

# If a patched version exists, evaluate:
# 1. Does it fix CVE-2024-27763?
# 2. Are there breaking API changes?
# 3. Should we switch back or keep vendored?
```

### Testing Changes
```bash
# Test import
python3 -c "from basicsr_tp import RRDBNet; print('✓')"

# Test functionality
python3 << EOF
from basicsr_tp import RRDBNet
import torch
model = RRDBNet(num_in_ch=3, num_out_ch=3)
x = torch.randn(1, 3, 64, 64)
with torch.no_grad():
    y = model(x)
print(f'✓ Forward pass: {x.shape} -> {y.shape}')
assert y.shape == torch.Size([1, 3, 256, 256])
EOF
```

## License

This vendored code is licensed under **Apache-2.0**, inherited from BasicSR.

**Original License:**
```
Copyright (c) 2018-2022 XPixelGroup

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

**Our Modifications:**
- Extracted RRDBNet architecture only
- Removed vulnerable SLURM utilities
- Inlined required helper functions
- Added security documentation

## Attribution

- **Original Author:** XPixelGroup (https://github.com/XPixelGroup)
- **Original Project:** BasicSR (https://github.com/XPixelGroup/BasicSR)
- **Vendored By:** Transformation Portal
- **Vendored Date:** 2025-11-23
- **Purpose:** Security mitigation (CVE-2024-27763)

## References

- **BasicSR Repository:** https://github.com/XPixelGroup/BasicSR
- **CVE-2024-27763:** https://nvd.nist.gov/vuln/detail/CVE-2024-27763
- **GitHub Advisory:** GHSA-86w8-vhw6-q9qq
- **ESRGAN Paper:** https://arxiv.org/abs/1809.00219
- **Real-ESRGAN Paper:** https://arxiv.org/abs/2107.10833
