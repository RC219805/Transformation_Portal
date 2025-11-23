# Security Policy

## Supported Versions

We actively maintain security updates for the latest version of Transformation Portal.

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < Latest| :x:                |

## Security Advisories

### CVE-2024-27763: BasicSR Command Injection Vulnerability

**Status:** ✅ **MITIGATED** (as of 2025-11-23)

#### Summary
BasicSR ≤ 1.4.2 contains a command injection vulnerability in its SLURM distributed computing utilities (`basicsr/utils/dist_util.py` line 44). An attacker who can control the `SLURM_NODELIST` environment variable can execute arbitrary commands via the unsanitized shell execution in `scontrol show hostname`.

#### CVSS Score
- **5.3** (Medium Severity)
- **Attack Vector:** Local
- **Attack Complexity:** Low
- **Privileges Required:** Low
- **User Interaction:** None

#### Affected Components
- BasicSR versions ≤ 1.4.2
- Direct dependency of Real-ESRGAN package

#### Risk Assessment for Transformation Portal
**Low Practical Risk** under normal usage:
- The repository uses BasicSR **only** for the RRDBNet architecture (Real-ESRGAN upscaling)
- SLURM distributed computing features are **never used**
- Single-node, non-SLURM workflows only
- Not exposed as a multi-tenant service
- No untrusted users can set environment variables in production

#### Mitigation Strategy
We have implemented **Option A: Vendored Security-Hardened Fork**

**What We Did:**
1. Created `basicsr_tp` package - a minimal, security-hardened vendored copy
2. Extracted **only** the RRDBNet architecture (required for Real-ESRGAN)
3. **Completely removed** all SLURM distributed utilities and vulnerable code
4. Updated all scripts to use `basicsr_tp.archs.rrdbnet_arch` instead of `basicsr.archs.rrdbnet_arch`
5. Removed `basicsr` from `requirements/ml.in` and `requirements/all.in`

**Files Changed:**
- `basicsr_tp/` - New vendored package (self-contained, no external dependencies)
- `scripts/pipelines/luxury_estate_master_pipeline.py` - Import updated
- `scripts/pipelines/test_luxury_estate_pipeline.py` - Import updated
- `requirements/ml.in` - Removed vulnerable `basicsr>=1.4.2,<2` dependency

**Verification:**
```bash
# Verify no imports from original basicsr
grep -r "from basicsr" --include="*.py" . | grep -v "basicsr_tp" | grep -v ".git"

# Should return no results (excluding vendored code and git history)
```

#### Benefits
✅ **Real mitigation** - Vulnerable SLURM code completely removed  
✅ **Clean security dashboard** - GitHub alerts will close once dependency tree updates  
✅ **No upstream wait** - Independent of XPixelGroup's patching timeline  
✅ **Minimal maintenance** - Only RRDBNet architecture, no training/distributed code  
✅ **Drop-in replacement** - Same API as original BasicSR  

#### Testing
All existing tests pass with vendored implementation:
```bash
# Test vendored package import
python3 -c "from basicsr_tp.archs.rrdbnet_arch import RRDBNet; print('✓ Import successful')"

# Test instantiation and forward pass
python3 -c "
from basicsr_tp import RRDBNet
import torch
model = RRDBNet(num_in_ch=3, num_out_ch=3)
x = torch.randn(1, 3, 64, 64)
y = model(x)
print(f'✓ Forward pass: {x.shape} -> {y.shape}')
"
```

#### References
- **GitHub Advisory:** GHSA-86w8-vhw6-q9qq
- **CVE ID:** CVE-2024-27763
- **NVD Entry:** https://nvd.nist.gov/vuln/detail/CVE-2024-27763
- **BasicSR Repository:** https://github.com/XPixelGroup/BasicSR
- **Vulnerable Code:** `basicsr/utils/dist_util.py` line 44 (unsanitized subprocess call)

#### Future Monitoring
We will monitor:
- XPixelGroup/BasicSR for official patches
- Real-ESRGAN dependency updates
- New security advisories in the ML ecosystem

If XPixelGroup releases a patched version (> 1.4.2), we will evaluate whether to:
- Continue using vendored `basicsr_tp` (minimal maintenance)
- Switch back to upstream BasicSR (if benefits outweigh vendoring costs)

---

## Reporting a Vulnerability

If you discover a security vulnerability in Transformation Portal, please report it to:
- **Email:** [Create GitHub Security Advisory](https://github.com/RC219805/Transformation_Portal/security/advisories/new)
- **Response Time:** We aim to respond within 48 hours
- **Disclosure Policy:** Coordinated disclosure - we will work with you to fix and responsibly disclose

### What to Include
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested mitigation (if any)

### What Happens Next
1. We acknowledge receipt within 48 hours
2. We investigate and validate the issue
3. We develop and test a fix
4. We coordinate disclosure timeline with you
5. We release a patch and security advisory

Thank you for helping keep Transformation Portal secure!
