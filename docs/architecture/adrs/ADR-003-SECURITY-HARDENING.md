# ADR-003: Security Hardening Strategy

**Status**: Proposed  
**Date**: 2025-12-08  
**Authors**: Transformation Portal Architect  
**Related PRs**: PR-1 (Security + Repo Hygiene)

---

## Context

Security audit reveals critical risks that must be addressed before further development:

### Critical Risks Identified

1. **CVE-2024-27763**: Vulnerable `basicsr` package (CVSS 9.8) allows command injection
   - **Current State**: Mitigated via vendored `basicsr_tp/` but no CI enforcement
   - **Risk**: Developers could accidentally install vulnerable version

2. **Sensitive Artifacts in Repository**:
   - `.bash_history` (may contain credentials, commands)
   - `.local_backup/` (client-specific work, potentially sensitive)
   - Client folders with proprietary data
   - Branch cleanup backups in root tree

3. **No CI Security Gates**:
   - Banned packages (`basicsr`, `realesrgan`, `gfpgan`) not enforced
   - No secret scanning in CI
   - No dependency vulnerability scanning

4. **Documentation Mismatch**:
   - README references Real-ESRGAN but it's removed/disabled
   - Security posture unclear to users
   - No clear "safe vs unsafe" boundary

### Compliance Requirements

- **Public Repository**: Must not expose client data, credentials
- **Open Source Standards**: Security policy, vulnerability disclosure process
- **Supply Chain Security**: Dependency provenance, vulnerability tracking

---

## Decision

We will implement a **Defense-in-Depth Security Strategy** across four layers:

### Layer 1: Repository Hygiene (Immediate)

**Actions**:
1. Purge sensitive artifacts from working tree and git history
2. Update `.gitignore` to prevent future accidents
3. Rotate any potentially exposed credentials
4. Document safe development practices

### Layer 2: CI Security Gates (Immediate)

**Actions**:
1. Enforce banned package policy in CI (fail on `basicsr`/`realesrgan`/`gfpgan`)
2. Add secret scanning (gitleaks, detect-secrets)
3. Add dependency vulnerability scanning (safety, bandit)
4. Require security checks to pass before merge

### Layer 3: Input Validation (Platform Core)

**Actions**:
1. Implement `core/security/` module with path validation
2. Add safe image loading (PIL verify, MIME type checking)
3. Add input sanitization for all user-provided data
4. Document security best practices for pipeline developers

### Layer 4: Service Security (Lux Depth V2)

**Actions**:
1. Add authentication (API key) to service mode
2. Add rate limiting to prevent DoS
3. Add file upload limits
4. Require HTTPS for production deployments

---

## Implementation

### 1. Repository Purge & Gitignore

**Artifacts to Remove**:
```bash
# Sensitive files
.bash_history
.local_backup/
.branch_cleanup_backup/

# Client-specific data
09_Client_Deliverables/  # Move to separate private repo
output_*_client_name/    # Rename to generic patterns

# Temporary backups
*.backup
*.old
*_BACKUP/
```

**Gitignore Additions**:
```gitignore
# Security: Never commit these
.bash_history
.zsh_history
.python_history
*.pem
*.key
*.env
.env.*
secrets/
credentials/

# Backups
.local_backup/
.backup/
*.backup
*_BACKUP/

# Client data
client_*/
*_client_*/
confidential/
```

**History Purge**:
```bash
# Use BFG Repo-Cleaner (safe, fast)
bfg --delete-files .bash_history
bfg --delete-folders .local_backup

# Or git filter-repo (more control)
git filter-repo --path .bash_history --invert-paths
git filter-repo --path .local_backup --invert-paths
```

**Credential Rotation**:
```bash
# Scan for exposed secrets
gitleaks detect --source . --verbose
trufflehog git file://. --only-verified

# Rotate any found credentials:
# - API keys (regenerate in provider dashboard)
# - SSH keys (generate new, update authorized_keys)
# - Database passwords (update in secret manager)
```

### 2. CI Security Gate

**New Script**: `scripts/ci/enforce_safe_deps.py`

```python
#!/usr/bin/env python3
"""Enforce safe dependency policy (CVE-2024-27763 mitigation)."""
import subprocess
import sys

BANNED_PACKAGES = ["basicsr", "realesrgan", "gfpgan"]

def check_installed_packages():
    """Check if banned packages are installed."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list"],
        capture_output=True,
        text=True
    )
    
    installed = result.stdout.lower()
    found_banned = []
    
    for pkg in BANNED_PACKAGES:
        if pkg in installed:
            found_banned.append(pkg)
    
    return found_banned

def check_imports():
    """Check if banned packages are imported."""
    import ast
    from pathlib import Path
    
    found_imports = []
    
    for py_file in Path("lux_depth_v2").rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in BANNED_PACKAGES:
                            found_imports.append((py_file, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    if node.module in BANNED_PACKAGES:
                        found_imports.append((py_file, node.module))
        except:
            pass
    
    return found_imports

def main():
    errors = []
    
    # Check installed packages
    banned_installed = check_installed_packages()
    if banned_installed:
        errors.append(f"❌ Banned packages installed: {banned_installed}")
    
    # Check imports
    banned_imports = check_imports()
    if banned_imports:
        errors.append(f"❌ Banned imports found: {banned_imports}")
    
    if errors:
        print("🔒 SECURITY GATE FAILED")
        for error in errors:
            print(error)
        print("\n📖 See lux_depth_v2/SECURITY.md for mitigation")
        sys.exit(1)
    
    print("✅ Security gate passed: No banned packages")
    sys.exit(0)

if __name__ == "__main__":
    main()
```

**CI Integration**: Update `.github/workflows/security-scan.yml`

```yaml
jobs:
  security-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Enforce Safe Dependencies
        run: python scripts/ci/enforce_safe_deps.py
      
      - name: Secret Scanning
        uses: trufflesecurity/trufflehog@v3
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}
          head: HEAD
      
      - name: Dependency Vulnerability Scan
        run: |
          pip install safety
          safety check --file requirements.txt --json
      
      - name: Code Security Scan
        run: |
          pip install bandit
          bandit -r lux_depth_v2/ -ll -f json -o bandit-report.json
```

### 3. Input Validation (Platform Core)

**Path Validator**: `core/security/paths.py`

```python
from pathlib import Path
from typing import Union
import re

class PathValidator:
    """Prevent path traversal and symlink attacks."""
    
    def __init__(self, allowed_base: Union[str, Path]):
        self.allowed_base = Path(allowed_base).resolve()
    
    def validate(self, user_input: Union[str, Path]) -> Path:
        """Validate path is safe."""
        path = Path(user_input).resolve()
        
        # Prevent path traversal (e.g., "../../../etc/passwd")
        if not path.is_relative_to(self.allowed_base):
            raise ValueError(
                f"Path traversal attempt: {user_input} outside {self.allowed_base}"
            )
        
        # Prevent symlink attacks
        if path.is_symlink():
            raise ValueError(f"Symlinks not allowed: {user_input}")
        
        # Prevent null bytes (C-style string termination attacks)
        if "\x00" in str(user_input):
            raise ValueError("Null bytes in path")
        
        return path
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Remove dangerous characters from filename."""
        # Allow: alphanumeric, underscore, hyphen, period
        safe = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)
        
        # Prevent double extensions (.png.exe)
        parts = safe.split('.')
        if len(parts) > 2:
            safe = f"{'_'.join(parts[:-1])}.{parts[-1]}"
        
        # Prevent hidden files
        if safe.startswith('.'):
            safe = '_' + safe[1:]
        
        return safe
```

**Image Validator**: `core/security/images.py`

```python
from pathlib import Path
from PIL import Image
import magic

ALLOWED_FORMATS = {"image/jpeg", "image/png", "image/tiff"}
MAX_IMAGE_SIZE = 1024 * 1024 * 500  # 500MB

def validate_image_file(file_path: Path) -> None:
    """Validate image file is safe to process."""
    # Check file size
    if file_path.stat().st_size > MAX_IMAGE_SIZE:
        raise ValueError(f"File too large: {file_path.stat().st_size} bytes")
    
    # Check MIME type (prevents .exe renamed to .jpg)
    mime = magic.from_file(str(file_path), mime=True)
    if mime not in ALLOWED_FORMATS:
        raise ValueError(f"Unsupported file type: {mime}")
    
    # Verify image integrity (detects corrupted/malicious files)
    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image: {e}")
    
    # Prevent decompression bombs
    Image.MAX_IMAGE_PIXELS = 933120000  # 30000 x 31104 (reasonable UHR)
```

### 4. Service Security (Lux Depth V2)

**API Key Authentication**:

```python
# lux_depth_v2/service.py
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
import os

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Verify API key from environment."""
    expected_key = os.getenv("LUX_DEPTH_API_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    if not api_key or api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key

@app.post("/v2/process")
async def process_endpoint(api_key: str = Depends(verify_api_key)):
    # Process with validated API key
    pass
```

**Rate Limiting**:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/v2/process")
@limiter.limit("10/minute")
async def process_endpoint(request: Request):
    pass
```

**File Upload Limits**:

```python
from starlette.middleware.base import BaseHTTPMiddleware

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_size: int = 100_000_000):  # 100MB
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size:
            return Response(status_code=413, content="Request too large")
        return await call_next(request)

app.add_middleware(RequestSizeLimitMiddleware)
```

---

## Consequences

### Positive

1. **Risk Elimination**: CVE-2024-27763 cannot be accidentally introduced
2. **Secret Protection**: No credentials in repository or history
3. **Attack Surface Reduction**: Input validation prevents common exploits
4. **Compliance**: Meets open source security standards
5. **Trust**: Users can verify security posture via CI badges
6. **Auditability**: Security decisions documented

### Negative

1. **Overhead**: Security checks add ~30 seconds to CI runtime
2. **Friction**: Developers must follow stricter practices
3. **History Rewrite**: Repository history purge requires force push
4. **Breaking Changes**: Insecure patterns no longer work

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| False positives in secret scanning | Low | Whitelist known safe patterns |
| CI gate blocks legitimate work | Medium | Clear error messages, override process |
| History purge breaks forks | High | Announce in advance, provide migration guide |
| Service security adds latency | Low | <10ms overhead from validation |

---

## Success Metrics

### Security Metrics

- ✅ **Zero Vulnerabilities**: No critical/high CVEs in dependencies
- ✅ **No Secrets**: Secret scanner passes on all branches
- ✅ **CI Enforcement**: 100% of PRs pass security gate
- ✅ **Service Security**: API key + rate limiting active in production

### Operational Metrics

- ✅ **CI Runtime**: <2 minutes for security checks
- ✅ **False Positive Rate**: <5% on secret scanning
- ✅ **Developer Satisfaction**: Positive feedback on security UX

---

## Rollout Plan

### Phase 1: Preparation (Day 1)

1. **Announce Changes**: Notify all contributors of upcoming history rewrite
2. **Backup Repository**: Create archive of current state
3. **Prepare Scripts**: Test purge scripts on fork

### Phase 2: Repository Purge (Day 1)

1. **Purge Artifacts**: Remove sensitive files from working tree
2. **Purge History**: Run BFG or git filter-repo
3. **Force Push**: Update remote repository
4. **Verify**: Confirm sensitive data removed

### Phase 3: CI Integration (Day 2)

1. **Add Security Scripts**: `scripts/ci/enforce_safe_deps.py`
2. **Update Workflows**: Integrate into `security-scan.yml`
3. **Test on Branch**: Verify gates work correctly
4. **Merge to Main**: Enable enforcement

### Phase 4: Documentation (Day 3)

1. **Update README**: Reflect actual security posture
2. **Security Guide**: Document best practices
3. **Migration Instructions**: Help forks update

---

## Migration Guide for Contributors

### For Contributors with Existing Clones

```bash
# Save local work
git branch backup-local-work

# Delete local repository
cd ..
rm -rf Transformation_Portal

# Re-clone after purge
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal

# Reapply local work (cherry-pick or manual merge)
```

### For Forked Repositories

```bash
# Backup fork
git clone https://github.com/YOUR_USERNAME/Transformation_Portal.git fork-backup

# Update fork to match upstream
git remote add upstream https://github.com/RC219805/Transformation_Portal.git
git fetch upstream
git reset --hard upstream/main
git push --force origin main
```

---

## References

- [CVE-2024-27763 Details](https://nvd.nist.gov/vuln/detail/CVE-2024-27763)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)
- [lux_depth_v2/SECURITY.md](../../lux_depth_v2/SECURITY.md)

---

**Decision**: ✅ **APPROVED (CRITICAL)**  
**Implementation**: PR-1 (Security + Repo Hygiene)  
**Timeline**: Days 1-3 (Week 1)  
**Next Review**: 2025-12-15 (1 week post-merge)
