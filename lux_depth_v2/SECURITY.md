# Security Guidelines for Lux Depth V2

**Last Updated**: 2025-12-06  
**Security Contact**: See root SECURITY.md

---

## Overview

This document outlines security considerations, mitigations, and best practices for the Lux Depth V2 module.

---

## 1. Known Vulnerabilities & Mitigations

### CVE-2024-27763: basicsr Command Injection

**Severity**: 🔴 **CRITICAL**  
**Affected Package**: `basicsr<=1.4.x`  
**Vulnerability**: Command injection via crafted file paths  
**CVSS Score**: 9.8 (Critical)

#### Mitigation

**Status**: ✅ **MITIGATED** (when using `requirements-repo.txt`)

The Transformation Portal repository uses a **vendored, patched version** of basicsr located at `/basicsr_tp/`. This version has the command injection vulnerability patched.

**Action Required**:
- ❌ **DO NOT** install `basicsr` or `realesrgan` via pip
- ✅ **DO** use `requirements-repo.txt` (references vendored basicsr_tp)
- ✅ **DO** use alternative upscaling backends (torchvision, Pillow)

**Verification**:
```bash
# Ensure basicsr is NOT in environment
pip list | grep basicsr
# Should show: basicsr-tp (vendored version) or nothing

# Verify using repo dependencies
pip install -r requirements-repo.txt
```

---

## 2. Service Mode Security

The `service.py` module exposes a FastAPI REST API endpoint. **Production deployments MUST implement security controls.**

### 2.1 Input Validation

**Risk**: Path traversal, arbitrary file access  
**Severity**: 🔴 **CRITICAL**

#### Required: Path Validation

All file paths from user input must be validated:

```python
from pathlib import Path

ALLOWED_BASE_DIR = Path("/data/uploads").resolve()

def validate_filepath(user_input: str) -> Path:
    """Validate and sanitize file paths to prevent directory traversal."""
    path = Path(user_input).resolve()
    
    # Prevent path traversal attacks
    if not path.is_relative_to(ALLOWED_BASE_DIR):
        raise ValueError(f"Path traversal attempt detected: {user_input}")
    
    # Prevent symlink attacks
    if path.is_symlink():
        raise ValueError(f"Symlinks are not allowed: {user_input}")
    
    return path
```

**Implementation Status**: ⚠️ **PENDING** - Must be added before production use

### 2.2 Authentication & Authorization

**Risk**: Unauthorized access, data exfiltration  
**Severity**: 🔴 **CRITICAL**

#### Recommended: API Key Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Verify API key from environment variable."""
    expected_key = os.getenv("LUX_DEPTH_API_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    if api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/v2/process")
async def process_endpoint(api_key: str = Depends(verify_api_key)):
    # Process with validated API key
    pass
```

**Environment Setup**:
```bash
export LUX_DEPTH_API_KEY="your-secret-key-here"
lux-depth-v2-service --service --port 8088
```

**Implementation Status**: ⚠️ **PENDING** - Optional but strongly recommended

### 2.3 Rate Limiting

**Risk**: Denial of service, resource exhaustion  
**Severity**: 🟡 **HIGH**

#### Required: Request Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/v2/process")
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def process_endpoint(request: Request):
    # Process with rate limiting
    pass
```

**Implementation Status**: ⚠️ **PENDING** - Must be added before production use

### 2.4 File Upload Limits

**Risk**: Denial of service via large uploads  
**Severity**: 🟡 **HIGH**

#### Required: Request Size Limits

```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_size: int = 100_000_000):  # 100MB default
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size:
            return Response(status_code=413, content="Request too large")
        return await call_next(request)

app.add_middleware(RequestSizeLimitMiddleware, max_size=100_000_000)
```

**Implementation Status**: ⚠️ **PENDING** - Must be added before production use

### 2.5 HTTPS/TLS

**Risk**: Man-in-the-middle attacks, credential interception  
**Severity**: 🔴 **CRITICAL** (for production)

#### Required for Production: Enable HTTPS

```bash
# Use reverse proxy (recommended)
nginx -> uvicorn (internal HTTP)

# Or use uvicorn with TLS (development only)
uvicorn lux_depth_v2.service:app \
  --host 0.0.0.0 \
  --port 8088 \
  --ssl-keyfile key.pem \
  --ssl-certfile cert.pem
```

**Implementation Status**: ⚠️ **USER RESPONSIBILITY** - Configure reverse proxy

---

## 3. Input Sanitization

### 3.1 Image File Validation

**Risk**: Malicious image files, code execution  
**Severity**: 🟡 **HIGH**

#### Recommended: File Format Validation

```python
from PIL import Image
import magic

ALLOWED_FORMATS = {"image/jpeg", "image/png", "image/tiff"}

def validate_image_file(file_path: Path) -> None:
    """Validate image file format and integrity."""
    # Check MIME type
    mime = magic.from_file(str(file_path), mime=True)
    if mime not in ALLOWED_FORMATS:
        raise ValueError(f"Unsupported file type: {mime}")
    
    # Verify image can be opened (detects corrupted/malicious files)
    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image: {e}")
```

**Implementation Status**: ⚠️ **PARTIAL** - Add magic library validation

### 3.2 Configuration Injection

**Risk**: Code injection via YAML/JSON configs  
**Severity**: 🟡 **HIGH**

#### Safe Configuration Loading

```python
import yaml

def safe_load_config(config_path: Path) -> dict:
    """Safely load YAML configuration without code execution."""
    with open(config_path, "r") as f:
        # Use safe_load to prevent YAML code execution
        return yaml.safe_load(f)
    
    # Never use yaml.load() or pickle.load() on untrusted input!
```

**Implementation Status**: ✅ **IMPLEMENTED** - config.py uses safe_load

---

## 4. Dependency Security

### 4.1 Vulnerability Scanning

**Required**: Regular dependency audits

```bash
# Install security tools
pip install safety bandit

# Scan dependencies for known vulnerabilities
safety check --file requirements-repo.txt --json

# Scan code for security issues
bandit -r lux_depth_v2/ -ll
```

**Automation**: Security scans are integrated into CI/CD (`security-scan.yml`)

### 4.2 Dependency Pinning

**Status**: ✅ **IMPLEMENTED** (via requirements-repo.txt)

- All dependencies reference repository's pinned versions (`requirements/ml.txt`)
- No version ranges that could pull in vulnerable updates
- Regular updates via Dependabot

---

## 5. Secure Deployment Checklist

### Development Environment
- [x] Use requirements-repo.txt (vendored dependencies)
- [x] No basicsr/realesrgan installed
- [ ] Service mode disabled (use CLI only)

### Production Environment
- [ ] HTTPS/TLS enabled (reverse proxy or uvicorn)
- [ ] API key authentication enabled
- [ ] Rate limiting configured (10-100 req/min per IP)
- [ ] File upload limits enforced (50-100MB max)
- [ ] Input validation for all user-provided paths
- [ ] File format validation (MIME type + PIL verify)
- [ ] Firewall rules restrict access to trusted IPs
- [ ] Logging enabled for security events
- [ ] Regular security scans (safety, bandit)
- [ ] Secrets managed via environment variables (not hardcoded)

---

## 6. Reporting Security Issues

**DO NOT** open public GitHub issues for security vulnerabilities.

**Contact**: See root `/SECURITY.md` for secure reporting instructions.

**Process**:
1. Email security contact (from root SECURITY.md)
2. Include: module (lux_depth_v2), vulnerability description, reproduction steps
3. Wait for acknowledgment (24-48 hours)
4. Coordinated disclosure after patch is available

---

## 7. Security Best Practices

### 7.1 Principle of Least Privilege

- Run service with dedicated user (not root)
- Limit filesystem access (read-only input, write-only output)
- Use container isolation (Docker) when possible

### 7.2 Defense in Depth

- Multiple layers of security (auth + rate limit + input validation)
- Assume all user input is malicious until validated
- Log security events for monitoring

### 7.3 Regular Updates

- Monitor security advisories (GitHub Security Advisories, CVE databases)
- Update dependencies monthly (or immediately for critical CVEs)
- Test security controls after updates

---

## 8. Resources

- **OWASP API Security Top 10**: https://owasp.org/www-project-api-security/
- **Python Security Best Practices**: https://python.readthedocs.io/en/stable/library/security.html
- **FastAPI Security**: https://fastapi.tiangolo.com/tutorial/security/
- **Repository Security Guide**: `/SECURITY.md`

---

**Version**: 1.0  
**Effective Date**: 2025-12-06  
**Next Review**: 2025-03-06 (quarterly)
