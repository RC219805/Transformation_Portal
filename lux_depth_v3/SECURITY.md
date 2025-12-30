# Security Guidelines for Lux Depth V3

**Last Updated**: 2025-12-19
**Module**: lux_depth_v3
**Security Contact**: See root SECURITY.md

---

## Overview

This document outlines security considerations, mitigations, and best practices for the Lux Depth V3 module.

---

## 1. Input Validation

### File Upload Security

**Implemented Controls:**

```python
# Maximum file size limit
MAX_FILE_SIZE_MB = 50  # Default in InputManager

# Maximum image dimension
MAX_IMAGE_DIMENSION = 4096  # Service mode limit

# Allowed file extensions
allowed_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
```

**Best Practices:**

1. **Always validate file sizes** before processing
2. **Check file extensions** against whitelist
3. **Verify image dimensions** to prevent memory exhaustion
4. **Use secure file paths** (no directory traversal)

### Example:

```python
from lux_depth_v3 import InputManager

# Secure input with size limit
manager = InputManager(max_file_size_mb=50)

try:
    manager.add_image(path=user_provided_path)
except ValueError as e:
    # Handle validation error
    print(f"Input validation failed: {e}")
```

---

## 2. Service Mode Security

### Rate Limiting

**Default Configuration:**

```python
RATE_LIMIT_REQUESTS_PER_MINUTE = 60
```

**Implementation:**

- Tracks request timestamps per client IP
- Rejects requests exceeding limit with HTTP 429
- Auto-cleanup of old timestamps

**Customization:**

```python
# In service.py
RATE_LIMIT_REQUESTS_PER_MINUTE = 120  # Increase if needed
```

### CORS Configuration

**Default (Development):**

```python
allow_origins=["*"]  # Permissive for development
```

**Production Configuration:**

```python
# Configure allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://app.yourdomain.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Path Traversal Protection

**Implemented:**

```python
# In download_depth endpoint
if not file_path.is_relative_to(output_dir):
    raise HTTPException(status_code=400, detail="Invalid filename")
```

**Never** construct file paths directly from user input without validation.

---

## 3. Dependency Security

### Verified Dependencies

All dependencies in `requirements.txt` are vetted for known vulnerabilities:

```
numpy>=1.24,<2.3.0      # Constrained for compatibility
Pillow>=10.0.0,<12      # Actively maintained, secure
torch>=2.0.0,<3         # Official PyTorch release
scipy>=1.15,<1.16       # Constrained for Python 3.10
```

### Dependency Scanning

Run security audit regularly:

```bash
# Using pip-audit
pip install pip-audit
pip-audit -r lux_depth_v3/requirements.txt

# Using safety
pip install safety
safety check -r lux_depth_v3/requirements.txt
```

### Update Policy

- **Critical vulnerabilities**: Patch immediately
- **High vulnerabilities**: Patch within 7 days
- **Medium/Low**: Review and patch in next release

---

## 4. Model Security

### Model Source Validation

**Official Source Only:**

```python
# Only use official DA3 models from trusted sources
model = DepthAnything3.from_pretrained(
    "depth-anything-3-metric-large",
    # Downloaded from official repository
)
```

**Do NOT:**
- Load models from untrusted sources
- Execute arbitrary code during model loading
- Use models without checksum verification

### Model Caching

**Secure Cache Directory:**

```python
# Default cache with user permissions
cache_dir = Path.home() / ".cache" / "lux_depth_v3"

# Ensure proper permissions (user-only read/write)
cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
```

---

## 5. Data Privacy

### Input Data Handling

**Principles:**

1. **No data retention** - Delete processed images after export
2. **In-memory processing** - Avoid unnecessary disk writes
3. **Secure deletion** - Overwrite sensitive data

**Example:**

```python
import os

# Secure file deletion
def secure_delete(file_path: Path):
    """Overwrite file before deletion."""
    if file_path.exists():
        # Overwrite with random data
        size = file_path.stat().st_size
        with open(file_path, "wb") as f:
            f.write(os.urandom(size))
        file_path.unlink()
```

### Metadata Sanitization

**Remove sensitive metadata:**

```python
from PIL import Image

def sanitize_image(img: Image.Image) -> Image.Image:
    """Remove EXIF and metadata."""
    # Create new image without metadata
    data = list(img.getdata())
    img_clean = Image.new(img.mode, img.size)
    img_clean.putdata(data)
    return img_clean
```

---

## 6. Service Deployment

### HTTPS/TLS

**Always use HTTPS in production:**

```bash
# With certificate
uvicorn lux_depth_v3.service:app \
  --host 0.0.0.0 \
  --port 8088 \
  --ssl-keyfile /path/to/key.pem \
  --ssl-certfile /path/to/cert.pem
```

### Reverse Proxy

**Recommended: Use nginx/Apache as reverse proxy:**

```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location /depth/ {
        proxy_pass http://127.0.0.1:8088;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Security headers
        add_header X-Content-Type-Options nosniff;
        add_header X-Frame-Options DENY;
        add_header X-XSS-Protection "1; mode=block";

        # Request size limits
        client_max_body_size 50M;
    }
}
```

### Authentication

**Add authentication layer:**

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token."""
    if credentials.credentials != os.getenv("API_TOKEN"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials

# Protected endpoint
@app.post("/depth/estimate")
async def estimate_depth(
    file: UploadFile,
    token: str = Depends(verify_token)
):
    # Process request
    ...
```

---

## 7. Error Handling

### Secure Error Messages

**Do NOT expose:**
- Internal paths
- Stack traces
- Configuration details

**Example:**

```python
try:
    result = engine.inference(img_input)
except Exception as e:
    # Log full error internally
    logger.error(f"Inference failed: {str(e)}", exc_info=True)

    # Return generic error to user
    raise HTTPException(
        status_code=500,
        detail="Processing failed. Please contact support."
    )
```

---

## 8. Logging & Monitoring

### Secure Logging

**Log security events:**

```python
import logging

logger = logging.getLogger("lux_depth_v3.security")

# Log authentication failures
logger.warning(
    "Authentication failed",
    extra={
        "client_ip": request.client.host,
        "endpoint": request.url.path,
    }
)

# Log rate limit violations
logger.warning(
    "Rate limit exceeded",
    extra={"client_ip": client_ip}
)
```

**Do NOT log:**
- API tokens
- User data
- Sensitive metadata

---

## 9. Known Vulnerabilities

### CVE Status

✅ **No known CVEs** in lux_depth_v3 dependencies (as of 2025-12-19)

### Related Modules

⚠️ **Note**: lux_depth_v2 has mitigated CVE-2024-27763 (basicsr command injection). Lux_depth_v3 does NOT use basicsr and is not affected.

---

## 10. Security Checklist

### Deployment Checklist

- [ ] HTTPS/TLS enabled
- [ ] Rate limiting configured
- [ ] CORS origins restricted
- [ ] Authentication implemented
- [ ] File size limits enforced
- [ ] Input validation active
- [ ] Error messages sanitized
- [ ] Logging configured
- [ ] Dependencies scanned
- [ ] Reverse proxy configured
- [ ] Security headers set
- [ ] Data retention policy defined

### Code Review Checklist

- [ ] No hardcoded credentials
- [ ] No SQL injection vectors
- [ ] No command injection vectors
- [ ] No path traversal vulnerabilities
- [ ] Input validation comprehensive
- [ ] Error handling secure
- [ ] Logging excludes sensitive data

---

## 11. Incident Response

### Security Issue Reporting

**Contact**: See root SECURITY.md

**Process**:
1. Report via security@transformationportal.com (if exists)
2. Include: affected component, impact, reproduction steps
3. Wait for acknowledgment (within 24 hours)
4. Coordinate disclosure timeline

### Patching Process

1. Develop fix in private branch
2. Test thoroughly
3. Release patch version
4. Notify users via security advisory
5. Update documentation

---

## 12. Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)

---

## Version History

- **2025-12-19**: Initial security guidelines for lux_depth_v3
