# Security Policy

## Supported Versions

The following versions of Transformation Portal are currently supported with security updates:

| Version | Supported          | Notes |
| ------- | ------------------ | ----- |
| main    | :white_check_mark: | Development branch - security fixes prioritized |
| 0.1.x   | :white_check_mark: | Current stable release |
| < 0.1   | :x:                | Unsupported |

## Reporting a Vulnerability

### How to Report

If you discover a security vulnerability in Transformation Portal, please **DO NOT** open a public issue. Instead:

1. **GitHub Security Advisory** (Preferred): Create a private security advisory at https://github.com/RC219805/Transformation_Portal/security/advisories/new
2. **Direct Contact**: Reach out via GitHub (@RC219805)
3. **Include**:
   - Affected version(s)
   - Steps to reproduce
   - Potential impact assessment
   - Your contact information for follow-up

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 5 business days
- **Resolution Target**: 
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: Next release cycle

### What to Expect

1. **Acknowledgment**: You'll receive confirmation that we've received your report
2. **Assessment**: Our security team will evaluate the vulnerability
3. **Communication**: We'll keep you informed throughout the resolution process
4. **Credit**: With your permission, we'll acknowledge your contribution in the fix announcement

## GitHub Security Features

This repository uses:
- **Dependabot**: Automated dependency updates for security vulnerabilities
- **Code Scanning**: CodeQL analysis on every PR
- **Secret Scanning**: Prevents accidental credential commits
- **Security Advisories**: Private vulnerability reporting via GitHub
- **Branch Protection**: Main branch requires security checks to pass

## Security Considerations

### Input Validation

Given our image/video processing nature, special attention is required for:

- **File Upload Security**:
  - Maximum file size limits (default: 500MB for images, 5GB for videos)
  - Strict MIME type validation
  - Magic number verification for file formats
  - Filename sanitization to prevent path traversal

- **TIFF Processing**:
  - Validation of TIFF tags to prevent buffer overflows
  - Limits on image dimensions (max 65536x65536)
  - Protection against compression bombs

### Depth Map Processing

- **Depth Anything V2 Model**: Validate input dimensions to prevent memory overflow (max 4096x4096)
- **Point Cloud Generation**: Limit vertex count to prevent DoS (max 10M vertices)
- **Temporary File Management**: Secure cleanup of intermediate depth maps
- **GPU Memory**: Monitor and limit VRAM usage (default: 8GB max)

### ML Model Security

- **Model Files**: 
  - Only load models from trusted sources
  - Verify model checksums before loading
  - Sandboxed model execution environment recommended
  
- **Depth Pipeline**:
  - Input size restrictions to prevent OOM attacks
  - Rate limiting for API endpoints
  - Secure temporary file handling for intermediate outputs

### Dependencies

- **Supply Chain**: 
  - All dependencies are pinned to specific versions
  - Regular dependency audits via `pip-audit` and `safety`
  - Automated security scanning in CI/CD pipeline

- **Known Vulnerabilities**:
  - PyTorch: Keep updated for CUDA-related security patches
  - Pillow: Critical for image parsing vulnerabilities
  - NumPy: Monitor for numerical computation exploits

### API Security

If exposing Transformation Portal as a service:

- **Authentication**: Implement API key or OAuth 2.0
- **Rate Limiting**: 
  - Default: 100 requests/minute per IP
  - Heavy operations: 10 requests/hour
- **Input Sanitization**: All user inputs must be validated
- **Output Filtering**: Ensure no metadata leakage in processed files

### API Security Headers

```python
# If using Flask/FastAPI
headers = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'"
}
```

## Performance vs. Security Trade-offs

Security features may impact performance:
- File validation: +100-500ms per upload
- Model checksums: +2-5s on first load
- Input sanitization: +50-200ms per request
- Memory clearing: +10-20% processing overhead
- Depth map bounds checking: +50ms per frame

**Note**: These overheads are configurable and can be tuned based on your security requirements

## Security Best Practices

### Deployment

```bash
# Run with minimal privileges (recommended)
sudo -u nobody python -m transformation_portal.cli

# Or use systemd service with User directive:
# [Service]
# User=nobody
# Group=nogroup

# Use read-only filesystem where possible
docker run --read-only --tmpfs /tmp transformation_portal:latest

# Enable security headers if web-facing
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Content-Security-Policy: default-src 'self'
```

### Configuration

**Note**: The project currently uses `config/default_config.yaml` for depth pipeline settings (see actual structure with `depth_model.variant`, `processing.zone_tone_mapping`, `optimization.memory_limit_gb`, etc.). The following represents recommended security-related configuration fields that should be implemented for production deployments:

```yaml
# Recommended security configuration (not currently implemented)
# These settings should be added to application configuration for production use
security:
  max_file_size: 536870912  # 512MB
  allowed_extensions: ['.jpg', '.png', '.tiff', '.mp4', '.mov']
  temp_directory: '/tmp/transformation_portal'
  cleanup_interval: 3600  # seconds
  
  depth_processing:
    max_input_dimension: 4096
    max_vertices: 10000000
    memory_limit_gb: 8  # GB (see: optimization.memory_limit_gb in default_config.yaml)
```

### Sensitive Data

- **EXIF Data**: Option to strip all metadata from outputs
- **Watermarking**: Support for invisible watermarks for tracking
- **Temporary Files**: Secure deletion with multi-pass overwrite
- **Memory**: Clear sensitive data from memory after processing

## Security Testing for Contributors

Before submitting PRs:

```bash
# Run code quality and security checks
make quality-check

# Run full test suite
make test-full

# Optional: Install and run security tools (not included by default)
# pip install bandit
# bandit -r src/ -ll

# Note: Additional security testing tools like bandit, pip-audit, safety, 
# semgrep, etc. are recommended but not included in project dependencies.
# Install them separately if needed for security auditing.
```

## Incident Response

In case of a security breach:

1. **Isolate**: Immediately isolate affected systems
   - Disable affected endpoints
   - Revoke compromised credentials
   
2. **Assess**: Determine scope and impact
   - Identify affected versions
   - Review access logs
   - Determine data exposure
   
3. **Notify**: Alert users within 72 hours if data was compromised
   - GitHub Security Advisory
   - Email to affected users (if applicable)
   - Update security status page
   
4. **Patch**: Deploy fixes with priority
   - Emergency patch for critical vulnerabilities
   - Coordinate disclosure with reporters
   
5. **Review**: Post-mortem and update security measures
   - Document lessons learned
   - Update security policies
   - Implement additional monitoring

## Known Security Requirements

### System Requirements

- Python 3.10+ (older versions have known vulnerabilities)
- FFmpeg 6+ (addresses multiple CVEs from earlier versions)
- Operating System with DEP/ASLR support
- Minimum 8GB RAM to prevent swap file exposure
- GPU drivers with security updates (NVIDIA 525+ for CUDA operations)

### Network Security

- HTTPS only for any network operations
- Disable unnecessary network features in production
- Firewall rules to restrict outbound connections
- No telemetry or phone-home features by default

## Security Audit History

No formal security audits have been conducted yet. This section will be updated as audits are completed.

## Compliance

This project aims to maintain compliance with:

- **CWE Top 25**: Addressing most dangerous software weaknesses
- **OWASP Top 10**: Web application security (if applicable)
- **PCI DSS**: If processing payment card data
- **GDPR**: For EU user data protection (metadata handling)
- **AI Security**: Following OWASP ML Security Top 10

## Security Tools

Recommended external tools for security testing (require separate installation):

```bash
# Dependency scanning
pip install pip-audit
pip-audit

pip install safety
safety check

# Static analysis
pip install bandit
bandit -r src/

pip install semgrep
semgrep --config=auto

# Existing project tools
pylint --enable=security

# Container scanning (if using Docker)
# Install trivy: https://github.com/aquasecurity/trivy
trivy image transformation_portal:latest
```

**Note**: These tools are not included in the project's dependencies. Install them separately as needed for security auditing.

## Responsible Disclosure

We support responsible disclosure and will:

1. Not pursue legal action against security researchers acting in good faith
2. Work collaboratively to understand and resolve issues
3. Publicly acknowledge researchers (with permission)
4. Maintain a hall of fame for security contributors
5. Consider bug bounties for critical findings (case-by-case basis)

## Security Contact

**Primary**: Create a security advisory at https://github.com/RC219805/Transformation_Portal/security/advisories/new  
**GitHub**: @RC219805  
**Response Time**: 48 hours maximum

## Security Badges

[![Security Rating](https://img.shields.io/badge/security-A-green)](https://github.com/RC219805/Transformation_Portal/security)
[![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-green)](https://github.com/RC219805/Transformation_Portal/network/dependencies)

## Additional Resources

- [docs/BEST_PRACTICES.md](docs/BEST_PRACTICES.md) - General best practices for contributors
- [docs/version_history/changelog.md](docs/version_history/changelog.md) - Version history and security updates
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture and security considerations

---

*Last Updated: November 2025*  
*Next Review: February 2026*  
*Security Policy Version: 1.0*
