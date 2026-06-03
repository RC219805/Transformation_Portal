# Security Policy

## Supported Versions

The following Transformation Portal release channels are currently supported with security updates:

| Version | Supported          | Notes |
| ------- | ------------------ | ----- |
| main    | :white_check_mark: | Active development branch; security fixes prioritized |
| Latest tagged release | :white_check_mark: | Supported for security updates until superseded by a newer tagged release |
| Older release tags | :x: | Unsupported unless an explicit security advisory or maintenance branch says otherwise |

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
- **Workflow Token Permissions**: All workflows use least-privilege `permissions:` declarations
  - `contents: read` (default) - Read-only repository access
  - `contents: write` - Only for dependency submission and automated PR creation
  - `security-events: write` - CodeQL and security scanning only
  - `pull-requests: write` - AI code review bot only

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
  - All dependencies use version constraints to balance security and compatibility
  - For security-critical deployments, consider strict version pinning (e.g., via lock files)
  - Regular dependency audits via `pip-audit` (governed scanner in CI)
  - Automated security scanning in CI/CD pipeline

- **Recent Security Updates**:

  **March 2026**:
  - **PyTorch CVE-2025-32434** - Critical RCE vulnerability via torch.load()
    - **Supported-lane remediation**: macOS Apple Silicon ML core lock rotates to `torch==2.8.0` / `torchvision==0.23.0`
    - **Retired-lane posture**: Linux and macOS Intel ML lanes are retired unsupported lanes and absent from installable requirements manifests
    - **Defense in depth**: Runtime enforcement of `weights_only=True` remains mandatory for all torch.load() calls
    - **Implementation**: Use `transformation_portal.core.security.torch_security.safe_load()`
  - **Hugging Face `Trainer` advisory GHSA-69w3-r845-3855**
    - **Disposition**: Managed inference paths do not use `transformers.Trainer`, `Seq2SeqTrainer`, `TrainingArguments`, `_load_rng_state`, or training-resume flows
    - **Action**: Dependabot alerts are dismissed as `not_used` with repo search evidence instead of forcing a `transformers` 5.x pre-release upgrade into inference stacks
  - **Pillow>=10.3.0** - Fixed CVE-2024-28219 (buffer overflow vulnerability)
  - **cryptography==47.0.0** - Current governed lock; includes the CVE-2026-26007 / GHSA-r6ph-v2qm-q3c2 SECT subgroup validation fix from 46.0.5
  - **black==26.3.1** - Fixed arbitrary file writes from unsanitized cache names
  - **Pygments==2.20.0** - Fixed CVE-2026-4539; the temporary pip-audit exception is retired
  - **Starlette==1.0.1** - Fixed CVE-2026-48710 / PYSEC-2026-161 BadHost request URL construction weakness

  **January 2026**:
  - **protobuf 6.34.0** - Fixed CVE-2026-0994 / GHSA-7gcm-g887-7qv7 (Dependabot #69)
  - **Workflow Hardening** - Stricter token permissions across all GitHub Actions workflows
  - **Quality Gate** - Fixed duplicate permissions block (aa555e0a)

- **Security vs Determinism Policy**:
  - Transformation Portal prioritizes **reproducibility** over latest versions (ADR-032)
  - Supported-lane security fixes prefer **controlled baseline rotations** over opportunistic broad upgrades
  - Version upgrades only occur during **controlled baseline rotations**
  - All torch.load() calls MUST use `weights_only=True` parameter

- **Known Vulnerabilities** (Mitigated):
  - Supported Apple Silicon lane runs on torch `2.8.0` / torchvision `0.23.0`
  - Linux and macOS Intel ML lanes are retired unsupported lanes and are absent from installable `requirements/*.in` / `requirements/*.txt` manifests
  - Historical retired-lane details live in `docs/governance/RETIRED_ML_LOCK_LANES_2026-04-30.md` and must not drive supported-lane remediation
  - All model loading uses safe_load() wrapper or explicit weights_only=True
  - Pygments CVE-2026-4539 is remediated by the governed `pygments==2.20.0` lock baseline; CI must not keep stale scanner exceptions for this CVE
  - Pillow: Critical for image parsing vulnerabilities
  - NumPy: Monitor for numerical computation exploits

- **Temporary CVE Exceptions**:
  - None active. New exceptions require an explicit expiry condition, tracked upstream issue, and matching CI/test coverage.

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
  max_file_size: 524288000  # 500 MiB (500 * 1024 * 1024 bytes)
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

# Install governed security tools from requirements/security.txt
pip install -r requirements/security.txt

# Run static security analysis
bandit -r src/ -ll

# Run dependency vulnerability scan
pip-audit
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

- Python 3.11+ (matches the package `requires-python` floor and CI support matrix)
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
- **PCI DSS**: Not applicable (no payment processing features)
- **GDPR**: For EU user data protection (metadata handling)
- **AI Security**: Following OWASP ML Security Top 10

## Security Tools

Security scanning tools are governed in CI via `requirements/security.txt`:

```bash
# Install governed security tools (bandit, pip-audit)
pip install -r requirements/security.txt

# Run dependency vulnerability scan
pip-audit

# Run static security analysis
bandit -r src/

# Additional optional tools (not governed)
pip install semgrep
semgrep --config=auto

# Existing project tools
pylint --enable=security

# Container scanning (if using Docker)
# Install trivy: https://github.com/aquasecurity/trivy
trivy image transformation_portal:latest
```

**Note**: `pip-audit` and `bandit` are the governed security tools installed from `requirements/security.txt` in CI. Additional tools like semgrep can be installed separately as needed for security auditing.

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

## Additional Resources

- [CONTRIBUTING.md](CONTRIBUTING.md) - Current contributor workflow and validation expectations
- [CHANGELOG.md](CHANGELOG.md) - Root change history with security-relevant entries
- [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) - System architecture and security considerations

---

*Last Updated: 2026-06-03*
*Next Review: 2026-09-03*
*Security Policy Version: 1.2*
