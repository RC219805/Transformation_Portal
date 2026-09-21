# Security Policy

## Supported Versions

The following Transformation Portal release channels are currently supported with security updates:

| Version | Supported          | Notes |
| ------- | ------------------ | ----- |
| main    | :white_check_mark: | Active development branch; security fixes prioritized |
| Latest semantic product release tag | :white_check_mark: | Supported for security updates until superseded by a newer semantic product release tag |
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
- **Branch Protection**: Repository policy and remotely configured required
  checks are distinct. Verify live settings before a merge; see
  [branch protection setup](docs/ci/BRANCH_PROTECTION_SETUP.md).
- **Workflow Token Permissions**: Workflow/job ``permissions`` declarations own
  token scope. Inspect the current workflow YAML before changing write access;
  this policy is not a complete inventory of jobs with write permissions.

## Security Considerations

### Input Validation

Given our image/video processing nature, special attention is required for:

- **File Upload Security**:
  - Bounded request and upload sizes. Current backend defaults are
    `TP_MAX_REQUEST_BYTES=1048576` and
    `TP_PORTAL_MAX_UPLOAD_REQUEST_BYTES=1048576`; raise them deliberately per
    deployment instead of assuming large media uploads are accepted by default.
  - Multipart guardrails: `TP_PORTAL_UPLOAD_MAX_FILES=256`,
    `TP_PORTAL_UPLOAD_MAX_FIELDS=32`, and
    `TP_PORTAL_UPLOAD_MAX_PART_BYTES=1048576` by default.
  - Strict MIME type validation
  - Magic number verification for file formats
  - Filename sanitization to prevent path traversal

- **TIFF Processing**: Validate supported formats and dimensions at the consuming
  boundary. Decoder limits and memory behavior vary by library and execution
  path; this repository does not enforce one universal image-dimension ceiling.

### Depth Map Processing

Input dimensions, point counts, GPU memory use, and intermediate storage depend
on the selected pipeline and runtime. Set deployment resource budgets and verify
them on representative inputs; there is no universal 8 GB VRAM cap or 10-million
vertex ceiling enforced across these paths. Removing temporary files does not
guarantee secure erasure from storage, backups, or process memory.

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
  - Install through the governed Make targets and exact target-owned locks in
    [requirements/README.md](requirements/README.md)
  - Do not substitute ad-hoc package upgrades for controlled lock regeneration
  - Regular dependency audits via `pip-audit` (governed scanner in CI)
  - Automated security scanning in CI/CD pipeline

- **Recorded Security Updates**:

  These entries retain historical remediation context. Current installed
  versions come from the applicable lock, and current alert status requires a
  fresh scanner or service check; this chronology is not an open-alert inventory.

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
  - **cryptography==50.0.0** - Current governed lock; includes the CVE-2026-69247 security fix
  - **black==26.3.1** - Fixed arbitrary file writes from unsanitized cache names
  - **Pygments==2.21.0** - Fixed CVE-2026-4539; the temporary pip-audit exception is retired
  - **Starlette==1.6.0** - Retains the governed CVE-2026-48710 / PYSEC-2026-161,
    StaticFiles, HTTPEndpoint, and form-parsing fixes; adds Starlette 1.5.1
    `FileResponse` range hardening and the 1.6.0 application/route
    `max_body_size` control

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
  - Supported Apple Silicon lane runs on torch `2.13.0` / torchvision `0.28.0`
  - Linux and macOS Intel ML lanes are retired unsupported lanes and are absent from installable `requirements/*.in` / `requirements/*.txt` manifests
  - Historical retired-lane details live in `docs/governance/RETIRED_ML_LOCK_LANES_2026-04-30.md` and must not drive supported-lane remediation
  - All model loading uses safe_load() wrapper or explicit weights_only=True
  - Pygments CVE-2026-4539 is remediated by the governed `pygments==2.21.0` lock baseline; CI must not keep stale scanner exceptions for this CVE
  - Pillow: Critical for image parsing vulnerabilities
  - NumPy: Monitor for numerical computation exploits

- **Temporary CVE Exceptions**:
  - None active. New exceptions require an explicit expiry condition, tracked upstream issue, and matching CI/test coverage.

### Open Dependency Risk: Accelerate Checkpoint Indexes

As reviewed on 2026-09-21 UTC, the supported Darwin arm64 ML lock contains
`accelerate==1.14.0`, affected by
[GHSA-4j2p-28q2-5m79](https://github.com/advisories/GHSA-4j2p-28q2-5m79).
Its sharded checkpoint loaders accept paths outside the checkpoint directory
and can block while opening a named pipe. Both dependency alerts remain open;
this is not a scanner exception or a completed remediation.

The advisory lists no patched release. Source inspection of
[Accelerate 1.15.0](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/modeling.py#L1936-L1944)
also shows shard entries joined without containment or regular-file validation.
Do not treat a version outside the advisory's listed range as proof of a fix.

The pinned optional ML closure uses Transformers `5.10.4` and Diffusers
`0.40.0`. Source review of those installed packages found no references to
`load_checkpoint_in_model` or `load_checkpoint_and_dispatch`; repository
production code does not call either affected API. Legitimate Accelerate
device mapping, dispatch and offload remain supported. The required
`Dependency Security` job now runs
`python scripts/validation/check_accelerate_loading.py`, which rejects direct
and aliased imports, statically resolved attribute aliases, and literal
`getattr` references to the affected APIs. Alias provenance is retained
conservatively across scopes and reimports so an unrelated name reuse cannot
hide a prohibited loader reference. This source gate does not inspect
third-party packages or dynamically computed Python expressions; repeat the
reachability review when the optional ML closure changes.

Repository-owned DA3 identity and manifest-local LLaVA resolution share bounded
shard-index parsing. It rejects duplicate keys, invalid/oversized maps, and
absolute, parent-traversing or Windows-style shard paths. The LLaVA boundary
also rejects missing/non-regular shards and symlink targets outside its model
snapshot or standard HF blob directory. Nonblocking descriptor opens reject
FIFOs before reads, including DA3 identity reads/hashes. Manifest resolution
requires a full commit SHA. Normal HF blob symlinks remain supported, and these
checks neither download alternate model variants nor copy entire model trees.
They do not isolate a model cache from another process running as the same
user: keep cache/runtime directories owner-controlled. Index validation is
not an immutable snapshot of the later model load and must not be used to
authorize the affected Accelerate APIs.

Until an upstream fix is verified, load only reviewed model checkpoints from
trusted sources in the optional ML environment. Do not pass untrusted local
checkpoint directories or shard indexes to Accelerate or integrations that
delegate checkpoint loading to it. Immutable revisions and `weights_only=True`
alone do not validate shard paths. Keep inference under an unprivileged account
with access limited to the intended model and asset directories.

Remediation requires verifying containment and non-regular-file rejection in
the released loader, regenerating the target-owned lock on native Darwin arm64,
and rerunning the ML lock and dependency-security checks. Keep the alerts open
until that evidence exists.

### Required Dependency Security Check

Main branch protection requires both `CI Gate` and `Dependency Security`, with
strict up-to-date checking and GitHub Actions app binding (`15368`), verified
2026-09-13 UTC. The security job runs on every pull request targeting `main`,
without path, draft, fork, or bot filters. It needs no repository secrets,
has read-only contents permission, disables persisted checkout credentials,
and has a 20-minute timeout. Scanner errors and findings fail the check;
pending fork approval cannot satisfy branch protection. Source contract tests
protect those trigger and failure semantics.

The job audits its installed core/CI environment. The optional native ML lock
is still tracked by Dependabot; requiring this job does not imply that every
optional environment has zero advisories or close the two Accelerate alerts.

### API Security

If exposing Transformation Portal as a service:

- **Authentication**: Protected `/v1` endpoints enforce API-key auth by
  default. Set `TP_API_KEY`, keep `TP_ENFORCE_JOB_API_KEY=true`, and only
  override `TP_API_KEY_HEADER` when the proxy/client contract requires it.
- **Rate Limiting**:
  - Default: 60 requests/minute per client via `TP_RATE_LIMIT_PER_MINUTE`
  - Admission cap: 4 concurrent jobs via `TP_MAX_CONCURRENT_JOBS`
- **Input Sanitization**: All user inputs must be validated
- **Output Filtering**: Ensure no metadata leakage in processed files

Signed tenant isolation has a service-backed regression at
`tests/orchestrator/test_tenant_dispatch_services.py`. Run
`PYTHONPATH=src:. ./.venv/bin/pytest tests/orchestrator/test_tenant_dispatch_services.py -q`
with `TP_TENANT_TEST_DATABASE_URL` set to a dedicated migrated test database
and `TP_DISPATCH_TEST_REDIS_URL` set to an isolated Redis service. The test
uses real archive inputs and an independent worker, verifies committed artifact
bytes, and exercises cross-tenant read, event, cancel and deletion denial. It
does not truncate the database; a skip without those services is not service
validation evidence.

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

Measure validation, checksum, and model-loading overhead for the target runtime
and input set. No fixed latency or percentage overhead has been established here.
Keep security-sensitive controls enabled while diagnosing performance; changes
need measured evidence and the applicable governance review.

## Security Best Practices

### Deployment

Run the backend under the deployment's unprivileged service account from the
repository root. Supply `TP_API_KEY` through that service's secret environment
before startup; `make run-backend-local-noreload` checks that it is set.

```bash
make run-backend-local-noreload
# Equivalent launch after loading the same environment:
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Configure least-privilege writable input/output/temp paths for the selected
workflow and keep the backend behind the managed front door. Use the maintained
[frontdoor guide](docs/guides/PORTAL_SECURE_FRONTDOOR_QUICKSTART.md) for HTTPS,
headers, sessions, and proxy authentication. A generic read-only container command
alone does not establish usable runtime or artifact storage.

### Configuration

Security-sensitive runtime controls are environment-backed and documented in
the root [`.env.example`](.env.example). Depth-pipeline defaults still live in
`config/default_config.yaml`, but API protection, request limits, filesystem
boundaries, and artifact-store controls are backend environment contracts:

```bash
TP_API_KEY="replace-with-strong-token"
TP_ENFORCE_JOB_API_KEY=true
TP_MAX_REQUEST_BYTES=1048576
TP_PORTAL_MAX_UPLOAD_REQUEST_BYTES=1048576
TP_RATE_LIMIT_PER_MINUTE=60
TP_MAX_CONCURRENT_JOBS=4
TP_ALLOWED_INPUT_ROOTS=.
TP_ALLOWED_OUTPUT_ROOTS=.
```

### Sensitive Data

- **EXIF Data**: Verify metadata behavior for the selected output writer;
  stripping metadata is not a universal promise across all artifact types.
- **Watermarking**: Support for invisible watermarks for tracking
- **Temporary Files and Memory**: Apply deployment retention and storage controls.
  The repository does not promise multi-pass secure deletion or complete memory
  erasure. Inspect intermediates and manifests before sharing artifacts.

## Security Testing for Contributors

Before submitting PRs:

```bash
# Run code quality and security checks
make quality-check

# Run full test suite
make test-full

# Install governed security tools from requirements/security.txt
.venv/bin/python -m pip install -r requirements/security.txt

# Run static security analysis
.venv/bin/bandit -r src/ -ll

# Run dependency vulnerability scan
.venv/bin/pip-audit
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
- Size memory for the selected workload and configure swap/storage protection
  at the operating-system level; a RAM minimum does not prevent swap exposure.
- GPU drivers with security updates (NVIDIA 525+ for CUDA operations)

### Network Security

- HTTPS only for any network operations
- Disable unnecessary network features in production
- Firewall rules to restrict outbound connections
- No telemetry or phone-home features by default

## Security Audit History

No completed formal security audit report is linked here. Scheduled dates and
passing automated scans do not establish a completed independent audit. Record
the report, scope, exact revision, and remediation evidence when one is completed.

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
.venv/bin/python -m pip install -r requirements/security.txt

# Run dependency vulnerability scan
.venv/bin/pip-audit

# Run static security analysis
.venv/bin/bandit -r src/

# Additional optional tools (install into an isolated security tooling env)
# python -m pip install semgrep
# semgrep --config=auto

# Existing project tools
make lint-parity

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

*Last source review: 2026-09-12*
*Previous scheduled review: 2026-09-03; a schedule alone is not completion evidence.*
*Security Policy Version: 1.2*
