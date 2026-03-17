"""
Core Security Module

Consolidated security validation and sanitization from:
- lux_depth_v2/hardening/
- src/transformation_portal/hardening/

Provides unified input validation, path traversal protection,
secure file handling, cryptographic signing, TLS, and tenant isolation.
"""

from .fs_guard import FSContext, FSGuard, FSPolicyError, get_fs_guard, set_fs_guard
from .model_lock import (
    ModelLockError,
    is_model_lock_strict_enabled,
    is_pinned_revision,
    load_model_lock_manifest,
    manifest_revision_for_repo,
    model_lock_manifest_path,
    resolve_model_lock_revision,
)
from .path import PathValidator, is_safe_path, safe_resolve_path
from .path_safety import (
    PathSafetyError,
    safe_cas_path,
    safe_join_file,
    safe_join_subpath,
    validate_safe_name,
    validate_sha256,
)
from .sanitization import SanitizationPolicy, sanitize_filename, validate_input_file
from .serialization import RestrictedUnpickler, safe_pickle_load
from .validation import InputValidator, ValidationError, ValidationResult

# Conditional imports for optional dependencies
try:
    from .signing import (
        CertificateSigner,
        CertificateVerifier,
        SignedCertificate,
        SigningError,
        generate_ed25519_keypair,
    )

    _SIGNING_AVAILABLE = True
except ImportError:
    _SIGNING_AVAILABLE = False

try:
    from .tls import (
        ThreadedTLSServer,
        TLSError,
        TLSTCPServer,
        create_client_ssl_context,
        create_server_ssl_context,
        create_tls_connection,
    )

    _TLS_AVAILABLE = True
except ImportError:
    _TLS_AVAILABLE = False

from .task_signing import SignedTask, TaskSigner, TaskSigningError, TaskVerifier
from .tenant import (
    TenantAwareFSGuard,
    TenantContext,
    TenantError,
    TenantManager,
    TenantPolicy,
    create_tenant_sandbox,
)

__all__ = [
    # Validation
    "InputValidator",
    "ValidationResult",
    "ValidationError",
    # Model Lock
    "ModelLockError",
    "load_model_lock_manifest",
    "manifest_revision_for_repo",
    "resolve_model_lock_revision",
    "is_pinned_revision",
    "is_model_lock_strict_enabled",
    "model_lock_manifest_path",
    # Path validation (legacy)
    "PathValidator",
    "safe_resolve_path",
    "is_safe_path",
    # Path safety (CodeQL-compliant)
    "PathSafetyError",
    "validate_safe_name",
    "validate_sha256",
    "safe_join_file",
    "safe_join_subpath",
    "safe_cas_path",
    # Zero-trust filesystem guard
    "FSGuard",
    "FSContext",
    "FSPolicyError",
    "get_fs_guard",
    "set_fs_guard",
    # Sanitization
    "SanitizationPolicy",
    "sanitize_filename",
    "validate_input_file",
    # Serialization
    "RestrictedUnpickler",
    "safe_pickle_load",
    # Signing (if available)
    "SigningError",
    "SignedCertificate",
    "CertificateSigner",
    "CertificateVerifier",
    "generate_ed25519_keypair",
    # TLS (if available)
    "TLSError",
    "TLSTCPServer",
    "ThreadedTLSServer",
    "create_server_ssl_context",
    "create_client_ssl_context",
    "create_tls_connection",
    # Task signing
    "TaskSigningError",
    "SignedTask",
    "TaskSigner",
    "TaskVerifier",
    # Multi-tenant
    "TenantError",
    "TenantContext",
    "TenantPolicy",
    "TenantManager",
    "TenantAwareFSGuard",
    "create_tenant_sandbox",
]
