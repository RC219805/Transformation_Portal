from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Sequence, Optional

from .exceptions import InputValidationError
from .policy import HardeningPolicy

# Platform Core integration for unified input validation
try:
    from transformation_portal.core.security.validation import (
        InputValidator as CoreInputValidator,
        ValidationError as CoreValidationError,
        ValidationResult
    )
    CORE_VALIDATION_AVAILABLE = True
except ImportError:
    CORE_VALIDATION_AVAILABLE = False
    CoreInputValidator = None
    CoreValidationError = None
    ValidationResult = None


def sniff_image_type(path: Path) -> str:
    """
    Lightweight magic-byte sniffing to prevent obvious wrong-file attacks.
    Returns one of: 'tiff', 'jpeg', 'png', 'unknown'
    """
    try:
        with path.open("rb") as f:
            head = f.read(16)
    except Exception as e:
        raise InputValidationError("Unable to read file header", path=str(path), details={"error": str(e)})

    # TIFF: II*\x00 or MM\x00*
    if len(head) >= 4 and (head[:4] == b"II*\x00" or head[:4] == b"MM\x00*"):
        return "tiff"
    # JPEG: FF D8 FF
    if len(head) >= 3 and head[:3] == b"\xFF\xD8\xFF":
        return "jpeg"
    # PNG signature
    if len(head) >= 8 and head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return "unknown"


def is_allowed_extension(path: Path, allowed_exts: Sequence[str]) -> bool:
    ext = path.suffix.lower()
    allowed = tuple(e.lower() for e in allowed_exts)
    return ext in allowed


def validate_image_file(path: Path, policy: HardeningPolicy, use_core: bool = True) -> None:
    """
    Enforce existence, extension allowlist, size cap, and magic-byte sniffing.
    
    Args:
        path: Path to image file
        policy: HardeningPolicy with validation rules
        use_core: If True and Platform Core available, use unified validator first
    
    Notes:
        - When use_core=True and core available, performs dual validation
        - Core validation provides baseline checks
        - Local validation adds lux_depth_v2-specific rules
        - Backward compatible with existing behavior
    """
    p = Path(path)
    
    # Phase 1: Use Platform Core validator if available (baseline checks)
    if use_core and CORE_VALIDATION_AVAILABLE:
        allowed_exts = policy.normalize_allowed_exts()
        max_size_mb = policy.max_input_bytes / (1024 * 1024)
        
        core_validator = CoreInputValidator(
            allowed_extensions=tuple(allowed_exts),
            max_size_mb=max_size_mb,
            enable_magic_bytes=True
        )
        
        try:
            result = core_validator.validate_file(p, strict=True)
        except CoreValidationError as e:
            # Convert core validation error to local format
            raise InputValidationError(
                str(e),
                path=str(p),
                details=e.details if hasattr(e, 'details') else {}
            )
    
    # Phase 2: Local validation (legacy + lux_depth_v2-specific rules)
    if not p.exists():
        raise InputValidationError("Input does not exist", path=str(p))
    if not p.is_file():
        raise InputValidationError("Input is not a file", path=str(p))

    policy.assert_input_under_allowed_roots(p)

    # Extension allowlist
    allowed_exts = policy.normalize_allowed_exts()
    if not is_allowed_extension(p, allowed_exts):
        raise InputValidationError(
            "Input extension not allowed",
            path=str(p),
            details={"ext": p.suffix.lower(), "allowed": list(allowed_exts)},
        )

    # Size cap
    try:
        size = p.stat().st_size
    except Exception as e:
        raise InputValidationError("Unable to stat input", path=str(p), details={"error": str(e)})

    if size > policy.max_input_bytes:
        raise InputValidationError(
            "Input exceeds max_input_bytes",
            path=str(p),
            details={"bytes": size, "max_bytes": policy.max_input_bytes},
        )

    # Magic bytes sniff (lux_depth_v2-specific implementation)
    kind = sniff_image_type(p)
    ext = p.suffix.lower()
    if ext in (".tif", ".tiff") and kind not in ("tiff", "unknown"):
        raise InputValidationError(
            "Input extension indicates TIFF but magic bytes disagree",
            path=str(p),
            details={"ext": ext, "sniffed": kind},
        )
    if ext in (".jpg", ".jpeg") and kind not in ("jpeg", "unknown"):
        raise InputValidationError(
            "Input extension indicates JPEG but magic bytes disagree",
            path=str(p),
            details={"ext": ext, "sniffed": kind},
        )
    if ext == ".png" and kind not in ("png", "unknown"):
        raise InputValidationError(
            "Input extension indicates PNG but magic bytes disagree",
            path=str(p),
            details={"ext": ext, "sniffed": kind},
        )

    # Optional MP cap (cheap estimate; real decode happens later)
    # Megapixel cap is not enforced here; real decode happens later if available.
    # Kept here for API completeness.


def safe_mkdir(path: Path, mode: int = 0o750) -> None:
    """
    Create directory with restrictive permissions where supported.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    try:
        p.chmod(mode)
    except Exception:
        # Windows / restricted FS: ignore
        pass


def safe_resolve_under(path: Path, root: Path) -> Path:
    """
    Resolve path and ensure it stays under root. Protects against traversal and symlink escape.
    """
    r = Path(root).expanduser().resolve()
    p = Path(path).expanduser().resolve()
    if not _is_relative_to(p, r):
        raise InputValidationError(
            "Path escapes allowed root",
            path=str(p),
            details={"root": str(r)},
        )
    return p


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def redact_path(s: str) -> str:
    """
    Conservative path redaction for logs. Keeps basename, strips parent dirs.
    """
    try:
        return Path(s).name
    except Exception:
        return "<redacted>"


def to_jsonable(obj):
    """
    Stable conversion for hashing/stamping:
    - dataclasses -> dict
    - Paths -> str
    - Enums -> name/value (handled by str fallback)
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return str(obj)
