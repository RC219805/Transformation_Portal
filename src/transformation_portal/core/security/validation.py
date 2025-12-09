"""
Input validation for pipeline security.

Consolidated from lux_depth_v2/hardening/safe_io.py and other sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Input validation error."""
    
    def __init__(self, message: str, path: Optional[Path] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.path = path
        self.details = details or {}


@dataclass
class ValidationResult:
    """Result of input validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    file_type: Optional[str] = None
    size_bytes: Optional[int] = None
    
    def __bool__(self) -> bool:
        """Validation result is True if valid."""
        return self.valid


class InputValidator:
    """
    Unified input validator for all pipelines.
    
    Validates file existence, extensions, size limits, and content.
    """
    
    def __init__(
        self,
        allowed_extensions: Tuple[str, ...] = (".tif", ".tiff", ".jpg", ".jpeg", ".png"),
        max_size_mb: float = 500.0,
        enable_magic_bytes: bool = True
    ):
        """
        Initialize validator.
        
        Args:
            allowed_extensions: Tuple of allowed file extensions
            max_size_mb: Maximum file size in MB
            enable_magic_bytes: Validate file content against extension
        """
        self.allowed_extensions = tuple(ext.lower() for ext in allowed_extensions)
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.enable_magic_bytes = enable_magic_bytes
    
    def validate_file(self, path: Path, strict: bool = True) -> ValidationResult:
        """
        Validate input file.
        
        Args:
            path: Path to file
            strict: If True, raise exception on validation failure
            
        Returns:
            ValidationResult
            
        Raises:
            ValidationError: If strict=True and validation fails
        """
        result = ValidationResult(valid=True)
        path = Path(path)
        
        # Check existence
        if not path.exists():
            result.valid = False
            result.errors.append(f"File does not exist: {path}")
            if strict:
                raise ValidationError("File does not exist", path=path)
            return result
        
        # Check it's a file
        if not path.is_file():
            result.valid = False
            result.errors.append(f"Path is not a file: {path}")
            if strict:
                raise ValidationError("Path is not a file", path=path)
            return result
        
        # Check extension
        if not self._check_extension(path):
            result.valid = False
            result.errors.append(
                f"Extension not allowed: {path.suffix} "
                f"(allowed: {', '.join(self.allowed_extensions)})"
            )
            if strict:
                raise ValidationError(
                    "Extension not allowed",
                    path=path,
                    details={"ext": path.suffix, "allowed": self.allowed_extensions}
                )
            return result
        
        # Check size
        try:
            size_bytes = path.stat().st_size
            result.size_bytes = size_bytes
            
            if size_bytes > self.max_size_bytes:
                result.valid = False
                result.errors.append(
                    f"File exceeds size limit: {size_bytes / (1024*1024):.1f}MB "
                    f"(max: {self.max_size_bytes / (1024*1024):.1f}MB)"
                )
                if strict:
                    raise ValidationError(
                        "File exceeds size limit",
                        path=path,
                        details={"size": size_bytes, "max": self.max_size_bytes}
                    )
                return result
        except Exception as e:
            result.valid = False
            result.errors.append(f"Failed to get file size: {e}")
            if strict:
                raise ValidationError("Failed to get file size", path=path, details={"error": str(e)})
            return result
        
        # Check magic bytes
        if self.enable_magic_bytes:
            file_type = self._detect_file_type(path)
            result.file_type = file_type
            
            if not self._verify_magic_bytes(path, file_type):
                result.warnings.append(
                    f"File extension ({path.suffix}) may not match content (detected: {file_type})"
                )
        
        return result
    
    def _check_extension(self, path: Path) -> bool:
        """Check if file extension is allowed."""
        return path.suffix.lower() in self.allowed_extensions
    
    def _detect_file_type(self, path: Path) -> str:
        """
        Detect file type from magic bytes.
        
        Returns one of: 'tiff', 'jpeg', 'png', 'unknown'
        """
        try:
            with open(path, "rb") as f:
                header = f.read(16)
        except Exception as e:
            logger.debug(f"Failed to read file header: {e}")
            return "unknown"
        
        # TIFF: II*\x00 (little-endian) or MM\x00* (big-endian)
        if len(header) >= 4:
            if header[:4] == b"II*\x00" or header[:4] == b"MM\x00*":
                return "tiff"
        
        # JPEG: FF D8 FF
        if len(header) >= 3:
            if header[:3] == b"\xFF\xD8\xFF":
                return "jpeg"
        
        # PNG: 89 50 4E 47 0D 0A 1A 0A
        if len(header) >= 8:
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                return "png"
        
        return "unknown"
    
    def _verify_magic_bytes(self, path: Path, detected_type: str) -> bool:
        """Verify file extension matches detected type."""
        ext = path.suffix.lower()
        
        # Map extensions to types
        ext_map = {
            ".tif": "tiff",
            ".tiff": "tiff",
            ".jpg": "jpeg",
            ".jpeg": "jpeg",
            ".png": "png",
        }
        
        expected_type = ext_map.get(ext)
        
        # Unknown is acceptable (might be unsupported format)
        if detected_type == "unknown":
            return True
        
        # Check match
        return expected_type == detected_type
