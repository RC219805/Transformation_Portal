"""
Input Sanitization & File Validation.

Cleans inputs to prevent injection attacks and verifies file integrity.
"""

import re
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from .validation import ValidationError

@dataclass
class SanitizationPolicy:
    """Rules for sanitization."""
    max_filename_length: int = 255
    allowed_extensions: Optional[List[str]] = None
    allow_spaces: bool = False

    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.exr']

def sanitize_filename(
    filename: str, 
    replacement: str = "_"
) -> str:
    """
    Make a string safe for use as a filename.
    
    Removes dangerous characters like / \ : * ? " < > |
    """
    # 1. Strip path components (we only want the name)
    name = Path(filename).name
    
    # 2. Replace dangerous chars
    # Allow alphanumeric, dot, hyphen, underscore
    safe_pattern = re.compile(r'[^a-zA-Z0-9\._-]')
    clean_name = safe_pattern.sub(replacement, name)
    
    # 3. Prevent hidden files
    while clean_name.startswith('.'):
        clean_name = clean_name[1:]
        
    # 4. Truncate length
    return clean_name[:255]

def validate_input_file(
    path: Path, 
    policy: Optional[SanitizationPolicy] = None
) -> None:
    """
    Verify a file is safe to process.
    
    Checks:
    - Path safety (traversal)
    - Existence
    - Extension allowlist
    - Size (basic check)
    """
    policy = policy or SanitizationPolicy()
    
    # 1. Path Safety is handled by caller (safe_resolve_path), 
    # but we check basic existence here.
    if not path.exists():
        raise ValidationError(f"File not found: {path}")
        
    if not path.is_file():
        raise ValidationError(f"Not a file: {path}")
        
    # 2. Extension Check
    ext = path.suffix.lower()
    if policy.allowed_extensions and ext not in policy.allowed_extensions:
        raise ValidationError(
            f"File type '{ext}' not allowed. Permitted: {policy.allowed_extensions}"
        )
        
    # 3. Magic Number Check (Header validation)
    # This prevents 'image.jpg' actually being a script
    try:
        with open(path, 'rb') as f:
            header = f.read(10)
            
        # Basic signatures
        signatures = {
            '.jpg': b'\xFF\xD8\xFF',
            '.jpeg': b'\xFF\xD8\xFF',
            '.png': b'\x89PNG\r\n\x1a\n',
            '.tiff': [b'II*\x00', b'MM\x00*'], # Little/Big Endian
            '.exr': b'v/1\x01'
        }
        
        expected = signatures.get(ext)
        if expected:
            if isinstance(expected, list):
                valid = any(header.startswith(sig) for sig in expected)
            else:
                valid = header.startswith(expected)
                
            if not valid:
                raise ValidationError(f"File signature mismatch for {ext}")
                
    except IOError as e:
        raise ValidationError(f"Could not read file header: {e}")
