"""Development utilities for the Transformation Portal.

This module provides tools for maintaining code quality, formatting,
and development workflow automation.

Key components:
- Formatting utilities (write_formatted, write_canonical)
- AST canonicalization (canonicalize_code, Canonicalizer)
- AST hashing (compute_ast_hash, are_semantically_equivalent)
- Equivalence indexing (ASTEquivalenceIndex)
- Deduplication (deduplicate_repo, DuplicationReport)
"""

from transformation_portal.dev.formatting import (
    CanonicalFileWriter,
    FormattedFileWriter,
    format_directory,
    format_file,
    write_canonical,
    write_formatted,
)

__all__ = [
    # Formatting
    "format_file",
    "format_directory",
    "write_formatted",
    "write_canonical",
    "FormattedFileWriter",
    "CanonicalFileWriter",
]
