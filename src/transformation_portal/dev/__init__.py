"""Development utilities for the Transformation Portal.

This module provides tools for maintaining code quality, formatting,
and development workflow automation.

Key components:
- Formatting utilities (write_formatted, write_canonical)
- AST canonicalization (canonicalize_code, Canonicalizer)
- AST hashing (compute_ast_hash, are_semantically_equivalent)
- Equivalence indexing (ASTEquivalenceIndex)
- Deduplication (deduplicate_repo, DuplicationReport)
- Test markers (add_pytest_import, add_pytestmark, process_file)
"""

from transformation_portal.dev.formatting import (
    CanonicalFileWriter,
    FormattedFileWriter,
    format_directory,
    format_file,
    write_canonical,
    write_formatted,
)
from transformation_portal.dev.test_markers import (
    DIRECTORY_MARKERS,
    IMPORT_PYTEST_PATTERN,
    MODULE_MARKER_PATTERN,
    PYTESTMARK_PATTERN,
    SKIP_DIRECTORIES,
    add_pytest_import,
    add_pytestmark,
    get_directory_marker,
    has_class_or_function_markers,
    has_existing_module_markers,
    has_test_functions,
    process_file,
)

__all__ = [
    # Formatting
    "format_file",
    "format_directory",
    "write_formatted",
    "write_canonical",
    "FormattedFileWriter",
    "CanonicalFileWriter",
    # Test markers
    "DIRECTORY_MARKERS",
    "SKIP_DIRECTORIES",
    "PYTESTMARK_PATTERN",
    "MODULE_MARKER_PATTERN",
    "IMPORT_PYTEST_PATTERN",
    "has_test_functions",
    "has_existing_module_markers",
    "has_class_or_function_markers",
    "get_directory_marker",
    "add_pytest_import",
    "add_pytestmark",
    "process_file",
]
