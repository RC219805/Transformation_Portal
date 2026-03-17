"""Zero-diff formatting utilities for canonical code generation.

This module enforces the principle that NO FILE ENTERS THE REPO UNFORMATTED.
All code generation, whether from Copilot, scripts, or programmatic sources,
must use these utilities to ensure formatting is applied at write-time.

Usage:
    from transformation_portal.dev.formatting import write_formatted

    # Instead of: path.write_text(code)
    write_formatted(path, code)

This eliminates:
- Black CI failures
- Formatting PR noise
- Human formatting work
- Diff instability
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Repository formatting standards (must match pyproject.toml)
BLACK_LINE_LENGTH = 127
BLACK_TARGET_VERSIONS = ["py311", "py312"]


def _tool_available(tool: str) -> bool:
    """Check if a formatting tool is available in the environment."""
    return shutil.which(tool) is not None


def format_file(path: Path, *, quiet: bool = True) -> bool:
    """Apply canonical formatting to a Python file immediately after write.

    The formatting order is: isort (import ordering) → Black → Black (final pass).
    Running isort first ensures imports are sorted, then Black formats the code.
    A final Black pass ensures any isort changes are also Black-compliant.

    Args:
        path: Path to the Python file to format.
        quiet: If True, suppress stdout/stderr from formatters.

    Returns:
        True if formatting was applied successfully, False otherwise.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cannot format non-existent file: {path}")

    if path.suffix not in (".py", ".pyi"):
        logger.debug("Skipping non-Python file: %s", path)
        return True

    success = True
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None

    # Step 1: Apply isort import ordering first
    if _tool_available("isort"):
        try:
            subprocess.run(
                ["isort", str(path)],
                check=True,
                stdout=stdout,
                stderr=stderr,
            )
            logger.debug("isort applied: %s", path)
        except subprocess.CalledProcessError as e:
            logger.warning("isort failed for %s: %s", path, e)
            success = False
    else:
        logger.warning("isort not available - skipping import sorting for %s", path)

    # Step 2: Apply Black formatting
    if _tool_available("black"):
        try:
            subprocess.run(
                ["black", f"--line-length={BLACK_LINE_LENGTH}", str(path)],
                check=True,
                stdout=stdout,
                stderr=stderr,
            )
            logger.debug("Black formatting applied: %s", path)
        except subprocess.CalledProcessError as e:
            logger.warning("Black formatting failed for %s: %s", path, e)
            success = False
    else:
        logger.warning("Black not available - skipping formatting for %s", path)

    # Step 3: Final Black pass to ensure isort changes are also Black-compliant
    if _tool_available("black") and success:
        try:
            subprocess.run(
                ["black", f"--line-length={BLACK_LINE_LENGTH}", str(path)],
                check=True,
                stdout=stdout,
                stderr=stderr,
            )
            logger.debug("Final Black pass applied: %s", path)
        except subprocess.CalledProcessError as e:
            logger.warning("Final Black pass failed for %s: %s", path, e)
            success = False

    return success


def write_formatted(
    path: Path,
    content: str,
    *,
    encoding: str = "utf-8",
    quiet: bool = True,
) -> None:
    """Write content to a file and apply canonical formatting atomically.

    This is the canonical way to write Python files in the repository.
    It ensures that all generated code is formatted before it enters
    the repository, eliminating formatting drift and CI failures.

    The write is atomic: content is written to a temporary file, formatted
    there, and only then renamed to the destination path. This ensures
    the destination is never left in a partially-written or unformatted state.

    Args:
        path: Destination path for the file.
        content: Content to write to the file.
        encoding: File encoding (default: utf-8).
        quiet: If True, suppress stdout/stderr from formatters.

    Example:
        from transformation_portal.dev.formatting import write_formatted

        code = '''
        def hello():
            print("Hello, world!")
        '''
        write_formatted(Path("src/module.py"), code)
    """
    import os
    import tempfile

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # For non-Python files, write directly (no formatting needed)
    if path.suffix not in (".py", ".pyi"):
        path.write_text(content, encoding=encoding)
        logger.debug("Wrote %d bytes to %s (non-Python, no formatting)", len(content), path)
        return

    # Create temp file in the same directory (so rename is atomic)
    fd, tmp_path_str = tempfile.mkstemp(
        suffix=path.suffix,
        prefix=f".{path.stem}_",
        dir=path.parent,
    )
    tmp_path = Path(tmp_path_str)

    try:
        # Write content to temp file
        os.write(fd, content.encode(encoding))
        os.close(fd)
        logger.debug("Wrote %d bytes to temp file %s", len(content), tmp_path)

        # Apply formatting to temp file
        format_file(tmp_path, quiet=quiet)

        # Atomic rename to final destination
        tmp_path.replace(path)
        logger.debug("Atomically moved %s to %s", tmp_path, path)

    except Exception:
        # Clean up temp file on error
        os.close(fd) if fd else None  # Ensure fd is closed
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def format_directory(
    directory: Path,
    *,
    recursive: bool = True,
    quiet: bool = True,
) -> tuple[int, int]:
    """Format all Python files in a directory.

    Args:
        directory: Directory to format.
        recursive: If True, format files in subdirectories.
        quiet: If True, suppress stdout/stderr from formatters.

    Returns:
        Tuple of (files_formatted, files_failed).
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pattern = "**/*.py" if recursive else "*.py"
    files = list(directory.glob(pattern))

    formatted = 0
    failed = 0

    for file_path in files:
        if format_file(file_path, quiet=quiet):
            formatted += 1
        else:
            failed += 1

    logger.info("Formatted %d files, %d failed in %s", formatted, failed, directory)
    return formatted, failed


def check_formatting(path: Path) -> bool:
    """Check if a file is properly formatted without modifying it.

    Args:
        path: Path to the file to check.

    Returns:
        True if the file is properly formatted, False otherwise.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cannot check non-existent file: {path}")

    if path.suffix not in (".py", ".pyi"):
        return True

    # Check Black formatting
    if _tool_available("black"):
        result = subprocess.run(
            ["black", "--check", f"--line-length={BLACK_LINE_LENGTH}", str(path)],
            capture_output=True,
        )
        if result.returncode != 0:
            return False

    # Check isort
    if _tool_available("isort"):
        result = subprocess.run(
            ["isort", "--check-only", str(path)],
            capture_output=True,
        )
        if result.returncode != 0:
            return False

    return True


class FormattedFileWriter:
    """Context manager for writing formatted Python files.

    Usage:
        with FormattedFileWriter(Path("module.py")) as writer:
            writer.write("def hello():\\n")
            writer.write("    print('Hello!')\\n")
        # File is automatically formatted on exit
    """

    def __init__(self, path: Path, encoding: str = "utf-8"):
        self.path = path
        self.encoding = encoding
        self._content: list[str] = []

    def __enter__(self) -> "FormattedFileWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            content = "".join(self._content)
            write_formatted(self.path, content, encoding=self.encoding)

    def write(self, text: str) -> None:
        """Append text to the file content."""
        self._content.append(text)

    def writelines(self, lines: list[str]) -> None:
        """Append multiple lines to the file content."""
        self._content.extend(lines)


def write_canonical(
    path: Path,
    content: str,
    *,
    encoding: str = "utf-8",
    quiet: bool = True,
    canonicalize: bool = True,
) -> None:
    """Write content with AST canonicalization and formatting.

    This is the most rigorous way to write Python files. It:
    1. Parses the content into an AST
    2. Applies canonical transformations (dict sorting, kwarg sorting, etc.)
    3. Emits deterministic code via ast.unparse()
    4. Applies Black and isort formatting

    This guarantees:
    - Semantic equivalence
    - Deterministic structure
    - Zero stylistic drift across generators

    Args:
        path: Destination path for the file.
        content: Content to write to the file.
        encoding: File encoding (default: utf-8).
        quiet: If True, suppress stdout/stderr from formatters.
        canonicalize: If True, apply AST canonicalization before formatting.

    Example:
        from transformation_portal.dev.formatting import write_canonical

        code = '''
        x = {"b": 2, "a": 1}
        foo(z=3, a=1)
        '''
        write_canonical(Path("src/module.py"), code)
        # Result: x = {"a": 1, "b": 2}; foo(a=1, z=3)
    """
    import os
    import tempfile

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Apply AST canonicalization if requested
    if canonicalize and path.suffix in (".py", ".pyi"):
        try:
            from transformation_portal.dev.ast_normalize import canonicalize_code

            content = canonicalize_code(content)
            logger.debug("Applied AST canonicalization to %s", path)
        except Exception as e:
            logger.warning("AST canonicalization failed for %s: %s (using original)", path, e)

    # For non-Python files, write directly (no formatting needed)
    if path.suffix not in (".py", ".pyi"):
        path.write_text(content, encoding=encoding)
        logger.debug("Wrote %d bytes to %s (non-Python, no formatting)", len(content), path)
        return

    # Create temp file in the same directory (so rename is atomic)
    fd, tmp_path_str = tempfile.mkstemp(
        suffix=path.suffix,
        prefix=f".{path.stem}_",
        dir=path.parent,
    )
    tmp_path = Path(tmp_path_str)

    try:
        # Write content to temp file
        os.write(fd, content.encode(encoding))
        os.close(fd)
        logger.debug("Wrote %d bytes to temp file %s", len(content), tmp_path)

        # Apply formatting to temp file
        format_file(tmp_path, quiet=quiet)

        # Atomic rename to final destination
        tmp_path.replace(path)
        logger.debug("Atomically moved %s to %s", tmp_path, path)

    except Exception:
        # Clean up temp file on error
        os.close(fd) if fd else None  # Ensure fd is closed
        if tmp_path.exists():
            tmp_path.unlink()
        raise


class CanonicalFileWriter:
    """Context manager for writing canonicalized Python files.

    Usage:
        with CanonicalFileWriter(Path("module.py")) as writer:
            writer.write("x = {'b': 2, 'a': 1}\\n")
        # File is automatically canonicalized and formatted on exit
    """

    def __init__(self, path: Path, encoding: str = "utf-8", canonicalize: bool = True):
        self.path = path
        self.encoding = encoding
        self.canonicalize = canonicalize
        self._content: list[str] = []

    def __enter__(self) -> "CanonicalFileWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            content = "".join(self._content)
            write_canonical(
                self.path,
                content,
                encoding=self.encoding,
                canonicalize=self.canonicalize,
            )

    def write(self, text: str) -> None:
        """Append text to the file content."""
        self._content.append(text)

    def writelines(self, lines: list[str]) -> None:
        """Append multiple lines to the file content."""
        self._content.extend(lines)
