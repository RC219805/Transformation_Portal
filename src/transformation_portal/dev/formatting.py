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
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Repository formatting standards (must match pyproject.toml)
BLACK_LINE_LENGTH = 127
BLACK_TARGET_VERSIONS = ["py311", "py312"]

# Directory names format_directory() never descends into: black's default
# exclude list plus vendored / cache trees. Explicit file paths handed to black
# bypass black's own exclude, so the walk filters them itself. Every hidden
# (dot-prefixed) path component below the formatted directory is skipped too.
EXCLUDED_DIR_NAMES = frozenset(
    {
        "__pycache__",
        "__pypackages__",
        "_build",
        "buck-out",
        "build",
        "dist",
        "node_modules",
        "site-packages",
        "venv",
    }
)


def _tool_available(tool: str) -> bool:
    """Check if a formatting tool is available in the environment."""
    return shutil.which(tool) is not None


def format_file(path: Path, *, quiet: bool = True) -> bool:
    """Apply canonical formatting to a Python file immediately after write.

    The formatting order is: isort (import ordering) → Black.
    Running isort first ensures imports are sorted; Black then formats the
    result, so isort's changes are already Black-compliant (Black is
    idempotent, so a second Black pass would only cost another subprocess).

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

    return success


def _current_umask() -> int:
    """Return the process umask without leaving it modified."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("Umask:"):
                return int(line.split()[1], 8)
    except (OSError, ValueError, IndexError):
        pass
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


def _destination_mode(path: Path) -> int:
    """Permission bits the written file should carry.

    An existing file keeps its mode (an executable script stays executable);
    a new file gets the regular ``0o666 & ~umask`` instead of mkstemp's 0o600.
    """
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        return 0o666 & ~_current_umask()


def _write_formatted_atomically(path: Path, content: str, *, encoding: str, quiet: bool) -> None:
    """Write ``content`` to a sibling temp file, format it, then rename it onto ``path``.

    Raises:
        RuntimeError: If a formatter failed; ``path`` is left untouched.
    """
    import tempfile

    mode = _destination_mode(path)

    # Create temp file in the same directory (so rename is atomic)
    fd, tmp_path_str = tempfile.mkstemp(
        suffix=path.suffix,
        prefix=f".{path.stem}_",
        dir=path.parent,
    )
    tmp_path = Path(tmp_path_str)

    try:
        # Write content to temp file (ensure fd is closed even on write failure)
        try:
            os.write(fd, content.encode(encoding))
        finally:
            os.close(fd)
        tmp_path.chmod(mode)
        logger.debug("Wrote %d bytes to temp file %s", len(content), tmp_path)

        # Apply formatting to temp file; never publish an unformatted file.
        if not format_file(tmp_path, quiet=quiet):
            raise RuntimeError(f"Formatting failed for {path}; destination left unchanged")

        # Atomic rename to final destination
        tmp_path.replace(path)
        logger.debug("Atomically moved %s to %s", tmp_path, path)

    except BaseException:
        # Clean up temp file on error (fd already closed by inner try/finally)
        tmp_path.unlink(missing_ok=True)
        raise


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
    An existing destination keeps its permission bits; a new one gets the
    usual ``0o666 & ~umask``.

    Args:
        path: Destination path for the file.
        content: Content to write to the file.
        encoding: File encoding (default: utf-8).
        quiet: If True, suppress stdout/stderr from formatters.

    Raises:
        RuntimeError: If isort or Black failed on the content (for example a
            syntax error); the destination is left unchanged.

    Example:
        from transformation_portal.dev.formatting import write_formatted

        code = '''
        def hello():
            print("Hello, world!")
        '''
        write_formatted(Path("src/module.py"), code)
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # For non-Python files, write directly (no formatting needed)
    if path.suffix not in (".py", ".pyi"):
        path.write_text(content, encoding=encoding)
        logger.debug("Wrote %d bytes to %s (non-Python, no formatting)", len(content), path)
        return

    _write_formatted_atomically(path, content, encoding=encoding, quiet=quiet)


def _is_excluded(file_path: Path, directory: Path) -> bool:
    """True when ``file_path`` lies in a hidden, vendored or build tree below ``directory``."""
    parts = file_path.relative_to(directory).parts
    return any(part.startswith(".") or part in EXCLUDED_DIR_NAMES for part in parts)


def format_directory(
    directory: Path,
    *,
    recursive: bool = True,
    quiet: bool = True,
) -> tuple[int, int]:
    """Format all Python files in a directory.

    Hidden paths (any dot-prefixed component, e.g. ``.venv`` or ``.git``) and
    the vendored / build trees in ``EXCLUDED_DIR_NAMES`` are skipped.

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
    files = [f for f in directory.glob(pattern) if f.is_file() and not _is_excluded(f, directory)]

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


def _has_comments(content: str) -> bool:
    """Return True when ``content`` contains a comment token (or cannot be tokenized)."""
    import io
    import tokenize

    try:
        return any(tok.type == tokenize.COMMENT for tok in tokenize.generate_tokens(io.StringIO(content).readline))
    except (tokenize.TokenError, SyntaxError):
        # Unknown structure: treat as commented so nothing is silently dropped.
        return True


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

    Content that contains comments is formatted but not canonicalized, because
    ast.unparse() would drop them.

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

    Raises:
        RuntimeError: If isort or Black failed on the content; the destination
            is left unchanged.
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Apply AST canonicalization if requested. ast.unparse() drops every
    # comment (shebang, licence header, ``# noqa`` / ``# nosec`` / ``# pragma``),
    # so sources that carry comments are only formatted, never canonicalized.
    if canonicalize and path.suffix in (".py", ".pyi") and _has_comments(content):
        logger.info("Skipping AST canonicalization for %s: source contains comments", path)
    elif canonicalize and path.suffix in (".py", ".pyi"):
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

    _write_formatted_atomically(path, content, encoding=encoding, quiet=quiet)


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
