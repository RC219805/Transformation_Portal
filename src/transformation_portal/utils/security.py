"""
Security Utilities for Transformation Portal

Provides validation functions to prevent common security vulnerabilities:
- Path traversal (CWE-22)
- Command injection (CWE-78)
- Resource exhaustion (CWE-400)

Usage:
    >>> from transformation_portal.utils.security import validate_filepath
    >>> safe_path = validate_filepath(
    ...     Path("user_input.jpg"),
    ...     allowed_dirs=[Path("/data")],
    ...     max_file_size=100*1024*1024  # 100MB
    ... )

See Also:
    - docs/architecture/adr/ADR-002-security-input-validation.md
    - SECURITY.md
"""

import os
import re
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple


class SecurityError(Exception):
    """Raised when security validation fails."""

    pass


# File size limits (bytes)
MAX_IMAGE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_VIDEO_SIZE = 10 * 1024 * 1024 * 1024  # 10GB
MAX_CONFIG_SIZE = 10 * 1024 * 1024  # 10MB

# Allowed file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
CONFIG_EXTENSIONS = {".yaml", ".yml", ".json", ".toml"}


def validate_filepath(
    filepath: Path,
    allowed_dirs: List[Path],
    max_file_size: Optional[int] = None,
    allowed_extensions: Optional[List[str]] = None,
    must_exist: bool = True,
) -> Path:
    """
    Validate file path against security constraints.

    Prevents path traversal attacks by ensuring the resolved path
    is within allowed directories.

    Args:
        filepath: Path to validate
        allowed_dirs: List of allowed parent directories
        max_file_size: Maximum file size in bytes (None = no limit)
        allowed_extensions: Whitelist of allowed extensions (None = any)
        must_exist: Whether file must exist (default True)

    Returns:
        Resolved, validated Path object

    Raises:
        SecurityError: If validation fails

    Example:
        >>> safe_path = validate_filepath(
        ...     Path("../../../etc/passwd"),
        ...     allowed_dirs=[Path("/data")]
        ... )
        Traceback (most recent call last):
        SecurityError: Path outside allowed directories
    """
    # Convert to Path object if string
    if isinstance(filepath, str):
        filepath = Path(filepath)

    # Resolve to absolute path (follows symlinks)
    try:
        if must_exist:
            resolved = filepath.resolve(strict=True)
        else:
            resolved = filepath.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise SecurityError(f"Cannot resolve path {filepath}: {e}")

    # Check file exists (if required)
    if must_exist and not resolved.exists():
        raise SecurityError(f"File does not exist: {filepath}")

    # Validate within allowed directories
    if not any(resolved.is_relative_to(d.resolve()) for d in allowed_dirs):
        raise SecurityError(f"Path {filepath} outside allowed directories: {allowed_dirs}")

    # Check file size (if file exists)
    if must_exist and max_file_size and resolved.is_file():
        file_size = resolved.stat().st_size
        if file_size > max_file_size:
            raise SecurityError(f"File {filepath} exceeds size limit: " f"{file_size} > {max_file_size} bytes")

    # Check extension
    if allowed_extensions:
        if resolved.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise SecurityError(f"File extension {resolved.suffix} not in whitelist: {allowed_extensions}")

    return resolved


def validate_image_path(filepath: Path, allowed_dirs: List[Path], max_size: int = MAX_IMAGE_SIZE) -> Path:
    """
    Validate image file path.

    Convenience wrapper for validate_filepath with image-specific defaults.

    Args:
        filepath: Path to image file
        allowed_dirs: List of allowed parent directories
        max_size: Maximum file size (default 100MB)

    Returns:
        Validated Path object

    Raises:
        SecurityError: If validation fails
    """
    return validate_filepath(
        filepath,
        allowed_dirs,
        max_file_size=max_size,
        allowed_extensions=list(IMAGE_EXTENSIONS),
    )


def validate_video_path(filepath: Path, allowed_dirs: List[Path], max_size: int = MAX_VIDEO_SIZE) -> Path:
    """
    Validate video file path.

    Args:
        filepath: Path to video file
        allowed_dirs: List of allowed parent directories
        max_size: Maximum file size (default 10GB)

    Returns:
        Validated Path object

    Raises:
        SecurityError: If validation fails
    """
    return validate_filepath(
        filepath,
        allowed_dirs,
        max_file_size=max_size,
        allowed_extensions=list(VIDEO_EXTENSIONS),
    )


def validate_config_path(filepath: Path, allowed_dirs: List[Path], max_size: int = MAX_CONFIG_SIZE) -> Path:
    """
    Validate configuration file path.

    Args:
        filepath: Path to config file
        allowed_dirs: List of allowed parent directories
        max_size: Maximum file size (default 10MB)

    Returns:
        Validated Path object

    Raises:
        SecurityError: If validation fails
    """
    return validate_filepath(
        filepath,
        allowed_dirs,
        max_file_size=max_size,
        allowed_extensions=list(CONFIG_EXTENSIONS),
    )


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for safe filesystem operations.

    Removes/replaces dangerous characters while preserving extension.

    Args:
        filename: Original filename
        max_length: Maximum filename length (default 255)

    Returns:
        Sanitized filename

    Example:
        >>> sanitize_filename("../../../etc/passwd")
        'etc_passwd'
        >>> sanitize_filename("file<script>.jpg")
        'file_script_.jpg'
    """
    # Remove path components
    filename = os.path.basename(filename)

    # Replace dangerous characters
    dangerous_chars = '<>:"|?*/\\;'
    for char in dangerous_chars:
        filename = filename.replace(char, "_")

    # Remove leading/trailing dots and spaces
    filename = filename.strip(". ")

    # Ensure not empty
    if not filename:
        filename = "unnamed"

    # Truncate if too long (preserve extension)
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        max_name_length = max_length - len(ext)
        filename = name[:max_name_length] + ext

    return filename


# Shell metacharacters that could enable command injection
# Includes: semicolon, ampersand, pipe, backtick, dollar, parentheses, angle brackets
DANGEROUS_SHELL_CHARS = set(";&|`$()<>")


def build_safe_command(executable: str, args: List[str], dangerous_chars: Optional[set] = None) -> List[str]:
    """
    Build safe command argument list for subprocess.

    Validates that arguments don't contain shell metacharacters
    that could be exploited via command injection.

    Args:
        executable: Command executable (e.g., 'ffmpeg')
        args: List of command arguments
        dangerous_chars: Set of characters to reject (default: DANGEROUS_SHELL_CHARS)

    Returns:
        List of command arguments (safe for subprocess.run without shell=True)

    Raises:
        SecurityError: If arguments contain dangerous characters

    Example:
        >>> cmd = build_safe_command('ffmpeg', ['-i', 'input.mp4', 'output.mp4'])
        >>> subprocess.run(cmd, check=True)  # Safe

        >>> cmd = build_safe_command('ffmpeg', ['-i', 'input.mp4; rm -rf /'])
        Traceback (most recent call last):
        SecurityError: Argument contains dangerous characters
    """
    if dangerous_chars is None:
        dangerous_chars = DANGEROUS_SHELL_CHARS

    # Validate executable
    if any(char in executable for char in dangerous_chars):
        raise SecurityError(f"Executable '{executable}' contains dangerous characters")

    # Validate all arguments
    for arg in args:
        if any(char in str(arg) for char in dangerous_chars):
            raise SecurityError(f"Argument '{arg}' contains dangerous characters: {dangerous_chars}")

    # Build command list
    return [executable] + [str(arg) for arg in args]


# Characters rejected inside FFmpeg filter graphs. The command is always an argv
# list (no shell), so this is defence in depth rather than injection prevention.
# Parentheses are NOT rejected: FFmpeg expressions need them ("min(1920,iw)").
# ';' stays rejected so a -vf value cannot grow into a multi-chain graph.
FILTER_GRAPH_REJECTED_CHARS = frozenset(DANGEROUS_SHELL_CHARS - {"(", ")"})

# Filters that read files, load code or accept runtime commands in ways a static
# check cannot confine: always rejected.
FFMPEG_UNCONFINABLE_FILTERS = frozenset(
    {
        "sendcmd",  # inline commands can re-init other filters with new file options
        "asendcmd",
        "zmq",  # network endpoint + runtime commands
        "azmq",
        "frei0r",  # loads a shared library by path
        "frei0r_src",
        "ladspa",
        "lv2",
        "libvmaf",  # compound model string embeds paths
    }
)

# Path-valued options of file-accessing filters (option names per the FFmpeg
# filter documentation): filter -> (named path options, index of the positional
# argument that is a path, or None). Any other positional argument of these
# filters is rejected, because FFmpeg maps positional arguments onto options in
# declaration order (e.g. drawtext's first positional argument is fontfile).
FFMPEG_FILTER_PATH_OPTIONS = {
    "drawtext": (frozenset({"fontfile", "textfile"}), None),
    "qrencode": (frozenset({"textfile"}), None),
    "subtitles": (frozenset({"filename", "f", "fontsdir"}), 0),
    "ass": (frozenset({"filename", "f", "fontsdir"}), 0),
    "movie": (frozenset({"filename"}), 0),
    "amovie": (frozenset({"filename"}), 0),
    "lut3d": (frozenset({"file"}), 0),
    "lut1d": (frozenset({"file"}), 0),
    "curves": (frozenset({"psfile"}), None),
    "selectivecolor": (frozenset({"psfile"}), None),
    "removelogo": (frozenset({"filename", "f"}), 0),
    "cover_rect": (frozenset({"cover"}), None),
    "find_rect": (frozenset({"object"}), None),
    "fieldhint": (frozenset({"hint"}), 0),
    "metadata": (frozenset({"file"}), None),
    "ametadata": (frozenset({"file"}), None),
    "psnr": (frozenset({"stats_file", "f"}), 0),
    "ssim": (frozenset({"stats_file", "f"}), 0),
    "vmafmotion": (frozenset({"stats_file"}), None),
    "signature": (frozenset({"filename"}), None),
    "vidstabdetect": (frozenset({"result"}), 0),
    "vidstabtransform": (frozenset({"input"}), 0),
    "lensfun": (frozenset({"db_path"}), None),
    "ocr": (frozenset({"datapath"}), 0),
    "sofalizer": (frozenset({"sofa"}), 0),
    "arnndn": (frozenset({"model", "m"}), 0),
    "derain": (frozenset({"model"}), None),
    "sr": (frozenset({"model"}), None),
    "dnn_processing": (frozenset({"model"}), None),
    "dnn_detect": (frozenset({"model", "labels"}), None),
    "dnn_classify": (frozenset({"model", "labels"}), None),
    "program_opencl": (frozenset({"source"}), 0),
    "openclsrc": (frozenset({"source"}), 0),
    "libplacebo": (frozenset({"custom_shader_path"}), None),
    "whisper": (frozenset({"model", "destination"}), 0),
}

# FFmpeg options that add inputs or filter graphs (or, FFmpeg >= 7, load an
# option value from a file with "-/opt"); build_ffmpeg_command only accepts
# them through its validated ``input_file`` / ``filters`` parameters.
_FFMPEG_GRAPH_OR_INPUT_ARG = re.compile(r"^-(?:i|vf|af|lavfi|filter(?:_complex)?(?:_script)?(?::.*)?|/.*)$")

_FILTER_WHITESPACE = " \n\t\r"
_FILTER_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
_URL_PROTOCOL_RE = re.compile(r"^[A-Za-z0-9+.\-]+:")


def _ffmpeg_get_token(text: str, pos: int, terminators: str) -> Tuple[str, int]:
    """Mirror FFmpeg ``av_get_token``: unescape ``\\x`` and ``'...'`` up to a terminator."""
    while pos < len(text) and text[pos] in _FILTER_WHITESPACE:
        pos += 1
    out: List[str] = []
    protected = 0  # prefix of ``out`` that trailing-whitespace trimming must keep
    while pos < len(text) and text[pos] not in terminators:
        char = text[pos]
        pos += 1
        if char == "\\" and pos < len(text):
            out.append(text[pos])
            pos += 1
            protected = len(out)
        elif char == "'":
            while pos < len(text) and text[pos] != "'":
                out.append(text[pos])
                pos += 1
            if pos < len(text):
                pos += 1
            protected = len(out)
        else:
            out.append(char)
    while len(out) > protected and out[-1] in _FILTER_WHITESPACE:
        out.pop()
    return "".join(out), pos


def _skip_link_labels(text: str, pos: int) -> int:
    while True:
        while pos < len(text) and text[pos] in _FILTER_WHITESPACE:
            pos += 1
        if pos >= len(text) or text[pos] != "[":
            return pos
        end = text.find("]", pos)
        if end < 0:
            raise SecurityError(f"Unterminated link label in filter graph: {text}")
        pos = end + 1


def _parse_filter_graph(filter_graph: str) -> List[Tuple[str, str]]:
    """Split a filter graph into ``(filter_name, unescaped_args)`` pairs, failing closed."""
    filters: List[Tuple[str, str]] = []
    pos = 0
    while True:
        pos = _skip_link_labels(filter_graph, pos)
        end = pos
        while end < len(filter_graph) and filter_graph[end] not in "=,;[":
            end += 1
        raw_name = filter_graph[pos:end].strip(_FILTER_WHITESPACE)
        name = raw_name.split("@", 1)[0]
        if not _FILTER_NAME_RE.match(name):
            raise SecurityError(f"Unparseable filter name {raw_name!r} in filter graph: {filter_graph}")
        pos = end
        args = ""
        if pos < len(filter_graph) and filter_graph[pos] == "=":
            args, pos = _ffmpeg_get_token(filter_graph, pos + 1, "[],;")
        filters.append((name, args))
        pos = _skip_link_labels(filter_graph, pos)
        if pos >= len(filter_graph):
            return filters
        if filter_graph[pos] not in ",;":
            raise SecurityError(f"Unparseable filter graph near offset {pos}: {filter_graph}")
        pos += 1


def _parse_filter_options(args: str) -> List[Tuple[Optional[str], str]]:
    """Split filter arguments into ``(key, value)`` pairs like FFmpeg (key None = positional)."""
    options: List[Tuple[Optional[str], str]] = []
    pos = 0
    while pos < len(args):
        key: Optional[str] = None
        scan = pos
        while scan < len(args) and args[scan] in _FILTER_WHITESPACE:
            scan += 1
        key_start = scan
        while scan < len(args) and args[scan].isascii() and (args[scan].isalnum() or args[scan] in "-_/."):
            scan += 1
        key_end = scan
        while scan < len(args) and args[scan] in _FILTER_WHITESPACE:
            scan += 1
        if scan < len(args) and args[scan] == "=":
            key = args[key_start:key_end]
            pos = scan + 1
        value, pos = _ffmpeg_get_token(args, pos, ":")
        options.append((key, value))
        if pos < len(args):
            pos += 1  # skip the ':' separator
    return options


def _validate_filter_path(filter_name: str, option: str, value: str, allowed_dirs: Optional[List[Path]]) -> None:
    if not value:
        return
    label = f"'{filter_name}' option {option!r}"
    if allowed_dirs is None:
        raise SecurityError(f"Filter {label} references a file but no allowed_dirs were given: {value}")
    if _URL_PROTOCOL_RE.match(value) or value.startswith("subfile,"):
        raise SecurityError(f"Filter {label} uses a URL/protocol, which is not allowed: {value}")
    try:
        validate_filepath(Path(value), allowed_dirs, must_exist=False)
    except SecurityError as exc:
        raise SecurityError(f"Filter {label} is not confined to allowed directories: {exc}") from exc


def _validate_filter_file_access(filter_graph: str, allowed_dirs: Optional[List[Path]]) -> None:
    for name, args in _parse_filter_graph(filter_graph):
        lowered = name.lower()
        if lowered in FFMPEG_UNCONFINABLE_FILTERS:
            raise SecurityError(f"Filter '{name}' is not allowed: its file, code or command access cannot be confined")
        path_options, positional_path_index = FFMPEG_FILTER_PATH_OPTIONS.get(lowered, (frozenset(), None))
        for index, (key, value) in enumerate(_parse_filter_options(args)):
            if key is not None and key.startswith("/"):
                # FFmpeg >= 7: "/option=path" loads the option value from a file.
                _validate_filter_path(name, key, value, allowed_dirs)
            elif key is not None:
                if key in path_options:
                    _validate_filter_path(name, key, value, allowed_dirs)
            elif path_options:
                if index != positional_path_index:
                    raise SecurityError(
                        f"Filter '{name}' takes file paths; use named options instead of positional argument {index}"
                    )
                _validate_filter_path(name, f"#{index}", value, allowed_dirs)


def build_ffmpeg_command(
    input_file: Path,
    output_file: Path,
    filters: Optional[List[str]] = None,
    codec: str = "libx264",
    additional_args: Optional[List[str]] = None,
    validate_paths: bool = True,
    allowed_dirs: Optional[List[Path]] = None,
) -> List[str]:
    """
    Build safe FFmpeg command with injection protection.

    Constructs FFmpeg command as argument list (no shell required),
    validates filter strings with :func:`validate_filter_graph` (including
    confinement of file-path filter options to ``allowed_dirs``), and
    optionally validates file paths.

    Args:
        input_file: Input file path
        output_file: Output file path
        filters: Optional list of video filter strings
        codec: Video codec (default 'libx264')
        additional_args: Optional additional FFmpeg arguments. Input and
            filter-graph options (``-i``, ``-vf``, ``-af``, ``-filter*``,
            ``-lavfi``, ``-/opt``) are rejected: pass them via ``input_file``
            and ``filters`` so they are validated.
        validate_paths: Whether to validate paths are safe (default True)
        allowed_dirs: Allowed directories (default: cwd when validate_paths=True).
            With validate_paths=False and no allowed_dirs, filters that
            reference files are rejected.

    Returns:
        List of command arguments safe for subprocess.run without shell=True

    Raises:
        SecurityError: If filters contain dangerous characters or unconfined
            file paths, additional_args smuggle inputs/filters, or paths invalid

    Example:
        >>> cmd = build_ffmpeg_command(
        ...     Path('input.mp4'),
        ...     Path('output.mp4'),
        ...     filters=['scale=1920:1080', 'fps=30']
        ... )
        >>> subprocess.run(cmd, check=True)  # Safe: no shell=True

    See Also:
        - docs/architecture/adr/ADR-002-security-input-validation.md
    """
    # Validate paths if requested
    if validate_paths:
        if allowed_dirs is None:
            allowed_dirs = [Path.cwd()]
        validate_filepath(input_file, allowed_dirs, must_exist=True)
        validate_filepath(output_file, allowed_dirs, must_exist=False)

    # Validate filter strings: rejected characters, then file-path confinement
    if filters:
        for filter_str in filters:
            if any(char in filter_str for char in FILTER_GRAPH_REJECTED_CHARS):
                raise SecurityError(
                    f"Filter contains dangerous characters: {filter_str}\n"
                    f"Dangerous characters: {set(FILTER_GRAPH_REJECTED_CHARS)}"
                )
        validate_filter_graph(",".join(filters), allowed_dirs=allowed_dirs)

    if additional_args:
        for arg in additional_args:
            if _FFMPEG_GRAPH_OR_INPUT_ARG.match(str(arg)):
                raise SecurityError(
                    f"additional_args may not contain input or filter-graph option {arg!r}; "
                    "use input_file / filters so they are validated"
                )

    # Build command list
    cmd = ["ffmpeg", "-i", str(input_file)]

    # Add filters if provided
    if filters:
        cmd.extend(["-vf", ",".join(filters)])

    # Add codec
    cmd.extend(["-c:v", codec])

    # Add additional arguments
    if additional_args:
        cmd.extend([str(arg) for arg in additional_args])

    # Add output file
    cmd.append(str(output_file))

    return cmd


def validate_filter_graph(filter_graph: str, allowed_dirs: Optional[List[Path]] = None) -> str:
    """
    Validate FFmpeg filter graph string for safety.

    Rejects the characters in ``FILTER_GRAPH_REJECTED_CHARS`` (parentheses are
    allowed, FFmpeg expressions need them), filters whose file/code/command
    access cannot be confined (``FFMPEG_UNCONFINABLE_FILTERS``), and path-valued
    options of file-accessing filters (``FFMPEG_FILTER_PATH_OPTIONS`` plus the
    FFmpeg >= 7 ``/option=file`` syntax) that resolve outside ``allowed_dirs``
    or name a URL/protocol. Without ``allowed_dirs`` any such path is rejected.

    Filters outside both tables are not inspected further: this is a
    denylist of documented file-accessing options, not a filter allowlist.

    Args:
        filter_graph: FFmpeg filter graph string
        allowed_dirs: Directories file-path filter options must stay within

    Returns:
        Validated filter graph string

    Raises:
        SecurityError: If filter graph contains dangerous patterns

    Example:
        >>> safe_graph = validate_filter_graph("scale='min(1920,iw)':-2,fps=30")
        >>> dangerous = validate_filter_graph("scale=1920; rm -rf /")
        Traceback (most recent call last):
        SecurityError: Filter graph contains dangerous characters
    """
    if any(char in filter_graph for char in FILTER_GRAPH_REJECTED_CHARS):
        raise SecurityError(
            f"Filter graph contains dangerous characters: {filter_graph}\n"
            f"Dangerous characters found: {sorted(c for c in FILTER_GRAPH_REJECTED_CHARS if c in filter_graph)}"
        )

    if filter_graph.strip(_FILTER_WHITESPACE):
        _validate_filter_file_access(filter_graph, allowed_dirs)

    return filter_graph


class TimeoutError(Exception):
    """Raised when operation exceeds timeout."""

    pass


@contextmanager
def timeout(seconds: int):
    """
    Context manager for operation timeout (Unix-only).

    Uses SIGALRM which only works on Unix-like systems (Linux, macOS).
    Windows users should use alternative timeout mechanisms.

    Args:
        seconds: Timeout in seconds

    Yields:
        None

    Raises:
        TimeoutError: If operation exceeds timeout
        NotImplementedError: If used on Windows

    Example:
        >>> with timeout(30):
        ...     slow_operation()  # Raises TimeoutError after 30s

    Warning:
        This implementation uses signal.SIGALRM which is not available
        on Windows. Use multiprocessing.Process with timeout parameter
        or threading-based solutions on Windows.
    """
    if not hasattr(signal, "SIGALRM"):
        raise NotImplementedError(
            "timeout() requires signal.SIGALRM (Unix-only). "
            "(TODO_INVENTORY.md §2.4) "
            "Use multiprocessing or threading-based timeout on Windows."
        )

    started = time.monotonic()
    own_deadline = started + seconds if seconds > 0 else None

    def timeout_handler(signum, frame):
        if own_deadline is not None and time.monotonic() >= own_deadline:
            raise TimeoutError(f"Operation exceeded {seconds}s timeout")
        # The alarm belongs to an enclosing timeout() whose deadline is earlier
        # than ours: let its handler decide.
        if callable(old_handler):
            old_handler(signum, frame)

    # Set signal handler. setitimer shares the ITIMER_REAL timer with
    # signal.alarm and returns the remaining time of an enclosing alarm, which
    # must not be cancelled by a nested timeout().
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    outer_remaining, _ = signal.setitimer(signal.ITIMER_REAL, 0)
    outer_deadline = started + outer_remaining if outer_remaining > 0 else None
    arm = [d for d in (own_deadline, outer_deadline) if d is not None]
    if arm:
        signal.setitimer(signal.ITIMER_REAL, max(min(arm) - time.monotonic(), 1e-3))

    try:
        yield
    finally:
        # Cancel our alarm, restore the old handler, then re-arm the enclosing
        # alarm with its remaining time (fire it promptly if already overdue).
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if outer_deadline is not None:
            signal.setitimer(signal.ITIMER_REAL, max(outer_deadline - time.monotonic(), 1e-3))


# Deprecated alias for backwards compatibility
from transformation_portal.compat import renamed_function


@renamed_function("validate_filepath")
def validate_file_path(*args, **kwargs):
    """Deprecated: Use validate_filepath instead."""
    return validate_filepath(*args, **kwargs)
