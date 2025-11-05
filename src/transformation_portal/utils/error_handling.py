#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robust error handling utilities for image and video processing.

This module provides utilities for graceful error handling, validation,
and recovery in processing pipelines.
"""
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass


class FileValidationError(ProcessingError):
    """Exception raised when file validation fails."""
    pass


class DependencyError(ProcessingError):
    """Exception raised when required dependencies are missing."""
    pass


class ConfigurationError(ProcessingError):
    """Exception raised when configuration is invalid."""
    pass


def validate_file_path(
    path: Union[str, Path],
    must_exist: bool = True,
    extensions: Optional[List[str]] = None
) -> Path:
    """Validate and normalize a file path.

    Args:
        path: File path to validate
        must_exist: If True, raise error if file doesn't exist
        extensions: List of allowed extensions (e.g., ['.jpg', '.png'])

    Returns:
        Validated Path object

    Raises:
        FileValidationError: If validation fails

    Example:
        path = validate_file_path('input.jpg', extensions=['.jpg', '.png'])
    """
    try:
        path_obj = Path(path).resolve()
    except (TypeError, ValueError) as e:
        raise FileValidationError(f"Invalid path format: {path}") from e

    if must_exist and not path_obj.exists():
        raise FileValidationError(f"File not found: {path_obj}")

    if extensions is not None:
        if path_obj.suffix.lower() not in [ext.lower() for ext in extensions]:
            raise FileValidationError(
                f"Invalid file extension {path_obj.suffix}. "
                f"Expected one of: {', '.join(extensions)}"
            )

    return path_obj


def validate_directory(
    path: Union[str, Path],
    create: bool = False,
    writable: bool = False
) -> Path:
    """Validate and optionally create a directory.

    Args:
        path: Directory path to validate
        create: If True, create directory if it doesn't exist
        writable: If True, check that directory is writable

    Returns:
        Validated Path object

    Raises:
        FileValidationError: If validation fails

    Example:
        output_dir = validate_directory('output/', create=True, writable=True)
    """
    try:
        path_obj = Path(path).resolve()
    except (TypeError, ValueError) as e:
        raise FileValidationError(f"Invalid path format: {path}") from e

    if not path_obj.exists():
        if create:
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path_obj}")
            except OSError as e:
                raise FileValidationError(
                    f"Failed to create directory: {path_obj}"
                ) from e
        else:
            raise FileValidationError(f"Directory not found: {path_obj}")

    if not path_obj.is_dir():
        raise FileValidationError(f"Path is not a directory: {path_obj}")

    if writable:
        test_file = path_obj / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except OSError as e:
            raise FileValidationError(
                f"Directory is not writable: {path_obj}"
            ) from e

    return path_obj


def check_dependency(
    module_name: str,
    package_name: Optional[str] = None,
    min_version: Optional[str] = None
) -> bool:
    """Check if a dependency is available and optionally verify version.

    Args:
        module_name: Name of the module to import
        package_name: Package name for error messages (defaults to module_name)
        min_version: Minimum required version (e.g., "2.0.0")

    Returns:
        True if dependency is available and meets version requirement

    Raises:
        DependencyError: If dependency is missing or version is too old

    Note:
        Version checking requires the 'packaging' module. If not available,
        version validation is skipped with a warning but the function still
        returns True if the module can be imported.

    Example:
        check_dependency('torch', package_name='torch', min_version='2.0.0')
    """
    package_name = package_name or module_name

    try:
        module = __import__(module_name)
    except ImportError as e:
        raise DependencyError(
            f"Required package '{package_name}' is not installed. "
            f"Install with: pip install {package_name}"
        ) from e

    if min_version is not None:
        try:
            from packaging import version
            module_version = getattr(module, '__version__', None)
            if module_version is None:
                logger.warning(
                    f"Cannot verify version of {package_name} "
                    "(no __version__ attribute)"
                )
            elif version.parse(module_version) < version.parse(min_version):
                raise DependencyError(
                    f"Package '{package_name}' version {module_version} is too old. "
                    f"Required: >={min_version}"
                )
        except ImportError:
            logger.warning("packaging module not available, skipping version check")

    return True


def safe_execute(
    func: Callable[..., T],
    *args,
    default: Optional[T] = None,
    error_message: Optional[str] = None,
    log_errors: bool = True,
    **kwargs
) -> Optional[T]:
    """Execute a function with error handling and optional default value.

    Args:
        func: Function to execute
        *args: Positional arguments for func
        default: Default value to return on error
        error_message: Custom error message to log
        log_errors: If True, log errors (set False for expected failures)
        **kwargs: Keyword arguments for func

    Returns:
        Function result or default value on error

    Example:
        result = safe_execute(
            process_image,
            'input.jpg',
            default=None,
            error_message="Failed to process image"
        )
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            msg = error_message or f"Error in {func.__name__}"
            logger.error(f"{msg}: {e}", exc_info=True)
        return default


def validate_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    name: str = "value"
) -> Union[int, float]:
    """Validate that a numeric value is within specified range.

    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        name: Name of the parameter for error messages

    Returns:
        The validated value

    Raises:
        ConfigurationError: If value is out of range

    Example:
        strength = validate_range(0.7, min_value=0.0, max_value=1.0, name="strength")
    """
    if min_value is not None and value < min_value:
        raise ConfigurationError(
            f"{name} must be >= {min_value}, got {value}"
        )

    if max_value is not None and value > max_value:
        raise ConfigurationError(
            f"{name} must be <= {max_value}, got {value}"
        )

    return value


def batch_with_error_handling(
    items: List[Any],
    process_func: Callable[[Any], T],
    skip_errors: bool = True,
    error_limit: Optional[int] = None
) -> List[T]:
    """Process items in batch with robust error handling.

    Args:
        items: List of items to process
        process_func: Function to apply to each item
        skip_errors: If True, skip failed items and continue. Failed items are
            logged at WARNING level and excluded from results.
        error_limit: Maximum number of errors before aborting (None = unlimited)

    Returns:
        List of successfully processed results. When skip_errors=True, the
        returned list will only contain successful results, so
        len(results) < len(items) is possible. Failed items are logged at WARNING
        level and excluded from the results.

    Raises:
        ProcessingError: If skip_errors=False and any item fails, or if
            error_limit is exceeded when skip_errors=True

    Example:
        results = batch_with_error_handling(
            image_paths,
            process_image,
            skip_errors=True,
            error_limit=10
        )
        # Note: len(results) may be less than len(image_paths) if some failed
    """
    results = []
    errors = []

    for i, item in enumerate(items):
        try:
            result = process_func(item)
            results.append(result)
        except Exception as e:
            error_msg = f"Error processing item {i}: {e}"
            logger.warning(error_msg)
            errors.append((i, item, e))

            if error_limit is not None and len(errors) >= error_limit:
                raise ProcessingError(
                    f"Error limit ({error_limit}) exceeded. "
                    f"Failed items: {len(errors)}/{len(items)}"
                )

            if not skip_errors:
                raise ProcessingError(error_msg) from e

    if errors:
        logger.warning(
            f"Completed with errors: {len(errors)}/{len(items)} items failed"
        )

    return results


def get_error_summary(errors: List[Exception]) -> str:
    """Generate a human-readable summary of errors.

    Args:
        errors: List of exceptions

    Returns:
        Formatted error summary string

    Example:
        summary = get_error_summary(errors)
        print(summary)
    """
    if not errors:
        return "No errors"

    error_types = {}
    for error in errors:
        error_type = type(error).__name__
        error_types[error_type] = error_types.get(error_type, 0) + 1

    summary_lines = [f"Total errors: {len(errors)}"]
    for error_type, count in sorted(error_types.items()):
        summary_lines.append(f"  - {error_type}: {count}")

    return "\n".join(summary_lines)


# Convenient aliases
validate_path = validate_file_path
validate_dir = validate_directory
safe_call = safe_execute
