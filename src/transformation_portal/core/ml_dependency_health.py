"""Helpers for optional ML dependency compatibility checks."""

from __future__ import annotations

import importlib
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional, Tuple

OPTIONAL_IMPORT_EXCEPTIONS = (
    ImportError,
    RuntimeError,
    TypeError,
    OSError,
    AttributeError,
)
MIN_SUPPORTED_TORCH_VERSION = (2, 13)
MIN_SUPPORTED_TORCH_VERSION_TEXT = "2.13.0"
MIN_SUPPORTED_TRANSFORMERS_VERSION = (5, 5)
MIN_SUPPORTED_TRANSFORMERS_VERSION_TEXT = "5.5.0"


def _version_tuple(raw_version: Any) -> Tuple[int, ...]:
    """Extract a best-effort numeric version tuple from a package version string."""
    text = str(raw_version or "").strip()
    if not text:
        return ()
    parts = []
    for token in text.replace("+", ".").replace("-", ".").split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            numeric_prefix = "".join(ch for ch in token if ch.isdigit())
            if numeric_prefix:
                parts.append(int(numeric_prefix))
            else:
                break
    return tuple(parts)


def _installed_version(distribution_name: str) -> Optional[str]:
    """Return an installed distribution version when available."""
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        module_name = distribution_name.strip().lower().replace("-", "_")
        module = sys.modules.get(module_name)
        module_version = getattr(module, "__version__", None) if module is not None else None
        if module_version:
            return str(module_version)
        return None


def ensure_dependency_importable(distribution_name: str) -> Any:
    """Import a dependency module and normalize import failures to ImportError.

    This is stricter than checking wheel metadata alone: backend availability
    should reject environments where the distribution is present but the module
    cannot be imported in the current interpreter.
    """
    module_name = distribution_name.strip().lower().replace("-", "_")
    if module_name in sys.modules and sys.modules[module_name] is None:
        raise ImportError(f"{distribution_name} package is installed but not importable in this Python process.")

    try:
        return importlib.import_module(module_name)
    except OPTIONAL_IMPORT_EXCEPTIONS as exc:
        raise ImportError(f"{distribution_name} package is installed but not importable in this Python process.") from exc


def _is_darwin_x86_64_runtime() -> bool:
    """Return True when the current interpreter is running on Intel macOS."""
    return sys.platform == "darwin" and platform.machine() == "x86_64"


def detect_transformers_torch_version_issue(
    torch_version: Optional[str],
    transformers_version: Optional[str],
) -> Optional[str]:
    """Describe a known incompatible torch/transformers/numpy version set."""
    if not torch_version or not transformers_version:
        return None

    torch_version_tuple = _version_tuple(torch_version)
    transformers_version_tuple = _version_tuple(transformers_version)
    details = []
    if torch_version_tuple and torch_version_tuple < MIN_SUPPORTED_TORCH_VERSION:
        details.append(
            f"installed torch {torch_version} is below the supported security baseline " f"{MIN_SUPPORTED_TORCH_VERSION_TEXT}"
        )
    if transformers_version_tuple and transformers_version_tuple < MIN_SUPPORTED_TRANSFORMERS_VERSION:
        details.append(
            f"installed transformers {transformers_version} is below the supported security baseline "
            f"{MIN_SUPPORTED_TRANSFORMERS_VERSION_TEXT}"
        )
    if (
        torch_version_tuple
        and transformers_version_tuple
        and torch_version_tuple < (2, 4)
        and transformers_version_tuple >= (5, 3)
    ):
        details.append(f"installed torch {torch_version} is below the minimum expected by transformers {transformers_version}")

    numpy_version = _installed_version("numpy")
    if (
        numpy_version
        and torch_version_tuple
        and torch_version_tuple < (2, 4)
        and _version_tuple(numpy_version) >= (2, 0)
        and _is_darwin_x86_64_runtime()
    ):
        details.append(
            f"numpy {numpy_version} may be incompatible with torch {torch_version} wheels compiled against NumPy 1.x"
        )

    if not details:
        return None

    message = (
        f"transformers {transformers_version} and torch {torch_version} are installed, but the version set is "
        "outside the repository's supported runtime envelope."
    )
    message = (
        f"{message} {'; '.join(details)}. Install a compatible ML stack via the supported Apple Silicon "
        "bootstrap/lockfile flow, "
        "or align torch and transformers to a mutually supported combination."
    )
    return message


def detect_transformers_torch_runtime_issue(
    torch_module: Any,
    transformers_module: Any,
) -> Optional[str]:
    """Describe an incompatible torch/transformers runtime state, if present."""
    if torch_module is None or transformers_module is None:
        return None

    try:
        from transformers.utils import is_torch_available
    except OPTIONAL_IMPORT_EXCEPTIONS:
        return None

    try:
        torch_backend_available = bool(is_torch_available())
    except OPTIONAL_IMPORT_EXCEPTIONS as exc:
        return "transformers could not validate its PyTorch backend " f"state: {exc}"

    if torch_backend_available:
        return None

    torch_version = str(getattr(torch_module, "__version__", "unknown"))
    transformers_version = str(getattr(transformers_module, "__version__", "unknown"))
    version_issue = detect_transformers_torch_version_issue(torch_version, transformers_version)

    message = (
        "transformers "
        f"{transformers_version} disabled its PyTorch backend while torch {torch_version} is installed. "
        "Depth model loading is unavailable in this environment."
    )
    if version_issue:
        message = f"{message} {version_issue}"
    message = (
        f"{message} Install a compatible ML stack via the supported Apple Silicon bootstrap/lockfile flow, "
        "or align torch and transformers to a mutually supported combination."
    )
    return message
