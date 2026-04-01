"""Helpers for optional ML dependency compatibility checks."""

from __future__ import annotations

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


def detect_transformers_torch_version_issue(
    torch_version: Optional[str],
    transformers_version: Optional[str],
) -> Optional[str]:
    """Describe a known incompatible torch/transformers/numpy version set."""
    if not torch_version or not transformers_version:
        return None

    details = []
    if _version_tuple(torch_version) and _version_tuple(torch_version) < (2, 4):
        details.append(f"installed torch {torch_version} is below the minimum expected by transformers {transformers_version}")

    numpy_version = _installed_version("numpy")
    if numpy_version and _version_tuple(numpy_version) >= (2, 0) and _version_tuple(torch_version) < (2, 4):
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
        f"{message} {'; '.join(details)}. Install a compatible ML stack via the repo bootstrap/lockfile flow, "
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
        f"{message} Install a compatible ML stack via the repo bootstrap/lockfile flow, "
        "or align torch and transformers to a mutually supported combination."
    )
    return message
