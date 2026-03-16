"""Secure serialization helpers.

Provides a restricted pickle loader that only permits a small allowlist of
global classes/functions needed for numpy array cache artifacts.
"""

from __future__ import annotations

import importlib
import logging
import pickle
from typing import Any, BinaryIO, Set, Tuple

logger = logging.getLogger(__name__)

# Allow only globals required for safe cache reconstruction across versions.
# `torch.Size` remains for backward compatibility with legacy caches written
# before `ReferenceImageEncoder.save_features()` normalized shape to tuples.
_ALLOWED_PICKLE_GLOBALS: Set[Tuple[str, str]] = {
    ("torch", "Size"),
    ("pathlib", "PosixPath"),
    ("pathlib", "WindowsPath"),
    ("pathlib", "Path"),
    ("collections", "OrderedDict"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
}


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that blocks arbitrary class/function loading."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _ALLOWED_PICKLE_GLOBALS:
            return getattr(importlib.import_module(module), name)
        logger.warning(
            f"Blocked forbidden pickle global: {module}.{name}. "
            "This may indicate a malicious pickle file or an unexpected data format."
        )
        raise pickle.UnpicklingError(f"Forbidden pickle global: {module}.{name}")


def safe_pickle_load(file_obj: BinaryIO) -> Any:
    """Load pickle data with restricted globals.

    Uses RestrictedUnpickler to prevent arbitrary code execution
    from malicious pickle files.

    Args:
        file_obj: Binary file object to read pickle data from

    Returns:
        Unpickled Python object

    Raises:
        pickle.UnpicklingError: If pickle contains forbidden globals
    """
    logger.debug("Loading pickle data with security restrictions")
    return RestrictedUnpickler(file_obj).load()
