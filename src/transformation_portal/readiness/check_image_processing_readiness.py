from __future__ import annotations

import importlib
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Tuple


def check_package(name: str) -> Tuple[bool, str]:
    """Return (installed, version_string_or_reason)."""
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        return False, str(e)

    version = getattr(mod, "__version__", None)
    if version is None:
        # Builtins like `sys` won't have __version__.
        return True, "unknown"
    return True, str(version)


def check_disk_space(path: str = ".") -> Dict[str, Any]:
    """Return disk space summary for the given path."""
    usage = shutil.disk_usage(path)
    return {
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
    }


def _callable_name(obj: Any) -> str:
    """Robust callable name (works for mocks, callables, partials)."""
    return getattr(obj, "__name__", obj.__class__.__name__)


def assess_capabilities() -> Dict[str, Any]:
    """
    Lightweight capability report.
    IMPORTANT: never assumes callables have __name__ (mocks often don't).
    """
    # These are deliberately conservative and dependency-light.
    checks = {
        "numpy": lambda: check_package("numpy")[0],
        "pillow": lambda: check_package("PIL")[0],
        "scipy": lambda: check_package("scipy")[0],
    }

    results: Dict[str, Any] = {"capabilities": {}}
    for key, fn in checks.items():
        try:
            results["capabilities"][key] = {"ok": bool(fn()), "check": _callable_name(fn)}
        except Exception as e:
            results["capabilities"][key] = {"ok": False, "error": str(e), "check": _callable_name(fn)}
    return results


def check_sample_images() -> Dict[str, Any]:
    """
    Placeholder for sample image checks.
    Kept minimal to avoid test/env coupling.
    """
    return {"ok": True, "count": 0}
