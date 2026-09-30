"""Bounded identity of the wheel-owned CMS used by photographic replay."""

from __future__ import annotations

import importlib
from pathlib import Path, PurePosixPath
from typing import Any

from transformation_portal.core.execution_plan_v2 import require_digest
from transformation_portal.depth.backends import da3_runtime_identity as identity

COLOR_PROCESSING_MODULES = frozenset(
    {
        "transformation_portal.lux_depth_v4.color_preparation",
        "transformation_portal.lux_depth_v4.color_preparation_evidence",
        "transformation_portal.lux_depth_v4.color_runtime",
    }
)
_SCHEMA = "tp.color.runtime.v1"


def color_runtime_identity() -> dict[str, Any]:
    """Bind imported CMS code to its actual bounded installed wheel closure.

    The supported imagecodecs wheel supplies LCMS. System-library discovery or
    an unrelated installed distribution cannot authorize a shadow CMS module.
    """
    package = importlib.import_module("imagecodecs")
    cms = importlib.import_module("imagecodecs._cms")
    entries: dict[str, dict[str, Any]] = {}
    token = identity._VERIFICATION_ENTRIES.set(entries)
    try:
        record = identity._distribution_record("imagecodecs", verify_record_hashes=False)
    finally:
        identity._VERIFICATION_ENTRIES.reset(token)
    origins = []
    for module in (package, cms):
        filename = getattr(module, "__file__", None)
        if not isinstance(filename, str):
            raise ValueError("Photographic CMS requires an installed module origin")
        origin = Path(filename).resolve(strict=True)
        if entries.get(str(origin), {}).get("kind") != "file":
            raise ValueError("Photographic CMS origin is outside its materialized wheel")
        origins.append(origin)
    package_root = origins[0].parent.parent
    bundled = {}
    for filename, entry in entries.items():
        path = Path(filename)
        if entry["kind"] != "file" or "lcms2" not in path.name.lower():
            continue
        if not any(suffix in path.name.lower() for suffix in (".dylib", ".so", ".dll")):
            continue
        if not path.is_relative_to(package_root):
            raise ValueError("Photographic CMS library is outside its wheel package root")
        bundled[path.relative_to(package_root).as_posix()] = identity._hash_regular_file(path)[0]
    if not bundled:
        raise ValueError("Photographic CMS requires wheel-bundled LCMS")
    result = {
        "schema": _SCHEMA,
        "imagecodecs_version": record["version"],
        "lcms_version": package.cms_version(),
        "wheel_record_sha256": record["record_sha256"],
        "wheel_files_sha256": record["installed_files_sha256"],
        "cms_extension_sha256": identity._hash_regular_file(origins[1])[0],
        "bundled_lcms": bundled,
    }
    validate_color_runtime_identity(result)
    return result


def validate_color_runtime_identity(payload: Any) -> None:
    """Validate portable CMS identity without importing or probing a runtime."""
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema",
            "imagecodecs_version",
            "lcms_version",
            "wheel_record_sha256",
            "wheel_files_sha256",
            "cms_extension_sha256",
            "bundled_lcms",
        }
        or payload["schema"] != _SCHEMA
    ):
        raise ValueError("Invalid photographic CMS runtime identity")
    for key in ("imagecodecs_version", "lcms_version"):
        if type(payload[key]) is not str or not 1 <= len(payload[key]) <= 128:
            raise ValueError("Photographic CMS versions must be bounded strings")
    if not payload["lcms_version"].startswith("lcms2 ") or len(payload["lcms_version"]) <= 6:
        raise ValueError("Photographic CMS requires the LCMS2 engine")
    for key in ("wheel_record_sha256", "wheel_files_sha256", "cms_extension_sha256"):
        require_digest(payload[key])
    bundled = payload["bundled_lcms"]
    if not isinstance(bundled, dict) or not 1 <= len(bundled) <= 8:
        raise ValueError("Photographic CMS requires a bounded bundled library inventory")
    for filename, digest in bundled.items():
        if (
            type(filename) is not str
            or not 1 <= len(filename) <= 512
            or "\\" in filename
            or PurePosixPath(filename).is_absolute()
            or any(part in {"", ".", ".."} for part in filename.split("/"))
            or "lcms2" not in PurePosixPath(filename).name.lower()
        ):
            raise ValueError("Invalid photographic CMS bundled library path")
        require_digest(digest)
