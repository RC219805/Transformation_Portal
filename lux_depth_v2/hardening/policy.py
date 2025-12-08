from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Tuple

from .exceptions import PolicyViolationError


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v.strip())
    except Exception:
        return default


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v.strip())
    except Exception:
        return default


@dataclass(frozen=True)
class HardeningPolicy:
    """
    Guardrails for Lux Depth V2 execution.

    Design:
    - Safe defaults for production.
    - Fully opt-in (wrapper/service factory).
    - Environment-variable overrides for ops.
    """

    # Input constraints (security + stability)
    max_input_bytes: int = 50 * 1024 * 1024  # 50MB default (matches architecture guidance)
    max_input_megapixels: Optional[float] = None  # None => no MP cap
    allowed_input_exts: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

    # Output constraints (prevent writing outside allowed root)
    enforce_output_within: Optional[str] = None  # e.g., "/var/lux/output"
    safe_dir_mode: int = 0o750

    # Observability / reproducibility
    stamp_reports: bool = True
    stamp_include_input_hash: bool = False  # hashing huge TIFFs can be slow; opt-in
    write_run_manifest: bool = True
    run_manifest_name: str = "run_manifest.json"

    # Service hardening (FastAPI)
    enable_rate_limit: bool = False
    rate_limit_per_minute: int = 60
    enable_request_ids: bool = True

    # Logging
    redact_paths: bool = True

    # CI / dependency guardrails (script enforces these in requirements)
    banned_packages: Tuple[str, ...] = ("basicsr", "realesrgan", "gfpgan")

    # Extra allowlisted roots for input (optional)
    allowed_input_roots: Tuple[str, ...] = field(default_factory=tuple)

    @staticmethod
    def from_json(path: Path) -> "HardeningPolicy":
        data = json.loads(path.read_text())
        # Keep strict: unknown keys ignored by dataclass init would error; so we filter.
        allowed = {f.name for f in HardeningPolicy.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in allowed}
        return HardeningPolicy(**filtered)

    @staticmethod
    def from_env(prefix: str = "LUX_HARDEN_") -> "HardeningPolicy":
        """
        Override selected fields via env vars, e.g.:
          LUX_HARDEN_MAX_INPUT_BYTES=60000000
          LUX_HARDEN_ENABLE_RATE_LIMIT=true
        """
        base = HardeningPolicy()
        overrides = {
            "max_input_bytes": _env_int(prefix + "MAX_INPUT_BYTES", base.max_input_bytes),
            "max_input_megapixels": _env_float(prefix + "MAX_INPUT_MEGAPIXELS", base.max_input_megapixels),
            "stamp_reports": _env_bool(prefix + "STAMP_REPORTS", base.stamp_reports),
            "stamp_include_input_hash": _env_bool(prefix + "STAMP_INCLUDE_INPUT_HASH", base.stamp_include_input_hash),
            "write_run_manifest": _env_bool(prefix + "WRITE_RUN_MANIFEST", base.write_run_manifest),
            "enable_rate_limit": _env_bool(prefix + "ENABLE_RATE_LIMIT", base.enable_rate_limit),
            "rate_limit_per_minute": _env_int(prefix + "RATE_LIMIT_PER_MINUTE", base.rate_limit_per_minute),
            "enable_request_ids": _env_bool(prefix + "ENABLE_REQUEST_IDS", base.enable_request_ids),
            "redact_paths": _env_bool(prefix + "REDACT_PATHS", base.redact_paths),
        }
        enforce = os.getenv(prefix + "ENFORCE_OUTPUT_WITHIN")
        if enforce:
            overrides["enforce_output_within"] = enforce
        roots = os.getenv(prefix + "ALLOWED_INPUT_ROOTS")
        if roots:
            overrides["allowed_input_roots"] = tuple(r.strip() for r in roots.split(",") if r.strip())

        return HardeningPolicy(**{**base.__dict__, **overrides})

    @staticmethod
    def load(
        path: Optional[Path] = None,
        env_prefix: str = "LUX_HARDEN_",
    ) -> "HardeningPolicy":
        """
        Load from JSON (if provided), then apply env overrides.
        """
        policy = HardeningPolicy.from_json(path) if path else HardeningPolicy()
        env_policy = HardeningPolicy.from_env(env_prefix)
        # Merge env overrides over json/base where env differs from default.
        default_policy = HardeningPolicy()
        merged = policy.__dict__.copy()
        for k, v in env_policy.__dict__.items():
            if v != default_policy.__dict__[k]:
                merged[k] = v
        return HardeningPolicy(**merged)

    def assert_output_allowed(self, output_dir: Path) -> None:
        if not self.enforce_output_within:
            return
        root = Path(self.enforce_output_within).expanduser().resolve()
        out = output_dir.expanduser().resolve()
        if not _is_relative_to(out, root):
            raise PolicyViolationError(
                message="Output directory violates enforce_output_within",
                rule="enforce_output_within",
                details={"output_dir": str(out), "allowed_root": str(root)},
            )

    def normalize_allowed_exts(self) -> Tuple[str, ...]:
        return tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in self.allowed_input_exts)

    def normalize_allowed_roots(self) -> Tuple[Path, ...]:
        return tuple(Path(r).expanduser().resolve() for r in self.allowed_input_roots)

    def assert_input_under_allowed_roots(self, input_path: Path) -> None:
        roots = self.normalize_allowed_roots()
        if not roots:
            return
        p = input_path.expanduser().resolve()
        if any(_is_relative_to(p, r) for r in roots):
            return
        raise PolicyViolationError(
            message="Input is not under allowed_input_roots",
            rule="allowed_input_roots",
            details={"input": str(p), "roots": [str(r) for r in roots]},
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def find_requirement_files(paths: Iterable[Path]) -> list[Path]:
    """
    Helper for CI scripts: return only files that look like requirements files.
    """
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob("**/*requirements*.txt")))
        elif p.is_file() and p.name.endswith(".txt"):
            out.append(p)
    return out
