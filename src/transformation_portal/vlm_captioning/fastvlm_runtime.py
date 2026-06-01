"""Subprocess-only FastVLM runtime wrapper and sidecar builder."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from transformation_portal.ingest.canonical_json import dumps_json

from .image_proxy import VLMImageProxy
from .parser import FastVLMCaptionParse, parse_fastvlm_caption

FASTVLM_MODEL_ROLES = {
    "default": "apple/FastVLM-1.5B-int8",
    "review": "apple/FastVLM-7B-int4",
    "smoke": "apple/FastVLM-0.5B-fp16",
}
FASTVLM_CHECKPOINT_DIRS = {
    "default": "FastVLM-1.5B-int8",
    "review": "FastVLM-7B-int4",
    "smoke": "FastVLM-0.5B-fp16",
}
DEFAULT_FASTVLM_PROMPT = (
    "Return exactly one conservative visual-description line using this exact format "
    "and equals signs: SCENE=<short>; MATERIALS=<clearly visible materials>; "
    "FEATURES=<clearly visible architectural or site features>; "
    "NATURAL=<clearly visible natural elements>; LIGHTING=<visibly supported lighting>; "
    "ISSUES=<directly visible image-quality or scene issues, or none>; "
    "UNCERTAIN=<items that may be present but are not visually clear>. "
    "Use the uppercase keys exactly as shown. Use equals signs, not colons. "
    "Do not infer use-case, time of day, weather, traffic, construction activity, people, funding, "
    "maintenance, purpose, ownership, market value, quality status, building condition, "
    "or property condition unless directly visible. Do not use JSON. Do not repeat items. "
    "Do not add extra commentary before or after the line."
)
REVIEW_FASTVLM_PROMPT = (
    "Return exactly one conservative visual audit line using this exact format "
    "and equals signs: SCENE=<short>; MATERIALS=<only clearly visible materials>; "
    "FEATURES=<only clearly visible architectural or site features>; "
    "NATURAL=<only clearly visible natural elements>; LIGHTING=<only visibly supported lighting>; "
    "ISSUES=<only directly visible image-quality or scene issues, or none>; "
    "UNCERTAIN=<visually ambiguous items only>. Use the uppercase keys exactly as shown. "
    "Use equals signs, not colons. Do not infer dusk, sunset, golden hour, weather, season, "
    "traffic, construction activity, people, funding, maintenance, purpose, ownership, market value, "
    "architectural intent, quality status, building condition, or property condition unless directly visible. "
    "If an issue is not directly visible, write ISSUES=none. Do not use JSON. Do not repeat items. "
    "Do not add extra commentary before or after the line."
)
FASTVLM_PROMPTS = {
    "default": DEFAULT_FASTVLM_PROMPT,
    "review": REVIEW_FASTVLM_PROMPT,
    "smoke": DEFAULT_FASTVLM_PROMPT,
}
ADVISORY_WARNING = "FastVLM output is advisory and may hallucinate objects or under-report quality issues."


def _find_repo_root(start: Path | None = None) -> Path:
    """Resolve the checkout root for repo-local optional runtimes."""
    current = (start or Path(__file__).resolve()).resolve()
    search_root = current if current.is_dir() else current.parent
    for parent in [search_root, *search_root.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    return Path(__file__).resolve().parents[3]


def default_fastvlm_runtime_root() -> Path:
    """Return the repo-local FastVLM runtime root."""
    return _find_repo_root() / ".runtime" / "fastvlm"


def _has_path_separator(candidate: str) -> bool:
    return os.sep in candidate or (os.altsep is not None and os.altsep in candidate)


def resolve_fastvlm_runtime_path(candidate: Path | str) -> Path:
    """Resolve runtime-local paths against the checkout root, not cwd."""
    path = Path(os.fspath(candidate).strip()).expanduser()
    if path.is_absolute():
        return path
    return _find_repo_root() / path


def _resolve_fastvlm_allowed_path(candidate: Path, allowed_roots: tuple[Path, ...]) -> Path:
    """Resolve ``candidate`` and require it to stay under one allowed root."""
    resolved = Path(os.path.realpath(candidate))
    for root in allowed_roots:
        root_real = Path(os.path.realpath(root))
        try:
            resolved.relative_to(root_real)
        except ValueError:
            continue
        return resolved
    raise ValueError("FastVLM model selector must be a safe model path under an allowed runtime root.")


def resolve_fastvlm_python_executable(candidate: Path | str) -> str:
    """Resolve a FastVLM Python executable path or PATH command."""
    value = os.fspath(candidate).strip()
    if not value:
        raise FileNotFoundError("FastVLM Python executable must be a non-empty path or command.")

    if value.startswith(".") or _has_path_separator(value):
        path = resolve_fastvlm_runtime_path(value)
        if not path.exists():
            raise FileNotFoundError(f"FastVLM Python executable not found: {path}")
        return os.path.abspath(os.fspath(path))

    resolved = shutil.which(value)
    if resolved is None:
        raise FileNotFoundError(f"FastVLM Python executable not found on PATH: {value}")
    return resolved


@dataclass(frozen=True)
class FastVLMRuntimeConfig:
    """Configuration for the isolated FastVLM subprocess runtime."""

    enabled: bool
    python_path: Path
    mlx_vlm_dir: Path
    model_path: Path
    max_tokens: int = 120
    temperature: float = 0.0
    timeout_seconds: int = 180
    strict: bool = False


@dataclass(frozen=True)
class FastVLMRuntimeResult:
    """Structured FastVLM invocation result."""

    success: bool
    status: str
    caption_parse: FastVLMCaptionParse
    raw_stdout: str
    raw_stderr: str
    returncode: int | None
    command: list[str]
    runtime_seconds: float
    error: str | None = None

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "returncode": self.returncode,
            "command": list(self.command),
            "runtime_seconds": self.runtime_seconds,
            "error": self.error,
            "stdout": self.raw_stdout,
            "stderr": self.raw_stderr,
        }


def resolve_fastvlm_model_id(model_path: Path | str, role: str | None = None) -> str:
    """Resolve a display model id from a role or checkpoint path."""
    if role in FASTVLM_MODEL_ROLES:
        return FASTVLM_MODEL_ROLES[str(role)]
    name = Path(model_path).name
    for known_role, directory_name in FASTVLM_CHECKPOINT_DIRS.items():
        if name == directory_name:
            return FASTVLM_MODEL_ROLES[known_role]
    return name or str(model_path)


def infer_fastvlm_model_role(model_path: Path | str, role: str | None = None) -> str:
    """Infer a governed FastVLM model role from an explicit role or checkpoint directory."""
    normalized_role = str(role or "").strip().lower()
    if normalized_role in FASTVLM_MODEL_ROLES:
        return normalized_role
    name = Path(model_path).name
    for known_role, directory_name in FASTVLM_CHECKPOINT_DIRS.items():
        if name == directory_name:
            return known_role
    return "default"


def prompt_for_fastvlm_model(model_path: Path | str, role: str | None = None) -> str:
    """Return the governed prompt for a FastVLM model role or checkpoint path."""
    model_role = infer_fastvlm_model_role(model_path, role)
    return FASTVLM_PROMPTS.get(model_role, DEFAULT_FASTVLM_PROMPT)


def resolve_fastvlm_model_path(
    selector: str,
    *,
    runtime_root: Path | None = None,
    allowed_roots: tuple[Path, ...] | None = None,
) -> Path:
    """Resolve ``default|review|smoke`` or an explicit model path."""
    normalized = str(selector or "default").strip()
    if not normalized or normalized.startswith("~") or "\x00" in normalized:
        raise ValueError("FastVLM model selector must be a role or safe model path.")
    root = runtime_root or default_fastvlm_runtime_root()
    safe_roots = allowed_roots or (root, _find_repo_root())
    role = normalized.lower()
    if role in FASTVLM_CHECKPOINT_DIRS:
        return _resolve_fastvlm_allowed_path(root / "checkpoints" / FASTVLM_CHECKPOINT_DIRS[role], (root,))
    model_path = Path(normalized)
    if model_path.is_absolute():
        return _resolve_fastvlm_allowed_path(model_path, safe_roots)
    if normalized.startswith(".") or _has_path_separator(normalized):
        return _resolve_fastvlm_allowed_path(_find_repo_root() / model_path, safe_roots)
    raise ValueError("FastVLM model selector must be a known role or safe model path.")


def config_from_env(*, model_role: str = "default") -> FastVLMRuntimeConfig:
    """Build FastVLM runtime config from TP_FASTVLM_* environment variables."""
    enabled_value = os.getenv("TP_FASTVLM_ENABLED", "0").strip().lower()
    model_env = "TP_FASTVLM_REVIEW_MODEL" if model_role == "review" else "TP_FASTVLM_MODEL"
    runtime_root = default_fastvlm_runtime_root()
    model_default = (
        runtime_root
        / "checkpoints"
        / FASTVLM_CHECKPOINT_DIRS.get(
            model_role,
            FASTVLM_CHECKPOINT_DIRS["default"],
        )
    )
    python_env = os.getenv("TP_FASTVLM_PYTHON")
    mlx_vlm_env = os.getenv("TP_FASTVLM_MLX_VLM_DIR")
    model_value = os.getenv(model_env)
    return FastVLMRuntimeConfig(
        enabled=enabled_value in {"1", "true", "yes", "on"},
        python_path=(
            Path(python_env.strip()) if python_env and python_env.strip() else runtime_root / ".venv-fastvlm/bin/python"
        ),
        mlx_vlm_dir=(
            resolve_fastvlm_runtime_path(mlx_vlm_env) if mlx_vlm_env and mlx_vlm_env.strip() else runtime_root / "mlx-vlm"
        ),
        model_path=(
            resolve_fastvlm_model_path(model_value, runtime_root=runtime_root)
            if model_value and model_value.strip()
            else model_default
        ),
        max_tokens=int(os.getenv("TP_FASTVLM_MAX_TOKENS", "120")),
        temperature=float(os.getenv("TP_FASTVLM_TEMPERATURE", "0.0")),
        timeout_seconds=int(os.getenv("TP_FASTVLM_TIMEOUT_SECONDS", "180")),
    )


def _failure_result(
    *,
    status: str,
    error: str,
    command: list[str] | None = None,
    raw_stdout: str = "",
    raw_stderr: str = "",
    returncode: int | None = None,
    runtime_seconds: float = 0.0,
) -> FastVLMRuntimeResult:
    return FastVLMRuntimeResult(
        success=False,
        status=status,
        caption_parse=parse_fastvlm_caption(raw_stdout),
        raw_stdout=raw_stdout,
        raw_stderr=raw_stderr,
        returncode=returncode,
        command=command or [],
        runtime_seconds=runtime_seconds,
        error=error,
    )


def _classify_nonzero_fastvlm_status(stdout: str, stderr: str) -> str:
    """Classify non-zero FastVLM exits that represent unavailable runtime state."""
    text = f"{stdout}\n{stderr}".lower()
    if "no metal device available" in text or "metal::load_device" in text:
        return "missing_runtime"
    return "error"


def run_fastvlm_caption(
    config: FastVLMRuntimeConfig,
    image_path: Path,
    *,
    prompt: str | None = None,
    model_role: str | None = None,
) -> FastVLMRuntimeResult:
    """Run FastVLM through ``python -m mlx_vlm.generate``.

    Failures are returned as structured advisory results unless
    ``config.strict`` is true.
    """
    image = Path(image_path)
    active_prompt = prompt if prompt is not None else prompt_for_fastvlm_model(config.model_path, model_role)
    command = [
        str(config.python_path),
        "-m",
        "mlx_vlm.generate",
        "--model",
        str(config.model_path),
        "--image",
        str(image),
        "--prompt",
        active_prompt,
        "--max-tokens",
        str(config.max_tokens),
        "--temperature",
        str(config.temperature),
    ]

    def fail(status: str, error: str) -> FastVLMRuntimeResult:
        result = _failure_result(status=status, error=error, command=command)
        if config.strict:
            raise RuntimeError(error)
        return result

    if not config.enabled:
        return fail("disabled", "FastVLM captioning is disabled.")
    try:
        command[0] = resolve_fastvlm_python_executable(config.python_path)
    except FileNotFoundError as exc:
        return fail("missing_runtime", str(exc))
    if not config.mlx_vlm_dir.exists() or not config.mlx_vlm_dir.is_dir():
        return fail("missing_runtime", f"FastVLM mlx-vlm directory not found: {config.mlx_vlm_dir}")
    if not config.model_path.exists():
        return fail("missing_model", f"FastVLM model path not found: {config.model_path}")
    if not image.exists() or not image.is_file():
        return fail("missing_image", f"FastVLM image path not found: {image}")
    if config.max_tokens < 1:
        return fail("invalid_config", "FastVLM max_tokens must be greater than zero.")
    if config.timeout_seconds < 1:
        return fail("invalid_config", "FastVLM timeout_seconds must be greater than zero.")

    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=str(config.mlx_vlm_dir),
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        runtime_seconds = time.monotonic() - start
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        result = _failure_result(
            status="timeout",
            error=f"FastVLM subprocess timed out after {config.timeout_seconds}s.",
            command=command,
            raw_stdout=stdout,
            raw_stderr=stderr,
            runtime_seconds=runtime_seconds,
        )
        if config.strict:
            raise RuntimeError(result.error)
        return result
    except OSError as exc:
        runtime_seconds = time.monotonic() - start
        result = _failure_result(
            status="error",
            error=f"FastVLM subprocess failed to start: {exc}",
            command=command,
            runtime_seconds=runtime_seconds,
        )
        if config.strict:
            raise RuntimeError(result.error)
        return result

    runtime_seconds = time.monotonic() - start
    parsed = parse_fastvlm_caption(completed.stdout)
    success = completed.returncode == 0
    status = "ok" if success else _classify_nonzero_fastvlm_status(completed.stdout, completed.stderr)
    error = None if success else f"FastVLM subprocess exited with code {completed.returncode}."
    result = FastVLMRuntimeResult(
        success=success,
        status=status,
        caption_parse=parsed,
        raw_stdout=completed.stdout,
        raw_stderr=completed.stderr,
        returncode=completed.returncode,
        command=command,
        runtime_seconds=runtime_seconds,
        error=error,
    )
    if config.strict and not success:
        raise RuntimeError(error or "FastVLM subprocess failed.")
    return result


def build_fastvlm_sidecar(
    *,
    enabled: bool,
    model_path: Path,
    image_proxy: VLMImageProxy,
    runtime_result: FastVLMRuntimeResult | None,
    model_role: str | None = None,
    model_id: str | None = None,
    extra_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build governed advisory sidecar JSON."""
    parsed = runtime_result.caption_parse if runtime_result is not None else parse_fastvlm_caption("")
    warnings = [ADVISORY_WARNING]
    warnings.extend(parsed.warnings)
    if extra_warnings:
        warnings.extend(extra_warnings)
    payload = {
        "vlm_captioning": {
            "schema_version": "vlm_captioning.v1",
            "enabled": bool(enabled),
            "role": "advisory",
            "provider": "fastvlm",
            "runtime": "mlx_vlm",
            "model": model_id or resolve_fastvlm_model_id(model_path, model_role),
            "model_role": model_role,
            "model_path": str(model_path),
            "image_proxy": image_proxy.to_dict(),
            "caption": parsed.caption,
            "raw_model_text": parsed.raw_text,
            "validated": bool(parsed.validated),
            "used_for_quality_gate": False,
            "warnings": warnings,
        }
    }
    if parsed.missing_keys:
        payload["vlm_captioning"]["missing_keys"] = list(parsed.missing_keys)
    if runtime_result is not None:
        payload["vlm_captioning"]["runtime_diagnostics"] = runtime_result.to_diagnostics()
    return payload


def dumps_sidecar(payload: Mapping[str, Any]) -> str:
    return dumps_json(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
