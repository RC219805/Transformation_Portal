"""Subprocess-only FastVLM runtime wrapper and sidecar builder."""

from __future__ import annotations

import os
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
    "Describe this image in one line using exactly this format: "
    "SCENE=<short>; MATERIALS=<items>; FEATURES=<items>; NATURAL=<items>; "
    "LIGHTING=<short>; ISSUES=<items>; UNCERTAIN=<items>."
)
ADVISORY_WARNING = "FastVLM output is advisory and may hallucinate objects or under-report quality issues."


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


def resolve_fastvlm_model_path(selector: str, *, runtime_root: Path | None = None) -> Path:
    """Resolve ``default|review|smoke`` or an explicit model path."""
    normalized = str(selector or "default").strip()
    role = normalized.lower()
    if role in FASTVLM_CHECKPOINT_DIRS:
        root = runtime_root or Path(".runtime/fastvlm")
        return root / "checkpoints" / FASTVLM_CHECKPOINT_DIRS[role]
    return Path(normalized)


def config_from_env(*, model_role: str = "default") -> FastVLMRuntimeConfig:
    """Build FastVLM runtime config from TP_FASTVLM_* environment variables."""
    enabled_value = os.getenv("TP_FASTVLM_ENABLED", "0").strip().lower()
    model_env = "TP_FASTVLM_REVIEW_MODEL" if model_role == "review" else "TP_FASTVLM_MODEL"
    model_default = Path(".runtime/fastvlm/checkpoints") / FASTVLM_CHECKPOINT_DIRS.get(
        model_role,
        FASTVLM_CHECKPOINT_DIRS["default"],
    )
    return FastVLMRuntimeConfig(
        enabled=enabled_value in {"1", "true", "yes", "on"},
        python_path=Path(os.getenv("TP_FASTVLM_PYTHON", ".runtime/fastvlm/.venv-fastvlm/bin/python")),
        mlx_vlm_dir=Path(os.getenv("TP_FASTVLM_MLX_VLM_DIR", ".runtime/fastvlm/mlx-vlm")),
        model_path=Path(os.getenv(model_env, str(model_default))),
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


def run_fastvlm_caption(
    config: FastVLMRuntimeConfig,
    image_path: Path,
    *,
    prompt: str = DEFAULT_FASTVLM_PROMPT,
) -> FastVLMRuntimeResult:
    """Run FastVLM through ``python -m mlx_vlm.generate``.

    Failures are returned as structured advisory results unless
    ``config.strict`` is true.
    """
    image = Path(image_path)
    command = [
        str(config.python_path),
        "-m",
        "mlx_vlm.generate",
        "--model",
        str(config.model_path),
        "--image",
        str(image),
        "--prompt",
        prompt,
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
    if not config.python_path.exists():
        return fail("missing_runtime", f"FastVLM Python executable not found: {config.python_path}")
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
    status = "ok" if success else "error"
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
