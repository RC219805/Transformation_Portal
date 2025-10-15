"""Utilities for rendering the Montecito coastal estate hero frame.

This helper wraps :mod:`lux_render_pipeline` with the exact guidance that the
Material Response review prescribed for reimagining the coastal estate aerial.
The command mirrors the reference shell invocation while exposing a Python API
that orchestration scripts can call directly.  The implementation purposely
loads the heavy diffusion stack lazily so test suites can stub the pipeline
module without importing ``diffusers`` or GPU bindings.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Mapping


COASTAL_ESTATE_PROMPT: str = (
    "luxury coastal estates Montecito golden hour, Spanish Mediterranean "
    "architecture, terracotta roofs glowing in sunset, emerald manicured lawns, "
    "Pacific Ocean with crystalline water gradients, palm trees casting long shadows, "
    "ultra-high-end real estate photography, architectural digest quality, "
    "material response technology, each surface expressing unique photonic signature"
)
"""Prompt tuned for the Montecito golden hour coastline brief."""


def _load_pipeline_module() -> ModuleType:
    """Import :mod:`lux_render_pipeline` lazily.

    The render stack pulls in ``diffusers`` and other GPU-heavy dependencies.
    Importing the module on demand keeps unit tests light and makes it easy to
    swap in fakes during validation.
    """

    return import_module("lux_render_pipeline")


_MANAGED_OPTION_PARAMS: Mapping[str, str] = {
    "input": "input_image",
    "out": "output_dir",
    "prompt": "prompt",
    "width": "width",
    "height": "height",
    "strength": "strength",
    "gs": "guidance_scale",
    "w4k": "export_4k",
    "use_realesrgan": "use_realesrgan",
    "brand_text": "brand_text",
    "logo": "brand_logo",
    "seed": "seed",
    "neg": "negative_prompt",
}


def render_coastal_estate(
    input_image: str,
    output_dir: str = "./transcended",
    *,
    prompt: str = COASTAL_ESTATE_PROMPT,
    width: int = 1536,
    height: int = 1024,
    strength: float = 0.35,
    guidance_scale: float = 8.0,
    seed: int | None = None,
    negative_prompt: str | None = None,
    export_4k: bool = False,
    use_realesrgan: bool = False,
    brand_text: str | None = None,
    brand_logo: str | None = None,
    extra_options: Mapping[str, Any] | None = None,
) -> None:
    """Run the coastal estate diffusion render with curated defaults.

    Parameters
    ----------
    input_image:
        Path to the base aerial capture.  Accepts any value understood by the
        CLI ``--input`` option (glob patterns are forwarded unchanged).
    output_dir:
        Destination directory mirroring the ``--out`` option.
    prompt:
        Positive prompt describing the coastal estate aesthetic.  Defaults to
        :data:`COASTAL_ESTATE_PROMPT` but can be overridden for experiments.
    width, height:
        Resolution used for the diffusion canvas.
    strength:
        How aggressively to deviate from the input capture.
    guidance_scale:
        Classifier-free guidance scale (``--gs``).
    seed:
        Optional deterministic seed.
    negative_prompt:
        Override for the negative prompt.  ``None`` falls back to the pipeline
        default.
    export_4k:
        Mirrors ``--w4k`` to enable the latent upscaling path.
    use_realesrgan:
        Mirrors ``--use_realesrgan`` for additional upscaling polish.
    brand_text, brand_logo:
        Caption and logo forwarded to the photo finishing stage.
    extra_options:
        Additional keyword arguments forwarded verbatim to
        :func:`lux_render_pipeline.main`.  Passing a key that collides with a
        managed argument raises :class:`ValueError` so the helper remains the
        single source of truth for the Material Response defaults.
    """

    pipeline_module = _load_pipeline_module()

    options: dict[str, Any] = {
        "input": input_image,
        "out": output_dir,
        "prompt": prompt,
        "width": width,
        "height": height,
        "strength": strength,
        "gs": guidance_scale,
        "w4k": export_4k,
        "use_realesrgan": use_realesrgan,
        "brand_text": brand_text,
        "logo": brand_logo,
    }

    if seed is not None:
        options["seed"] = seed
    if negative_prompt is not None:
        options["neg"] = negative_prompt

    if extra_options is not None:
        conflicting_keys = [key for key in extra_options if key in _MANAGED_OPTION_PARAMS]
        if conflicting_keys:
            formatted = ", ".join(sorted(f"'{key}'" for key in conflicting_keys))
            raise ValueError(f"extra option {formatted} conflicts with managed argument")

        for key, value in extra_options.items():
            options[key] = value

    pipeline_module.main(**options)


__all__ = ["COASTAL_ESTATE_PROMPT", "render_coastal_estate"]

