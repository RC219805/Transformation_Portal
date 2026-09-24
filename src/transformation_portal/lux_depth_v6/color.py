"""Explicit, deterministic grading and separate bounded SDR rendering.

The working master remains extended linear sRGB. Oklab chroma uses Bjorn
Ottosson's public-domain 2021 matrices: https://bottosson.github.io/posts/oklab/.
The display transform is a versioned local recipe, not ACES, AgX, or HDR output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np

from transformation_portal.core.image_artifact import ImageMaster, metadata_payload

MAX_TILE_PIXELS = 65536
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
_RGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float64,
)
_LMS_TO_LAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)
_LAB_TO_LMS = np.array(
    [[1, 0.3963377774, 0.2158037573], [1, -0.1055613458, -0.0638541728], [1, -0.0894841775, -1.2914855480]], dtype=np.float64
)
_LMS_TO_RGB = np.array(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ],
    dtype=np.float64,
)


def _number(value: Any, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-boolean number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite non-boolean number") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite non-boolean number")
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return result


@dataclass(frozen=True)
class GradeRecipe:
    """Global controls; white balance is explicit RGB gain, never inferred Kelvin."""

    exposure_stops: float = 0.0
    white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)
    contrast: float = 1.0
    pivot: float = 0.18
    saturation: float = 1.0

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("exposure_stops", -8, 8),
            ("contrast", 0.25, 4),
            ("pivot", 0.001, 1),
            ("saturation", 0, 2),
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name, lower, upper))
        if not isinstance(self.white_balance, (tuple, list)) or len(self.white_balance) != 3:
            raise ValueError("white_balance must contain three RGB gains")
        object.__setattr__(self, "white_balance", tuple(_number(v, "white_balance gain", 0.25, 4) for v in self.white_balance))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "tp.lux.grade.v1",
            "exposure_stops": self.exposure_stops,
            "white_balance": list(self.white_balance),
            "contrast": self.contrast,
            "pivot": self.pivot,
            "saturation": self.saturation,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GradeRecipe:
        expected = {"schema", "exposure_stops", "white_balance", "contrast", "pivot", "saturation"}
        if not isinstance(payload, Mapping) or set(payload) != expected or payload["schema"] != "tp.lux.grade.v1":
            raise ValueError("Unsupported grading recipe")
        return cls(**{key: value for key, value in payload.items() if key != "schema"})


@dataclass(frozen=True)
class RenderRecipe:
    """SDR display transform; clip_srgb is an explicit legacy comparison mode."""

    mode: str = "perceptual_srgb"
    shoulder: float = 0.8

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {"perceptual_srgb", "soft_srgb", "clip_srgb"}:
            raise ValueError("Unsupported display rendering mode")
        object.__setattr__(self, "shoulder", _number(self.shoulder, "shoulder", 0.1, 0.95))

    def to_payload(self) -> dict[str, Any]:
        return {"schema": "tp.lux.sdr_render.v1", "mode": self.mode, "shoulder": self.shoulder}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RenderRecipe:
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"schema", "mode", "shoulder"}
            or payload["schema"] != "tp.lux.sdr_render.v1"
        ):
            raise ValueError("Unsupported display rendering recipe")
        return cls(mode=payload["mode"], shoulder=payload["shoulder"])


def linear_srgb_to_oklab(values: np.ndarray) -> np.ndarray:
    """Ottosson 2021 linear-sRGB transform; signed cube roots retain finite negatives."""
    values = np.asarray(values, dtype=np.float64)
    return np.cbrt(values @ _RGB_TO_LMS.T) @ _LMS_TO_LAB.T


def oklab_to_linear_srgb(values: np.ndarray) -> np.ndarray:
    """Inverse of the named Oklab transform, without clipping scene values."""
    values = np.asarray(values, dtype=np.float64)
    return ((values @ _LAB_TO_LMS.T) ** 3) @ _LMS_TO_RGB.T


def _tiles(master: ImageMaster) -> Iterator[tuple[slice, slice]]:
    height, width = master.shape
    tile_width = min(width, MAX_TILE_PIXELS)
    tile_height = max(1, MAX_TILE_PIXELS // tile_width)
    for y in range(0, height, tile_height):
        for x in range(0, width, tile_width):
            yield slice(y, min(y + tile_height, height)), slice(x, min(x + tile_width, width))


def _result(master: ImageMaster, pixels: np.ndarray, metadata_key: str, settings: dict[str, Any]) -> ImageMaster:
    metadata = metadata_payload(master.metadata)
    metadata[metadata_key] = settings
    return ImageMaster(pixels, master.source_sha256, master.source_bit_depth, master.alpha, metadata, master.source_icc)


def grade_master(master: ImageMaster, recipe: GradeRecipe) -> tuple[ImageMaster, dict[str, Any]]:
    """Apply exposure, RGB gains, signed luminance contrast, then Oklab chroma.

    Contrast maps Y to sign(Y)*pivot*(abs(Y)/pivot)**contrast and scales RGB
    by the luminance ratio; exactly zero luminance is unchanged. Nonopaque RGB
    is protected bit-for-bit. Default controls preserve every pixel bit-for-bit.
    """
    if not isinstance(master, ImageMaster) or not isinstance(recipe, GradeRecipe):
        raise ValueError("Grading requires an ImageMaster and GradeRecipe")
    if master.metadata.get("color_domain") == "display_linear_srgb":
        raise ValueError("Display-rendered pixels cannot authorize scene grading")
    settings = recipe.to_payload()
    output = master.pixels.copy()
    changed = protected_count = 0
    max_delta = 0.0
    gain = np.exp2(recipe.exposure_stops) * np.asarray(recipe.white_balance, np.float64)
    identity = np.all(gain == 1) and recipe.contrast == recipe.saturation == 1
    for ys, xs in _tiles(master):
        baseline = master.pixels[ys, xs]
        allowed = np.ones(baseline.shape[:2], bool) if master.alpha is None else master.alpha[ys, xs] == 1
        protected_count += int(np.count_nonzero(~allowed))
        if identity or not allowed.any():
            continue
        candidate = baseline[allowed].astype(np.float64) * gain
        if recipe.contrast != 1:
            luminance = candidate @ _LUMA
            nonzero = luminance != 0
            multiplier = np.ones_like(luminance)
            multiplier[nonzero] = (np.abs(luminance[nonzero]) / recipe.pivot) ** (recipe.contrast - 1)
            candidate *= multiplier[:, None]
        if recipe.saturation != 1:
            # Rounded Oklab matrices assign tiny chroma to exact RGB neutrals.
            # Preserve the already exposed/balanced/contrasted values: changing
            # their saturation must not add a tint or overflow finite HDR gray.
            neutral = np.all(candidate == candidate[:, :1], axis=1)
            lab = linear_srgb_to_oklab(candidate)
            lab[:, 1:] *= recipe.saturation
            chroma_adjusted = oklab_to_linear_srgb(lab)
            chroma_adjusted[neutral] = candidate[neutral]
            candidate = chroma_adjusted
        if not np.isfinite(candidate).all() or np.any(np.abs(candidate) > np.finfo(np.float32).max):
            raise ValueError("Grade exceeds finite float32 master representation")
        stored = candidate.astype(np.float32)
        output[ys, xs][allowed] = stored
        delta = np.abs(stored.astype(np.float64) - baseline[allowed].astype(np.float64))
        max_delta = max(max_delta, float(delta.max(initial=0)))
        changed += int(np.count_nonzero(np.any(stored != baseline[allowed], axis=-1)))
    result = _result(master, output, "grade", settings)
    return result, {
        "schema": "tp.lux.grade_receipt.v1",
        "recipe": settings,
        "input_master_sha256": master.content_hash(),
        "output_master_sha256": result.content_hash(),
        "changed_pixels": changed,
        "protected_pixels": protected_count,
        "max_abs_delta": max_delta,
        "working_space": "extended_linear_srgb",
        "chroma_space": "oklab_2021",
        "contrast": "signed_luminance_power_zero_unchanged",
        "white_balance": "explicit_rgb_gains",
        "max_scratch_tile_pixels": MAX_TILE_PIXELS,
        "quality_acceptance": "unestablished",
    }


def _soft_render(values: np.ndarray, shoulder: float) -> tuple[np.ndarray, int, int]:
    """Map luminance, then contract linear RGB chroma into the display cube."""
    luminance = values @ _LUMA
    positive = luminance > 0
    mapped = np.maximum(luminance, 0)
    high = luminance > shoulder
    mapped[high] = shoulder - (1 - shoulder) * np.expm1(-(luminance[high] - shoulder) / (1 - shoulder))
    scaled = np.zeros_like(values)
    np.multiply(values, np.divide(mapped, luminance, out=np.zeros_like(mapped), where=positive)[..., None], out=scaled)
    chroma = scaled - mapped[..., None]
    factor = np.ones_like(mapped)
    for channel in range(3):
        value = chroma[..., channel]
        bound = np.ones_like(mapped)
        np.divide(1 - mapped, value, out=bound, where=value > 0)
        np.divide(-mapped, value, out=bound, where=value < 0)
        factor = np.minimum(factor, bound)
    factor = np.clip(factor, 0, 1)
    rendered = mapped[..., None] + factor[..., None] * chroma
    # Preserve the neutral region exactly and clamp only numerical boundary error.
    unchanged = (luminance <= shoulder) & np.all((values >= 0) & (values <= 1), axis=-1)
    rendered[unchanged] = values[unchanged]
    return np.clip(rendered, 0, 1), int(high.sum()), int(np.count_nonzero(factor < 1))


def _perceptual_render(values: np.ndarray, shoulder: float) -> tuple[np.ndarray, int, int]:
    """Tone-map Oklab lightness and contract chroma at constant lightness/hue.

    A C1 rational shoulder acts on L**3, a relative lightness measure, not
    physical luminance. Scaling a/b together retains the named Oklab hue.
    Fixed-iteration gamut contraction avoids channel clipping except rounding.
    Signed RGB is a numerical extension; perceptual validation uses positive
    RGB and excludes near-neutral samples whose hue is undefined.
    """
    lab = linear_srgb_to_oklab(values)
    lightness = lab[..., 0]
    brightness = np.maximum(lightness, 0) ** 3
    high = brightness > shoulder
    mapped = brightness.copy()
    delta = brightness[high] - shoulder
    mapped[high] = shoulder + (1 - shoulder) * (delta / (delta + 1 - shoulder))
    mapped_lightness = np.cbrt(mapped)
    ratio = np.divide(mapped_lightness, lightness, out=np.zeros_like(lightness), where=lightness > 0)
    lab *= ratio[..., None]
    lab[..., 0] = mapped_lightness
    rendered = oklab_to_linear_srgb(lab)
    # Unmodified in-gamut colors must not accumulate matrix round-trip error.
    unchanged = ~high & np.all((values >= 0) & (values <= 1), axis=-1)
    rendered[unchanged] = values[unchanged]
    outside = ~unchanged & np.any((rendered < 0) | (rendered > 1), axis=-1)
    if outside.any():
        selected = lab[outside]
        lower = np.zeros(len(selected), dtype=np.float64)
        upper = np.ones(len(selected), dtype=np.float64)
        trial = selected.copy()
        # 24 bisections resolve chroma to float32 precision. The lower bound is
        # always in gamut; the neutral fallback is exact, including at white.
        safe = np.repeat((selected[:, :1] ** 3), 3, axis=1)
        for _ in range(24):
            middle = (lower + upper) * 0.5
            trial[:, 1:] = selected[:, 1:] * middle[:, None]
            candidate = oklab_to_linear_srgb(trial)
            valid = np.all((candidate >= 0) & (candidate <= 1), axis=1)
            lower = np.where(valid, middle, lower)
            upper = np.where(valid, upper, middle)
            safe[valid] = candidate[valid]
        rendered[outside] = safe
    # Equal RGB is exactly neutral, rather than colored by rounded matrices.
    neutral = (values[..., 0] == values[..., 1]) & (values[..., 1] == values[..., 2])
    neutral_values = np.maximum(values[..., 0], 0).copy()
    neutral_high = neutral_values > shoulder
    delta = neutral_values[neutral_high] - shoulder
    neutral_values[neutral_high] = shoulder + (1 - shoulder) * (delta / (delta + 1 - shoulder))
    rendered[neutral] = np.repeat(neutral_values[neutral, None], 3, axis=-1)
    rendered[unchanged] = values[unchanged]
    return np.clip(rendered, 0, 1), int(high.sum()), int(outside.sum())


def render_master(master: ImageMaster, recipe: RenderRecipe) -> tuple[ImageMaster, dict[str, Any]]:
    """Create a separate bounded display-linear master for TIFF and PNG encoding.

    Perceptual rendering preserves the named Oklab hue within measured numeric
    tolerances; this is not a universal perceptual guarantee. Soft rendering
    retains the earlier linear-RGB comparison. Alpha is retained unchanged.
    """
    if not isinstance(master, ImageMaster) or not isinstance(recipe, RenderRecipe):
        raise ValueError("Rendering requires an ImageMaster and RenderRecipe")
    if master.metadata.get("color_domain") == "display_linear_srgb":
        raise ValueError("Display rendering cannot be applied twice")
    output = np.empty_like(master.pixels)
    changed = high_count = gamut_count = low_count = 0
    for ys, xs in _tiles(master):
        baseline = master.pixels[ys, xs]
        values = baseline.astype(np.float64)
        low_count += int(np.count_nonzero(values @ _LUMA <= 0))
        if recipe.mode == "perceptual_srgb":
            rendered, high, gamut = _perceptual_render(values, recipe.shoulder)
            high_count += high
            gamut_count += gamut
        elif recipe.mode == "soft_srgb":
            rendered, high, gamut = _soft_render(values, recipe.shoulder)
            high_count += high
            gamut_count += gamut
        else:
            rendered = np.clip(values, 0, 1)
            high_count += int(np.count_nonzero(np.any(values > 1, axis=-1)))
            gamut_count += int(np.count_nonzero(np.any((values < 0) | (values > 1), axis=-1)))
        output[ys, xs] = rendered.astype(np.float32)
        changed += int(np.count_nonzero(np.any(output[ys, xs] != baseline, axis=-1)))
    settings = recipe.to_payload()
    metadata = metadata_payload(master.metadata)
    metadata.update({"display_render": settings, "color_domain": "display_linear_srgb"})
    result = ImageMaster(output, master.source_sha256, master.source_bit_depth, master.alpha, metadata, master.source_icc)
    return result, {
        "schema": "tp.lux.sdr_render_receipt.v1",
        "recipe": settings,
        "input_master_sha256": master.content_hash(),
        "output_master_sha256": result.content_hash(),
        "changed_pixels": changed,
        "highlight_mapped_pixels": high_count,
        "gamut_compressed_pixels": gamut_count,
        "nonpositive_luminance_pixels": low_count,
        "output_domain": "display_linear_srgb",
        "gamut_method": {
            "perceptual_srgb": "oklab_constant_lightness_hue_chroma_contraction_24",
            "soft_srgb": "linear_luminance_neutral_chroma_contraction",
            "clip_srgb": "independent_channel_clip",
        }[recipe.mode],
        "max_scratch_tile_pixels": MAX_TILE_PIXELS,
        "quality_acceptance": "unestablished",
    }
