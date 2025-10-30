"""Tests for texture and sky environment integration in material response finishing."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image

# ``lux_render_pipeline`` depends on heavy diffusion stacks. Provide small stubs
# so we can import the module and exercise the finishing helpers without the
# runtime dependencies.
torch_stub = types.ModuleType("torch")
torch_cuda = types.ModuleType("torch.cuda")

torch_cuda.is_available = lambda: False  # type: ignore[attr-defined]
torch_cuda.manual_seed_all = lambda seed: None  # type: ignore[attr-defined]


class _Generator:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def manual_seed(self, seed: int) -> "_Generator":
        return self


torch_stub.cuda = torch_cuda
torch_stub.Generator = _Generator
torch_stub.manual_seed = lambda seed: None
torch_stub.float16 = float
torch_stub.float32 = float
torch_stub.inference_mode = lambda: (lambda fn: fn)

sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.cuda", torch_cuda)


diffusers_stub = types.ModuleType("diffusers")


class _DummyPipeline:
    def __init__(self) -> None:
        self.scheduler = types.SimpleNamespace(config={})

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # type: ignore[override]
        return cls()

    def to(self, device: str):  # pragma: no cover - unused in tests
        return self

    def __call__(self, *args, **kwargs):  # pragma: no cover - unused in tests
        dummy_image = Image.new("RGB", (8, 8), color=0)
        return types.SimpleNamespace(images=[dummy_image])


class _DummyScheduler:
    config: dict[str, object] = {}

    @classmethod
    def from_config(cls, config):  # type: ignore[override]
        inst = cls()
        inst.config = config
        return inst


diffusers_stub.ControlNetModel = _DummyPipeline
diffusers_stub.StableDiffusionControlNetImg2ImgPipeline = _DummyPipeline
diffusers_stub.StableDiffusionLatentUpscalePipeline = _DummyPipeline
diffusers_stub.UniPCMultistepScheduler = _DummyScheduler
diffusers_stub.StableDiffusionXLControlNetPipeline = _DummyPipeline
diffusers_stub.StableDiffusionXLImg2ImgPipeline = _DummyPipeline

sys.modules.setdefault("diffusers", diffusers_stub)


controlnet_aux_stub = types.ModuleType("controlnet_aux")


class _CannyDetector:
    def __call__(self, image: Image.Image) -> Image.Image:
        return image


class _MidasDetector:
    def __init__(self, model_type: str = "dpt_large") -> None:
        self.model_type = model_type

    def __call__(self, image: Image.Image) -> Image.Image:
        return image


controlnet_aux_stub.CannyDetector = _CannyDetector
controlnet_aux_stub.MidasDetector = _MidasDetector

sys.modules.setdefault("controlnet_aux", controlnet_aux_stub)

# Import after stubs are in place to prevent lux_render_pipeline from loading heavy ML dependencies during test setup
from lux_render_pipeline import apply_material_response_finishing  # pylint: disable=wrong-import-position


def _make_texture(path: Path, color: tuple[int, int, int]) -> None:
    tile = np.zeros((2, 2, 3), dtype=np.uint8)
    tile[..., 0] = color[0]
    tile[..., 1] = color[1]
    tile[..., 2] = color[2]
    Image.fromarray(tile, mode="RGB").save(path)


def test_floor_texture_blend_enriches_lower_rows(tmp_path: Path) -> None:
    rgb = np.full((8, 8, 3), 0.2, dtype=np.float32)
    texture_path = tmp_path / "floor.jpg"
    _make_texture(texture_path, (255, 0, 0))

    result = apply_material_response_finishing(
        rgb.copy(),
        texture_boost=0.0,
        ambient_occlusion=0.0,
        highlight_warmth=0.0,
        haze_strength=0.0,
        floor_plank_contrast=0.0,
        floor_specular=0.0,
        floor_contact_shadow=0.0,
        floor_texture_path=str(texture_path),
        floor_texture_strength=1.0,
        textile_contrast=0.0,
        leather_sheen=0.0,
        fireplace_glow=0.0,
        fireplace_glow_radius=1.0,
        window_reflection=0.0,
        bedding_relief=0.0,
        wall_texture_path=None,
        wall_texture_strength=0.0,
        wall_texture=0.0,
        painting_integration=0.0,
        window_light_wrap=0.0,
        pool_texture_path=None,
        pool_texture_strength=0.0,
        exterior_atmosphere=0.0,
        sky_environment_path=None,
        sky_environment_strength=0.0,
    )

    assert result[-1, :, 0].mean() > 0.5
    assert np.allclose(result[0, :, :], rgb[0, :, :], atol=1e-2)


def test_sky_environment_tints_exterior_region(tmp_path: Path) -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.float32)
    rgb[:3, 4:, :] = np.array([0.3, 0.35, 0.9], dtype=np.float32)

    sky_path = tmp_path / "sky.jpg"
    _make_texture(sky_path, (64, 128, 255))

    result = apply_material_response_finishing(
        rgb.copy(),
        texture_boost=0.0,
        ambient_occlusion=0.0,
        highlight_warmth=0.0,
        haze_strength=0.0,
        floor_plank_contrast=0.0,
        floor_specular=0.0,
        floor_contact_shadow=0.0,
        textile_contrast=0.0,
        leather_sheen=0.0,
        fireplace_glow=0.0,
        fireplace_glow_radius=1.0,
        window_reflection=0.0,
        bedding_relief=0.0,
        wall_texture_path=None,
        wall_texture_strength=0.0,
        wall_texture=0.0,
        painting_integration=0.0,
        window_light_wrap=0.0,
        pool_texture_path=None,
        pool_texture_strength=0.0,
        exterior_atmosphere=0.0,
        sky_environment_path=str(sky_path),
        sky_environment_strength=1.0,
    )

    sky_region = result[:3, 4:, :]
    interior_region = result[5:, :4, :]

    assert sky_region[..., 2].mean() > 0.2
    assert np.allclose(interior_region, rgb[5:, :4, :], atol=1e-2)
