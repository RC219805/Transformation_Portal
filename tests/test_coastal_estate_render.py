from __future__ import annotations

import types

import pytest

import coastal_estate_render as cer


class StubPipeline(types.SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(calls=[])

    def main(self, **kwargs):  # type: ignore[override]
        self.calls.append(kwargs)


@pytest.fixture()
def stub_pipeline(monkeypatch):
    stub = StubPipeline()
    monkeypatch.setattr(cer, "_load_pipeline_module", lambda: stub)
    return stub


def test_render_coastal_estate_uses_material_response_defaults(stub_pipeline: StubPipeline):
    cer.render_coastal_estate("coastal_estate.jpg")

    assert stub_pipeline.calls == [
        {
            "input": "coastal_estate.jpg",
            "out": "./transcended",
            "prompt": cer.COASTAL_ESTATE_PROMPT,
            "width": 1536,
            "height": 1024,
            "strength": 0.35,
            "gs": 8.0,
            "w4k": False,
            "use_realesrgan": False,
            "brand_text": None,
            "logo": None,
        }
    ]


def test_render_coastal_estate_allows_overrides(stub_pipeline: StubPipeline):
    cer.render_coastal_estate(
        "input.png",
        "./lux",
        strength=0.42,
        guidance_scale=7.3,
        seed=99,
        negative_prompt="bad",
        export_4k=True,
        use_realesrgan=True,
        brand_text="The Estate",
        brand_logo="logo.png",
        extra_options={"steps": 40},
    )

    assert stub_pipeline.calls == [
        {
            "input": "input.png",
            "out": "./lux",
            "prompt": cer.COASTAL_ESTATE_PROMPT,
            "width": 1536,
            "height": 1024,
            "strength": 0.42,
            "gs": 7.3,
            "w4k": True,
            "use_realesrgan": True,
            "brand_text": "The Estate",
            "logo": "logo.png",
            "seed": 99,
            "neg": "bad",
            "steps": 40,
        }
    ]


def test_render_coastal_estate_rejects_conflicting_extra_options(stub_pipeline: StubPipeline):
    with pytest.raises(ValueError) as excinfo:
        cer.render_coastal_estate("estate.tif", extra_options={"input": "other"})

    assert "conflicts" in str(excinfo.value)


@pytest.mark.parametrize(
    "extra_key",
    ["brand_text", "logo", "seed", "neg"],
)
def test_render_coastal_estate_rejects_optional_managed_keys(extra_key: str, stub_pipeline: StubPipeline):
    with pytest.raises(ValueError) as excinfo:
        cer.render_coastal_estate("estate.jpg", extra_options={extra_key: "value"})

    assert "conflicts" in str(excinfo.value)
