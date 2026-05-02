from __future__ import annotations

import sys
from pathlib import Path

import pytest

from transformation_portal.vlm_captioning.fastvlm_runtime import (
    FastVLMRuntimeConfig,
    build_fastvlm_sidecar,
    run_fastvlm_caption,
)
from transformation_portal.vlm_captioning.image_proxy import build_vlm_image_proxy

pytestmark = pytest.mark.unit


def _write_fake_mlx_module(runtime_dir: Path, body: str) -> None:
    package_dir = runtime_dir / "mlx_vlm"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "generate.py").write_text(body, encoding="utf-8")


def _config(tmp_path: Path, runtime_dir: Path) -> tuple[FastVLMRuntimeConfig, Path]:
    model = tmp_path / "model"
    model.mkdir()
    image = tmp_path / "image.png"
    image.write_bytes(b"not-a-real-image")
    return (
        FastVLMRuntimeConfig(
            enabled=True,
            python_path=Path(sys.executable),
            mlx_vlm_dir=runtime_dir,
            model_path=model,
            max_tokens=12,
            timeout_seconds=3,
        ),
        image,
    )


def test_runtime_success_output(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    _write_fake_mlx_module(
        runtime_dir,
        "print('SCENE=Pool; MATERIALS=stone, tile; FEATURES=steps; NATURAL=sky; LIGHTING=daylight; ISSUES=none; UNCERTAIN=none.')\n",
    )
    config, image = _config(tmp_path, runtime_dir)

    result = run_fastvlm_caption(config, image)

    assert result.success is True
    assert result.status == "ok"
    assert result.caption_parse.validated is True
    assert result.caption_parse.caption["scene"] == "Pool"
    assert result.raw_stdout
    assert result.raw_stderr == ""


def test_runtime_timeout(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    _write_fake_mlx_module(runtime_dir, "import time\ntime.sleep(5)\n")
    config, image = _config(tmp_path, runtime_dir)
    config = FastVLMRuntimeConfig(**{**config.__dict__, "timeout_seconds": 1})

    result = run_fastvlm_caption(config, image)

    assert result.success is False
    assert result.status == "timeout"
    assert "timed out" in (result.error or "")


def test_runtime_nonzero_exit(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    _write_fake_mlx_module(
        runtime_dir,
        "import sys\nprint('SCENE=Pool; MATERIALS=tile;')\nprint('boom', file=sys.stderr)\nsys.exit(7)\n",
    )
    config, image = _config(tmp_path, runtime_dir)

    result = run_fastvlm_caption(config, image)

    assert result.success is False
    assert result.status == "error"
    assert result.returncode == 7
    assert "boom" in result.raw_stderr


def test_runtime_malformed_output_returns_partial_parse(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    _write_fake_mlx_module(runtime_dir, "print('SCENE=Patio; MATERIALS=stone.')\n")
    config, image = _config(tmp_path, runtime_dir)

    result = run_fastvlm_caption(config, image)

    assert result.success is True
    assert result.caption_parse.validated is False
    assert result.caption_parse.caption == {"scene": "Patio", "materials": ["stone"]}


def test_runtime_missing_model_path(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    config = FastVLMRuntimeConfig(
        enabled=True,
        python_path=Path(sys.executable),
        mlx_vlm_dir=runtime_dir,
        model_path=tmp_path / "missing-model",
    )

    result = run_fastvlm_caption(config, image)

    assert result.success is False
    assert result.status == "missing_model"


def test_runtime_missing_runtime_path(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    config = FastVLMRuntimeConfig(
        enabled=True,
        python_path=Path(sys.executable),
        mlx_vlm_dir=tmp_path / "missing-runtime",
        model_path=model,
    )

    result = run_fastvlm_caption(config, image)

    assert result.success is False
    assert result.status == "missing_runtime"


def test_sidecar_is_advisory_and_preserves_diagnostics(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    from PIL import Image

    Image.new("RGB", (10, 10), (1, 2, 3)).save(source)
    proxy = build_vlm_image_proxy(source, tmp_path / "out")
    runtime_dir = tmp_path / "runtime"
    _write_fake_mlx_module(
        runtime_dir,
        "print('SCENE=Pool; MATERIALS=tile; FEATURES=steps; NATURAL=sky; LIGHTING=daylight; ISSUES=none; UNCERTAIN=none.')\n",
    )
    config, _image = _config(tmp_path, runtime_dir)
    result = run_fastvlm_caption(config, proxy.proxy_path)

    sidecar = build_fastvlm_sidecar(
        enabled=True,
        model_path=config.model_path,
        image_proxy=proxy,
        runtime_result=result,
        model_role="default",
    )

    payload = sidecar["vlm_captioning"]
    assert payload["role"] == "advisory"
    assert payload["used_for_quality_gate"] is False
    assert payload["runtime_diagnostics"]["stdout"]
