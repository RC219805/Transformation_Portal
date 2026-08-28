from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from transformation_portal.vlm_captioning.fastvlm_runtime import (
    DEFAULT_FASTVLM_PROMPT,
    REVIEW_FASTVLM_PROMPT,
    FastVLMRuntimeConfig,
    build_fastvlm_sidecar,
    config_from_env,
    infer_fastvlm_model_role,
    prompt_for_fastvlm_model,
    resolve_fastvlm_model_path,
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


def _command_prompt(command: list[str]) -> str:
    index = command.index("--prompt")
    return command[index + 1]


def test_governed_prompts_follow_model_role_and_checkpoint_path(tmp_path: Path) -> None:
    review_model = tmp_path / "FastVLM-7B-int4"
    default_model = tmp_path / "FastVLM-1.5B-int8"
    smoke_model = tmp_path / "FastVLM-0.5B-fp16"
    custom_model = tmp_path / "custom-model"

    assert infer_fastvlm_model_role(custom_model, "review") == "review"
    assert infer_fastvlm_model_role(review_model) == "review"
    assert infer_fastvlm_model_role(default_model) == "default"
    assert infer_fastvlm_model_role(smoke_model) == "smoke"
    assert infer_fastvlm_model_role(custom_model) == "default"
    assert prompt_for_fastvlm_model(review_model) == REVIEW_FASTVLM_PROMPT
    assert prompt_for_fastvlm_model(custom_model, "review") == REVIEW_FASTVLM_PROMPT
    assert prompt_for_fastvlm_model(default_model) == DEFAULT_FASTVLM_PROMPT
    assert prompt_for_fastvlm_model(smoke_model) == DEFAULT_FASTVLM_PROMPT
    assert prompt_for_fastvlm_model(custom_model) == DEFAULT_FASTVLM_PROMPT
    assert "Do not infer dusk" in REVIEW_FASTVLM_PROMPT
    assert "unless directly visible" in DEFAULT_FASTVLM_PROMPT


def test_runtime_selects_review_prompt_from_model_role(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    _write_fake_mlx_module(
        runtime_dir,
        "print('SCENE=Pool; MATERIALS=stone; FEATURES=steps; NATURAL=sky; LIGHTING=daylight; ISSUES=none; UNCERTAIN=none.')\n",
    )
    config, image = _config(tmp_path, runtime_dir)

    result = run_fastvlm_caption(config, image, model_role="review")

    assert result.success is True
    assert _command_prompt(result.command) == REVIEW_FASTVLM_PROMPT


def test_runtime_selects_review_prompt_from_checkpoint_path(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    _write_fake_mlx_module(
        runtime_dir,
        "print('SCENE=Pool; MATERIALS=stone; FEATURES=steps; NATURAL=sky; LIGHTING=daylight; ISSUES=none; UNCERTAIN=none.')\n",
    )
    config, image = _config(tmp_path, runtime_dir)
    review_model = tmp_path / "FastVLM-7B-int4"
    review_model.mkdir()
    config = FastVLMRuntimeConfig(**{**config.__dict__, "model_path": review_model})

    result = run_fastvlm_caption(config, image)

    assert result.success is True
    assert _command_prompt(result.command) == REVIEW_FASTVLM_PROMPT


def test_runtime_explicit_prompt_overrides_governed_prompt(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    _write_fake_mlx_module(
        runtime_dir,
        "print('SCENE=Pool; MATERIALS=stone; FEATURES=steps; NATURAL=sky; LIGHTING=daylight; ISSUES=none; UNCERTAIN=none.')\n",
    )
    config, image = _config(tmp_path, runtime_dir)

    result = run_fastvlm_caption(config, image, prompt="CUSTOM_PROMPT", model_role="review")

    assert result.success is True
    assert _command_prompt(result.command) == "CUSTOM_PROMPT"


def test_runtime_empty_prompt_is_still_explicit_override(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    _write_fake_mlx_module(
        runtime_dir,
        "print('SCENE=Pool; MATERIALS=stone; FEATURES=steps; NATURAL=sky; LIGHTING=daylight; ISSUES=none; UNCERTAIN=none.')\n",
    )
    config, image = _config(tmp_path, runtime_dir)

    result = run_fastvlm_caption(config, image, prompt="", model_role="review")

    assert result.success is True
    assert _command_prompt(result.command) == ""


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
    assert not list(runtime_dir.rglob("*.pyc"))


def test_default_runtime_paths_resolve_from_repo_root_when_cwd_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TP_FASTVLM_PYTHON", raising=False)
    monkeypatch.delenv("TP_FASTVLM_MLX_VLM_DIR", raising=False)
    monkeypatch.delenv("TP_FASTVLM_MODEL", raising=False)
    repo_root = Path(__file__).resolve().parents[2]

    model_path = resolve_fastvlm_model_path("default")
    config = config_from_env()

    assert model_path == repo_root / ".runtime/fastvlm/checkpoints/FastVLM-1.5B-int8"
    assert config.python_path == repo_root / ".runtime/fastvlm/.venv-fastvlm/bin/python"
    assert config.mlx_vlm_dir == repo_root / ".runtime/fastvlm/mlx-vlm"
    assert config.model_path == repo_root / ".runtime/fastvlm/checkpoints/FastVLM-1.5B-int8"


def test_model_path_allows_explicit_paths_under_safe_runtime_root(tmp_path: Path) -> None:
    runtime_root = tmp_path / "fastvlm"
    model_path = runtime_root / "checkpoints" / "custom-model"

    resolved = resolve_fastvlm_model_path(
        str(model_path),
        runtime_root=runtime_root,
        allowed_roots=(runtime_root,),
    )

    assert resolved == Path(os.path.realpath(model_path))


def test_model_path_rejects_absolute_paths_outside_safe_runtime_root(tmp_path: Path) -> None:
    runtime_root = tmp_path / "fastvlm"
    outside_model = tmp_path / "outside" / "model"

    with pytest.raises(ValueError, match="safe model path"):
        resolve_fastvlm_model_path(
            str(outside_model),
            runtime_root=runtime_root,
            allowed_roots=(runtime_root,),
        )


def test_model_path_rejects_symlink_escape_from_safe_runtime_root(tmp_path: Path) -> None:
    runtime_root = tmp_path / "fastvlm"
    outside_root = tmp_path / "outside"
    runtime_root.mkdir()
    outside_root.mkdir()
    outside_model = outside_root / "model"
    outside_model.mkdir()
    symlink_model = runtime_root / "model-link"
    symlink_model.symlink_to(outside_model, target_is_directory=True)

    with pytest.raises(ValueError, match="safe model path"):
        resolve_fastvlm_model_path(
            str(symlink_model),
            runtime_root=runtime_root,
            allowed_roots=(runtime_root,),
        )


def test_model_path_rejects_unknown_bare_selector(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="known role or safe model path"):
        resolve_fastvlm_model_path("not-a-role", runtime_root=tmp_path)


def test_runtime_accepts_bare_python_executable_from_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_dir = tmp_path / "runtime"
    _write_fake_mlx_module(
        runtime_dir,
        "print('SCENE=Pool; MATERIALS=stone; FEATURES=steps; NATURAL=sky; LIGHTING=daylight; ISSUES=none; UNCERTAIN=none.')\n",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.symlink_to(Path(sys.executable))
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    config, image = _config(tmp_path, runtime_dir)
    config = FastVLMRuntimeConfig(**{**config.__dict__, "python_path": Path("python3")})

    result = run_fastvlm_caption(config, image)

    assert result.success is True
    assert Path(result.command[0]).resolve() == Path(sys.executable).resolve()


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


def test_runtime_classifies_headless_metal_failure_as_missing_runtime(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    _write_fake_mlx_module(
        runtime_dir,
        "import sys\n"
        "print('RuntimeError: [metal::load_device] No Metal device available.', file=sys.stderr)\n"
        "sys.exit(1)\n",
    )
    config, image = _config(tmp_path, runtime_dir)

    result = run_fastvlm_caption(config, image)

    assert result.success is False
    assert result.status == "missing_runtime"
    assert "No Metal device available" in result.raw_stderr


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
