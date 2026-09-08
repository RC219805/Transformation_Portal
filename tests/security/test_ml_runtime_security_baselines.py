"""Contract tests for optional ML runtime security baselines."""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.security]

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("relative_path", "required_pins"),
    [
        (
            "config/fastvlm_runtime_requirements.txt",
            (
                "pip==26.2.1",
                "pillow==12.3.0",
                "torch==2.13.0",
                "torchvision==0.28.0",
                "transformers==5.10.1",
                "datasets==5.0.1",
                "gradio==6.22.0",
                "gradio_client==2.6.0",
                "huggingface_hub==1.16.0",
                # Gradio 6.22.0 requires tomlkit<0.15.0; move these together.
                "tomlkit==0.14.0",
            ),
        ),
        (
            "requirements/da3-runtime-darwin-arm64.txt",
            ("pillow==12.3.0", "torch==2.13.0", "torchvision==0.28.0", "transformers==5.10.1"),
        ),
        (
            "requirements/ml-core-darwin-arm64.txt",
            ("torch==2.13.0", "torchvision==0.28.0", "transformers==5.10.4"),
        ),
        (
            "scripts/setup/install_depth_pro_runtime.sh",
            ('"pillow==12.3.0"', '"torch==2.13.0"', '"torchvision==0.28.0"'),
        ),
    ],
)
def test_optional_ml_runtimes_use_current_security_baselines(
    relative_path: str,
    required_pins: tuple[str, ...],
) -> None:
    content = (REPO_ROOT / relative_path).read_text(encoding="utf-8").lower()

    for pin in required_pins:
        assert pin in content

    assert "pillow==12.2.0" not in content
    assert "torch==2.12.0" not in content
    assert "torch==2.12.1" not in content


def test_fastvlm_runtime_does_not_keep_stale_datasets_pin() -> None:
    content = (REPO_ROOT / "config/fastvlm_runtime_requirements.txt").read_text(encoding="utf-8").lower()

    assert "datasets==4.8.5" not in content


def test_supported_ml_metadata_rejects_vulnerable_transformers_series() -> None:
    supported_range = "transformers>=5.10.1,<5.11"

    for relative_path in ("requirements/ml-core.in", "requirements/ml-core-darwin-arm64.in"):
        content = (REPO_ROOT / relative_path).read_text(encoding="utf-8").lower()
        assert supported_range in content
        assert "transformers>=5.5.0,<5.6" not in content

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert pyproject.count(supported_range) == 2
    assert "transformers>=5.5.0,<5.6" not in pyproject


def test_fastvlm_mlx_runtime_pins_stay_coupled() -> None:
    content = (REPO_ROOT / "config/fastvlm_runtime_requirements.txt").read_text(encoding="utf-8").lower()
    mlx_versions = [line.removeprefix("mlx==") for line in content.splitlines() if line.startswith("mlx==")]
    mlx_metal_versions = [line.removeprefix("mlx-metal==") for line in content.splitlines() if line.startswith("mlx-metal==")]

    assert len(mlx_versions) == 1, "FastVLM requirements must contain exactly one exact mlx pin"
    assert len(mlx_metal_versions) == 1, "FastVLM requirements must contain exactly one exact mlx-metal pin"
    assert mlx_versions == mlx_metal_versions, "mlx and mlx-metal must use the same exact version on Darwin"
