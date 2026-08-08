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
            ("pillow==12.3.0", "torch==2.13.0", "torchvision==0.28.0", "transformers==5.8.0"),
        ),
        (
            "scripts/setup/install_da3_runtime.sh",
            ('"pillow==12.3.0"', '"torch==2.13.0"', '"torchvision==0.28.0"', '"transformers==5.5.0"'),
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
