from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

from transformation_portal.vlm_captioning.__main__ import main

pytestmark = pytest.mark.unit


def test_standalone_cli_writes_proxy_sidecar_and_raw(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    Image.new("RGB", (20, 10), (30, 60, 90)).save(source)
    runtime_dir = tmp_path / "runtime"
    package_dir = runtime_dir / "mlx_vlm"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "generate.py").write_text(
        "print('SCENE=Pool; MATERIALS=tile; FEATURES=steps; NATURAL=sky; LIGHTING=daylight; ISSUES=none; UNCERTAIN=none.')\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "FastVLM-1.5B-int8"
    model_path.mkdir()
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "--input-image",
            str(source),
            "--output-dir",
            str(output_dir),
            "--model-path",
            str(model_path),
            "--fastvlm-python",
            sys.executable,
            "--mlx-vlm-dir",
            str(runtime_dir),
            "--proxy-format",
            "png",
            "--max-side-px",
            "1600",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "image_proxy.png").is_file()
    assert (output_dir / "vlm_captioning.sidecar.json").is_file()
    assert (output_dir / "vlm_captioning.raw.txt").read_text(encoding="utf-8").startswith("SCENE=Pool")
