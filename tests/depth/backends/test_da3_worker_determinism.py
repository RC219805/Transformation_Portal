"""Per-input DA3 sky sampling is deterministic inside isolated workers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from transformation_portal.depth.backends.da3_worker import _input_inference_seed

pytestmark = pytest.mark.unit


def test_input_seed_binds_rgb_content_and_dimensions_not_file_encoding(tmp_path: Path) -> None:
    image = Image.new("RGB", (20, 20), (100, 120, 200))
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    image.save(first, compress_level=0)
    image.save(second, compress_level=9)
    assert first.read_bytes() != second.read_bytes()
    with Image.open(first) as left, Image.open(second) as right:
        assert _input_inference_seed(left) == _input_inference_seed(right)
    seed = _input_inference_seed(image)
    image.putpixel((0, 0), (100, 120, 201))
    assert _input_inference_seed(image) != seed
    assert _input_inference_seed(Image.new("RGB", (10, 40), (100, 120, 200))) != seed


@pytest.mark.ml
def test_isolated_workers_repeat_sky_quantile_without_changing_parent_rng(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "sky.png"
    Image.new("RGB", (16, 16), (100, 120, 200)).save(source)
    # Exercise the actual worker prediction boundary with the upstream sky
    # algorithm's >100000-value sampling branch, without model downloads.
    script = r"""
import hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from transformation_portal.depth.backends import da3_worker
class SkyEngine:
    def predict(self, image):
        depth = torch.linspace(0.01, 199.0, 250000)
        indices = torch.randint(0, depth.numel(), (100000,), device=depth.device)
        sky_depth = torch.quantile(depth[indices], 0.99).item()
        return SimpleNamespace(depth_map=np.full((16,16), sky_depth, dtype=np.float32),
            original_image=np.asarray(image), metadata={"device":"cpu", "sky_depth":sky_depth,
            "sample_sha256":hashlib.sha256(indices.numpy().tobytes()).hexdigest()})
da3_worker._build_inference_engine = lambda **kwargs: SkyEngine()
output=Path(sys.argv[2])
da3_worker._run_inference(input_image=Path(sys.argv[1]), output_depth=output.with_suffix(".npy"),
    output_json=output, model_variant_name="METRIC_LARGE", model_key=None, device="cpu", use_coreml=False, non_commercial_ok=True)
"""
    state = torch.random.get_rng_state().clone()
    outputs = []
    root = Path(__file__).resolve().parents[3]
    for i in range(2):
        output = tmp_path / f"result_{i}.json"
        completed = subprocess.run(
            [sys.executable, "-c", script, str(source), str(output)],
            env={**os.environ, "PYTHONPATH": f"{root / 'src'}:{root}", "OMP_NUM_THREADS": "1"},
            capture_output=True,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr.decode()
        outputs.append(json.loads(output.read_text())["metadata"])
    assert outputs[0]["sample_sha256"] == outputs[1]["sample_sha256"]
    assert outputs[0]["sky_depth"] == outputs[1]["sky_depth"]
    assert outputs[0]["inference_seed_policy"] == "tp.da3.rgb-seed.v1"
    assert outputs[0]["inference_seed"] == _input_inference_seed(Image.open(source))
    assert torch.equal(state, torch.random.get_rng_state())
