"""Core-only planning and imports do not require the execution supervisor."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_core_imports_and_cpu_planning_without_psutil(tmp_path):
    source_root = Path(__file__).resolve().parents[2] / "src"
    script = r"""
import importlib.abc
import pathlib
import sys

class MissingSupervisor(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "psutil" or fullname.startswith("psutil."):
            raise ModuleNotFoundError("No module named 'psutil'")

sys.meta_path.insert(0, MissingSupervisor())
from PIL import Image
from transformation_portal.lux_depth_v4 import LuxDepthV4Request, prepare
from transformation_portal.lux_depth_v4 import backend, pipeline, publication, raw

root = pathlib.Path(sys.argv[1])
inputs = root / "inputs"
inputs.mkdir()
Image.new("RGB", (14, 14)).save(inputs / "image.jpg")
output = root / "output"
prepared = prepare(LuxDepthV4Request(inputs, output, device="cpu"))
assert prepared.plan.to_payload()["device"] == "cpu"
assert not output.exists()
try:
    backend.require_process_supervisor()
except RuntimeError as error:
    assert "make install-ml-core" in str(error)
else:
    raise AssertionError("Missing process supervision must fail closed at execution")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(source_root)},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
