"""Diagnostic output must not authorize runtimes or recommend unmanaged repairs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.regression]
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_utility_missing_diffusers_points_to_governed_lane(monkeypatch, capsys):
    module = _load_script("model_install_diagnostic", "scripts/setup/install_models.py")
    monkeypatch.setitem(sys.modules, "diffusers", None)

    assert module.install_controlnet_models(dry_run=True) == 0
    output = capsys.readouterr().out
    assert "make install-ml-core" in output
    assert "Darwin arm64" in output
    assert "pip install diffusers torch" not in output


def test_model_utility_summary_distinguishes_isolated_workers(monkeypatch, capsys):
    module = _load_script("model_summary_diagnostic", "scripts/setup/install_models.py")
    monkeypatch.setattr(sys, "argv", ["install_models.py", "--dry-run", "--all"])
    for name in (
        "install_depth_models",
        "install_realesrgan_weights",
        "install_controlnet_models",
        "install_stable_diffusion_models",
    ):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: 0)

    assert module.main() == 0
    output = capsys.readouterr().out
    assert "requirements/README.md" in output
    assert "own installers" in output
    assert "pip install accelerate" not in output
    assert "pip install torch" not in output


@pytest.mark.parametrize("missing_depth_pro", [False, True])
def test_ml_diagnostic_preserves_import_status_without_claiming_inference(monkeypatch, capsys, missing_depth_pro):
    module = _load_script("ml_scope_diagnostic", "scripts/verification/verify_ml_deps.py")
    for name in ("numpy", "PIL", "cv2", "torchvision", "transformers", "diffusers", "depth_pro"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__version__="99.0"))
    torch = SimpleNamespace(
        __version__="99.0",
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "transformation_portal.depth.backends.synthetic",
        SimpleNamespace(SyntheticDepthBackend=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformation_portal.depth.backends.depth_pro",
        SimpleNamespace(DepthProBackend=object),
    )
    if missing_depth_pro:
        monkeypatch.setitem(sys.modules, "depth_pro", None)

    assert module.check_dependencies() == (1 if missing_depth_pro else 0)
    output = capsys.readouterr().out
    assert "Isolated runtime authorization and inference are not checked" in output
    assert "READY FOR APEX" not in output
    assert "ALL DEPENDENCIES VERIFIED" not in output
    if missing_depth_pro:
        assert "owning runtime" in output
    else:
        assert "version warnings" in output
