"""Tests for style preset asset validation and reporting."""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

import transformation_portal.style_transfer.ip_adapter as ip_adapter_module
from transformation_portal.style_transfer.ip_adapter import IPAdapterStyleTransfer
from transformation_portal.style_transfer.style_presets import (
    ArchitecturalStylePresets,
    PresetAssetsNotBundledError,
)

pytestmark = pytest.mark.unit


def test_resolve_reference_image_fails_when_asset_bundle_missing(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ArchitecturalStylePresets, "ASSETS_ROOT", tmp_path)

    with pytest.raises(PresetAssetsNotBundledError, match="not bundled") as exc_info:
        ArchitecturalStylePresets.resolve_reference_image("architectural_digest")

    assert exc_info.value.error_code == "asset_bundle_missing"


def test_apply_preset_style_resolves_bundled_reference_asset(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    asset_path = tmp_path / "references" / "editorial" / "architectural_digest.jpg"
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color="white").save(asset_path)
    monkeypatch.setattr(ArchitecturalStylePresets, "ASSETS_ROOT", tmp_path)

    style_transfer = object.__new__(IPAdapterStyleTransfer)
    style_transfer.transfer_style = Mock(return_value="styled")

    result = IPAdapterStyleTransfer.apply_preset_style(
        style_transfer,
        content_image="content.png",
        preset="architectural_digest",
        strength=0.55,
    )

    assert result == "styled"
    assert style_transfer.transfer_style.call_args.kwargs["style_reference"] == str(asset_path)


def test_transfer_style_reports_flux_img2img_backend() -> None:
    style_transfer = object.__new__(IPAdapterStyleTransfer)
    style_transfer.device = "cpu"
    style_transfer.flux_model_revision = "rev-123"
    style_transfer.last_capability_report = None
    style_transfer._load_image = lambda image: Image.new("RGB", (4, 4), color="white")
    style_transfer.encode_reference_image = lambda image: object()
    style_transfer.flux_pipe = Mock(return_value=type("Result", (), {"images": [Image.new("RGB", (4, 4), color="black")]})())

    result = IPAdapterStyleTransfer.transfer_style(
        style_transfer,
        content_image=Image.new("RGB", (4, 4), color="white"),
        style_reference=Image.new("RGB", (4, 4), color="black"),
    )

    assert isinstance(result, Image.Image)
    assert style_transfer.last_capability_report["executed_backend"] == "flux_img2img"
    assert style_transfer.last_capability_report["model_revision"] == "rev-123"


def test_ip_adapter_module_import_stays_lazy_without_optional_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = Path(ip_adapter_module.__file__).resolve()
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        if name in {"diffusers", "transformers"}:
            raise ModuleNotFoundError(f"No module named '{name}'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    spec = importlib.util.spec_from_file_location("test_ip_adapter_lazy_import", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.IPAdapterStyleTransfer.__name__ == "IPAdapterStyleTransfer"


def test_ip_adapter_init_fails_closed_when_optional_runtime_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ip_adapter_module, "torch", None)
    monkeypatch.setattr(ip_adapter_module, "_TORCH_IMPORT_ERROR", ModuleNotFoundError("No module named 'torch'"))

    with pytest.raises(ImportError, match="requires torch, transformers, and diffusers"):
        IPAdapterStyleTransfer()
