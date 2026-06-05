from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "ad_editorial_post_pipeline.py"


def _load_tool(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setitem(sys.modules, "rawpy", ModuleType("rawpy"))
    reportlab_module = ModuleType("reportlab")
    reportlab_lib_module = ModuleType("reportlab.lib")
    reportlab_pagesizes_module = ModuleType("reportlab.lib.pagesizes")
    reportlab_pagesizes_module.A4 = (595.27, 841.89)
    reportlab_utils_module = ModuleType("reportlab.lib.utils")
    reportlab_utils_module.ImageReader = object
    reportlab_pdfgen_module = ModuleType("reportlab.pdfgen")
    reportlab_canvas_module = ModuleType("reportlab.pdfgen.canvas")

    class _Canvas:
        def __init__(self, *args, **kwargs) -> None:
            pass

    reportlab_canvas_module.Canvas = _Canvas
    monkeypatch.setitem(sys.modules, "reportlab", reportlab_module)
    monkeypatch.setitem(sys.modules, "reportlab.lib", reportlab_lib_module)
    monkeypatch.setitem(sys.modules, "reportlab.lib.pagesizes", reportlab_pagesizes_module)
    monkeypatch.setitem(sys.modules, "reportlab.lib.utils", reportlab_utils_module)
    monkeypatch.setitem(sys.modules, "reportlab.pdfgen", reportlab_pdfgen_module)
    monkeypatch.setitem(sys.modules, "reportlab.pdfgen.canvas", reportlab_canvas_module)

    spec = importlib.util.spec_from_file_location("ad_editorial_post_pipeline_under_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cfg(module: ModuleType, project_root: Path, *, selects: dict | None = None, metadata: dict | None = None):
    input_raw_dir = project_root / "input_raw"
    input_raw_dir.mkdir(parents=True, exist_ok=True)
    return module.PipelineConfig(
        project_name="SmithResidence",
        project_root=project_root,
        input_raw_dir=input_raw_dir,
        backup_raw_dir=None,
        rename={"enabled": False},
        selects=selects or {"use_csv": True},
        icc={},
        processing={"workers": 1},
        styles={"natural": {"exposure": 0, "contrast": 0, "saturation": 0}},
        consistency={"target_median": 0.42, "wb_neutralize": True},
        retouch={"dust_remove": False, "hotspot_reduce": False},
        export={
            "web_long_edge_px": 2500,
            "jpeg_quality": 96,
            "sharpen_web_amount": 0.35,
            "sharpen_print_amount": 0.1,
        },
        metadata=metadata or {},
        deliver={"zip": False},
    )


def test_selects_csv_relative_config_path_resolves_under_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_tool(monkeypatch)
    project_root = tmp_path / "project"
    cfg = _cfg(module, project_root, selects={"use_csv": True, "csv_path": "DOCS/selects.csv"})
    layout = module.Layout.build(cfg)

    csv_path = module.ensure_selects_csv(cfg, layout, [project_root / "RAW" / "Originals" / "image.CR3"])

    assert csv_path == project_root / "DOCS" / "selects.csv"
    assert csv_path.exists()
    assert not (PROJECT_ROOT / "DOCS" / "selects.csv").exists()


def test_selects_csv_absolute_config_path_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_tool(monkeypatch)
    project_root = tmp_path / "project"
    external_selects = tmp_path / "selects" / "keepers.csv"
    cfg = _cfg(module, project_root, selects={"use_csv": True, "csv_path": str(external_selects)})
    layout = module.Layout.build(cfg)

    csv_path = module.ensure_selects_csv(cfg, layout, [project_root / "RAW" / "Originals" / "image.CR3"])

    assert csv_path == external_selects
    assert external_selects.exists()


def test_filter_selects_reads_relative_config_path_from_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_tool(monkeypatch)
    project_root = tmp_path / "project"
    cfg = _cfg(module, project_root, selects={"use_csv": True, "csv_path": "DOCS/selects.csv"})
    selects_path = project_root / "DOCS" / "selects.csv"
    selects_path.parent.mkdir(parents=True)
    selects_path.write_text("filename,keep,notes\nkeep.CR3,1,\nskip.CR3,0,\n", encoding="utf-8")

    files = [project_root / "RAW" / "Originals" / "keep.CR3", project_root / "RAW" / "Originals" / "skip.CR3"]

    assert module.filter_selects(cfg, files) == [files[0]]


def test_metadata_csv_relative_config_path_resolves_under_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_tool(monkeypatch)
    project_root = tmp_path / "project"
    cfg = _cfg(module, project_root, metadata={"csv_path": "DOCS/metadata.csv"})

    metadata_path = module.project_relative_path(
        cfg.project_root,
        cfg.metadata.get("csv_path"),
        cfg.project_root / "DOCS" / "metadata.csv",
    )

    assert metadata_path == project_root / "DOCS" / "metadata.csv"


def test_tiff_output_names_use_canonical_tif_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_tool(monkeypatch)

    assert module.TIFF_SUFFIX == ".tif"
    assert module.tiff_filename("image") == "image.tif"
    assert module.tiff_filename("image", "_HDR") == "image_HDR.tif"
    assert module.tiff_filename("image", "_PANO") == "image_PANO.tif"


def test_active_pipeline_source_no_longer_uses_truncated_ti_suffix() -> None:
    source = TOOL_PATH.read_text(encoding="utf-8")

    assert '".ti"' not in source
    assert '"*.ti"' not in source


def test_contact_sheet_names_use_pdf_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_tool(monkeypatch)

    assert module.PDF_SUFFIX == ".pdf"
    assert module.contact_sheet_filename("natural") == "contact_natural.pdf"
    assert Path(module.contact_sheet_filename("natural")).suffix != ".pd"


def test_load_image_float_preserves_uint16_range(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_tool(monkeypatch)
    image = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
    monkeypatch.setattr(module.Image, "open", lambda _path: image)

    result = module.load_image_float(Path("intermediate.tif"))

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.array([[[0.0, 32768 / 65535.0, 1.0]]], dtype=np.float32))


def test_load_image_float_preserves_uint8_range(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_tool(monkeypatch)
    image = np.array([[[0, 128, 255]]], dtype=np.uint8)
    monkeypatch.setattr(module.Image, "open", lambda _path: image)

    result = module.load_image_float(Path("preview.jpg"))

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.array([[[0.0, 128 / 255.0, 1.0]]], dtype=np.float32))


def test_active_pipeline_intermediate_reloads_use_bit_depth_aware_loader() -> None:
    source = TOOL_PATH.read_text(encoding="utf-8")

    assert "np.array(Image.open(p)).astype(np.float32) / 255.0" not in source
    assert "np.array(Image.open(pt)).astype(np.float32) / 255.0" not in source
    assert "load_image_float(p)" in source
    assert "load_image_float(pt)" in source
