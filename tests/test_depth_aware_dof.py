import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.transformation_portal.depth.depth_aware_dof import DepthAwareDofOptions, main, run_depth_aware_dof

tifffile = pytest.importorskip("tifffile")

pytestmark = [pytest.mark.unit]


def _write_fixture(tmp_path: Path, *, height: int = 40, width: int = 60) -> tuple[Path, Path, Path]:
    yy, xx = np.indices((height, width))
    checker = ((xx + yy) % 2).astype(np.float32)
    gradient = xx.astype(np.float32) / max(width - 1, 1)
    rgb01 = np.stack(
        [
            0.20 + checker * 0.45,
            0.25 + gradient * 0.50,
            0.30 + (1.0 - gradient) * 0.35,
        ],
        axis=-1,
    )
    source = tmp_path / "source.tiff"
    tifffile.imwrite(source, np.rint(rgb01 * 65535.0).astype(np.uint16), photometric="rgb")

    depth = np.tile(np.linspace(2.0, 30.0, width, dtype=np.float32), (height, 1))
    depth_path = tmp_path / "source_depth.npy"
    np.save(depth_path, depth)

    metadata = tmp_path / "source_depth_metadata.json"
    metadata.write_text(json.dumps({"model": "depth_pro", "stats": {"convention": "higher_is_farther"}}), encoding="utf-8")
    return source, depth_path, metadata


def test_run_depth_aware_dof_uses_metadata_convention_and_writes_package(tmp_path):
    source, depth_path, metadata = _write_fixture(tmp_path)

    result = run_depth_aware_dof(
        DepthAwareDofOptions(
            source=source,
            depth_npy=depth_path,
            metadata=metadata,
            out_dir=tmp_path / "out",
            focus_depth=9.0,
        )
    )

    assert result.depth_convention == "higher-is-farther"
    assert result.production_tiff.exists()
    assert result.preview_jpeg.exists()
    assert result.diagnostic_contact_sheet.exists()
    assert result.summary_json.exists()
    assert result.package_zip.exists()

    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert summary["depth"]["convention"] == "higher-is-farther"
    assert summary["depth"]["metadata_model"] == "depth_pro"
    assert summary["outputs"]["production_tiff"]["sha256"] == result.artifact_hashes["production_tiff"]
    assert "sha256" not in summary["outputs"]["summary_json"]
    assert "sha256" not in summary["outputs"]["package_zip"]


def test_output_filenames_reflect_source_depth_and_preview_size(tmp_path):
    source, depth_path, metadata = _write_fixture(tmp_path)

    result = run_depth_aware_dof(
        DepthAwareDofOptions(
            source=source,
            depth_npy=depth_path,
            metadata=metadata,
            out_dir=tmp_path / "out",
            focus_depth=9.0,
            preview_long_edge=512,
        )
    )

    assert result.production_tiff.name == "source_depth_aware_DOF_16bit.tiff"
    assert result.preview_jpeg.name == "source_depth_aware_DOF_preview_512px.jpg"


def test_missing_metadata_convention_fails_closed(tmp_path):
    source, depth_path, _metadata = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="Depth convention is required"):
        run_depth_aware_dof(
            DepthAwareDofOptions(
                source=source,
                depth_npy=depth_path,
                out_dir=tmp_path / "out",
            )
        )


def test_depth_npy_requires_float_and_disallows_pickle(tmp_path):
    source, _depth_path, metadata = _write_fixture(tmp_path)
    object_depth = tmp_path / "object_depth.npy"
    np.save(object_depth, np.array([[{"not": "safe"}]], dtype=object))

    with pytest.raises(ValueError, match="Object arrays|pickle"):
        run_depth_aware_dof(
            DepthAwareDofOptions(
                source=source,
                depth_npy=object_depth,
                metadata=metadata,
                out_dir=tmp_path / "out",
            )
        )


def test_empty_depth_npy_fails_closed(tmp_path):
    source, _depth_path, metadata = _write_fixture(tmp_path)
    empty_depth = tmp_path / "empty_depth.npy"
    np.save(empty_depth, np.empty((0, 12), dtype=np.float32))

    with pytest.raises(ValueError, match="non-empty with positive height and width"):
        run_depth_aware_dof(
            DepthAwareDofOptions(
                source=source,
                depth_npy=empty_depth,
                metadata=metadata,
                out_dir=tmp_path / "out",
            )
        )


def test_depth_source_dimension_mismatch_fails(tmp_path):
    source, _depth_path, metadata = _write_fixture(tmp_path)
    mismatched_depth = tmp_path / "mismatched_depth.npy"
    np.save(mismatched_depth, np.ones((12, 12), dtype=np.float32))

    with pytest.raises(ValueError, match="dimension mismatch"):
        run_depth_aware_dof(
            DepthAwareDofOptions(
                source=source,
                depth_npy=mismatched_depth,
                metadata=metadata,
                out_dir=tmp_path / "out",
            )
        )


def test_optional_inputs_must_be_files(tmp_path):
    source, depth_path, _metadata = _write_fixture(tmp_path)
    metadata_dir = tmp_path / "metadata_dir"
    metadata_dir.mkdir()

    with pytest.raises(ValueError, match="metadata is not a file"):
        run_depth_aware_dof(
            DepthAwareDofOptions(
                source=source,
                depth_npy=depth_path,
                metadata=metadata_dir,
                depth_convention="higher-is-farther",
                out_dir=tmp_path / "out",
            )
        )


def test_16bit_tiff_output_preserves_uint16(tmp_path):
    source, depth_path, metadata = _write_fixture(tmp_path)

    result = run_depth_aware_dof(
        DepthAwareDofOptions(
            source=source,
            depth_npy=depth_path,
            metadata=metadata,
            out_dir=tmp_path / "out",
            focus_depth=9.0,
        )
    )

    output = tifffile.imread(result.production_tiff)
    assert output.dtype == np.uint16
    assert output.shape == (40, 60, 3)


def _write_float_fixture(tmp_path: Path, *, highlight: float) -> tuple[Path, Path, np.ndarray]:
    rgb = np.full((24, 32, 3), 0.40, dtype=np.float32)
    rgb[0, 0] = highlight
    source = tmp_path / "render.tif"
    tifffile.imwrite(source, rgb, photometric="rgb")
    depth_path = tmp_path / "render_depth.npy"
    np.save(depth_path, np.full((24, 32), 5.0, dtype=np.float32))
    return source, depth_path, rgb


@pytest.mark.parametrize("highlight", [0.9, 8.0])
def test_float_tiff_output_keeps_source_value_range(tmp_path, highlight):
    """Constant in-focus depth with no haze must round-trip a float render at its own scale.

    A scene-linear highlight above 1.0 used to divide the whole frame by its maximum and the
    32-bit TIFF and preview were written that way (body 0.40 -> 0.05).
    """

    source, depth_path, rgb = _write_float_fixture(tmp_path, highlight=highlight)

    result = run_depth_aware_dof(
        DepthAwareDofOptions(
            source=source,
            depth_npy=depth_path,
            out_dir=tmp_path / "out",
            depth_convention="higher-is-farther",
            haze_strength=0.0,
        )
    )

    output = tifffile.imread(result.production_tiff)
    assert output.dtype == np.float32
    np.testing.assert_allclose(output, rgb, rtol=1e-6, atol=1e-7)
    preview = np.asarray(Image.open(result.preview_jpeg).convert("RGB"), dtype=np.float32)
    assert abs(float(np.median(preview)) - 0.40 * 255.0) <= 2.0


def test_options_accept_str_paths(tmp_path):
    source, depth_path, metadata = _write_fixture(tmp_path)

    result = run_depth_aware_dof(
        DepthAwareDofOptions(
            source=str(source),  # type: ignore[arg-type]
            depth_npy=str(depth_path),  # type: ignore[arg-type]
            metadata=str(metadata),  # type: ignore[arg-type]
            out_dir=str(tmp_path / "out"),  # type: ignore[arg-type]
            focus_depth=9.0,
        )
    )

    assert result.production_tiff == tmp_path / "out" / "source_depth_aware_DOF_16bit.tiff"
    assert result.production_tiff.is_file()


def test_protection_mask_keeps_architecture_region_stable(tmp_path):
    source, depth_path, metadata = _write_fixture(tmp_path)
    protect_mask = tmp_path / "protect.png"
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[12:28, 18:38] = 255
    Image.fromarray(mask, mode="L").save(protect_mask)

    result = run_depth_aware_dof(
        DepthAwareDofOptions(
            source=source,
            depth_npy=depth_path,
            metadata=metadata,
            protect_mask=protect_mask,
            out_dir=tmp_path / "out",
            focus_depth=12.0,
        )
    )

    before = tifffile.imread(source).astype(np.float32) / 65535.0
    after = tifffile.imread(result.production_tiff).astype(np.float32) / 65535.0
    protected_delta = np.abs(after[12:28, 18:38] - before[12:28, 18:38]).mean()
    far_delta = np.abs(after[:, 48:58] - before[:, 48:58]).mean()
    assert protected_delta < far_delta * 0.5


def test_cli_writes_expected_outputs(tmp_path, capsys):
    source, depth_path, metadata = _write_fixture(tmp_path)
    out_dir = tmp_path / "out"

    exit_code = main(
        [
            "--source",
            str(source),
            "--depth-npy",
            str(depth_path),
            "--metadata",
            str(metadata),
            "--out-dir",
            str(out_dir),
            "--focus-depth",
            "9.0",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["production_tiff"]).exists()
    assert Path(payload["preview_jpeg"]).exists()
    assert Path(payload["diagnostic_contact_sheet"]).exists()
    assert Path(payload["summary_json"]).exists()
    assert Path(payload["package_zip"]).exists()


def test_legacy_pipeline_depth_tools_reexports_depth_aware_api():
    from src.transformation_portal.pipelines import depth_tools

    assert depth_tools.DepthAwareDofOptions is DepthAwareDofOptions
    assert depth_tools.run_depth_aware_dof is run_depth_aware_dof
