import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams, GaussianSplat, Scene3D
from transformation_portal.spatial_ai.reconstruction.export_ply import PLYExportConfig, PLYExporter
from transformation_portal.spatial_ai.reconstruction.geometric_validator import GeometricValidator
from transformation_portal.spatial_ai.reconstruction.mesh_exporter import MeshExporter

pytestmark = pytest.mark.unit


def _camera(
    *,
    camera_id: str | None = None,
    width: int = 8,
    height: int = 6,
    distortion: np.ndarray | None = None,
) -> CameraParams:
    return CameraParams(
        intrinsics=np.eye(3, dtype=np.float32),
        extrinsics=np.eye(4, dtype=np.float32),
        width=width,
        height=height,
        camera_id=camera_id,
        distortion=distortion,
    )


def _splats(positions: np.ndarray | None = None, colors: np.ndarray | None = None) -> GaussianSplat:
    if positions is None:
        positions = np.array(
            [
                [1.0, 1.0, 2.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 2.0],
            ],
            dtype=np.float32,
        )
    if colors is None:
        colors = np.array(
            [
                [1.0, 0.5, 0.0],
                [0.0, 0.5, 1.0],
                [0.25, 0.75, 0.5],
            ],
            dtype=np.float32,
        )

    count = len(positions)
    return GaussianSplat(
        positions=positions.astype(np.float32),
        colors=colors.astype(np.float32),
        scales=np.full((count, 3), 0.1, dtype=np.float32),
        rotations=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1)),
        opacities=np.full((count, 1), 0.8, dtype=np.float32),
    )


def _scene(*, splats: GaussianSplat | None = None, rmse: float = 0.02) -> Scene3D:
    return Scene3D(
        splats=splats or _splats(),
        cameras=[_camera(camera_id="cam-a"), _camera(camera_id="cam-b")],
        metadata={"source": "cpu-fill-gaps"},
        convergence="converged",
        iteration=12,
        rmse=rmse,
    )


def test_geometric_validator_uses_pre_rendered_images_and_validates_shapes() -> None:
    validator = GeometricValidator()
    scene = _scene()
    reference_images = [
        np.zeros((4, 4, 3), dtype=np.float32),
        np.ones((4, 4, 3), dtype=np.float32),
    ]
    rendered_images = [
        np.full((4, 4, 3), 0.5, dtype=np.float32),
        np.full((4, 4, 3), 0.25, dtype=np.float32),
    ]

    expected = np.sqrt(
        (np.mean((rendered_images[0] - reference_images[0]) ** 2) + np.mean((rendered_images[1] - reference_images[1]) ** 2))
        / 2
    )

    assert validator.compute_rmse(scene, reference_images, rendered_images=rendered_images) == pytest.approx(expected)

    with pytest.raises(ValueError, match="Number of rendered images"):
        validator.compute_rmse(scene, reference_images, rendered_images=rendered_images[:1])

    with pytest.raises(ValueError, match="Image shape mismatch"):
        validator.compute_rmse(
            scene,
            reference_images,
            rendered_images=[rendered_images[0], np.zeros((2, 2, 3), dtype=np.float32)],
        )


def test_geometric_validator_depth_consistency_branches() -> None:
    validator = GeometricValidator()
    splats = _splats(
        positions=np.array(
            [
                [2.0, 2.0, 2.0],
                [4.0, 4.0, 2.0],
                [20.0, 20.0, 2.0],
            ],
            dtype=np.float32,
        )
    )
    scene = Scene3D(
        splats=splats,
        cameras=[_camera(width=4, height=4), _camera(width=4, height=4)],
        metadata={},
        convergence="converged",
        iteration=1,
        rmse=0.02,
    )
    depth_map = np.zeros((4, 4), dtype=np.float32)
    depth_map[1, 1] = 2.0
    depth_map[2, 2] = 4.0

    assert validator.compute_depth_consistency(scene, [depth_map, depth_map], threshold=0.1) == pytest.approx(0.5)
    assert (
        validator.compute_depth_consistency(scene, [np.zeros_like(depth_map), np.zeros_like(depth_map)], threshold=0.1) == 0.0
    )

    with pytest.raises(ValueError, match="Number of depth maps"):
        validator.compute_depth_consistency(scene, [depth_map])


def test_geometric_validator_reprojection_and_coverage_edge_cases() -> None:
    validator = GeometricValidator()
    scene = _scene()

    assert validator.compute_reprojection_error(scene, view_idx=0, points_2d=None) == 0.0

    coverage = validator.compute_coverage(scene)
    assert coverage["mean_points_per_view"] == pytest.approx(3.0)
    assert coverage["min_points_per_view"] == 3
    assert coverage["max_points_per_view"] == 3

    with pytest.raises(ValueError, match="view index"):
        validator.compute_reprojection_error(scene, view_idx=4)

    with pytest.raises(ValueError, match="Number of 2D points"):
        validator.compute_reprojection_error(scene, view_idx=0, points_2d=np.zeros((1, 2), dtype=np.float32))


def _nonidentity_projection_scene() -> Scene3D:
    # A 90-degree rotation about Y plus translation maps world coordinates to
    # camera coordinates (z - 1, y - 2, 6 - x).
    camera = CameraParams(
        intrinsics=np.array([[4.0, 0.0, 2.0], [0.0, 6.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        extrinsics=np.array(
            [[0.0, 0.0, 1.0, -1.0], [0.0, 1.0, 0.0, -2.0], [-1.0, 0.0, 0.0, 6.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        width=8,
        height=7,
    )
    # Known camera coordinates are (0, 0, 2), (1, 1, 3), (-1, -2, 4),
    # and (12, 0, 5). The last point projects outside the image.
    positions = np.array([[4.0, 2.0, 1.0], [3.0, 3.0, 2.0], [2.0, 0.0, 0.0], [1.0, 2.0, 13.0]], dtype=np.float32)
    return Scene3D(
        splats=_splats(positions=positions, colors=np.full((4, 3), 0.5, dtype=np.float32)),
        cameras=[
            camera,
            CameraParams(
                intrinsics=camera.intrinsics.copy(),
                extrinsics=np.eye(4, dtype=np.float32),
                width=8,
                height=7,
            ),
        ],
        metadata={},
        convergence="converged",
        iteration=1,
        rmse=0.02,
    )


def test_geometric_validator_nonidentity_projection_uses_world_to_camera() -> None:
    scene = _nonidentity_projection_scene()
    # u = 2 + 4*x_camera/z_camera; v = 3 + 6*y_camera/z_camera.
    expected = [[2.0, 3.0], [10.0 / 3.0, 5.0], [1.0, 0.0], [58.0 / 5.0, 3.0]]

    projected = GeometricValidator()._project_points(scene.splats.positions, scene.cameras[0])

    np.testing.assert_allclose(projected, expected, rtol=1e-6, atol=1e-6)


def test_geometric_validator_nonidentity_depths_use_camera_axis() -> None:
    scene = _nonidentity_projection_scene()
    positions = np.vstack([scene.splats.positions, np.array([[8.0, 2.0, 1.0]], dtype=np.float32)])
    # The added point is (0, 0, -2) in camera coordinates. Preserve the
    # existing signed-depth behavior rather than introducing visibility culling.
    expected = [[2.0, 3.0, 2.0], [10.0 / 3.0, 5.0, 3.0], [1.0, 0.0, 4.0], [58.0 / 5.0, 3.0, 5.0], [2.0, 3.0, -2.0]]

    projected = GeometricValidator()._project_depths(positions, scene.cameras[0])

    np.testing.assert_allclose(projected, expected, rtol=1e-6, atol=1e-6)


def test_geometric_validator_nonidentity_public_reprojection_error() -> None:
    scene = _nonidentity_projection_scene()
    reference = np.array([[2.0, 3.0], [10.0 / 3.0, 5.0], [1.0, 0.0], [58.0 / 5.0, 3.0]], dtype=np.float32)
    validator = GeometricValidator()

    assert validator.compute_reprojection_error(scene, 0, points_2d=reference) == pytest.approx(0.0, abs=1e-6)
    assert validator.compute_reprojection_error(scene, 0, points_2d=reference + [3.0, 4.0]) == pytest.approx(5.0)


def test_geometric_validator_nonidentity_public_depth_consistency() -> None:
    scene = _nonidentity_projection_scene()
    depth_map = np.zeros((7, 8), dtype=np.float32)
    # Independently known pixel/depth pairs; the fourth point is out of bounds.
    depth_map[3, 2], depth_map[5, 3], depth_map[0, 1] = 2.0, 3.0, 4.0
    no_reference = np.zeros_like(depth_map)
    validator = GeometricValidator()

    assert validator.compute_depth_consistency(scene, [depth_map, no_reference]) == pytest.approx(1.0)
    depth_map[5, 3] = 6.0
    assert validator.compute_depth_consistency(scene, [depth_map, no_reference]) == pytest.approx(2.0 / 3.0)


def test_geometric_validator_nonidentity_public_coverage() -> None:
    # The rotated camera sees three points; the identity camera sees only the
    # fourth (the others are outside its image with the same intrinsics).
    coverage = GeometricValidator().compute_coverage(_nonidentity_projection_scene())

    assert coverage == {"mean_points_per_view": 2.0, "min_points_per_view": 1, "max_points_per_view": 3, "coverage_std": 1.0}


@pytest.mark.parametrize(
    ("rmse", "grade"),
    [
        (0.005, "A"),
        (0.015, "B"),
        (0.035, "C"),
        (0.08, "D"),
    ],
)
def test_geometric_validator_quality_grades_without_runtime_rendering(rmse: float, grade: str) -> None:
    result = GeometricValidator().validate_scene(_scene(rmse=rmse), reference_images=None, depth_maps=None)

    assert result["quality_grade"] == grade
    assert result["depth_consistency"] is None


def test_mesh_exporter_ascii_ply_without_attributes(tmp_path: Path) -> None:
    output_path = tmp_path / "scene_ascii.ply"

    MeshExporter().export_ply(_scene(), output_path, include_attributes=False, binary=False)

    content = output_path.read_text()
    assert "format ascii 1.0" in content
    assert "property float scale_x" not in content
    assert "property float opacity" not in content
    assert "1.000000 0.500000 0.000000" in content


def test_mesh_exporter_obj_subsamples_vertices_and_materials(tmp_path: Path) -> None:
    positions = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
            [4.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    colors = np.tile(np.array([[0.2, 0.4, 0.6]], dtype=np.float32), (len(positions), 1))
    scene = _scene(splats=_splats(positions=positions, colors=colors))
    output_path = tmp_path / "subsampled.obj"

    MeshExporter().export_obj(scene, output_path, vertex_colors=True, subsample_factor=2)

    obj_lines = output_path.read_text().splitlines()
    mtl_text = output_path.with_suffix(".mtl").read_text()
    assert "mtllib subsampled.mtl" in obj_lines
    assert sum(line.startswith("v ") for line in obj_lines) == 3
    assert sum(line.startswith("usemtl color_") for line in obj_lines) == 3
    assert mtl_text.count("newmtl color_") == 3


def test_mesh_exporter_obj_without_vertex_colors_skips_material_file(tmp_path: Path) -> None:
    output_path = tmp_path / "points.obj"

    MeshExporter().export_obj(_scene(), output_path, vertex_colors=False)

    assert "mtllib" not in output_path.read_text()
    assert not output_path.with_suffix(".mtl").exists()


def test_mesh_exporter_export_cameras_writes_defaults_and_distortion(tmp_path: Path) -> None:
    scene = Scene3D(
        splats=_splats(),
        cameras=[
            _camera(distortion=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)),
            _camera(camera_id="explicit"),
        ],
        metadata={"job": "unit"},
        convergence="converged",
        iteration=7,
        rmse=0.01,
    )
    output_path = tmp_path / "cameras.json"

    MeshExporter().export_cameras(scene, output_path)

    payload = json.loads(output_path.read_text())
    assert payload["cameras"][0]["camera_id"] == "camera_000"
    assert payload["cameras"][0]["distortion"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert payload["cameras"][1]["camera_id"] == "explicit"
    assert payload["scene_metadata"]["num_gaussians"] == scene.splats.num_gaussians


def test_ply_exporter_defaults_bad_color_scale_and_writes_sidecar(tmp_path: Path) -> None:
    output_path = tmp_path / "splats.ply"
    scene = _scene()

    PLYExporter(PLYExportConfig(binary=False, include_attributes=False, color_scale=0)).export(
        scene,
        output_path,
        write_sidecar=True,
        additional_metadata={"request_id": "req-1"},
    )

    ply_text = output_path.read_text()
    sidecar = json.loads(output_path.with_suffix(".provenance.json").read_text())
    assert "255 127 0" in ply_text
    assert sidecar["request_metadata"] == {"request_id": "req-1"}
    assert sidecar["quality_score"] == scene.quality_score


class _MalformedSplats:
    def __init__(self, **overrides: Any) -> None:
        self.positions = np.zeros((2, 3), dtype=np.float32)
        self.colors = np.zeros((2, 3), dtype=np.float32)
        self.scales = np.ones((2, 3), dtype=np.float32)
        self.rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 1))
        self.opacities = np.ones(2, dtype=np.float32)
        for key, value in overrides.items():
            setattr(self, key, value)

    @property
    def num_gaussians(self) -> int:
        return len(self.positions)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("positions", np.zeros((2, 2), dtype=np.float32), "Expected positions shape"),
        ("colors", np.zeros((2, 4), dtype=np.float32), "Expected colors shape"),
        ("scales", np.zeros((2, 2), dtype=np.float32), "Expected scales shape"),
        ("rotations", np.zeros((2, 3), dtype=np.float32), "Expected rotations shape"),
        ("opacities", np.zeros((1,), dtype=np.float32), "Expected opacities first dimension"),
    ],
)
def test_ply_exporter_rejects_malformed_binary_shapes(
    tmp_path: Path,
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    scene = SimpleNamespace(splats=_MalformedSplats(**{field: value}))

    with pytest.raises(ValueError, match=message):
        PLYExporter().export(scene, tmp_path / f"bad_{field}.ply")
