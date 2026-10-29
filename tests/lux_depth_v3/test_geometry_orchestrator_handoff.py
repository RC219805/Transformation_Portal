"""Corrected V3 geometry survives real V2 execution and Materials handoff.

Synthetic coordinate photographs and controlled model/Materials boundaries prove
spatial consistency and authority. They do not run inference or establish
representative photographic quality, runtime or production acceptance.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile
from PIL import Image

from tests.lux_depth_v3.test_geometry_corrected_boundary import _orient, _source, _support
from transformation_portal.depth.backends.protocol import DepthResult, LicenseType
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.execution_plan_adapter import LuxExecutionPlanAuthorityError
from transformation_portal.lux_depth_v3.geometry import NORMALIZED_GEOMETRY_RECIPE
from transformation_portal.lux_depth_v3.input_manager import ImageInput
from transformation_portal.lux_depth_v3.preprocessing import preprocess_geometry_snapshot

pytestmark = pytest.mark.unit


def _orchestrator(tmp_path, monkeypatch, *, materials_engine=None, **config_overrides):
    import transformation_portal.lux_depth_v3.orchestrator as module

    class CoordinateBackend:
        name = "da3"
        license_type = LicenseType.COMMERCIAL

        @staticmethod
        def ensure_available():
            return None

        @staticmethod
        def compute(image):
            encoded = np.asarray(image).copy()
            return DepthResult(
                depth_map=encoded[..., 0].astype(np.float32) + 1,
                original_image=encoded,
                metadata={},
                depth_units="relative",
                backend_id="da3",
                device="cpu",
            )

    backend = CoordinateBackend()

    class FixtureRegistry:
        @staticmethod
        def get_backend(_backend_id, _config=None, **_kwargs):
            return backend

    monkeypatch.setattr(module, "DepthBackendRegistry", FixtureRegistry)
    if materials_engine is not None:
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.materials_v3.MaterialsV3Engine",
            lambda _config: materials_engine,
        )
    values = {
        "geometry_recipe": NORMALIZED_GEOMETRY_RECIPE,
        "depth_backend": "da3",
        "depth_device": "cpu",
        "depth_operational_fallback_chain": ("da3",),
        "depth_fallback": "fail",
        "enable_v2": True,
        "v2_input_color": "srgb",
        "enable_materials_v3": materials_engine is not None,
        "enable_depth_cache": False,
        "save_float_depth": True,
        "output_bit_depth": 8,
    }
    values.update(config_overrides)
    orchestrator = module.EnhanceOrchestrator(EnhanceConfig(**values), tmp_path / "outputs")
    orchestrator.postprocessor = SimpleNamespace(process=lambda result: result)
    return orchestrator


def _depth_stage(orchestrator, source):
    depth = orchestrator.depth_dir / "coordinates_depth.png"
    metadata, *_ = orchestrator._compute_depth_stage(
        image_input=ImageInput(source),
        output_key=Path("coordinates"),
        depth_path=depth,
        float_depth_path=depth.with_suffix(".npy"),
        manifest_path=orchestrator.manifests_dir / "coordinates.json",
        skip_depth=False,
    )
    return depth, metadata


def _v2_arguments(orchestrator, source, depth, metadata):
    return {
        "image_input": ImageInput(source),
        "source_image_input": ImageInput(source),
        "depth_path": depth,
        "output_key": Path("coordinates"),
        "v2_log_path": orchestrator.output_root / "v2.log",
        "manifest_path": orchestrator.manifests_dir / "coordinates.json",
        "skip_depth": False,
        "geometry_transform": metadata.stats["geometry"],
        "depth_artifact_sha256": metadata.stats["depth_png_sha256"],
    }


def _materials_carrier(orchestrator, source, monkeypatch):
    geometry = preprocess_geometry_snapshot(source)

    def process(*, image, **_kwargs):
        enhanced = image.copy()
        enhanced[..., 0] += 0.05
        return {
            "enhanced_image": enhanced,
            "material_masks": {},
            "materials_v3_metadata": {"segmentation_metadata": {}},
            "materials_v3_pixel_ops": {"applied": []},
        }

    orchestrator.materials_v3_engine = SimpleNamespace(process=process)
    monkeypatch.setattr(
        "transformation_portal.lux_depth_v3.segmentation_backend.segment_materials", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        "transformation_portal.lux_depth_v3.segmentation_backend.get_last_segmentation_runtime_metadata", lambda: {}
    )
    result, _, carrier = orchestrator._run_materials_v3_stage(
        preprocessed_array=geometry.pixels,
        source_path=source,
        depth_map=np.ones(geometry.transform.model_shape, dtype=np.float32),
        output_key=Path("carrier"),
        artifact_shape=geometry.transform.normalized_shape,
        geometry_result=geometry,
    )
    return result, carrier


@pytest.mark.parametrize("materials_handoff", [False, True])
def test_corrected_stage_a_to_actual_v2_subprocess_preserves_unsupported_borders(tmp_path, monkeypatch, materials_handoff):
    source, stored = _source(tmp_path, orientation=6)
    original_bytes = source.read_bytes()
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    depth, metadata = _depth_stage(orchestrator, source)
    arguments = _v2_arguments(orchestrator, source, depth, metadata)
    input_bytes = original_bytes
    if materials_handoff:
        materials, carrier = _materials_carrier(orchestrator, source, monkeypatch)
        input_bytes = carrier.read_bytes()
        assert input_bytes != original_bytes
        arguments.update(
            image_input=ImageInput(carrier),
            materials_input_path=carrier,
            materials_v3_result=materials,
            geometry_input_sha256=materials["geometry_carrier_sha256"],
        )
    captured = []
    actual_run = orchestrator.v2_runner.run

    def record_actual_run(**kwargs):
        captured.append(json.loads(kwargs["geometry_handoff_path"].read_text()))
        assert kwargs["depth_dir"] is None
        assert kwargs["geometry_recipe"] == NORMALIZED_GEOMETRY_RECIPE
        return actual_run(**kwargs)

    monkeypatch.setattr(orchestrator.v2_runner, "run", record_actual_run)
    result, runtime, report_path = orchestrator._run_v2_stage(**arguments)
    assert result["status"] == "success"
    assert result["processing_version"] == "1.3.0"
    assert result["depth_consumed"] is True
    assert runtime > 0
    assert report_path.is_file()
    receipt = result["geometry"]
    assert receipt["handoff"] == captured[0]
    assert receipt["handoff_identity_sha256"] == captured[0]["identity_sha256"]
    assert receipt["handoff"]["source_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert receipt["handoff"]["v2_input_sha256"] == hashlib.sha256(input_bytes).hexdigest()
    assert json.loads(report_path.read_text())["geometry"] == receipt

    canonical = _orient(stored, 6)
    support = _support(canonical.shape[:2])
    with Image.open(result["output"]) as published:
        assert published.getexif().get(274, 1) == 1
        actual = np.asarray(published).copy()
    assert actual.shape == canonical.shape
    difference = np.abs(actual.astype(np.int16) - canonical.astype(np.int16))
    assert difference[~support].max() <= 1
    assert np.any(difference[support] > 1)
    assert not list((orchestrator.output_root / "temp").glob("geometry-depth-*"))
    assert source.read_bytes() == original_bytes


@pytest.mark.parametrize("failure", ["different-original", "missing-geometry", "missing-depth"])
def test_corrected_v2_rejects_missing_or_wrong_source_authority_before_child(tmp_path, monkeypatch, failure):
    source, _ = _source(tmp_path, orientation=6)
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    depth, metadata = _depth_stage(orchestrator, source)
    arguments = _v2_arguments(orchestrator, source, depth, metadata)
    if failure == "different-original":
        wrong = tmp_path / "same-encoded-shape.png"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (45, 31), color=(19, 53, 91)).save(wrong, exif=exif)
        arguments["source_image_input"] = ImageInput(wrong)
    elif failure == "missing-geometry":
        arguments["geometry_transform"] = None
    else:
        arguments["depth_path"] = None
    child = Mock(side_effect=AssertionError("Unverified source must not launch V2"))
    monkeypatch.setattr(orchestrator.v2_runner, "run", child)
    with pytest.raises(LuxExecutionPlanAuthorityError):
        orchestrator._run_v2_stage(**arguments)
    child.assert_not_called()
    assert not list(orchestrator.v2_dir.glob("*_report.json"))
    assert depth.is_file()


def test_corrected_v2_rejects_same_shape_depth_replacement_after_stage_a(tmp_path, monkeypatch):
    source, _ = _source(tmp_path, orientation=6)
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    depth, metadata = _depth_stage(orchestrator, source)
    expected_digest = metadata.stats["depth_png_sha256"]
    with Image.fromarray(np.full((45, 31), 12345, dtype=np.uint16)) as replacement:
        replacement.save(depth)
    assert hashlib.sha256(depth.read_bytes()).hexdigest() != expected_digest
    child = Mock(side_effect=AssertionError("Changed published depth must not launch V2"))
    monkeypatch.setattr(orchestrator.v2_runner, "run", child)
    with pytest.raises(LuxExecutionPlanAuthorityError):
        orchestrator._run_v2_stage(**_v2_arguments(orchestrator, source, depth, metadata))
    child.assert_not_called()


@pytest.mark.parametrize("carrier_kind", ["original-source", "materials-carrier"])
def test_corrected_v2_input_cannot_change_after_its_producer(tmp_path, monkeypatch, carrier_kind):
    source, _ = _source(tmp_path, orientation=6)
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    depth, metadata = _depth_stage(orchestrator, source)
    arguments = _v2_arguments(orchestrator, source, depth, metadata)
    if carrier_kind == "original-source":
        with Image.open(source) as image:
            original_shape = image.size
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", original_shape, color=(19, 53, 91)).save(source, exif=exif)
    else:
        result, carrier = _materials_carrier(orchestrator, source, monkeypatch)
        assert result["geometry_carrier_sha256"] == hashlib.sha256(carrier.read_bytes()).hexdigest()
        arguments.update(
            image_input=ImageInput(carrier),
            materials_input_path=carrier,
            materials_v3_result=result,
            geometry_input_sha256=result["geometry_carrier_sha256"],
        )
        Image.new("RGB", (31, 45), color=(19, 53, 91)).save(carrier)
    child = Mock(side_effect=AssertionError("Changed photographic input must not launch V2"))
    monkeypatch.setattr(orchestrator.v2_runner, "run", child)
    with pytest.raises(LuxExecutionPlanAuthorityError):
        orchestrator._run_v2_stage(**arguments)
    child.assert_not_called()


def test_corrected_authorized_masks_cannot_silently_disappear_when_serialization_fails(tmp_path, monkeypatch):
    source, _ = _source(tmp_path, orientation=6)
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    depth, metadata = _depth_stage(orchestrator, source)
    masks = {"glass": np.ones((45, 31), dtype=np.float32)}
    monkeypatch.setattr(
        "transformation_portal.lux_depth_v3.materials_v3.authorized_v2_material_masks",
        lambda *_args, **_kwargs: masks,
    )
    serializer = Mock(return_value=None)
    monkeypatch.setattr(orchestrator, "_serialize_material_masks", serializer)
    child = Mock(side_effect=AssertionError("Authorized mask loss must not launch V2"))
    monkeypatch.setattr(orchestrator.v2_runner, "run", child)
    arguments = _v2_arguments(orchestrator, source, depth, metadata)
    arguments["materials_v3_result"] = {"material_masks": masks}
    with pytest.raises(LuxExecutionPlanAuthorityError):
        orchestrator._run_v2_stage(**arguments)
    serializer.assert_called_once()
    child.assert_not_called()


@pytest.mark.parametrize("replacement_kind", ["archive-only", "producer-array-and-archive"])
def test_corrected_authorized_masks_cannot_be_replaced_by_another_valid_npz(tmp_path, monkeypatch, replacement_kind):
    source, _ = _source(tmp_path, orientation=6)
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    depth, metadata = _depth_stage(orchestrator, source)
    mask = np.zeros((45, 31), dtype=np.float32)
    mask[3:9, 4:10] = 1
    masks = {"glass": mask}
    monkeypatch.setattr(
        "transformation_portal.lux_depth_v3.materials_v3.authorized_v2_material_masks",
        lambda *_args, **_kwargs: masks,
    )
    actual_serialize = orchestrator._serialize_material_masks
    replacements = []

    def replace_serialized_mask(*args, **kwargs):
        artifact = actual_serialize(*args, **kwargs)
        assert artifact is not None
        replacement = np.ascontiguousarray(mask[:, ::-1])
        assert replacement.shape == mask.shape
        assert replacement.dtype == mask.dtype
        assert not np.array_equal(replacement, mask)
        if replacement_kind == "producer-array-and-archive":
            # A mutable producer alias cannot update the authority after its
            # authorized inventory was frozen before serialization began.
            mask[...] = replacement
        np.savez_compressed(artifact, glass=replacement)
        replacements.append(artifact)
        return artifact

    monkeypatch.setattr(orchestrator, "_serialize_material_masks", replace_serialized_mask)
    child = Mock(side_effect=AssertionError("Replaced authorized mask must not launch V2"))
    monkeypatch.setattr(orchestrator.v2_runner, "run", child)
    arguments = _v2_arguments(orchestrator, source, depth, metadata)
    arguments["materials_v3_result"] = {"material_masks": masks}
    with pytest.raises(LuxExecutionPlanAuthorityError):
        orchestrator._run_v2_stage(**arguments)
    child.assert_not_called()
    assert len(replacements) == 1
    assert not replacements[0].exists()


def test_corrected_negative_depth_normalizes_real_support_without_filling_unobserved_borders(tmp_path, monkeypatch):
    source, _ = _source(tmp_path, orientation=6)
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    observed_depth = []

    def negative_depth(image):
        encoded = np.asarray(image).copy()
        values = np.add.outer(np.arange(42, dtype=np.float32) * 0.5, np.arange(28, dtype=np.float32) * 0.25) - 100
        assert values.shape == encoded.shape[:2]
        assert np.isfinite(values).all() and values.max() < 0
        observed_depth.append(values)
        return DepthResult(
            depth_map=values,
            original_image=encoded,
            metadata={},
            depth_units="relative",
            backend_id="da3",
            device="cpu",
        )

    monkeypatch.setattr(orchestrator.depth_backend, "compute", negative_depth)
    depth, metadata = _depth_stage(orchestrator, source)
    assert len(observed_depth) == 1
    with Image.open(depth) as published:
        encoded = np.asarray(published, dtype=np.uint16)
    support = np.zeros((45, 31), dtype=bool)
    support[1:43, 1:29] = True
    assert np.all(encoded[~support] == 0)
    p01, p99 = np.percentile(observed_depth[0].astype(np.float64), [1, 99])
    expected = np.clip((observed_depth[0].astype(np.float64) - p01) / (p99 - p01), 0, 1)
    expected_u16 = np.rint(expected * 65535).astype(np.uint16)
    difference = np.abs(encoded[1:43, 1:29].astype(np.int32) - expected_u16.astype(np.int32))
    assert difference.max() <= 1
    normalization = metadata.scaling["normalization"]
    assert normalization["mode"] == "percentile_1_99"
    assert normalization["source_min"] == float(observed_depth[0].min())
    assert normalization["source_max"] == float(observed_depth[0].max())
    assert normalization["source_max"] < 0


@pytest.mark.parametrize("mutation", ["identity", "payload", "processing-version"])
def test_parent_rejects_child_receipt_that_does_not_match_frozen_handoff(tmp_path, monkeypatch, mutation):
    source, _ = _source(tmp_path, orientation=6)
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    depth, metadata = _depth_stage(orchestrator, source)
    calls = []

    def return_mutated_receipt(**kwargs):
        payload = json.loads(kwargs["geometry_handoff_path"].read_text())
        calls.append(payload)
        report = {
            "status": "success",
            "processing_version": "1.3.0",
            "geometry": {"handoff_identity_sha256": payload["identity_sha256"], "handoff": deepcopy(payload)},
        }
        if mutation == "identity":
            report["geometry"]["handoff_identity_sha256"] = "f" * 64
        elif mutation == "payload":
            report["geometry"]["handoff"]["source_sha256"] = "f" * 64
        else:
            report["processing_version"] = "1.2.0"
        return report

    monkeypatch.setattr(orchestrator.v2_runner, "run", return_mutated_receipt)
    with pytest.raises(LuxExecutionPlanAuthorityError, match="exact verified geometry execution receipt"):
        orchestrator._run_v2_stage(**_v2_arguments(orchestrator, source, depth, metadata))
    assert len(calls) == 1
    assert not list((orchestrator.output_root / "temp").glob("geometry-depth-*"))
    assert depth.is_file()


def test_corrected_published_artifact_reuse_is_disabled_before_legacy_discovery(tmp_path, monkeypatch):
    source, _ = _source(tmp_path, orientation=6)
    orchestrator = _orchestrator(tmp_path, monkeypatch)
    depth, _ = _depth_stage(orchestrator, source)
    manifest = orchestrator.manifests_dir / "coordinates.json"
    report = orchestrator.v2_dir / "coordinates_report.json"
    manifest.write_text("untrusted historical manifest must not be read")
    report.write_text("untrusted historical report must not be read")
    monkeypatch.setattr(
        orchestrator,
        "_load_existing_manifest",
        lambda *_args, **_kwargs: pytest.fail("Corrected mode must not authorize published legacy outputs"),
    )
    assert orchestrator.should_skip_depth(depth, manifest, ImageInput(source)) is False
    assert orchestrator.should_skip_v2(report, manifest, ImageInput(source), True) is False


def test_corrected_prepared_cache_reuses_native_depth_but_republishes_artifacts(tmp_path, monkeypatch):
    import transformation_portal.lux_depth_v3.orchestrator as module
    from tests.lux_depth_v3.test_depth_cache_runtime import _evidence
    from transformation_portal.core.execution_identity_v3 import BackendRuntimeIdentity, ExecutionIdentityV3
    from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
    from transformation_portal.stage_graph.registry import StageRegistryIdentifier

    root = tmp_path / "inputs"
    root.mkdir()
    source, _ = _source(root, orientation=6)
    prepared = prepare_lux_execution(
        EnhanceConfig(
            geometry_recipe=NORMALIZED_GEOMETRY_RECIPE,
            depth_backend="da2",
            depth_device="cpu",
            depth_operational_fallback_chain=("da2",),
            depth_fallback="fail",
            allow_synthetic_fallback=False,
            enable_depth_cache=True,
            enable_v2=False,
        ),
        root,
        [source],
    )
    node = next(node for node in prepared.plan.nodes if node.stage_registry_id is StageRegistryIdentifier.LUX_DEPTH)
    seed = ExecutionIdentityV3.from_plan(
        prepared.plan,
        stage_node_id=node.node_id,
        candidate_id="da2",
        input_id=prepared.plan.inputs[0].input_id,
    )
    evidence = _evidence(
        BackendRuntimeIdentity.from_seed(
            seed,
            materialized_weights_sha256="3" * 64,
            dependency_lock_sha256="4" * 64,
            interpreter_identity_sha256="5" * 64,
            platform_identity_sha256="6" * 64,
            accelerator_identity_sha256="7" * 64,
            source_identity_sha256="8" * 64,
        )
    )
    model_calls = []

    class FixtureBackend:
        name = "da2"
        license_type = LicenseType.COMMERCIAL

        @staticmethod
        def ensure_available():
            return None

        @staticmethod
        def prepare_cache_runtime_identity(**_kwargs):
            return evidence

        @staticmethod
        def verify_prepared_cache_runtime_identity(*, runtime_identity_sha256):
            return runtime_identity_sha256 == evidence.runtime_identity_sha256

        @staticmethod
        def compute(image):
            encoded = np.asarray(image).copy()
            model_calls.append(encoded)
            return DepthResult(
                depth_map=encoded[..., 0].astype(np.float32) + 1,
                original_image=encoded,
                metadata={"runtime_identity_sha256": evidence.runtime_identity_sha256},
                depth_units="relative",
                backend_id="da2",
                device="cpu",
            )

    class FixtureRegistry:
        @staticmethod
        def get_backend(_backend_id, _config=None, **_kwargs):
            return FixtureBackend()

    monkeypatch.setattr(module, "DepthBackendRegistry", FixtureRegistry)
    actual_writer = module.atomic_write_depth_u16_png_with_stats
    publications = []

    def record_publication(*args, **kwargs):
        publications.append(Path(args[0]))
        return actual_writer(*args, **kwargs)

    monkeypatch.setattr(module, "atomic_write_depth_u16_png_with_stats", record_publication)
    orchestrator = module.EnhanceOrchestrator.from_prepared(prepared, tmp_path / "outputs")
    orchestrator.postprocessor = SimpleNamespace(process=lambda result: result)
    first = orchestrator.enhance_batch(root, input_files=[source])[0]
    assert first["status"] == "ok"
    first_bytes = Path(first["depth_path"]).read_bytes()
    second = orchestrator.enhance_batch(root, input_files=[source])[0]
    assert second["status"] == "ok"
    assert len(model_calls) == 1
    assert model_calls[0].shape == (42, 28, 3)
    assert first["attempts"][0]["cached"] is False
    assert second["attempts"][0]["cached"] is True
    assert len(publications) == 2
    assert Path(second["depth_path"]).read_bytes() == first_bytes
    assert orchestrator.depth_cache.stats()["entry_count"] == 1


@pytest.mark.parametrize("shape,origin", [((31, 45), (1, 1)), ((29, 11), (0, 0))])
def test_corrected_materials_lifts_only_supported_delta_and_nearest_masks(tmp_path, monkeypatch, shape, origin):
    source, stored = _source(tmp_path, shape=shape)
    geometry = preprocess_geometry_snapshot(source)
    observed = []
    window = np.zeros(geometry.transform.model_shape, dtype=np.float32)
    window[2:7, 3:10] = 1
    padding = np.zeros(geometry.transform.model_shape, dtype=np.float32)
    if shape == (29, 11):
        padding[:, [0, -2, -1]] = 1

    def process(*, image, segmentation_result, depth_map):
        observed.append(image.copy())
        enhanced = image.copy()
        enhanced[..., 0] += window * 0.05 + padding * 0.4
        return {
            "enhanced_image": enhanced,
            "material_masks": {"glass": window, "metal": padding},
            "materials_v3_metadata": {"segmentation_metadata": {}},
            "materials_v3_pixel_ops": {"applied": []},
        }

    orchestrator = _orchestrator(tmp_path, monkeypatch, materials_engine=SimpleNamespace(process=process))
    monkeypatch.setattr(
        "transformation_portal.lux_depth_v3.segmentation_backend.segment_materials", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        "transformation_portal.lux_depth_v3.segmentation_backend.get_last_segmentation_runtime_metadata", lambda: {}
    )
    result, _, carrier = orchestrator._run_materials_v3_stage(
        preprocessed_array=geometry.pixels,
        source_path=source,
        depth_map=np.ones(geometry.transform.model_shape, dtype=np.float32),
        output_key=Path("materials"),
        artifact_shape=shape,
        geometry_result=geometry,
    )
    assert len(observed) == 1
    expected_mask = np.zeros(shape, dtype=np.float32)
    top, left = origin
    # Mixed-width padding consumes one model column at the left. A source crop
    # consumes one native row/column instead, so the two mappings must differ.
    x_offset = -1 if shape == (29, 11) else left
    expected_mask[top + 2 : top + 7, 3 + x_offset : 10 + x_offset] = 1
    np.testing.assert_array_equal(result["material_masks"]["glass"], expected_mask)
    np.testing.assert_array_equal(result["material_masks"]["metal"], np.zeros(shape))
    expected_image = stored.astype(np.float32) / 255
    expected_image[..., 0] += expected_mask * 0.05
    np.testing.assert_allclose(result["enhanced_image"], expected_image, atol=0.000001, rtol=0)
    assert result["materials_v3_metadata"]["segmentation_metadata"]["geometry"] == geometry.transform.to_payload()
    assert carrier.is_file()


@pytest.mark.parametrize("failure", ["changed-source", "changed-model-pixels", "alpha", "sixteen-bit"])
def test_corrected_materials_rejects_unsupported_source_before_segmentation_or_engine(tmp_path, monkeypatch, failure):
    if failure == "sixteen-bit":
        source = tmp_path / "source16.tif"
        tifffile.imwrite(source, np.full((31, 45, 3), 20000, np.uint16), photometric="rgb", metadata=None)
    else:
        source, _ = _source(tmp_path, alpha=failure == "alpha")
    geometry = preprocess_geometry_snapshot(source)
    model_pixels = geometry.pixels
    if failure == "changed-source":
        Image.new("RGB", (45, 31), color=(19, 53, 91)).save(source)
    elif failure == "changed-model-pixels":
        model_pixels = geometry.pixels[:, ::-1].copy()
    engine = Mock()
    segmentation = Mock(side_effect=AssertionError("Rejected source must not execute segmentation"))
    orchestrator = _orchestrator(tmp_path, monkeypatch, materials_engine=engine)
    monkeypatch.setattr("transformation_portal.lux_depth_v3.segmentation_backend.segment_materials", segmentation)
    with pytest.raises(LuxExecutionPlanAuthorityError):
        orchestrator._run_materials_v3_stage(
            preprocessed_array=model_pixels,
            source_path=source,
            depth_map=np.ones(geometry.transform.model_shape, dtype=np.float32),
            output_key=Path("rejected"),
            artifact_shape=geometry.transform.normalized_shape,
            geometry_result=geometry,
        )
    engine.process.assert_not_called()
    segmentation.assert_not_called()
    assert not (orchestrator.output_root / "temp").exists()


@pytest.mark.parametrize("failure", ["alpha", "sixteen-bit"])
def test_corrected_materials_source_limit_rejects_before_any_depth_compute(tmp_path, monkeypatch, failure):
    if failure == "sixteen-bit":
        source = tmp_path / "source16.tif"
        tifffile.imwrite(source, np.full((31, 45, 3), 20000, np.uint16), photometric="rgb", metadata=None)
    else:
        source, _ = _source(tmp_path, alpha=True)
    engine = Mock()
    orchestrator = _orchestrator(tmp_path, monkeypatch, materials_engine=engine)
    model = Mock(side_effect=AssertionError("Unsupported Materials source must reject before depth inference"))
    monkeypatch.setattr(orchestrator.depth_backend, "compute", model)
    with pytest.raises(LuxExecutionPlanAuthorityError, match="opaque 8-bit"):
        _depth_stage(orchestrator, source)
    model.assert_not_called()
    engine.process.assert_not_called()
    assert not list(orchestrator.depth_dir.glob("coordinates_depth.*"))


@pytest.mark.parametrize("unsupported", ["generate_pbr", "enable_reconstruction", "spatial_ai_linear_ingest"])
def test_corrected_recipe_rejects_unversioned_output_consumers_at_construction(tmp_path, monkeypatch, unsupported):
    with pytest.raises(ValueError, match="does not support"):
        _orchestrator(tmp_path, monkeypatch, **{unsupported: True})
    assert not (tmp_path / "outputs").exists()
