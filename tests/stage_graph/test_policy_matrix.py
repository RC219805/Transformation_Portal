"""PolicyEngine and sub-policy decision matrix tests."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from transformation_portal.stage_graph.policy import (
    CachingPolicy,
    DevicePolicy,
    PolicyEngine,
    ProcessingPolicy,
    QualityPolicy,
    QualityPreset,
    SceneType,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("stage_name", "policy", "expected"),
    [
        (
            "depth_estimation",
            DevicePolicy(has_cuda=True, has_mps=True, has_coreml=True),
            "coreml",
        ),
        (
            "depth_estimation",
            DevicePolicy(
                has_cuda=True,
                has_mps=True,
                has_coreml=True,
                prefer_coreml_depth=False,
            ),
            "cuda",
        ),
        ("enhance", DevicePolicy(has_cuda=True, has_mps=True), "cuda"),
        ("enhance", DevicePolicy(has_mps=True), "mps"),
        ("enhance", DevicePolicy(), "cpu"),
        (
            "enhance",
            DevicePolicy(has_cuda=True, has_mps=True, prefer_gpu=False),
            "cpu",
        ),
    ],
)
def test_device_policy_select_device_matrix(
    stage_name: str,
    policy: DevicePolicy,
    expected: str,
) -> None:
    assert policy.select_device(stage_name) == expected


def test_device_policy_can_use_batch_respects_memory_headroom() -> None:
    policy = DevicePolicy(available_memory_gb=8.0)

    assert policy.can_use_batch(batch_size=1, image_size_mp=1.0) is True
    assert policy.can_use_batch(batch_size=512, image_size_mp=16.0) is False
    assert policy.can_use_batch(batch_size=1, image_size_mp=512 / 3) is False


@pytest.mark.parametrize(
    ("preset", "expected", "unchanged_fields"),
    [
        (
            QualityPreset.DRAFT,
            {
                "upscale_factor": 1.0,
                "enhancement_strength": 0.3,
                "clarity_strength": 0.2,
                "enable_materials": False,
            },
            ("material_strength",),
        ),
        (
            QualityPreset.STANDARD,
            {
                "upscale_factor": 1.0,
                "enhancement_strength": 0.5,
                "clarity_strength": 0.4,
                "enable_materials": True,
                "material_strength": 0.5,
            },
            (),
        ),
        (
            QualityPreset.HIGH,
            {
                "upscale_factor": 2.0,
                "enhancement_strength": 0.7,
                "clarity_strength": 0.6,
                "enable_materials": True,
                "material_strength": 0.7,
            },
            (),
        ),
        (
            QualityPreset.PRODUCTION,
            {
                "upscale_factor": 2.0,
                "enhancement_strength": 0.8,
                "clarity_strength": 0.7,
                "enable_materials": True,
                "material_strength": 0.8,
            },
            (),
        ),
    ],
)
def test_quality_policy_apply_preset_exact_fields(
    preset: QualityPreset,
    expected: dict[str, float | bool],
    unchanged_fields: tuple[str, ...],
) -> None:
    policy = QualityPolicy()
    before = {field_name: getattr(policy, field_name) for field_name in unchanged_fields}

    policy.apply_preset(preset)

    assert policy.preset is preset
    for field_name, expected_value in expected.items():
        assert getattr(policy, field_name) == expected_value
    for field_name, before_value in before.items():
        assert getattr(policy, field_name) == before_value


@pytest.mark.parametrize(
    ("stage_name", "policy", "expected"),
    [
        ("depth_estimation", CachingPolicy(cache_depth_maps=True), True),
        ("depth_estimation", CachingPolicy(cache_depth_maps=False), False),
        ("material_response", CachingPolicy(cache_material_masks=True), True),
        ("segmentation", CachingPolicy(cache_material_masks=False), False),
        ("enhance", CachingPolicy(cache_enhanced=False), False),
        ("upscale", CachingPolicy(cache_enhanced=True), True),
        ("white_balance", CachingPolicy(), True),
    ],
)
def test_caching_policy_should_cache_stage_matrix(
    stage_name: str,
    policy: CachingPolicy,
    expected: bool,
) -> None:
    assert policy.should_cache_stage(stage_name) is expected


def test_caching_policy_disabled_overrides_stage_specific_flags() -> None:
    policy = CachingPolicy(
        enabled=False,
        cache_depth_maps=True,
        cache_material_masks=True,
        cache_enhanced=True,
    )

    assert policy.should_cache_stage("depth_estimation") is False
    assert policy.should_cache_stage("unknown_stage") is False


def test_processing_policy_preserves_injected_subpolicies() -> None:
    device = DevicePolicy(has_cuda=True)
    quality = QualityPolicy(preset=QualityPreset.PRODUCTION)
    caching = CachingPolicy(enabled=False)

    policy = ProcessingPolicy(device=device, quality=quality, caching=caching)

    assert policy.device is device
    assert policy.quality is quality
    assert policy.caching is caching


@pytest.mark.parametrize(
    ("scene_type", "expected"),
    [
        (
            SceneType.AERIAL,
            {
                "enhancement_strength": 0.5,
                "clarity_strength": pytest.approx(0.48),
                "material_strength": pytest.approx(0.4),
                "enable_materials": True,
            },
        ),
        (
            SceneType.INTERIOR,
            {
                "enhancement_strength": 0.5,
                "clarity_strength": 0.4,
                "material_strength": 0.8,
                "enable_materials": True,
            },
        ),
        (
            SceneType.EXTERIOR,
            {
                "enhancement_strength": pytest.approx(0.55),
                "clarity_strength": 0.4,
                "material_strength": 0.5,
                "enable_materials": True,
            },
        ),
        (
            SceneType.MIXED,
            {
                "enhancement_strength": 0.5,
                "clarity_strength": 0.4,
                "material_strength": 0.5,
                "enable_materials": True,
            },
        ),
        (
            SceneType.UNKNOWN,
            {
                "enhancement_strength": 0.5,
                "clarity_strength": 0.4,
                "material_strength": 0.5,
                "enable_materials": True,
            },
        ),
    ],
)
def test_policy_engine_scene_adjustment_matrix(
    monkeypatch: pytest.MonkeyPatch,
    scene_type: SceneType,
    expected: dict[str, float | bool],
) -> None:
    engine = PolicyEngine()
    monkeypatch.setattr(engine, "_detect_devices", lambda device_policy: None)

    policy = engine.create_policy(
        quality_preset=QualityPreset.STANDARD,
        scene_type=scene_type,
    )

    assert policy.scene_type is scene_type
    for field_name, expected_value in expected.items():
        assert getattr(policy.quality, field_name) == expected_value


def test_policy_engine_create_policy_composes_preset_scene_and_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = PolicyEngine()

    def fake_detect_devices(device_policy: DevicePolicy) -> None:
        device_policy.has_cuda = True
        device_policy.available_memory_gb = 12.0

    monkeypatch.setattr(engine, "_detect_devices", fake_detect_devices)

    policy = engine.create_policy(
        quality_preset=QualityPreset.HIGH,
        scene_type=SceneType.AERIAL,
        config={
            "device": "cpu",
            "cache_dir": str(tmp_path / "cache"),
            "cache_enabled": False,
            "upscale_factor": 3.0,
            "enhancement_strength": 0.9,
            "enable_parallel": False,
            "max_workers": 2,
        },
    )

    assert policy.scene_type is SceneType.AERIAL
    assert policy.device.has_cuda is True
    assert policy.device.available_memory_gb == 12.0
    assert policy.device.prefer_gpu is False
    assert policy.device.select_device("enhance") == "cpu"
    assert policy.caching.cache_dir == tmp_path / "cache"
    assert policy.caching.enabled is False
    assert policy.quality.upscale_factor == 3.0
    assert policy.quality.enhancement_strength == 0.9
    assert policy.quality.clarity_strength == pytest.approx(0.72)
    assert policy.quality.material_strength == pytest.approx(0.56)
    assert policy.enable_parallel is False
    assert policy.max_workers == 2


def test_policy_engine_create_policy_defaults_without_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = PolicyEngine()
    monkeypatch.setattr(engine, "_detect_devices", lambda device_policy: None)

    policy = engine.create_policy()

    assert policy.scene_type is SceneType.UNKNOWN
    assert policy.quality.preset is QualityPreset.STANDARD
    assert policy.quality.upscale_factor == 1.0
    assert policy.quality.enhancement_strength == 0.7
    assert policy.quality.clarity_strength == 0.5
    assert policy.quality.enable_materials is True
    assert policy.quality.material_strength == 0.6
    assert policy.device.prefer_gpu is True
    assert policy.caching.enabled is True
    assert policy.enable_parallel is True
    assert policy.max_workers == 4


def test_detect_devices_keeps_defaults_when_optional_runtimes_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "coremltools", None)
    monkeypatch.setitem(sys.modules, "psutil", None)
    policy = DevicePolicy(available_memory_gb=8.0)

    PolicyEngine()._detect_devices(policy)

    assert policy.has_cuda is False
    assert policy.has_mps is False
    assert policy.has_coreml is False
    assert policy.available_memory_gb == 8.0


def test_detect_devices_handles_torch_without_cuda_or_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "coremltools", None)
    monkeypatch.setitem(sys.modules, "psutil", None)
    policy = DevicePolicy(available_memory_gb=8.0)

    PolicyEngine()._detect_devices(policy)

    assert policy.has_cuda is False
    assert policy.has_mps is False
    assert policy.has_coreml is False
    assert policy.available_memory_gb == 8.0


def test_detect_devices_reads_fake_cuda_mps_coreml_and_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True),
        ),
    )
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=16 * 1024**3),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "coremltools", ModuleType("coremltools"))
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    policy = DevicePolicy()

    PolicyEngine()._detect_devices(policy)

    assert policy.has_cuda is True
    assert policy.has_mps is True
    assert policy.has_coreml is True
    assert policy.available_memory_gb == 16.0


def test_apply_config_overrides_supported_fields(tmp_path: Path) -> None:
    policy = ProcessingPolicy()
    cache_dir = tmp_path / "policy-cache"

    PolicyEngine()._apply_config(
        policy,
        {
            "device": "cpu",
            "cache_dir": str(cache_dir),
            "cache_enabled": False,
            "upscale_factor": 2.5,
            "enhancement_strength": 0.65,
            "enable_parallel": False,
            "max_workers": 6,
        },
    )

    assert policy.device.prefer_gpu is False
    assert policy.caching.cache_dir == cache_dir
    assert policy.caching.enabled is False
    assert policy.quality.upscale_factor == 2.5
    assert policy.quality.enhancement_strength == 0.65
    assert policy.enable_parallel is False
    assert policy.max_workers == 6


def test_apply_config_leaves_gpu_preference_for_non_cpu_device() -> None:
    policy = ProcessingPolicy()

    PolicyEngine()._apply_config(policy, {"device": "cuda"})

    assert policy.device.prefer_gpu is True
