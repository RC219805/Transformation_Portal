"""Fixed native reservations require exact runtime, source and ownership bounds."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from transformation_portal.lux_depth_v4.color_preparation import output_srgb_icc
from transformation_portal.photographic_delivery import native_resources as resources
from transformation_portal.photographic_delivery.native_resource_inputs import inspect_native_raster

pytestmark = pytest.mark.unit


@pytest.fixture(name="evidence")
def fixture_evidence():
    buffer = io.BytesIO()
    tifffile.imwrite(
        buffer,
        np.zeros((7, 9, 3), np.uint16),
        photometric="rgb",
        iccprofile=output_srgb_icc(),
        metadata={"color_space": "srgb", "color_domain": "display_encoded_srgb"},
    )
    raw = buffer.getvalue()
    return inspect_native_raster(raw, hashlib.sha256(raw).hexdigest())


@pytest.fixture(name="matching_contract")
def fixture_matching_contract(monkeypatch):
    # Synthetic closure values exercise qualification logic, never runtime/RSS evidence.
    runtime = {"python": "test-runtime", "extensions": {"test-extension": "a" * 64}}
    sources = {"test-source.py": "b" * 64}
    monkeypatch.setattr(resources, "_runtime_identity", lambda: copy.deepcopy(runtime))
    monkeypatch.setattr(resources, "_source_identity", lambda: dict(sources))
    contract = resources.capture_contract()
    monkeypatch.setattr(resources, "RESOURCE_CONTRACT", contract)
    return runtime, sources, contract


def _qualify(evidence, **overrides):
    args = {"regions": 1, "provider_scope_sha256": "c" * 64, "state": {"version": 1, "revisions": []}, "response_owned": True}
    args.update(overrides)
    return resources.qualify_native_resources(evidence, **args)


def test_native_profile_stays_unqualified_without_reviewed_data(evidence, monkeypatch):
    monkeypatch.setattr(resources, "RESOURCE_CONTRACT", {})
    assert resources.runtime_profile_identity() is None
    assert _qualify(evidence) is None
    assert resources.profile_identity() == hashlib.sha256(b"{}").hexdigest()


def test_native_profile_qualification_is_fixed_frozen_and_source_bound(evidence, matching_contract):
    qualified = _qualify(evidence)
    assert qualified is not None
    assert qualified.profile_id == "tp.photo.native_sdr_local.v1"
    assert qualified.profile_identity == resources.profile_identity() == resources.runtime_profile_identity()
    assert qualified.source_sha256 == evidence.source_sha256
    assert qualified.raster_identity == evidence.content_hash()
    assert qualified.memory_bytes == 6 * 1024**3
    assert qualified.analytical_memory_bytes < qualified.memory_bytes
    with pytest.raises(FrozenInstanceError):
        qualified.memory_bytes = 0
    payload = qualified.to_payload()
    payload["memory_bytes"] = 0
    assert qualified.memory_bytes == resources.MEMORY_BYTES


@pytest.mark.parametrize("change", ["source", "runtime", "limit", "profile", "schema", "extra"])
def test_native_profile_closure_or_contract_drift_fails_closed(evidence, matching_contract, change):
    runtime, sources, contract = matching_contract
    old_identity = resources.profile_identity()
    if change == "source":
        sources["test-source.py"] = "d" * 64
    elif change == "runtime":
        runtime["extensions"]["test-extension"] = "d" * 64
    elif change == "limit":
        contract["limits"]["memory_bytes"] += 1
    elif change == "profile":
        contract["profile_id"] = "unknown"
    elif change == "schema":
        contract["schema"] = "unknown"
    else:
        contract["unreviewed"] = True
    assert resources.runtime_profile_identity() is None
    assert _qualify(evidence) is None
    if change in {"source", "runtime"}:
        assert resources.profile_identity() == old_identity
    else:
        assert resources.profile_identity() != old_identity


@pytest.mark.parametrize("provider", [None, "", "bad", "A" * 64, "g" * 64, True])
def test_native_profile_provider_scope_must_be_an_exact_sha(evidence, matching_contract, provider):
    assert _qualify(evidence, provider_scope_sha256=provider) is None


@pytest.mark.parametrize("regions", [0, 5, 1.0, True, None])
def test_native_profile_region_count_is_closed(evidence, matching_contract, regions):
    assert _qualify(evidence, regions=regions) is None


@pytest.mark.parametrize("owned", [False, None, 1, "true"])
def test_native_profile_requires_response_ownership(evidence, matching_contract, owned):
    assert _qualify(evidence, response_owned=owned) is None


@pytest.mark.parametrize(
    "change",
    [
        {"source_sha256": "bad"},
        {"profile_sha256": "a" * 64},
        {"layout_sha256": "bad"},
        {"width": 1},
        {"alpha": 1},
        {"source_bit_depth": 32},
        {"orientation": 9},
        {"byte_order": "unknown"},
        {"container": "png"},
        {"header_bytes": 65537},
        {"tag_count": 33},
        {"strip_count": 65},
        {"raster_bytes": 1},
        {"size_bytes": 1},
        {"source_width": 65537},
    ],
)
def test_native_profile_revalidates_evidence_construction(evidence, matching_contract, change):
    assert _qualify(replace(evidence, **change)) is None


@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("regions", [1, 2, 3, 4])
def test_native_profile_maximum_conditional_array_phases_fit_fixed_lease(evidence, matching_contract, alpha, bits, regions):
    pixels, header = 30_000_000, evidence.header_bytes
    raster = pixels * (4 if alpha else 3) * (bits // 8)
    # Analytical geometry only: no full-resolution test image or measured observation.
    maximum = replace(
        evidence,
        width=6000,
        height=5000,
        source_width=6000,
        source_height=5000,
        alpha=alpha,
        source_bit_depth=bits,
        tag_count=18,
        raster_bytes=raster,
        size_bytes=header + raster,
    )
    qualified = _qualify(maximum, regions=regions)
    assert qualified is not None
    assert qualified.memory_bytes == 6 * resources.GIB
    assert qualified.analytical_memory_bytes < qualified.memory_bytes
    if alpha and regions == 4:
        expected = 95 * pixels + 2 * 65536 + 1444 * resources.MIB + resources.GIB
        assert qualified.analytical_memory_bytes == expected
    assert resources.ACCEPTANCE_PEAK_BYTES * 3 // 2 == resources.MEMORY_BYTES


def test_native_profile_rejects_large_heap_nodes_depth_and_cycles(evidence, matching_contract):
    assert _qualify(evidence, state={"large": "x" * resources.MAX_STATE_BYTES}) is None
    assert _qualify(evidence, state={"many": [0] * resources.MAX_STATE_NODES}) is None
    nested = {}
    for _ in range(resources.MAX_STATE_DEPTH + 1):
        nested = {"child": nested}
    assert _qualify(evidence, state=nested) is None
    cyclic = {}
    cyclic["self"] = cyclic
    assert _qualify(evidence, state=cyclic) is None
    shared = {"values": [1, 2, 3]}
    assert _qualify(evidence, state={"one": shared, "two": shared}) is not None


@pytest.mark.parametrize("value", [float("inf"), float("nan"), (1, 2), {1, 2}, b"bytes"])
def test_native_profile_state_is_a_json_graph(evidence, matching_contract, value):
    assert _qualify(evidence, state={"invalid": value}) is None
    assert _qualify(evidence, state={1: "invalid key"}) is None


def test_native_profile_does_not_invoke_arbitrary_size_protocols(evidence, matching_contract):
    class Unexpected:
        def __sizeof__(self):
            raise AssertionError("Arbitrary state object was inspected")

    assert _qualify(evidence, state={"invalid": Unexpected()}) is None
    assert resources.metadata_heap_is_bounded({"version": 1})
    assert not resources.metadata_heap_is_bounded({"invalid": Unexpected()})


def test_native_runtime_capture_is_lazy_and_requires_pinned_host_and_threads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported runtime reached compiled-module discovery")

    monkeypatch.setattr(resources.importlib.util, "find_spec", forbidden)
    monkeypatch.setattr(resources.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(resources.platform, "python_version", lambda: "3.12.15")
    monkeypatch.setattr(resources.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(resources.platform, "machine", lambda: "arm64")
    for name in resources._THREAD_ENV:
        monkeypatch.setenv(name, "1")
    assert resources._runtime_identity() is None
    monkeypatch.setattr(resources.platform, "python_version", lambda: "3.12.14")
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert resources._runtime_identity() is None
    with pytest.raises(ValueError, match="pinned runtime"):
        resources.capture_contract()


def test_native_runtime_missing_files_and_packages_keep_fallback(monkeypatch):
    monkeypatch.setattr(resources, "RESOURCE_CONTRACT", {"nonempty": True})

    def missing():
        raise FileNotFoundError("Missing compiled extension")

    monkeypatch.setattr(resources, "_runtime_identity", missing)
    assert resources.runtime_profile_identity() is None


def test_native_runtime_revalidates_compiled_binary_hashes_without_package_changes(monkeypatch, tmp_path):
    extension, executable = tmp_path / "extension.so", tmp_path / "python"
    extension.write_bytes(b"compiled extension before")
    executable.write_bytes(b"trusted python executable")
    for function, value in (
        ("python_implementation", "CPython"),
        ("python_version", "3.12.14"),
        ("system", "Darwin"),
        ("machine", "arm64"),
    ):
        monkeypatch.setattr(resources.platform, function, lambda value=value: value)
    for name in resources._THREAD_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(resources.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(extension)))
    monkeypatch.setattr(resources.importlib.metadata, "version", lambda name: "fixed-package-version")
    monkeypatch.setattr(resources.sys, "executable", str(executable))
    monkeypatch.setattr(resources, "_source_identity", lambda: {"source.py": "a" * 64})
    contract = resources.capture_contract()
    monkeypatch.setattr(resources, "RESOURCE_CONTRACT", contract)
    assert resources.runtime_profile_identity() == resources.profile_identity()
    assert set(contract["runtime"]["extensions"]) == set(resources._EXTENSIONS)
    extension.write_bytes(b"compiled extension changed")
    assert resources.runtime_profile_identity() is None


def test_native_source_closure_excludes_only_pure_data_authority_and_includes_loader():
    paths = resources._SOURCE_PATHS
    assert "src/transformation_portal/photographic_delivery/native_resources.py" in paths
    assert "src/transformation_portal/photographic_delivery/native_resource_inputs.py" in paths
    assert "src/transformation_portal/photographic_delivery/native_resource_contract.py" not in paths
    assert "app.py" in paths
    root = Path(resources.__file__).resolve().parents[3]
    assert all((root / path).is_file() for path in paths)
    actual = resources._source_identity()
    assert set(actual) == set(paths) and all(len(digest) == 64 for digest in actual.values())


@pytest.mark.parametrize(
    "name",
    ["app", "transformation_portal.photographic_delivery.models", "transformation_portal.photographic_delivery"],
)
def test_native_profile_rejects_loaded_sources_from_a_different_checkout(monkeypatch, tmp_path, name):
    monkeypatch.setattr(resources, "_runtime_identity", lambda: {"runtime": "source-closure-test"})
    monkeypatch.setattr(resources, "RESOURCE_CONTRACT", resources.capture_contract())
    assert resources.runtime_profile_identity() == resources.profile_identity()
    alternate = tmp_path / "alternate.py"
    alternate.write_text("ALTERNATE_SOURCE = True\n")
    spec = importlib.util.spec_from_file_location(name, alternate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(resources.sys.modules, name, module)
    assert module.ALTERNATE_SOURCE is True
    with pytest.raises(ValueError, match="Loaded native resource source differs"):
        resources._source_identity()
    assert resources.runtime_profile_identity() is None


def test_native_profile_checks_loaded_file_bytes_without_importing_unloaded_app(monkeypatch):
    monkeypatch.delitem(resources.sys.modules, "app", raising=False)
    identities = resources._source_identity()
    assert "app.py" in identities and "app" not in resources.sys.modules
