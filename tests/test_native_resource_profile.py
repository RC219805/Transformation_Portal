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
from packaging.requirements import Requirement

from transformation_portal.ingest.canonical_json import dumps_json
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
    runtime = {"python": "test-runtime", "extensions": {"test-extension": "a" * 64}, "packages": {"packaging": "26.3"}}
    sources = {"test-source.py": "b" * 64}
    monkeypatch.setattr(resources, "_runtime_identity", lambda: copy.deepcopy(runtime))
    monkeypatch.setattr(resources, "_source_identity", lambda: dict(sources))
    contract = resources.capture_contract()
    monkeypatch.setattr(resources, "RESOURCE_CONTRACT", contract)
    return runtime, sources, contract


@pytest.fixture(name="synthetic_cache_authority")
def fixture_synthetic_cache_authority(monkeypatch):
    # Only the marked source-closure tests use synthetic cache authority. Their
    # source-byte, origin and schema checks remain real; this is no host/runtime
    # qualification. Executable-cache rejection tests never request this fixture.
    monkeypatch.setattr(resources, "_verify_executable_caches", lambda *_args, **_kwargs: None)


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
    assert qualified.profile_id == "tp.photo.native_sdr_managed.v2"
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


@pytest.mark.parametrize("change", ["source", "runtime", "package", "limit", "profile", "schema", "extra"])
def test_native_profile_closure_or_contract_drift_fails_closed(evidence, matching_contract, change):
    runtime, sources, contract = matching_contract
    old_identity = resources.profile_identity()
    if change == "source":
        sources["test-source.py"] = "d" * 64
    elif change == "runtime":
        runtime["extensions"]["test-extension"] = "d" * 64
    elif change == "package":
        runtime["packages"]["packaging"] = "26.4"
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
    if change in {"source", "runtime", "package"}:
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
    monkeypatch.setattr(resources, "_runtime_packages", lambda: {"test-package": "1.0"})
    monkeypatch.setattr(resources, "_runtime_package_payloads", lambda: {"test-package": {"sha256": "e" * 64}})
    monkeypatch.setattr(resources.sys, "executable", str(executable))
    monkeypatch.setattr(resources, "_source_identity", lambda: {"source.py": "a" * 64})
    contract = resources.capture_contract()
    monkeypatch.setattr(resources, "RESOURCE_CONTRACT", contract)
    assert resources.runtime_profile_identity() == resources.profile_identity()
    assert set(contract["runtime"]["extensions"]) == set(resources._EXTENSIONS)
    extension.write_bytes(b"compiled extension changed")
    assert resources.runtime_profile_identity() is None


@pytest.mark.usefixtures("synthetic_cache_authority")
def test_native_source_closure_excludes_only_pure_data_authority_and_includes_loader():
    paths = resources._SOURCE_PATHS
    assert "src/transformation_portal/photographic_delivery/native_resources.py" in paths
    assert "src/transformation_portal/photographic_delivery/native_resource_inputs.py" in paths
    assert "src/transformation_portal/photographic_delivery/native_resource_contract.py" not in paths
    assert "app.py" in paths
    assert "pyproject.toml" in paths and "requirements/base.in" in paths
    assert {
        "src/transformation_portal/lux_depth_v4/runtime.py",
        "src/transformation_portal/lux_depth_v4/_bytecode.py",
        "src/transformation_portal/depth/backends/da3_runtime_identity.py",
    }.issubset(paths)
    root = Path(resources.__file__).resolve().parents[3]
    assert all((root / path).is_file() for path in paths)
    actual = resources._source_identity()
    assert set(actual) == set(paths) and all(len(digest) == 64 for digest in actual.values())


@pytest.mark.usefixtures("synthetic_cache_authority")
def test_native_profile_rejects_schema_resources_from_another_source_closure(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "_runtime_identity", lambda: {"runtime": "schema-closure-test"})
    monkeypatch.setattr(resources, "RESOURCE_CONTRACT", resources.capture_contract())
    monkeypatch.setattr(resources, "files", lambda package: tmp_path)
    (tmp_path / "plan.v2.schema.json").write_bytes(b"{}")
    assert resources.runtime_profile_identity() is None


@pytest.mark.parametrize(
    "name",
    ["app", "transformation_portal.photographic_delivery.models", "transformation_portal.photographic_delivery"],
)
@pytest.mark.usefixtures("synthetic_cache_authority")
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


@pytest.mark.usefixtures("synthetic_cache_authority")
def test_native_profile_checks_loaded_file_bytes_without_importing_unloaded_app(monkeypatch):
    monkeypatch.delitem(resources.sys.modules, "app", raising=False)
    identities = resources._source_identity()
    assert "app.py" in identities and "app" not in resources.sys.modules


def _write_runtime_requirements(root, dependencies, base):
    (root / "pyproject.toml").write_text(
        "[project]\ndependencies = [" + ",".join(dumps_json(value) for value in dependencies) + "]\n", encoding="utf-8"
    )
    (root / "requirements").mkdir()
    (root / "requirements/base.in").write_text(base, encoding="utf-8")


@pytest.mark.parametrize("system,opencv", [("Darwin", "opencv-python"), ("Linux", "opencv-python-headless")])
def test_native_core_requirements_preserve_markers_extras_and_inline_comments(tmp_path, system, opencv):
    _write_runtime_requirements(
        tmp_path,
        [
            "Pillow>=13,<14",
            "SQLAlchemy[asyncio]>=2.0.50,<2.2",
            "opencv-python>=4.8,<5; platform_system != 'Linux'",
            "opencv-python-headless>=4.8,<5; platform_system == 'Linux'",
        ],
        "# Security floor\nPillow>=13.0.0,<14 # patched codec\ncertifi>=2026.10.19\n",
    )
    requirements = resources._active_runtime_requirements(tmp_path, {"platform_system": system})
    names = [value.name for value in requirements]
    assert names.count("Pillow") == 2 and "certifi" in names
    assert {name for name in names if name.startswith("opencv-")} == {opencv}
    assert next(value for value in requirements if value.name == "SQLAlchemy").extras == {"asyncio"}
    assert resources._requirement_line("name; platform_system == 'Darwin # marker' # comment") == (
        "name; platform_system == 'Darwin # marker'"
    )


def test_native_runtime_packages_capture_full_active_core_union_and_canonical_names(monkeypatch):
    requirements = (Requirement("pillow>=13,<14"), Requirement("certifi>=2026.10.19"), Requirement("Some_Package>=2"))
    monkeypatch.setattr(resources, "_active_runtime_requirements", lambda: requirements)
    versions = {"Pillow": "13.0.0", "certifi": "2026.10.19", "Some_Package": "2.0"}
    observed = []

    def version(name):
        observed.append(name)
        return versions.get(name, "1.0")

    monkeypatch.setattr(resources.importlib.metadata, "version", version)
    packages = resources._runtime_packages()
    assert set(resources._PACKAGES).issubset(packages)
    assert packages["certifi"] == "2026.10.19" and packages["Some_Package"] == "2.0"
    assert observed.count("Pillow") == 1 and "pillow" not in observed
    resources._validate_runtime_packages({"some-package": "2.0"}, (Requirement("Some_Package>=2"),))


def test_native_inactive_python_and_extra_markers_do_not_require_unselected_packages(tmp_path):
    _write_runtime_requirements(
        tmp_path,
        ["Pillow>=13,<14", "certifi>=9999; python_version < '3.12'", "redis>=9999; extra == 'ml'"],
        "Pillow>=13,<14\n",
    )
    requirements = resources._active_runtime_requirements(tmp_path, {"python_version": "3.12", "extra": ""})
    assert {value.name for value in requirements} == {"Pillow"}
    resources._validate_runtime_packages({"Pillow": "13.0.0"}, requirements)


def test_native_profile_rejects_new_security_floor_with_unchanged_installed_packages(monkeypatch):
    requirements = [Requirement("Pillow>=13,<14"), Requirement("certifi>=2026.10.19")]
    monkeypatch.setattr(resources, "_active_runtime_requirements", lambda: tuple(requirements))
    versions = {"Pillow": "13.0.0", "certifi": "2026.10.19"}
    monkeypatch.setattr(resources.importlib.metadata, "version", lambda name: versions.get(name, "1.0"))
    monkeypatch.setattr(resources, "_runtime_identity", lambda: {"packages": resources._runtime_packages()})
    monkeypatch.setattr(resources, "_source_identity", lambda: {"source.py": "a" * 64})
    monkeypatch.setattr(resources, "RESOURCE_CONTRACT", resources.capture_contract())
    assert resources.runtime_profile_identity() == resources.profile_identity()
    requirements.append(Requirement("certifi>=2026.11.1"))
    with pytest.raises(ValueError, match="declared core requirement"):
        resources.capture_contract()
    assert resources.runtime_profile_identity() is None


@pytest.mark.parametrize(
    "packages,requirement",
    [
        ({"Pillow": "12.3.0"}, "Pillow>=13.0.0,<14"),
        ({"Pillow": "14.0.0"}, "Pillow>=13.0.0,<14"),
        ({"certifi": "2026.7.22"}, "certifi>=2026.10.19"),
        ({"fastapi": "0.141.0"}, "fastapi==0.141.1"),
        ({}, "Pillow>=13.0.0,<14"),
        ({"Pillow": "13.0.0"}, "Pillow @ https://example.invalid/Pillow.whl"),
    ],
)
def test_native_runtime_packages_reject_unsupported_or_unverifiable_declarations(packages, requirement):
    with pytest.raises(ValueError, match="declared core requirement"):
        resources._validate_runtime_packages(packages, (Requirement(requirement),))


@pytest.mark.parametrize("base", ["-r other.in\n", "--extra-index-url https://example.invalid\n", "not a requirement\n"])
def test_native_runtime_requirements_reject_unsupported_pip_directives_and_invalid_lines(tmp_path, base):
    _write_runtime_requirements(tmp_path, ["Pillow>=13,<14"], base)
    with pytest.raises(ValueError):
        resources._active_runtime_requirements(tmp_path)


def test_native_runtime_requirements_reject_missing_core_declarations(tmp_path):
    _write_runtime_requirements(tmp_path, [], "Pillow>=13,<14\n")
    with pytest.raises(ValueError, match="declared core dependency"):
        resources._active_runtime_requirements(tmp_path)


@pytest.mark.parametrize("marker", ["dependency_groups", "extras"])
def test_native_runtime_marker_contexts_without_core_environment_fail_closed(monkeypatch, tmp_path, marker):
    _write_runtime_requirements(tmp_path, [f"Pillow>=13; {marker} == 'photo'"], "Pillow>=13,<14\n")
    active_requirements = resources._active_runtime_requirements
    monkeypatch.setattr(resources, "_active_runtime_requirements", lambda: active_requirements(tmp_path))
    monkeypatch.setattr(resources, "_runtime_identity", lambda: {"packages": resources._runtime_packages()})
    monkeypatch.setattr(resources, "RESOURCE_CONTRACT", {"nonempty": True})
    with pytest.raises(ValueError, match="declared marker context"):
        resources.capture_contract()
    assert resources.runtime_profile_identity() is None


@pytest.mark.parametrize("problem", ["unsupported", "missing_package", "missing_parser"])
def test_native_runtime_capture_and_matching_fail_closed_before_binary_discovery(monkeypatch, problem):
    for function, value in (
        ("python_implementation", "CPython"),
        ("python_version", "3.12.14"),
        ("system", "Darwin"),
        ("machine", "arm64"),
    ):
        monkeypatch.setattr(resources.platform, function, lambda value=value: value)
    for name in resources._THREAD_ENV:
        monkeypatch.setenv(name, "1")

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported packages reached compiled-module discovery")

    monkeypatch.setattr(resources.importlib.util, "find_spec", forbidden)
    monkeypatch.setattr(resources, "RESOURCE_CONTRACT", {"nonempty": True})
    if problem == "missing_parser":
        monkeypatch.setitem(resources.sys.modules, "packaging.requirements", None)
        expected_error = ImportError
    else:
        monkeypatch.setattr(resources, "_active_runtime_requirements", lambda: (Requirement("Pillow>=13,<14"),))

        def version(name):
            if problem == "missing_package":
                raise resources.importlib.metadata.PackageNotFoundError(name)
            return "12.3.0" if name == "Pillow" else "1.0"

        monkeypatch.setattr(resources.importlib.metadata, "version", version)
        expected_error = ImportError if problem == "missing_package" else ValueError
    with pytest.raises(expected_error):
        resources.capture_contract()
    assert resources.runtime_profile_identity() is None


@pytest.mark.usefixtures("synthetic_cache_authority")
def test_native_declaration_source_hashes_do_not_discover_python_modules(monkeypatch):
    for name in ("pyproject.toml", "requirements.base.in"):
        monkeypatch.setitem(resources.sys.modules, name, SimpleNamespace(__file__="/not-a-python-module"))
    identities = resources._source_identity()
    assert len(identities["pyproject.toml"]) == len(identities["requirements/base.in"]) == 64


def test_reviewed_native_contract_satisfies_current_declared_core_requirements():
    runtime = resources.RESOURCE_CONTRACT["runtime"]
    assert set(resources._PACKAGES).issubset(runtime["packages"])
    environment = {
        "implementation_name": "cpython",
        "implementation_version": runtime["python"],
        "platform_system": runtime["system"],
        "platform_machine": runtime["machine"],
        "platform_release": runtime["release"],
        "platform_version": runtime["version"],
        "platform_python_implementation": runtime["implementation"],
        "python_full_version": runtime["python"],
        "python_version": ".".join(runtime["python"].split(".")[:2]),
        "sys_platform": "darwin",
        "os_name": "posix",
    }
    resources._validate_runtime_packages(runtime["packages"], resources._active_runtime_requirements(environment=environment))
