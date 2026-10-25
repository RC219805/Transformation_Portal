from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "scripts" / "utilities" / "package_verified_delivery.py"
from transformation_portal.core import verified_delivery as packager


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _products(output: Path) -> tuple[Path, Path, Path]:
    return output, Path(f"{output}.sha256"), Path(f"{output}.receipt.json")


@pytest.fixture
def package_case(tmp_path: Path) -> dict[str, Any]:
    source = tmp_path / "source"
    source.mkdir()
    payloads = {
        "photos/photo-02.tif": b"retained TIFF bytes\x00\xff",
        "photos/photo-01.png": b"preview PNG bytes\x00\x01",
        "report.txt": b"Existing verification evidence; no new quality claim.\n",
    }
    archive_names = {
        "photos/photo-02.tif": "tiffs/photo-02.tif",
        "photos/photo-01.png": "previews/photo-01.png",
        "report.txt": "verification/report.txt",
    }
    for name, content in payloads.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = {
        "schema": "tp.delivery.package_spec.v1",
        "archive_root": "verified-cpu-batch",
        "files": [
            {
                "source": name,
                "archive_path": archive_names[name],
                "sha256": _sha256(content),
                "size_bytes": len(content),
            }
            for name, content in payloads.items()
        ],
    }
    manifest_path = tmp_path / "spec.json"
    _write_manifest(manifest_path, manifest)
    publish = tmp_path / "publish"
    publish.mkdir()
    return {
        "source": source,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "output": publish / "verified.zip",
        "payloads": payloads,
    }


def _build(case: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return packager.build_package(case["manifest_path"], case["source"], case["output"], **kwargs)


def _assert_rejected_without_publication(case: dict[str, Any]) -> None:
    _write_manifest(case["manifest_path"], case["manifest"])
    with pytest.raises(packager.PackageError):
        _build(case)
    assert list(case["output"].parent.iterdir()) == []


def test_source_cli_shim_preserves_supported_library_api() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("delivery_source_shim", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PackageError is packager.PackageError
    assert module.build_package is packager.build_package
    assert module.main is packager.main


def test_installed_library_without_source_checkout_uses_only_stdlib(package_case: dict[str, Any], tmp_path: Path) -> None:
    import shutil

    site = tmp_path / "installed-site"
    package = site / "transformation_portal"
    core = package / "core"
    core.mkdir(parents=True)
    source = PROJECT_ROOT / "src" / "transformation_portal"
    for relative in ("__init__.py", "core/__init__.py", "core/verified_delivery.py"):
        shutil.copyfile(source / relative, package / relative)
    case = package_case
    # -I and -S disable the editable checkout, site packages, environment Python
    # paths, and third-party dependencies. Only this installed package is added.
    code = """import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
from transformation_portal.core import verified_delivery
assert pathlib.Path(verified_delivery.__file__).is_relative_to(sys.argv[1])
assert 'numpy' not in sys.modules and 'PIL' not in sys.modules and 'torch' not in sys.modules
result = verified_delivery.build_package(*map(pathlib.Path, sys.argv[2:5]))
print(json.dumps(result))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            code,
            str(site),
            str(case["manifest_path"]),
            str(case["source"]),
            str(case["output"]),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "verified"
    assert all(path.is_file() for path in _products(case["output"]))


def test_build_packages_only_explicit_members_and_verifiable_evidence(package_case: dict[str, Any]) -> None:
    case = package_case
    (case["source"] / "unselected-secret.txt").write_text("must stay out", encoding="utf-8")
    receipt = _build(case)
    assert receipt["status"] == "verified"
    assert receipt["scope"] == "byte_integrity_not_semantic_verification"
    assert receipt["source_file_count"] == len(case["payloads"])
    output, checksum, receipt_path = _products(case["output"])
    assert all(path.is_file() for path in (output, checksum, receipt_path))
    assert isinstance(json.loads(receipt_path.read_text(encoding="utf-8")), dict)
    digest, checksum_name = checksum.read_text(encoding="utf-8").strip().split("  ", 1)
    assert digest == _sha256(output.read_bytes())
    assert checksum_name == output.name

    archive_root = case["manifest"]["archive_root"]
    manifest_name = f"{archive_root}/_package/manifest.json"
    sums_name = f"{archive_root}/_package/SHA256SUMS.txt"
    expected = {
        f"{archive_root}/{item['archive_path']}": case["payloads"][item["source"]] for item in case["manifest"]["files"]
    }
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == sorted([*expected, manifest_name, sums_name])
        assert archive.testzip() is None
        for name, content in expected.items():
            assert archive.read(name) == content
        expected_manifest = dict(case["manifest"])
        expected_manifest["files"] = sorted(expected_manifest["files"], key=lambda entry: entry["archive_path"])
        assert json.loads(archive.read(manifest_name)) == expected_manifest
        for entry in archive.infolist():
            assert entry.date_time == (1980, 1, 1, 0, 0, 0)
            assert entry.external_attr >> 16 & 0o777 == 0o644
        checksum_lines = archive.read(sums_name).decode("utf-8").splitlines()
        recorded = dict(line.split("  ", 1)[::-1] for line in checksum_lines)
        assert set(recorded) == {*expected, manifest_name}
        for name, digest in recorded.items():
            assert digest == _sha256(archive.read(name))


def test_same_inputs_produce_identical_zip_despite_source_metadata_changes(package_case: dict[str, Any]) -> None:
    case = package_case
    _build(case)
    original_zip = case["output"].read_bytes()
    for name in case["payloads"]:
        path = case["source"] / name
        os.utime(path, (1_700_000_000, 1_700_000_000))
        path.chmod(0o600)
    case["manifest"]["files"].reverse()
    _write_manifest(case["manifest_path"], case["manifest"])
    case["output"] = case["output"].with_name("second.zip")
    _build(case)
    assert case["output"].read_bytes() == original_zip


def test_streamed_package_and_copy_preserve_source_bytes(
    package_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    case = package_case
    monkeypatch.setattr(packager, "CHUNK_SIZE", 7)
    content = bytes(range(256)) * 129 + b"last partial chunk"
    item = case["manifest"]["files"][0]
    source = case["source"] / item["source"]
    source.write_bytes(content)
    case["payloads"][item["source"]] = content
    item.update(sha256=_sha256(content), size_bytes=len(content))
    _write_manifest(case["manifest_path"], case["manifest"])
    before = {name: (case["source"] / name).read_bytes() for name in case["payloads"]}
    destination = case["output"].parent.parent / "delivery-copy"
    destination.mkdir()

    _build(case, copy_to=destination)

    for product in _products(case["output"]):
        assert (destination / product.name).read_bytes() == product.read_bytes()
    assert {name: (case["source"] / name).read_bytes() for name in before} == before
    with zipfile.ZipFile(case["output"]) as archive:
        archive_name = f"{case['manifest']['archive_root']}/{item['archive_path']}"
        assert archive.read(archive_name) == content


@pytest.mark.parametrize("product_index", [0, 1, 2], ids=["zip", "checksum", "receipt"])
def test_existing_local_product_is_never_overwritten(package_case: dict[str, Any], product_index: int) -> None:
    case = package_case
    blocked = _products(case["output"])[product_index]
    blocked.write_bytes(b"existing delivery must survive")
    with pytest.raises(packager.PackageError):
        _build(case)
    assert blocked.read_bytes() == b"existing delivery must survive"
    assert sorted(case["output"].parent.iterdir()) == [blocked]


@pytest.mark.parametrize("product_index", [0, 1, 2], ids=["zip", "checksum", "receipt"])
def test_copy_collision_preserves_destination_and_publishes_nothing(package_case: dict[str, Any], product_index: int) -> None:
    case = package_case
    destination = case["output"].parent.parent / "delivery-copy"
    destination.mkdir()
    blocked = destination / _products(case["output"])[product_index].name
    blocked.write_bytes(b"finalized delivery")
    with pytest.raises(packager.PackageError):
        _build(case, copy_to=destination)
    assert blocked.read_bytes() == b"finalized delivery"
    assert list(destination.iterdir()) == [blocked]
    assert list(case["output"].parent.iterdir()) == []


def test_late_copy_collision_rolls_back_own_publications_and_preserves_foreign_file(
    package_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    case = package_case
    destination = case["output"].parent.parent / "delivery-copy"
    destination.mkdir()
    foreign = destination / case["output"].name
    copy_identity = destination.stat().st_dev, destination.stat().st_ino
    real_link = packager.os.link
    collision_created = False

    def insert_collision(source: str, target: str, **kwargs: Any) -> None:
        nonlocal collision_created
        target_fd = kwargs.get("dst_dir_fd")
        if target_fd is not None:
            target_stat = os.fstat(target_fd)
            if (target_stat.st_dev, target_stat.st_ino) == copy_identity and not collision_created:
                # Local publication completed; the copy destination was only preflighted.
                assert all(path.is_file() for path in _products(case["output"]))
                foreign.write_bytes(b"delivery created by another writer")
                collision_created = True
        real_link(source, target, **kwargs)

    monkeypatch.setattr(packager.os, "link", insert_collision)
    with pytest.raises(packager.PackageError):
        _build(case, copy_to=destination)
    assert collision_created
    assert foreign.read_bytes() == b"delivery created by another writer"
    assert list(destination.iterdir()) == [foreign]
    assert list(case["output"].parent.iterdir()) == []


def test_post_link_destination_replacement_is_preserved_during_rollback(
    package_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    case = package_case
    real_link = packager.os.link
    replaced = False

    def replace_just_linked_file(source: str, target: str, **kwargs: Any) -> None:
        nonlocal replaced
        real_link(source, target, **kwargs)
        if target == case["output"].name and not replaced:
            # Another writer replaces the link before the publisher can stat it.
            case["output"].unlink()
            case["output"].write_bytes(b"foreign replacement after publication link")
            replaced = True

    monkeypatch.setattr(packager.os, "link", replace_just_linked_file)
    with pytest.raises(packager.PackageError):
        _build(case)
    assert replaced
    assert case["output"].read_bytes() == b"foreign replacement after publication link"
    assert list(case["output"].parent.iterdir()) == [case["output"]]
    assert all(not path.exists() for path in _products(case["output"])[1:])


@pytest.mark.parametrize("replacement_scope", ["parent", "stage"])
def test_replaced_publication_namespace_is_rejected_without_deleting_foreign_files(
    package_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, replacement_scope: str
) -> None:
    case = package_case
    real_publish = packager._publish
    moved = case["output"].parent.with_name("retained-original-parent")
    replacement_stage: Path | None = None

    def replace_namespace_then_publish(groups: list[Any], names: list[str]) -> None:
        nonlocal replacement_stage, moved
        stage = groups[0]
        original_stage = stage.parent / stage.name
        if replacement_scope == "parent":
            stage.parent.rename(moved)
            stage.parent.mkdir()
        else:
            moved = stage.parent / "retained-original-stage"
            original_stage.rename(moved)
        replacement_stage = stage.parent / stage.name
        replacement_stage.mkdir()
        (replacement_stage / "foreign-sentinel.txt").write_bytes(b"another writer owns this directory")
        (replacement_stage / names[0]).write_bytes(b"not the verified ZIP")
        real_publish(groups, names)

    monkeypatch.setattr(packager, "_publish", replace_namespace_then_publish)
    with pytest.raises(packager.PackageError):
        _build(case)

    assert replacement_stage is not None
    assert (replacement_stage / "foreign-sentinel.txt").read_bytes() == b"another writer owns this directory"
    assert (replacement_stage / case["output"].name).read_bytes() == b"not the verified ZIP"
    assert {path.name for path in replacement_stage.iterdir()} == {"foreign-sentinel.txt", case["output"].name}
    assert all(not path.exists() for path in _products(case["output"]))
    assert moved.is_dir()
    assert list(moved.iterdir()) == []


@pytest.mark.parametrize("mismatch", ["hash", "size", "missing"])
def test_changed_or_missing_selected_file_fails_before_publication(package_case: dict[str, Any], mismatch: str) -> None:
    case = package_case
    item = case["manifest"]["files"][0]
    if mismatch == "hash":
        item["sha256"] = "0" * 64
    elif mismatch == "size":
        item["size_bytes"] += 1
    else:
        (case["source"] / item["source"]).unlink()
    _assert_rejected_without_publication(case)


@pytest.mark.parametrize(
    ("field", "name"),
    [
        ("source", "../outside.txt"),
        ("source", "/outside.txt"),
        ("source", "photos\\photo-02.tif"),
        ("archive_path", "../outside.txt"),
        ("archive_path", "/outside.txt"),
        ("archive_path", "tiffs\\photo-02.tif"),
        ("archive_path", "tiffs/../photo-02.tif"),
        ("archive_path", "tiffs/photo\n02.tif"),
        ("archive_path", "tiffs/photo\x7f02.tif"),
        ("archive_path", "_package/manifest.json"),
        ("archive_path", "_PACKAGE/custom.txt"),
    ],
)
def test_unsafe_source_and_archive_names_fail_closed(package_case: dict[str, Any], field: str, name: str) -> None:
    case = package_case
    case["manifest"]["files"][0][field] = name
    _assert_rejected_without_publication(case)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("tiffs/photo.tif", "tiffs/photo.tif"),
        ("tiffs/Photo.tif", "TIFFS/photo.TIF"),
        ("tiffs", "tiffs/photo.tif"),
        ("TIFFS", "tiffs/photo.tif"),
    ],
    ids=["duplicate", "casefold", "file-directory-prefix", "casefold-prefix"],
)
def test_colliding_archive_names_are_rejected(package_case: dict[str, Any], first: str, second: str) -> None:
    case = package_case
    case["manifest"]["files"][0]["archive_path"] = first
    case["manifest"]["files"][1]["archive_path"] = second
    _assert_rejected_without_publication(case)


@pytest.mark.parametrize("symlink_kind", ["source-file", "source-directory", "source-root"])
def test_symlink_sources_and_roots_are_rejected(package_case: dict[str, Any], symlink_kind: str) -> None:
    case = package_case
    source = case["source"]
    if symlink_kind == "source-root":
        root_link = source.with_name("source-link")
        root_link.symlink_to(source, target_is_directory=True)
        case["source"] = root_link
    elif symlink_kind == "source-directory":
        original = source / "photos"
        retained = source.with_name("retained-photos")
        original.rename(retained)
        original.symlink_to(retained, target_is_directory=True)
    else:
        original = source / case["manifest"]["files"][0]["source"]
        retained = source.with_name("retained-photo.tif")
        original.rename(retained)
        original.symlink_to(retained)
    with pytest.raises(packager.PackageError):
        _build(case)
    assert list(case["output"].parent.iterdir()) == []


def test_source_swapped_to_symlink_after_open_is_rejected(
    package_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    case = package_case
    item = case["manifest"]["files"][0]
    original = case["source"] / item["source"]
    outside = case["source"].parent / "outside.tif"
    outside.write_bytes(original.read_bytes())
    real_open = packager.os.open
    swapped = False

    def swap_after_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == original.name and "dir_fd" in kwargs and not swapped:
            original.unlink()
            original.symlink_to(outside)
            swapped = True
        return descriptor

    monkeypatch.setattr(packager.os, "open", swap_after_open)
    with pytest.raises(packager.PackageError):
        _build(case)
    assert swapped
    assert original.is_symlink()
    assert outside.read_bytes() == case["payloads"][item["source"]]
    assert list(case["output"].parent.iterdir()) == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO sources require POSIX")
def test_fifo_source_is_rejected_without_blocking(package_case: dict[str, Any]) -> None:
    case = package_case
    item = case["manifest"]["files"][0]
    source = case["source"] / item["source"]
    source.unlink()
    os.mkfifo(source)
    item.update(sha256=_sha256(b""), size_bytes=0)
    _write_manifest(case["manifest_path"], case["manifest"])
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--manifest",
            str(case["manifest_path"]),
            "--source-root",
            str(case["source"]),
            "--output",
            str(case["output"]),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2
    assert "regular file" in result.stderr
    assert "Traceback" not in result.stderr
    assert list(case["output"].parent.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "tp.delivery.package_spec.v99"),
        ("archive_root", "nested/root"),
        ("archive_root", ".."),
        ("files", []),
    ],
)
def test_invalid_spec_contract_is_rejected(package_case: dict[str, Any], field: str, value: Any) -> None:
    case = package_case
    case["manifest"][field] = value
    _assert_rejected_without_publication(case)


@pytest.mark.parametrize(("field", "value"), [("size_bytes", True), ("size_bytes", -1), ("sha256", "F" * 64)])
def test_manifest_sizes_and_digests_are_strict(package_case: dict[str, Any], field: str, value: Any) -> None:
    case = package_case
    case["manifest"]["files"][0][field] = value
    _assert_rejected_without_publication(case)


@pytest.mark.parametrize("field", ["source", "archive_path"])
def test_surrogate_path_names_raise_package_errors(package_case: dict[str, Any], field: str) -> None:
    case = package_case
    case["manifest"]["files"][0][field] = "photos/invalid-\ud800.tif"
    _assert_rejected_without_publication(case)


@pytest.mark.parametrize("limit", ["bytes", "files"])
def test_manifest_resource_limits_fail_before_publication(
    package_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    case = package_case
    if limit == "bytes":
        monkeypatch.setattr(packager, "MAX_MANIFEST_BYTES", case["manifest_path"].stat().st_size - 1)
    else:
        monkeypatch.setattr(packager, "MAX_FILES", len(case["manifest"]["files"]) - 1)
    with pytest.raises(packager.PackageError):
        _build(case)
    assert list(case["output"].parent.iterdir()) == []


def test_unknown_spec_fields_and_duplicate_json_keys_are_rejected(package_case: dict[str, Any]) -> None:
    case = package_case
    case["manifest"]["unexpected"] = "not silently accepted"
    _assert_rejected_without_publication(case)
    case["manifest"].pop("unexpected")
    text = json.dumps(case["manifest"])
    case["manifest_path"].write_text(text.replace('"schema":', '"schema":"wrong", "schema":', 1), encoding="utf-8")
    with pytest.raises(packager.PackageError):
        _build(case)
    assert list(case["output"].parent.iterdir()) == []


@pytest.mark.parametrize("invalid_manifest", ["oversized-integer", "deep-nesting", "array-root"])
def test_invalid_manifests_return_package_errors_without_cli_tracebacks(
    package_case: dict[str, Any], invalid_manifest: str
) -> None:
    case = package_case
    if invalid_manifest == "oversized-integer":
        text = '{"schema":' + "9" * 5000 + ',"archive_root":"batch","files":[]}'
    elif invalid_manifest == "deep-nesting":
        text = "[" * 20_000 + "0" + "]" * 20_000
    else:
        text = "[]"
    case["manifest_path"].write_text(text, encoding="utf-8")
    with pytest.raises(packager.PackageError):
        _build(case)
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--manifest",
            str(case["manifest_path"]),
            "--source-root",
            str(case["source"]),
            "--output",
            str(case["output"]),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    # Decoder limits vary across Python versions; schema rejection is valid too.
    assert result.stderr.startswith("error: ")
    assert result.stdout == ""
    assert list(case["output"].parent.iterdir()) == []


def test_copy_requires_an_existing_directory(package_case: dict[str, Any]) -> None:
    case = package_case
    destination = case["output"].parent.parent / "missing-copy-directory"
    with pytest.raises(packager.PackageError):
        _build(case, copy_to=destination)
    assert not destination.exists()
    assert list(case["output"].parent.iterdir()) == []


def test_output_must_be_a_zip_outside_the_selected_source_tree(package_case: dict[str, Any]) -> None:
    case = package_case
    case["output"] = case["source"] / "package.zip"
    before = sorted(path.relative_to(case["source"]) for path in case["source"].rglob("*"))
    with pytest.raises(packager.PackageError):
        _build(case)
    assert sorted(path.relative_to(case["source"]) for path in case["source"].rglob("*")) == before
    case["output"] = case["source"].parent / "publish" / "package.tar"
    with pytest.raises(packager.PackageError):
        _build(case)
    assert not case["output"].exists()


def test_optimized_cli_keeps_hash_validation_enabled(package_case: dict[str, Any]) -> None:
    case = package_case
    case["manifest"]["files"][0]["sha256"] = "0" * 64
    _write_manifest(case["manifest_path"], case["manifest"])
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            str(TOOL_PATH),
            "--manifest",
            str(case["manifest_path"]),
            "--source-root",
            str(case["source"]),
            "--output",
            str(case["output"]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert any(word in result.stderr.lower() for word in ("sha256", "sha-256", "hash", "digest"))
    assert list(case["output"].parent.iterdir()) == []
