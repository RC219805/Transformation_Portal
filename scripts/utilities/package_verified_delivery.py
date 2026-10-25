#!/usr/bin/env python3
"""Package explicitly selected, checksum-bound files without reprocessing them.

POSIX, Python 3.11+, standard library only. Supply an explicit manifest:

  {"schema":"tp.delivery.package_spec.v1","archive_root":"delivery-example",
   "files":[{"source":"example.txt","archive_path":"documents/example.txt",
   "sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
   "size_bytes":3}]}

This selects exactly the bytes ``abc`` (no newline). Source names are relative
to --source-root; archive paths are relative to archive_root. Save the spec as
.runtime/package-demo/spec.json, then run from the repository root:

  mkdir -p .runtime/package-demo/source .runtime/package-demo/output .runtime/package-demo/copy
  printf abc > .runtime/package-demo/source/example.txt
  python3 scripts/utilities/package_verified_delivery.py \\
    --manifest .runtime/package-demo/spec.json \\
    --source-root .runtime/package-demo/source \\
    --output .runtime/package-demo/output/example.zip \\
    --copy-to .runtime/package-demo/copy

The output parent and optional copy directory must already exist outside
source-root; copy-to must differ from the output parent.
Outputs are NEW.zip, NEW.zip.sha256, NEW.zip.receipt.json;
optional copy-to receives the same three verified files. Existing names are
never replaced. Private staging and full verification precede publication.
Each file is published atomically; the multi-file/cross-directory publication
is not crash-atomic. Recoverable publication failures roll back owned links.

The ZIP includes original selected bytes, _package/manifest.json (normalized
selection), and _package/SHA256SUMS.txt under archive_root. Checksum lines use
full ZIP member names and exclude the checksum file itself. From the directory
containing example.zip, verify and extract with:

  shasum -a 256 -c example.zip.sha256
  unzip example.zip
  shasum -a 256 -c delivery-example/_package/SHA256SUMS.txt

The manifest limit is 8 MiB and 1..10,000 entries. Paths must be portable,
normalized relative POSIX paths. Symlink roots/entries/destinations, nonregular
inputs, unsafe or colliding archive names, changed files and SHA/size mismatches
fail closed. Keep generated packages ignored, for example under .runtime/.

This checks byte integrity against caller-supplied hashes only. It does not
discover files, run models, reprocess images, or replay native generations.
Curated exports need their own mappings and are not automatically replayable.
It establishes no photographic, semantic, licensing, physical-depth, or
production acceptance. ZIP byte reproducibility is scoped to the same Python
and compression runtime. Success prints a JSON receipt; errors exit 2.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import stat
import sys
import unicodedata
import uuid
import zipfile
from pathlib import Path
from typing import BinaryIO, Iterator, NamedTuple

CHUNK_SIZE = 4 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_FILES = 10_000
SCHEMA = "tp.delivery.package_spec.v1"
RESERVED = "_package"
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK


class PackageError(ValueError):
    """Invalid specification, changed input, or unsafe publication request."""


class _Stage(NamedTuple):
    parent: Path
    parent_fd: int
    name: str
    fd: int
    expected: dict


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise PackageError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _relative_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise PackageError("paths must be nonempty relative POSIX names of at most 1024 UTF-8 bytes")
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value):
        raise PackageError("control characters are forbidden in paths")
    if len(value.encode("utf-8")) > 1024:
        raise PackageError("paths must be at most 1024 UTF-8 bytes")
    if any(char in value for char in '\\:*?<>|"') or unicodedata.normalize("NFC", value) != value:
        raise PackageError(f"nonportable path: {value!r}")
    for part in value.split("/"):
        device = part.split(".", 1)[0].upper()
        if (
            part in {"", ".", ".."}
            or part.endswith((" ", "."))
            or len(part.encode("utf-8")) > 255
            or device in {"CON", "PRN", "AUX", "NUL"}
            or re.fullmatch(r"(?:COM|LPT)[1-9]", device)
        ):
            raise PackageError(f"unsafe path: {value!r}")
    return value


def _load_spec(path: Path) -> dict:
    with os.fdopen(os.open(path, READ_FLAGS), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise PackageError("manifest must be a regular file")
        raw = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise PackageError("manifest exceeds 8 MiB")
    try:
        spec = json.loads(raw, object_pairs_hook=_unique_object)
    except PackageError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise PackageError(f"invalid JSON manifest: {exc}") from exc
    if not isinstance(spec, dict) or set(spec) != {"schema", "archive_root", "files"} or spec["schema"] != SCHEMA:
        raise PackageError(f"manifest must contain schema={SCHEMA!r}, archive_root, and files")
    root = _relative_name(spec["archive_root"])
    if "/" in root:
        raise PackageError("archive_root must be one directory name")
    rows = spec["files"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_FILES:
        raise PackageError(f"files must contain between 1 and {MAX_FILES} entries")
    names = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"source", "archive_path", "sha256", "size_bytes"}:
            raise PackageError("each file requires source, archive_path, sha256, and size_bytes")
        _relative_name(row["source"])
        name = _relative_name(row["archive_path"]).casefold()
        if name.split("/", 1)[0] == RESERVED or name in names:
            raise PackageError(f"reserved or colliding archive path: {row['archive_path']}")
        names.add(name)
        if not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise PackageError("sha256 must be 64 lowercase hexadecimal characters")
        if type(row["size_bytes"]) is not int or not 0 <= row["size_bytes"] <= 2**63 - 1:
            raise PackageError("size_bytes must be a nonnegative integer below 2**63")
    for name in names:
        parts = name.split("/")
        if any("/".join(parts[:index]) in names for index in range(1, len(parts))):
            raise PackageError(f"archive file/directory collision: {name}")
    return {**spec, "files": sorted(rows, key=lambda row: row["archive_path"])}


def _fingerprint(value: os.stat_result) -> tuple:
    return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns


@contextlib.contextmanager
def _source_file(root_fd: int, relative: str) -> Iterator[BinaryIO]:
    """Walk beneath a pinned root descriptor without following any symlinks."""
    with contextlib.ExitStack() as stack:
        directory = root_fd
        parts = relative.split("/")
        for part in parts[:-1]:
            directory = os.open(part, DIRECTORY_FLAGS, dir_fd=directory)
            stack.callback(os.close, directory)
        fd = os.open(parts[-1], READ_FLAGS, dir_fd=directory)
        stream = stack.enter_context(os.fdopen(fd, "rb"))
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise PackageError(f"source is not a regular file: {relative}")
        yield stream
        named = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)
        if _fingerprint(before) != _fingerprint(os.fstat(fd)) or _fingerprint(before) != _fingerprint(named):
            raise PackageError(f"source changed during read: {relative}")


def _stream_hash(stream: BinaryIO, destination: BinaryIO | None = None, *, max_bytes: int | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(CHUNK_SIZE if max_bytes is None else min(CHUNK_SIZE, max_bytes - size + 1)):
        digest.update(chunk)
        size += len(chunk)
        if max_bytes is not None and size > max_bytes:
            raise PackageError("input exceeds declared size")
        if destination is not None:
            destination.write(chunk)
    return digest.hexdigest(), size


def _check_digest(actual: tuple[str, int], expected: dict, label: str) -> None:
    if actual != (expected["sha256"], expected["size_bytes"]):
        raise PackageError(f"SHA-256 or size mismatch: {label}")


def _info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info._compresslevel = 1
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _write_archive(stage: _Stage, filename: str, spec: dict, root_fd: int) -> dict:
    prefix = spec["archive_root"] + "/"
    selected = {prefix + row["archive_path"]: row for row in spec["files"]}
    manifest_name = prefix + RESERVED + "/manifest.json"
    sums_name = prefix + RESERVED + "/SHA256SUMS.txt"
    manifest = _canonical(spec)
    expected = {name: {"sha256": row["sha256"], "size_bytes": row["size_bytes"]} for name, row in selected.items()}
    expected[manifest_name] = {"sha256": hashlib.sha256(manifest).hexdigest(), "size_bytes": len(manifest)}
    sums = "".join(f"{row['sha256']}  {name}\n" for name, row in sorted(expected.items())).encode("utf-8")
    generated = {manifest_name: manifest, sums_name: sums}
    expected[sums_name] = {"sha256": hashlib.sha256(sums).hexdigest(), "size_bytes": len(sums)}
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with os.fdopen(os.open(filename, flags, 0o644, dir_fd=stage.fd), "w+b") as raw:
        with zipfile.ZipFile(raw, "w", allowZip64=True) as archive:
            for name in sorted(expected):
                if name in generated:
                    archive.writestr(_info(name), generated[name])
                    continue
                row = selected[name]
                with _source_file(root_fd, row["source"]) as source:
                    if os.fstat(source.fileno()).st_size != row["size_bytes"]:
                        raise PackageError(f"source size mismatch: {row['source']}")
                    with archive.open(_info(name), "w", force_zip64=True) as target:
                        _check_digest(_stream_hash(source, target, max_bytes=row["size_bytes"]), row, row["source"])
        raw.flush()
        os.fsync(raw.fileno())
        raw.seek(0)
        # Read every member through EOF: zipfile checks CRC as well.
        with zipfile.ZipFile(raw) as archive:
            if archive.namelist() != sorted(expected):
                raise PackageError("archive inventory mismatch")
            for name, row in expected.items():
                with archive.open(name) as source:
                    _check_digest(_stream_hash(source, max_bytes=row["size_bytes"]), row, name)
        raw.seek(0)
        digest, size = _stream_hash(raw)
    return {"sha256": digest, "size_bytes": size}


def _directory(path: Path) -> Path:
    if path.is_symlink():
        raise PackageError(f"symlink directories are not allowed: {path}")
    result = path.resolve(strict=True)
    if not result.is_dir():
        raise PackageError(f"not a directory: {path}")
    return result


def _preflight(directory: Path, names: list[str]) -> None:
    for name in names:
        if os.path.lexists(directory / name):
            raise PackageError(f"destination already exists: {directory / name}")


def _copy_verified(source: _Stage, target: _Stage, name: str, expected: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with (
        _source_file(source.fd, name) as stream,
        os.fdopen(os.open(name, flags, 0o644, dir_fd=target.fd), "wb") as destination,
    ):
        _check_digest(_stream_hash(stream, destination, max_bytes=expected["size_bytes"]), expected, name)
        destination.flush()
        os.fsync(destination.fileno())
    with _source_file(target.fd, name) as stream:
        _check_digest(_stream_hash(stream, max_bytes=expected["size_bytes"]), expected, name)


def _identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _check_stage(stage: _Stage) -> None:
    if stage.parent.is_symlink() or _identity(stage.parent.stat()) != _identity(os.fstat(stage.parent_fd)):
        raise PackageError(f"publication directory changed: {stage.parent}")
    named = os.stat(stage.name, dir_fd=stage.parent_fd, follow_symlinks=False)
    if _identity(named) != _identity(os.fstat(stage.fd)):
        raise PackageError("private staging directory changed")


@contextlib.contextmanager
def _staging(parent: Path, parent_fd: int, names: list[str], expected: dict) -> Iterator[_Stage]:
    name = ".tp-package-" + uuid.uuid4().hex
    os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    fd = os.open(name, DIRECTORY_FLAGS, dir_fd=parent_fd)
    stage = _Stage(parent, parent_fd, name, fd, expected)
    try:
        _check_stage(stage)
        yield stage
    finally:
        # Cleanup stays anchored to the original private directory, even if a
        # parent/name was replaced. Never recursively delete a replacement.
        try:
            for filename in names:
                try:
                    os.unlink(filename, dir_fd=fd)
                except FileNotFoundError:
                    pass
            try:
                named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if _identity(named) == _identity(os.fstat(fd)):
                    os.rmdir(name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        finally:
            os.close(fd)


def _write_metadata(stage: _Stage, name: str, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with os.fdopen(os.open(name, flags, 0o644, dir_fd=stage.fd), "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    stage.expected[name] = {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _publish(groups: list[_Stage], names: list[str]) -> None:
    """No-replace hard links publish complete files; roll back only our links."""
    published = []
    try:
        for stage in groups:
            _check_stage(stage)
            for name in names:
                with _source_file(stage.fd, name) as source:
                    expected = stage.expected[name]
                    _check_digest(_stream_hash(source, max_bytes=expected["size_bytes"]), expected, name)
                    source_identity = _identity(os.fstat(source.fileno()))
                    os.link(name, name, src_dir_fd=stage.fd, dst_dir_fd=stage.parent_fd, follow_symlinks=False)
                    published.append((stage.parent_fd, name, *source_identity))
                    actual = os.stat(name, dir_fd=stage.parent_fd, follow_symlinks=False)
                    if _identity(actual) != source_identity:
                        raise PackageError("staged file changed during publication")
            _check_stage(stage)
            os.fsync(stage.parent_fd)
        for stage in groups:
            _check_stage(stage)
    except BaseException:
        for fd, name, device, inode in reversed(published):
            try:
                current = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == (device, inode):
                    os.unlink(name, dir_fd=fd)
            except FileNotFoundError:
                continue
        raise


def build_package(manifest_path: Path, source_root: Path, output: Path, copy_to: Path | None = None) -> dict:
    """Validate, stage, verify, and publish a new ZIP and optional verified copy."""
    try:
        spec = _load_spec(Path(manifest_path))
        root = _directory(Path(source_root))
        root_identity = _identity(root.stat())
        output = Path(output).absolute()
        parent = _directory(output.parent)
        parent_identity = _identity(parent.stat())
        _relative_name(output.name)
        if output.suffix != ".zip":
            raise PackageError("output must have a .zip suffix")
        output = parent / output.name
        destination = _directory(Path(copy_to)) if copy_to is not None else None
        destination_identity = _identity(destination.stat()) if destination is not None else None
        for directory in [parent, *([destination] if destination is not None else [])]:
            if directory.is_relative_to(root):
                raise PackageError("output and copy directories must be outside source-root")
        if destination == parent:
            raise PackageError("copy-to must differ from the output directory")
        names = [output.name, output.name + ".sha256", output.name + ".receipt.json"]
        _preflight(parent, names)
        if destination is not None:
            _preflight(destination, names)
        with contextlib.ExitStack() as stack:
            root_fd = os.open(root, DIRECTORY_FLAGS)
            stack.callback(os.close, root_fd)
            if _identity(os.fstat(root_fd)) != root_identity:
                raise PackageError("source-root directory changed")
            parent_fd = os.open(parent, DIRECTORY_FLAGS)
            stack.callback(os.close, parent_fd)
            if _identity(os.fstat(parent_fd)) != parent_identity:
                raise PackageError("output directory changed")
            expected = {}
            stage = stack.enter_context(_staging(parent, parent_fd, names, expected))
            archive = _write_archive(stage, output.name, spec, root_fd)
            expected[output.name] = archive
            receipt = {
                "schema": "tp.delivery.package_receipt.v1",
                "status": "verified",
                "scope": "byte_integrity_not_semantic_verification",
                "archive": {"path": str(output), **archive},
                "source_file_count": len(spec["files"]),
                "member_count": len(spec["files"]) + 2,
                "copied_to": str(destination) if destination is not None else None,
            }
            _write_metadata(stage, names[1], f"{archive['sha256']}  {output.name}\n".encode("utf-8"))
            _write_metadata(stage, names[2], _canonical(receipt))
            groups = [stage]
            if destination is not None:
                copy_fd = os.open(destination, DIRECTORY_FLAGS)
                stack.callback(os.close, copy_fd)
                if _identity(os.fstat(copy_fd)) != destination_identity:
                    raise PackageError("copy directory changed")
                copy_stage = stack.enter_context(_staging(destination, copy_fd, names, expected))
                for name in names:
                    _copy_verified(stage, copy_stage, name, expected[name])
                groups.append(copy_stage)
            _publish(groups, names)
            return receipt
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise PackageError(f"package operation failed: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        epilog=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, required=True, help=f"explicit {SCHEMA} JSON selection")
    parser.add_argument(
        "--source-root", type=Path, required=True, help="read-only source directory; symlink inputs are rejected"
    )
    parser.add_argument("--output", type=Path, required=True, help="new .zip outside source-root; parent must exist")
    parser.add_argument("--copy-to", type=Path, help="optional existing directory for a verified ZIP/checksum/receipt copy")
    args = parser.parse_args(argv)
    try:
        receipt = build_package(args.manifest, args.source_root, args.output, args.copy_to)
    except PackageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
