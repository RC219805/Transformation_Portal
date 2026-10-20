"""Materialized parent runtime for photographic transforms and verified cache use."""

from __future__ import annotations

import importlib.util
import marshal
import os
import stat
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from transformation_portal.core.execution_plan_v2 import digest_payload
from transformation_portal.depth.backends import da3_runtime_identity as identity
from transformation_portal.lux_depth_v4._bytecode import source_cache_semantic_digest

_CODEC_MODULES = ("deflate", "imagecodecs", "psutil", "tifffile")
_MAX_DIRECTORY_ENTRIES = 65536
_MAX_BYTECODE_BYTES = 16 * 1024 * 1024
_MAX_BYTECODE_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_BYTECODE_VALIDATION_BYTES = 128 * 1024 * 1024
_BytecodeReceipts = dict[str, tuple[tuple[int, int, int, int, int], tuple[int, int, int, int, int]]]
_BytecodeBudget = tuple[list[int], list[int], set[str]]
_CacheNamespace = dict[str, tuple[int, int, int] | None]


def _bytecode_configuration() -> tuple:
    """Bind cache selection without making source-equivalent caches output authority."""
    if len(sys.path) > identity._MAX_IMPORT_SEARCH_PATHS:
        raise RuntimeError("Photographic bytecode import roots exceed their inventory bound")
    prefix = sys.pycache_prefix
    if prefix is not None and (type(prefix) is not str or not prefix or len(prefix) > 4096):
        raise RuntimeError("Photographic bytecode prefix is invalid")
    cwd = os.getcwd() if prefix is not None and not os.path.isabs(prefix) else None
    effective = None if prefix is None else os.path.abspath(prefix)
    return (
        prefix,
        effective,
        cwd,
        sys.implementation.cache_tag,
        sys.flags.optimize,
        importlib.util.MAGIC_NUMBER,
        tuple(sys.path),
    )


def _source_cache_path(filename: str, optimization: int) -> Path:
    cache = importlib.util.cache_from_source(filename, optimization="" if optimization == 0 else str(optimization))
    return Path(os.path.abspath(cache))


def _bytecode_budget() -> _BytecodeBudget:
    return [_MAX_DIRECTORY_ENTRIES], [_MAX_BYTECODE_VALIDATION_BYTES], set()


def _bytecode_entry(budget: _BytecodeBudget, path: Path) -> None:
    name = str(path)
    if name in budget[2]:
        return
    budget[0][0] -= 1
    if budget[0][0] < 0:
        raise RuntimeError("Photographic directory inventory exceeds entry bound")
    budget[2].add(name)


def _cache_ancestry(path: Path, namespace: _CacheNamespace, budget: _BytecodeBudget) -> None:
    """Stat only selected ancestors; never enumerate the global cache prefix."""
    parents = tuple(reversed(path.parents))
    if len(parents) > 32:
        raise RuntimeError("Photographic bytecode namespace exceeds depth bound")
    for parent in parents:
        name = str(parent)
        _bytecode_entry(budget, parent)
        try:
            observed = parent.lstat()
        except FileNotFoundError:
            if namespace.get(name) is not None:
                raise RuntimeError(f"Photographic bytecode namespace changed: {parent}") from None
            namespace[name] = None
            continue
        if not stat.S_ISDIR(observed.st_mode):
            raise RuntimeError(f"Photographic bytecode ancestry is not a directory: {parent}")
        current = (stat.S_IFMT(observed.st_mode), observed.st_dev, observed.st_ino)
        if namespace.get(name) not in (None, current):
            raise RuntimeError(f"Photographic bytecode namespace changed: {parent}")
        namespace[name] = current


def _source_compilation_filenames(source: Path) -> tuple[str, ...]:
    """Include pip's lexical RECORD paths without trusting a cached code object.

    Wheel scripts can be compiled as site-packages/../../../bin/script.py. Every
    candidate comes from the already bound import roots and resolves to the same
    source; arbitrary filenames from the bytecode are never parsed or accepted.
    """
    candidates = [str(source)]
    for entry in sys.path[:64]:
        if not isinstance(entry, str) or not entry or not Path(entry).is_absolute():
            continue
        candidate = str(Path(entry) / os.path.relpath(source, entry))
        if len(candidate) <= 4096 and candidate not in candidates and Path(candidate).resolve() == source:
            candidates.append(candidate)
    return tuple(candidates)


def _read_bytecode_input(path: Path, maximum: int) -> tuple[bytes, tuple[int, int, int, int, int]]:
    """Read a bounded immutable file without following links or parsing bytecode."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= maximum:
            raise RuntimeError(f"Photographic bytecode input exceeds its regular-file bound: {path}")
        payload = handle.read(maximum + 1)
        after = os.fstat(handle.fileno())
        if len(payload) != before.st_size or _stat_identity(after) != _stat_identity(before):
            raise RuntimeError(f"Photographic bytecode input changed during verification: {path}")
        if _stat_identity(path.lstat()) != _stat_identity(before):
            raise RuntimeError(f"Photographic bytecode namespace changed during verification: {path}")
    return payload, _stat_identity(before)


def _verify_source_bytecode(
    path: Path,
    receipts: _BytecodeReceipts,
    remaining_bytes: list[int],
    *,
    source: Path | None = None,
    optimization: int | None = None,
    expected_source_identity: tuple[int, int, int, int, int] | None = None,
) -> bool:
    """Verify current-interpreter caches using compiled source, never marshal.loads.

    Other interpreter tags and orphan caches cannot satisfy normal source imports.
    Current caches are compared with this interpreter's compilation of their
    corresponding source. This bounds hostile payloads before allocation and
    avoids unmarshalling attacker-controlled code objects entirely.
    """
    if source is None:
        cache_name = str(path)
        if sys.pycache_prefix is not None:
            prefix = Path(os.path.abspath(sys.pycache_prefix))
            if path.is_relative_to(prefix):
                # source_from_cache uses the interpreter's lexical prefix;
                # filesystem discovery uses its effective absolute spelling.
                cache_name = os.path.join(sys.pycache_prefix, str(path.relative_to(prefix)))
        try:
            source = Path(os.path.abspath(importlib.util.source_from_cache(cache_name)))
        except ValueError:
            return False
        optimization = next(
            (level for level in range(3) if _source_cache_path(str(source), level) == path),
            None,
        )
    if optimization is None or not source.exists():
        return False
    cache_stat, source_stat = path.lstat(), source.lstat()
    if not stat.S_ISREG(cache_stat.st_mode) or not stat.S_ISREG(source_stat.st_mode):
        raise RuntimeError(f"Photographic bytecode must bind a regular source file: {path}")
    if expected_source_identity is not None and _stat_identity(source_stat) != expected_source_identity:
        raise RuntimeError(f"Photographic bytecode source changed after materialization: {source}")
    expected_stats = (_stat_identity(cache_stat), _stat_identity(source_stat))
    if receipts.get(str(path)) == expected_stats:
        return True
    remaining_bytes[0] -= cache_stat.st_size + source_stat.st_size
    if remaining_bytes[0] < 0:
        raise RuntimeError("Photographic bytecode verification exceeds its aggregate byte bound")
    cached, cache_identity = _read_bytecode_input(path, _MAX_BYTECODE_BYTES)
    source_bytes, source_identity = _read_bytecode_input(source, _MAX_BYTECODE_SOURCE_BYTES)
    if expected_source_identity is not None and source_identity != expected_source_identity:
        raise RuntimeError(f"Photographic bytecode source changed after materialization: {source}")
    if len(cached) >= 16 and cached[:4] == importlib.util.MAGIC_NUMBER:
        try:
            cached_digest = source_cache_semantic_digest(cached[16:])
            matches = False
            for filename in _source_compilation_filenames(source):
                compiled = compile(source_bytes, filename, "exec", dont_inherit=True, optimize=optimization)
                expected = marshal.dumps(compiled)
                if len(expected) > _MAX_BYTECODE_BYTES:
                    raise RuntimeError(f"Photographic compiled bytecode exceeds its bound: {source}")
                if cached_digest == source_cache_semantic_digest(expected):
                    matches = True
                    break
        except (SyntaxError, ValueError, RecursionError, OverflowError) as exc:
            raise RuntimeError(f"Photographic bytecode source cannot be compiled safely: {source}") from exc
        if not matches:
            raise RuntimeError(f"Photographic cached bytecode differs from its source: {path}")
    receipts[str(path)] = (cache_identity, source_identity)
    return True


def _selected_source_caches(sources: tuple[Path, ...]) -> tuple[tuple[Path, Path, int], ...]:
    """Derive only cache paths which admitted source imports can actually select."""
    selected: dict[str, tuple[Path, Path, int]] = {}
    for source in sources:
        for filename in _source_compilation_filenames(source):
            for optimization in range(3):
                cache = _source_cache_path(filename, optimization)
                candidate = cache, source, optimization
                if str(cache) in selected and selected[str(cache)] != candidate:
                    raise RuntimeError("Photographic cache path has conflicting source authority")
                selected[str(cache)] = candidate
                if len(selected) > _MAX_DIRECTORY_ENTRIES:
                    raise RuntimeError("Photographic directory inventory exceeds entry bound")
    return tuple(selected.values())


def _verify_selected_caches(
    caches: tuple[tuple[Path, Path, int], ...],
    receipts: _BytecodeReceipts,
    namespace: _CacheNamespace,
    budget: _BytecodeBudget,
    *,
    source_identities: dict[str, tuple[int, int, int, int, int]] | None = None,
) -> None:
    for cache, source, optimization in caches:
        _bytecode_entry(budget, cache)
        _cache_ancestry(cache, namespace, budget)
        _cache_ancestry(source, namespace, budget)
        expected_source = None if source_identities is None else source_identities[str(source)]
        if expected_source is not None:
            observed_source = source.lstat()
            if not stat.S_ISREG(observed_source.st_mode) or _stat_identity(observed_source) != expected_source:
                raise RuntimeError(f"Photographic bytecode source changed after materialization: {source}")
        try:
            cache.lstat()
        except FileNotFoundError:
            pass  # Positive absence permits normal source compilation.
        else:
            if not _verify_source_bytecode(
                cache,
                receipts,
                budget[1],
                source=source,
                optimization=optimization,
                expected_source_identity=expected_source,
            ):
                raise RuntimeError(f"Photographic selected bytecode lost its bound source: {source}")
        _cache_ancestry(cache, namespace, budget)
        _cache_ancestry(source, namespace, budget)


def _directory_inventory(
    path: Path,
    *,
    bytecode_receipts: _BytecodeReceipts | None = None,
    validation_budget: _BytecodeBudget | None = None,
) -> tuple:
    """Bind namespace membership while permitting only verified source-cache churn.

    A source-equivalent pyc may be added or rewritten by ordinary lazy imports.
    Every executable current-interpreter cache is revalidated when its own or its
    source's file identity changes. Other namespace entries stay authoritative.
    Already materialized wheel bytecode retains strict file checks separately.
    """
    receipts = {} if bytecode_receipts is None else bytecode_receipts
    path = Path(os.path.abspath(path))
    budget = _bytecode_budget() if validation_budget is None else validation_budget
    configuration = _bytecode_configuration()
    prefix = None if configuration[1] is None else Path(configuration[1])

    def inventory(directory: Path, *, cache_tree: bool = False, depth: int = 0) -> tuple:
        if depth > 32:
            raise RuntimeError("Photographic bytecode namespace exceeds depth bound")
        result: list[tuple[Any, ...]] = []
        with os.scandir(directory) as children:
            for child in children:
                _bytecode_entry(budget, Path(child.path))
                observed = child.stat(follow_symlinks=False)
                kind = stat.S_IFMT(observed.st_mode)
                prefixed = prefix is not None and directory.is_relative_to(prefix)
                if child.name.endswith(".pyc") and (directory.name == "__pycache__" or prefixed):
                    verified = _verify_source_bytecode(Path(child.path), receipts, budget[1])
                    if stat.S_ISREG(observed.st_mode) and (directory.name == "__pycache__" or verified):
                        continue
                if stat.S_ISDIR(observed.st_mode) and (child.name == "__pycache__" or cache_tree):
                    contents = inventory(Path(child.path), cache_tree=True, depth=depth + 1)
                    if child.name == "__pycache__" and not contents:
                        continue
                    result.append((child.name, kind, observed.st_dev, observed.st_ino, contents))
                else:
                    result.append((child.name, kind, observed.st_dev, observed.st_ino))
        return tuple(sorted(result))

    if prefix is not None and path.is_relative_to(prefix):
        _cache_ancestry(path / "_selected_cache_namespace", {}, budget)
    result = inventory(path, cache_tree=path.name == "__pycache__")
    if _bytecode_configuration() != configuration:
        raise RuntimeError("Photographic bytecode configuration changed")
    return result


def _stat_identity(observed: os.stat_result) -> tuple[int, int, int, int, int]:
    return observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns, observed.st_ctime_ns


def _expected_stat(entry: dict) -> tuple[int, int, int, int, int]:
    return entry["device"], entry["inode"], entry["size_bytes"], entry["mtime_ns"], entry["ctime_ns"]


class PhotographyRuntime:
    """Bind the parent closure, TIFF codecs and process supervisor at stage seams."""

    def __init__(self) -> None:
        self.bytecode_configuration = _bytecode_configuration()
        self.parent = identity.prepare_parent_output_runtime_identity()
        entries = {entry["path"]: entry for entry in self.parent.verification_entries}
        token = identity._VERIFICATION_ENTRIES.set(entries)
        try:
            installed = identity._installed_distribution_index()
            records = []
            self.modules = {}
            for name in _CODEC_MODULES:
                spec = importlib.util.find_spec(name)
                self.modules[name] = None if spec is None else spec.origin
                if spec is None:
                    if name == "tifffile":
                        raise RuntimeError("V4 photographic output requires tifffile")
                    continue
                if name not in installed:
                    raise RuntimeError(f"Photographic codec {name} has no distribution identity")
                # Capture this distribution's own closure. A shadow module must
                # not borrow authorization from an unrelated installed wheel.
                distribution_entries: dict[str, dict[str, Any]] = {}
                distribution_token = identity._VERIFICATION_ENTRIES.set(distribution_entries)
                try:
                    record = identity._distribution_record(name, distribution=installed[name][0], verify_record_hashes=False)
                finally:
                    identity._VERIFICATION_ENTRIES.reset(distribution_token)
                origin = None if spec.origin is None else str(Path(spec.origin).resolve(strict=True))
                if origin is None or distribution_entries.get(origin, {}).get("kind") != "file":
                    raise RuntimeError(f"Photographic codec {name} origin is outside its materialized distribution")
                for path, entry in distribution_entries.items():
                    if path in entries and entries[path] != entry:
                        raise RuntimeError(f"Photographic runtime changed during materialization: {path}")
                    entries[path] = entry
                records.append(record)
            self.codec_imports = identity._import_environment_payload(_CODEC_MODULES)
        finally:
            identity._VERIFICATION_ENTRIES.reset(token)
        self.entries = tuple(entries.values())
        self.bytecode_receipts: _BytecodeReceipts = {}
        self.bytecode_namespace: _CacheNamespace = {}
        self.bytecode_source_identities = {
            entry["path"]: _expected_stat(entry)
            for entry in self.entries
            if entry["kind"] == "file" and entry["path"].endswith(".py")
        }
        self.selected_caches = _selected_source_caches(tuple(Path(path) for path in self.bytecode_source_identities))
        budget = _bytecode_budget()
        self.directory_inventories = {}
        for entry in self.entries:
            if entry["kind"] != "directory":
                continue
            directory_path = Path(entry["path"])
            # Materialization must finish from a stable inventory. Later cache
            # writes may change directory timestamps, but never code membership.
            if _stat_identity(directory_path.stat()) != _expected_stat(entry):
                raise RuntimeError(f"Photographic runtime changed during materialization: {directory_path}")
            self.directory_inventories[entry["path"]] = _directory_inventory(
                directory_path, bytecode_receipts=self.bytecode_receipts, validation_budget=budget
            )
        _verify_selected_caches(
            self.selected_caches,
            self.bytecode_receipts,
            self.bytecode_namespace,
            budget,
            source_identities=self.bytecode_source_identities,
        )
        if _bytecode_configuration() != self.bytecode_configuration:
            raise RuntimeError("Photographic bytecode configuration changed")
        parent_payload = asdict(self.parent)
        parent_payload.pop("verification_entries")
        self.payload = {
            "schema": "tp.photography.runtime.v1",
            "parent": parent_payload,
            "codecs": records,
            "codec_import_environment": self.codec_imports,
        }
        self.sha256 = digest_payload(self.payload)
        self.source_sha256 = self.parent.source_identity_sha256

    def verify(self) -> None:
        if _bytecode_configuration() != self.bytecode_configuration:
            raise RuntimeError("Photographic bytecode configuration changed")
        if not identity.verify_parent_output_runtime_identity(self.parent):
            raise RuntimeError("Photographic runtime import or platform identity changed")
        if identity._import_environment_payload(_CODEC_MODULES) != self.codec_imports:
            raise RuntimeError("Photographic codec import identity changed")
        budget = _bytecode_budget()
        for entry in self.entries:
            path = Path(entry["path"])
            observed = path.stat()
            expected = _expected_stat(entry)
            actual = _stat_identity(observed)
            kind = stat.S_ISREG(observed.st_mode) if entry["kind"] == "file" else stat.S_ISDIR(observed.st_mode)
            if not kind or str(path.resolve(strict=True)) != str(path):
                raise RuntimeError(f"Photographic runtime bytes changed after materialization: {path}")
            if entry["kind"] == "directory":
                if (
                    actual[:2] != expected[:2]
                    or _directory_inventory(path, bytecode_receipts=self.bytecode_receipts, validation_budget=budget)
                    != self.directory_inventories[entry["path"]]
                ):
                    raise RuntimeError(f"Photographic runtime namespace changed after materialization: {path}")
            elif actual != expected:
                raise RuntimeError(f"Photographic runtime bytes changed after materialization: {path}")
        _verify_selected_caches(
            self.selected_caches,
            self.bytecode_receipts,
            self.bytecode_namespace,
            budget,
            source_identities=self.bytecode_source_identities,
        )
        if _bytecode_configuration() != self.bytecode_configuration:
            raise RuntimeError("Photographic bytecode configuration changed")
