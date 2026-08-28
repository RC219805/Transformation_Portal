#!/usr/bin/env python3
"""Shared FastVLM runtime manifest validation helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TRUSTED_FASTVLM_MODEL_REPOS = {
    "apple/FastVLM-0.5B-fp16",
    "apple/FastVLM-1.5B-int8",
    "apple/FastVLM-7B-int4",
}
TRUSTED_RUNTIME_SOURCES = {
    "ml_fastvlm": "https://github.com/apple/ml-fastvlm.git",
    "mlx_vlm": "https://github.com/Blaizzy/mlx-vlm.git",
}
TRUSTED_MLX_VLM_PATCH_SOURCE = "ml_fastvlm"
TRUSTED_MLX_VLM_PATCH_PATH = "model_export/fastvlm_mlx-vlm.patch"
TRUSTED_RUNTIME_SOURCE_TARGETS = {
    "ml_fastvlm": "ml-fastvlm",
    "mlx_vlm": "mlx-vlm",
}
FASTVLM_RUNTIME_IMPORTS = ("datasets", "huggingface_hub", "mlx_vlm")
GIT_SUBPROCESS_TIMEOUT_SECONDS = 30
HEX_DIGITS = set("0123456789abcdef")
_ALLOWED_LOCAL_GIT_CONFIG_KEYS = {
    "core.bare",
    "core.filemode",
    "core.ignorecase",
    "core.logallrefupdates",
    "core.precomposeunicode",
    "core.repositoryformatversion",
    "remote.origin.fetch",
    "remote.origin.url",
}
_GIT_ENVIRONMENT_DENYLIST = {
    "SSH_ASKPASS",
    "SSH_ASKPASS_REQUIRE",
}


class ManifestError(RuntimeError):
    """Raised when the FastVLM runtime manifest violates governance."""


class RuntimeVerificationError(RuntimeError):
    """Raised when the local FastVLM runtime does not match the manifest."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_manifest_path() -> Path:
    return repo_root() / "config" / "fastvlm_runtime_manifest.json"


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    manifest_path = Path(path) if path is not None else default_manifest_path()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"FastVLM runtime manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"FastVLM runtime manifest is invalid JSON: {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError("FastVLM runtime manifest must be a JSON object.")
    return payload


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in HEX_DIGITS for char in text.lower())


def _is_git_revision(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 40 and all(char in HEX_DIGITS for char in text.lower())


def _safe_relative_parts(path_text: Any) -> tuple[str, ...]:
    text = str(path_text or "").strip().replace("\\", "/")
    if not text or text.startswith("/") or "\x00" in text:
        raise ManifestError(f"Unsafe FastVLM manifest path: {path_text!r}")
    parts = tuple(part for part in text.split("/") if part)
    if any(part in {".", ".."} for part in parts):
        raise ManifestError(f"Unsafe FastVLM manifest path: {path_text!r}")
    return parts


def safe_child(root: Path, path_text: Any) -> Path:
    parts = _safe_relative_parts(path_text)
    resolved_root = Path(os.path.realpath(root))
    candidate = resolved_root.joinpath(*parts)
    resolved_candidate = Path(os.path.realpath(candidate))
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ManifestError(f"FastVLM manifest path escapes runtime root: {path_text!r}") from exc
    return candidate


def runtime_root(manifest: Mapping[str, Any], *, override: Path | str | None = None) -> Path:
    if override is not None:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_absolute() else repo_root() / candidate
    root_text = str(manifest.get("runtime_root") or ".runtime/fastvlm")
    return safe_child(repo_root(), root_text)


def selected_model_roles(
    manifest: Mapping[str, Any],
    *,
    models: str | Sequence[str] | None = None,
    all_models: bool = False,
    default_roles: Sequence[str] = ("smoke", "default"),
) -> list[str]:
    model_map = manifest.get("models")
    if not isinstance(model_map, dict) or not model_map:
        raise ManifestError("FastVLM runtime manifest must define models.")
    if all_models:
        roles = list(model_map.keys())
    elif isinstance(models, str) and models.strip():
        roles = [role.strip() for role in models.split(",") if role.strip()]
    elif models:
        roles = [str(role).strip() for role in models if str(role).strip()]
    else:
        roles = list(default_roles)
    unknown = sorted(set(roles) - set(model_map.keys()))
    if unknown:
        raise ManifestError(f"Unknown FastVLM model role(s): {', '.join(unknown)}")
    return roles


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != "fastvlm-runtime.v1":
        errors.append("schema_version must be fastvlm-runtime.v1")
    try:
        safe_child(repo_root(), manifest.get("runtime_root") or ".runtime/fastvlm")
    except ManifestError as exc:
        errors.append(str(exc))

    sources = manifest.get("runtime_sources")
    if not isinstance(sources, dict):
        errors.append("runtime_sources must be an object")
    else:
        extra_sources = sorted(set(sources) - set(TRUSTED_RUNTIME_SOURCES))
        if extra_sources:
            errors.append(f"runtime_sources contains unknown key(s): {', '.join(extra_sources)}")
        for name, expected_url in TRUSTED_RUNTIME_SOURCES.items():
            source = sources.get(name)
            if not isinstance(source, dict):
                errors.append(f"runtime_sources.{name} must be an object")
                continue
            if source.get("repo_url") != expected_url:
                errors.append(f"runtime_sources.{name}.repo_url is not allowlisted")
            if not _is_git_revision(source.get("revision")):
                errors.append(f"runtime_sources.{name}.revision must be a pinned 40-hex revision")
            try:
                _safe_relative_parts(source.get("target_dir"))
            except ManifestError as exc:
                errors.append(str(exc))
            if source.get("target_dir") != TRUSTED_RUNTIME_SOURCE_TARGETS[name]:
                errors.append(f"runtime_sources.{name}.target_dir must be " f"{TRUSTED_RUNTIME_SOURCE_TARGETS[name]}")
            patch = source.get("patch")
            if name != "mlx_vlm":
                if patch is not None:
                    errors.append(f"runtime_sources.{name}.patch is not supported")
                continue
            if not isinstance(patch, dict):
                errors.append("runtime_sources.mlx_vlm.patch must be an object")
                continue
            if patch.get("source") != TRUSTED_MLX_VLM_PATCH_SOURCE:
                errors.append("runtime_sources.mlx_vlm.patch.source must be ml_fastvlm")
            if patch.get("path") != TRUSTED_MLX_VLM_PATCH_PATH:
                errors.append("runtime_sources.mlx_vlm.patch.path must be " f"{TRUSTED_MLX_VLM_PATCH_PATH}")
            try:
                _safe_relative_parts(patch.get("path"))
            except ManifestError as exc:
                errors.append(str(exc))
            if not _is_sha256(patch.get("sha256")):
                errors.append("runtime_sources.mlx_vlm.patch.sha256 must be a SHA-256 hex digest")
            if not _is_git_revision(patch.get("patched_tree")):
                errors.append("runtime_sources.mlx_vlm.patch.patched_tree must be a 40-hex Git tree")

    models = manifest.get("models")
    if not isinstance(models, dict) or not models:
        errors.append("models must be a non-empty object")
    else:
        for role, model in models.items():
            if not isinstance(model, dict):
                errors.append(f"models.{role} must be an object")
                continue
            repo_id = model.get("repo_id")
            if repo_id not in TRUSTED_FASTVLM_MODEL_REPOS:
                errors.append(f"models.{role}.repo_id is not allowlisted")
            if not _is_git_revision(model.get("revision")):
                errors.append(f"models.{role}.revision must be a pinned 40-hex revision")
            try:
                _safe_relative_parts(model.get("target_dir"))
            except ManifestError as exc:
                errors.append(str(exc))
            required_files = model.get("required_files")
            if not isinstance(required_files, list) or not required_files:
                errors.append(f"models.{role}.required_files must be a non-empty list")
                continue
            for index, entry in enumerate(required_files):
                if not isinstance(entry, dict):
                    errors.append(f"models.{role}.required_files[{index}] must be an object")
                    continue
                try:
                    _safe_relative_parts(entry.get("path"))
                except ManifestError as exc:
                    errors.append(str(exc))
                if not _is_sha256(entry.get("sha256")):
                    errors.append(f"models.{role}.required_files[{index}].sha256 must be a SHA-256 hex digest")
                size = entry.get("size_bytes")
                if not isinstance(size, int) or size <= 0:
                    errors.append(f"models.{role}.required_files[{index}].size_bytes must be a positive integer")
    return errors


def require_valid_manifest(manifest: Mapping[str, Any]) -> None:
    errors = validate_manifest(manifest)
    if errors:
        raise ManifestError("; ".join(errors))


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_not_symlink(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeVerificationError(f"FastVLM manifest path must not be a symlink: {path}")


def model_target_dir(manifest: Mapping[str, Any], role: str, *, root: Path | None = None) -> Path:
    model_map = manifest.get("models")
    if not isinstance(model_map, dict) or role not in model_map:
        raise ManifestError(f"Unknown FastVLM model role: {role}")
    runtime = root or runtime_root(manifest)
    model = model_map[role]
    if not isinstance(model, dict):
        raise ManifestError(f"models.{role} must be an object")
    return safe_child(runtime, model.get("target_dir"))


def verify_model_role(manifest: Mapping[str, Any], role: str, *, root: Path | None = None) -> list[str]:
    target = model_target_dir(manifest, role, root=root)
    return verify_model_files(manifest, role, target)


def verify_model_files(manifest: Mapping[str, Any], role: str, target: Path) -> list[str]:
    errors: list[str] = []
    model = manifest["models"][role]
    if not target.is_dir():
        return [f"FastVLM model role {role} missing directory: {target}"]
    for entry in model["required_files"]:
        file_path = safe_child(target, entry["path"])
        try:
            _ensure_not_symlink(file_path)
            if not file_path.is_file():
                errors.append(f"FastVLM model role {role} missing required file: {file_path}")
                continue
            actual_size = file_path.stat().st_size
            if actual_size != int(entry["size_bytes"]):
                errors.append(
                    f"FastVLM model role {role} file size mismatch for {entry['path']}: "
                    f"{actual_size} != {entry['size_bytes']}"
                )
                continue
            actual_sha = compute_file_sha256(file_path)
            if actual_sha != str(entry["sha256"]).lower():
                errors.append(
                    f"FastVLM model role {role} SHA-256 mismatch for {entry['path']}: " f"{actual_sha} != {entry['sha256']}"
                )
        except (OSError, ManifestError, RuntimeVerificationError) as exc:
            errors.append(str(exc))
    return errors


def secure_git_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an environment that cannot redirect Git or load ambient config."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_") and key.upper() not in _GIT_ENVIRONMENT_DENYLIST
    }
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
    )
    if extra:
        environment.update(extra)
    return environment


def run_secure_git(
    args: Sequence[str],
    *,
    git_dir: Path | None = None,
    work_tree: Path | None = None,
    cwd: Path | None = None,
    allow_file_protocol: bool = False,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    """Run Git with hooks, ambient configuration, and redirecting env disabled."""

    command = [
        "git",
        "--no-pager",
        "--literal-pathspecs",
        "-c",
        f"core.hooksPath={os.devnull}",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "gc.auto=0",
        "-c",
        "maintenance.auto=false",
        "-c",
        "credential.helper=",
        "-c",
        "protocol.ext.allow=never",
        "-c",
        f"protocol.file.allow={'always' if allow_file_protocol else 'never'}",
    ]
    if git_dir is not None:
        command.append(f"--git-dir={git_dir}")
    if work_tree is not None:
        command.append(f"--work-tree={work_tree}")
    command.extend(str(arg) for arg in args)
    try:
        return subprocess.run(
            command,
            check=False,
            cwd=str(cwd or work_tree) if (cwd is not None or work_tree is not None) else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
            env=secure_git_environment(),
            timeout=GIT_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeVerificationError(f"FastVLM Git command timed out after {GIT_SUBPROCESS_TIMEOUT_SECONDS}s") from exc


def _require_git_output(
    args: Sequence[str],
    *,
    git_dir: Path | None = None,
    work_tree: Path | None = None,
    cwd: Path | None = None,
    allow_file_protocol: bool = False,
) -> str:
    completed = run_secure_git(
        args,
        git_dir=git_dir,
        work_tree=work_tree,
        cwd=cwd,
        allow_file_protocol=allow_file_protocol,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise RuntimeVerificationError(f"FastVLM Git verification failed: {detail}")
    return completed.stdout.strip()


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _ensure_no_symlink_components(path: Path) -> None:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise RuntimeVerificationError(f"FastVLM source path is missing: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeVerificationError(f"FastVLM source path must not contain symlinks: {current}")


def _ensure_git_metadata_is_local(git_dir: Path) -> None:
    for forbidden in (
        git_dir / "commondir",
        git_dir / "gitdir",
        git_dir / "config.worktree",
        git_dir / "info" / "grafts",
        git_dir / "objects" / "info" / "alternates",
        git_dir / "refs" / "replace",
    ):
        if forbidden.exists() or forbidden.is_symlink():
            raise RuntimeVerificationError(f"FastVLM Git metadata redirection is not permitted: {forbidden}")

    def inspect_directory(directory: Path) -> None:
        try:
            entries = os.scandir(directory)
        except OSError as exc:
            raise RuntimeVerificationError(f"FastVLM Git metadata could not be read: {directory}") from exc
        with entries:
            for entry in entries:
                metadata = entry.stat(follow_symlinks=False)
                path = Path(entry.path)
                if stat.S_ISLNK(metadata.st_mode):
                    raise RuntimeVerificationError(f"FastVLM Git metadata must not contain symlinks: {path}")
                if stat.S_ISDIR(metadata.st_mode):
                    inspect_directory(path)
                elif not stat.S_ISREG(metadata.st_mode):
                    raise RuntimeVerificationError(f"FastVLM Git metadata contains unsupported filesystem entry: {path}")

    inspect_directory(git_dir)

    hooks_dir = git_dir / "hooks"
    if hooks_dir.is_dir():
        active_hooks = sorted(path.name for path in hooks_dir.iterdir() if not path.name.endswith(".sample"))
        if active_hooks:
            raise RuntimeVerificationError("FastVLM Git checkout contains active hooks: " + ", ".join(active_hooks))

    exclude_path = git_dir / "info" / "exclude"
    if exclude_path.is_file():
        active_excludes = [
            line.strip()
            for line in exclude_path.read_text(encoding="utf-8", errors="strict").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if active_excludes:
            raise RuntimeVerificationError("FastVLM Git checkout contains active local exclude rules")


def _is_allowed_local_git_config_key(key: str) -> bool:
    if key in _ALLOWED_LOCAL_GIT_CONFIG_KEYS:
        return True
    if key.startswith("branch."):
        return key.rsplit(".", 1)[-1] in {"merge", "remote"}
    return False


def _git_config_values(config_path: Path, key: str) -> list[str]:
    completed = run_secure_git(
        ["config", "--file", str(config_path), "--no-includes", "--get-all", key],
    )
    if completed.returncode == 1:
        return []
    if completed.returncode != 0:
        raise RuntimeVerificationError(f"FastVLM Git config could not be inspected: {config_path}")
    return completed.stdout.splitlines()


def inspect_source_checkout(
    path: Path,
    *,
    expected_origin: str,
    require_head: bool = True,
) -> tuple[Path, Path, str]:
    """Validate checkout boundaries and return ``(worktree, git_dir, HEAD)``."""

    target = _lexical_absolute(path)
    _ensure_no_symlink_components(target)
    if not target.is_dir():
        raise RuntimeVerificationError(f"FastVLM runtime source missing directory: {target}")
    git_dir = target / ".git"
    try:
        git_metadata = git_dir.lstat()
    except FileNotFoundError as exc:
        raise RuntimeVerificationError(f"FastVLM runtime source is not a standalone Git checkout: {target}") from exc
    if not stat.S_ISDIR(git_metadata.st_mode):
        raise RuntimeVerificationError(f"FastVLM runtime source .git must be a real directory: {git_dir}")
    _ensure_git_metadata_is_local(git_dir)

    config_path = git_dir / "config"
    try:
        config_metadata = config_path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeVerificationError(f"FastVLM Git config is missing: {config_path}") from exc
    if not stat.S_ISREG(config_metadata.st_mode):
        raise RuntimeVerificationError(f"FastVLM Git config must be a regular file: {config_path}")

    keys_output = _require_git_output(
        ["config", "--file", str(config_path), "--no-includes", "--null", "--name-only", "--list"]
    )
    config_keys = [key for key in keys_output.split("\0") if key]
    unsafe_keys = sorted(key for key in config_keys if not _is_allowed_local_git_config_key(key))
    if unsafe_keys:
        raise RuntimeVerificationError("FastVLM Git config contains unsafe key(s): " + ", ".join(unsafe_keys))

    origin_values = _git_config_values(config_path, "remote.origin.url")
    if origin_values != [expected_origin]:
        actual_origin = origin_values[0] if len(origin_values) == 1 else "missing or ambiguous"
        raise RuntimeVerificationError(f"FastVLM runtime source origin mismatch: {actual_origin} != {expected_origin}")

    absolute_git_dir = _require_git_output(["rev-parse", "--absolute-git-dir"], git_dir=git_dir, work_tree=target)
    if Path(os.path.realpath(absolute_git_dir)) != git_dir:
        raise RuntimeVerificationError(f"FastVLM Git directory resolves outside the source checkout: {git_dir}")
    top_level = _require_git_output(["rev-parse", "--show-toplevel"], git_dir=git_dir, work_tree=target)
    if Path(os.path.realpath(top_level)) != target:
        raise RuntimeVerificationError(f"FastVLM Git worktree resolves outside the source checkout: {target}")
    head = ""
    if require_head:
        head = _require_git_output(["rev-parse", "--verify", "HEAD^{commit}"], git_dir=git_dir, work_tree=target)
    return target, git_dir, head


def _git_tree_entries(target: Path, git_dir: Path, treeish: str) -> dict[str, tuple[str, str]]:
    completed = run_secure_git(
        ["ls-tree", "-rz", "--full-tree", treeish],
        git_dir=git_dir,
        work_tree=target,
        text=False,
    )
    if completed.returncode != 0:
        raise RuntimeVerificationError(f"FastVLM pinned Git tree is unavailable: {treeish}")
    entries: dict[str, tuple[str, str]] = {}
    for raw_record in completed.stdout.split(b"\0"):
        if not raw_record:
            continue
        metadata, separator, raw_path = raw_record.partition(b"\t")
        if not separator:
            raise RuntimeVerificationError("FastVLM pinned Git tree contains malformed entries")
        fields = metadata.decode("ascii", errors="strict").split()
        if len(fields) != 3:
            raise RuntimeVerificationError("FastVLM pinned Git tree contains malformed metadata")
        mode, object_type, object_id = fields
        try:
            relative_path = raw_path.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RuntimeVerificationError("FastVLM pinned Git tree contains a non-UTF-8 path") from exc
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise RuntimeVerificationError(
                f"FastVLM pinned Git tree contains unsupported entry: {relative_path} ({mode} {object_type})"
            )
        _safe_relative_parts(relative_path)
        if relative_path in entries:
            raise RuntimeVerificationError(f"FastVLM pinned Git tree contains duplicate path: {relative_path}")
        entries[relative_path] = (mode, object_id)
    return entries


def _filesystem_entries(target: Path) -> tuple[dict[str, Path], set[str]]:
    files: dict[str, Path] = {}
    directories: set[str] = set()

    def visit(directory: Path, relative_parts: tuple[str, ...]) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise RuntimeVerificationError(f"FastVLM source directory could not be read: {directory}") from exc
        for entry in entries:
            if not relative_parts and entry.name == ".git":
                continue
            if "\\" in entry.name or entry.name in {".", ".."}:
                raise RuntimeVerificationError(f"FastVLM source contains unsafe path component: {entry.name!r}")
            relative = "/".join((*relative_parts, entry.name))
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeVerificationError(f"FastVLM source contains symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
                visit(Path(entry.path), (*relative_parts, entry.name))
            elif stat.S_ISREG(metadata.st_mode):
                files[relative] = Path(entry.path)
            else:
                raise RuntimeVerificationError(f"FastVLM source contains unsupported filesystem entry: {relative}")

    visit(target, ())
    return files, directories


def _git_blob_oid(path: Path) -> tuple[str, os.stat_result]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeVerificationError(f"FastVLM source entry must be a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeVerificationError(f"FastVLM source changed while being verified: {path}")
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(f"blob {opened.st_size}\0".encode("ascii"))
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.lstat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(opened, field) != getattr(after_open, field) for field in stable_fields) or any(
        getattr(opened, field) != getattr(after_path, field) for field in stable_fields
    ):
        raise RuntimeVerificationError(f"FastVLM source changed while being verified: {path}")
    return digest.hexdigest(), after_path


def verify_checkout_tree(target: Path, git_dir: Path, expected_tree: str) -> None:
    expected_files = _git_tree_entries(target, git_dir, expected_tree)
    actual_files, actual_directories = _filesystem_entries(target)
    expected_paths = set(expected_files)
    actual_paths = set(actual_files)
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    expected_directories: set[str] = set()
    for relative_path in expected_paths:
        parts = relative_path.split("/")[:-1]
        expected_directories.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))
    extra_directories = sorted(actual_directories - expected_directories)
    if missing or extra or extra_directories:
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(missing[:5]))
        if extra:
            details.append("ungoverned=" + ", ".join(extra[:5]))
        if extra_directories:
            details.append("ungoverned_dirs=" + ", ".join(extra_directories[:5]))
        raise RuntimeVerificationError("FastVLM source tree does not match its governed tree: " + "; ".join(details))

    for relative_path, (expected_mode, expected_oid) in expected_files.items():
        actual_oid, metadata = _git_blob_oid(actual_files[relative_path])
        if actual_oid != expected_oid:
            raise RuntimeVerificationError(f"FastVLM source content mismatch: {relative_path}")
        if os.name != "nt":
            actual_executable = bool(metadata.st_mode & 0o111)
            expected_executable = expected_mode == "100755"
            if actual_executable != expected_executable:
                raise RuntimeVerificationError(f"FastVLM source executable-mode mismatch: {relative_path}")


def verify_source_checkout(
    path: Path,
    *,
    expected_origin: str,
    expected_revision: str,
    expected_tree: str | None = None,
) -> None:
    target, git_dir, head = inspect_source_checkout(path, expected_origin=expected_origin)
    if head != expected_revision:
        raise RuntimeVerificationError(f"FastVLM runtime source revision mismatch: {head} != {expected_revision}")
    treeish = expected_tree or f"{expected_revision}^{{tree}}"
    verify_checkout_tree(target, git_dir, treeish)


def verify_runtime_sources(manifest: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    runtime = root or runtime_root(manifest)
    errors: list[str] = []
    sources = manifest.get("runtime_sources")
    if not isinstance(sources, dict):
        return ["FastVLM manifest runtime_sources must be an object"]
    source_targets: dict[str, Path] = {}
    for name in TRUSTED_RUNTIME_SOURCES:
        source = sources.get(name)
        if not isinstance(source, dict):
            errors.append(f"FastVLM runtime source {name} is missing or malformed")
            continue
        try:
            target = safe_child(runtime, source["target_dir"])
        except ManifestError as exc:
            errors.append(str(exc))
            continue
        source_targets[name] = target
        patch = source.get("patch")
        expected_tree = str(patch.get("patched_tree")) if isinstance(patch, dict) else None
        try:
            verify_source_checkout(
                target,
                expected_origin=TRUSTED_RUNTIME_SOURCES[name],
                expected_revision=str(source["revision"]),
                expected_tree=expected_tree,
            )
        except (ManifestError, RuntimeVerificationError, OSError, UnicodeError) as exc:
            errors.append(f"FastVLM runtime source {name} failed verification: {exc}")

        if not isinstance(patch, dict):
            continue

        patch_source = source_targets.get(str(patch.get("source")))
        if patch_source is None:
            errors.append(f"FastVLM runtime source {name} patch source is unavailable")
            continue
        try:
            patch_path = safe_child(patch_source, patch.get("path"))
            _ensure_no_symlink_components(patch_path)
            if not patch_path.is_file():
                errors.append(f"FastVLM runtime source {name} patch is missing: {patch_path}")
            else:
                actual_patch_sha = compute_file_sha256(patch_path)
                if actual_patch_sha != str(patch.get("sha256") or "").lower():
                    errors.append(
                        f"FastVLM runtime source {name} patch digest mismatch: " f"{actual_patch_sha} != {patch.get('sha256')}"
                    )
        except (ManifestError, RuntimeVerificationError, OSError) as exc:
            errors.append(str(exc))

    return errors


def python_runtime_path(manifest: Mapping[str, Any], *, root: Path | None = None) -> Path:
    runtime = root or runtime_root(manifest)
    python_config = manifest.get("python") or {}
    if not isinstance(python_config, dict):
        raise ManifestError("FastVLM manifest python section must be an object")
    venv_dir = safe_child(runtime, python_config.get("venv_dir") or ".venv-fastvlm")
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def verify_python_runtime(manifest: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    try:
        python_path = python_runtime_path(manifest, root=root)
    except ManifestError as exc:
        return [str(exc)]
    if not python_path.is_file():
        return [f"FastVLM Python executable missing: {python_path}"]
    if os.name != "nt" and not os.access(python_path, os.X_OK):
        return [f"FastVLM Python executable is not executable: {python_path}"]
    return []


def verify_python_imports(manifest: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    runtime = root or runtime_root(manifest)
    try:
        python_path = python_runtime_path(manifest, root=runtime)
        sources = manifest.get("runtime_sources")
        if not isinstance(sources, dict) or not isinstance(sources.get("mlx_vlm"), dict):
            raise ManifestError("FastVLM manifest mlx_vlm source is missing or malformed")
        mlx_vlm_dir = safe_child(runtime, sources["mlx_vlm"]["target_dir"])
    except ManifestError as exc:
        return [str(exc)]
    python_errors = verify_python_runtime(manifest, root=runtime)
    if python_errors:
        return python_errors
    import_lines = (
        "import importlib, sys\n"
        f"modules = {FASTVLM_RUNTIME_IMPORTS!r}\n"
        "missing = []\n"
        "loaded = {}\n"
        "for name in modules:\n"
        "    try:\n"
        "        loaded[name] = importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        missing.append(f'{name}: {type(exc).__name__}: {exc}')\n"
        f"expected_mlx_vlm_root = {str(mlx_vlm_dir)!r}\n"
        "mlx_vlm_module = loaded.get('mlx_vlm')\n"
        "if mlx_vlm_module is not None:\n"
        "    try:\n"
        "        from pathlib import Path\n"
        "        Path(mlx_vlm_module.__file__).resolve().relative_to(Path(expected_mlx_vlm_root).resolve())\n"
        "    except (AttributeError, TypeError, ValueError) as exc:\n"
        "        missing.append(f'mlx_vlm import origin: {type(exc).__name__}: {exc}')\n"
        "datasets_module = loaded.get('datasets')\n"
        "if datasets_module is not None:\n"
        "    try:\n"
        "        dataset = datasets_module.Dataset.from_dict({'value': [1, 2]})\n"
        "        mapped = dataset.map(\n"
        "            lambda row: {'doubled': row['value'] * 2},\n"
        "            keep_in_memory=True,\n"
        "            load_from_cache_file=False,\n"
        "        )\n"
        "        if dataset.column_names != ['value'] or mapped.column_names != ['value', 'doubled']:\n"
        "            raise AssertionError('unexpected datasets columns')\n"
        "        if mapped[1]['doubled'] != 4:\n"
        "            raise AssertionError('unexpected datasets map result')\n"
        "    except Exception as exc:\n"
        "        missing.append(f'datasets API smoke: {type(exc).__name__}: {exc}')\n"
        "print('\\n'.join(missing), file=sys.stderr)\n"
        "sys.exit(1 if missing else 0)"
    )
    python_environment = os.environ.copy()
    python_environment.pop("PYTHONHOME", None)
    python_environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(mlx_vlm_dir),
            "PYTHONSAFEPATH": "1",
        }
    )
    try:
        completed = subprocess.run(
            [str(python_path), "-c", import_lines],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            env=python_environment,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stderr if isinstance(exc.stderr, str) else "") or (exc.stdout if isinstance(exc.stdout, str) else "")
        return ["FastVLM Python import smoke timed out after 30s: " + (output.strip() or str(python_path))]
    if completed.returncode == 0:
        return []
    return [
        "FastVLM Python import smoke failed: "
        + (completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}")
    ]


def verify_runtime(
    manifest: Mapping[str, Any],
    *,
    roles: Iterable[str],
    root: Path | None = None,
    include_sources: bool = True,
    include_python: bool = True,
) -> list[str]:
    require_valid_manifest(manifest)
    errors: list[str] = []
    if include_sources:
        errors.extend(verify_runtime_sources(manifest, root=root))
    if include_python:
        errors.extend(verify_python_runtime(manifest, root=root))
    for role in roles:
        errors.extend(verify_model_role(manifest, role, root=root))
    return errors


def allow_patterns_for_role(manifest: Mapping[str, Any], role: str) -> list[str]:
    model = manifest["models"][role]
    patterns = sorted({str(entry["path"]) for entry in model["required_files"]})
    return patterns


def add_common_manifest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest",
        default=str(default_manifest_path()),
        help="FastVLM runtime manifest path (default: %(default)s)",
    )
    parser.add_argument(
        "--runtime-root",
        default="",
        help="Optional runtime root override (default: manifest runtime_root)",
    )
    parser.add_argument(
        "--models",
        default="smoke,default",
        help="Comma-separated model roles to verify or install (default: %(default)s)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Select all manifest model roles",
    )
