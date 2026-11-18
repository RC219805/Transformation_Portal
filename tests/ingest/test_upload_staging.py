"""Function-level tests for the staged portal upload helpers.

Every malformed upload must surface as ``UploadStagingError`` (which the HTTP
route maps to a typed 4xx envelope) and must leave no batch directory behind.
"""

from __future__ import annotations

import errno
import io
import os
from pathlib import Path

import pytest

from transformation_portal.ingest import upload_staging
from transformation_portal.ingest.upload_staging import IncomingUpload, UploadStagingError

pytestmark = [pytest.mark.unit, pytest.mark.security]

NOW = 1_700_000_000.0


def _upload(name: str, payload: bytes = b"data") -> IncomingUpload:
    return IncomingUpload(filename=name, stream=io.BytesIO(payload))


def _stage(tmp_path: Path, uploads: list[IncomingUpload], **kwargs: object) -> upload_staging.StagedUploadResult:
    return upload_staging.stage_upload_batch(upload_root=tmp_path / "uploads", uploads=uploads, now=NOW, **kwargs)


def _batches_on_disk(tmp_path: Path) -> list[str]:
    upload_root = tmp_path / "uploads"
    return sorted(path.name for path in upload_root.iterdir()) if upload_root.exists() else []


def _name_max(directory: Path) -> int:
    try:
        return int(os.pathconf(str(directory), "PC_NAME_MAX"))
    except (AttributeError, OSError, ValueError):
        pytest.skip("filesystem does not report a file name limit")


# ---------------------------------------------------------------------------
# Relative paths that cannot coexist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "names",
    [
        ["a", "a/b"],
        ["a/b", "a"],
        ["shoot/day/raw/0001.dng", "shoot/day"],
    ],
    ids=["file-then-directory", "directory-then-file", "nested-directory-then-file"],
)
def test_path_used_as_file_and_directory_is_rejected(tmp_path: Path, names: list[str]) -> None:
    with pytest.raises(UploadStagingError) as exc_info:
        _stage(tmp_path, [_upload(name) for name in names])

    assert exc_info.value.reason == "conflicting_relative_path"
    assert exc_info.value.status_code == 400
    assert exc_info.value.field == "files"
    assert _batches_on_disk(tmp_path) == []


def test_paths_that_only_share_a_name_prefix_are_staged(tmp_path: Path) -> None:
    names = ["shoot/a", "shoot/a.jpg", "shoot/a.jpg.xmp", "shoot/ab/c.jpg"]

    result = _stage(tmp_path, [_upload(name, name.encode()) for name in names])

    assert result.file_count == 4
    for name in names:
        assert (result.input_dir / name).read_bytes() == name.encode()


def test_names_differing_only_in_case_never_overwrite_each_other(tmp_path: Path) -> None:
    (tmp_path / "CaseProbe").write_text("", encoding="utf-8")
    case_insensitive = (tmp_path / "caseprobe").exists()
    uploads = [_upload("IMG_0001.JPG", b"upper"), _upload("img_0001.jpg", b"lower")]

    if case_insensitive:
        with pytest.raises(UploadStagingError) as exc_info:
            _stage(tmp_path, uploads)
        assert exc_info.value.reason == "duplicate_relative_path"
        assert _batches_on_disk(tmp_path) == []
    else:
        result = _stage(tmp_path, uploads)
        assert (result.input_dir / "IMG_0001.JPG").read_bytes() == b"upper"
        assert (result.input_dir / "img_0001.jpg").read_bytes() == b"lower"


def test_destination_that_already_exists_is_never_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for a volume that folds two relative paths into one file.

    A directory symlink makes ``alias/frame.jpg`` and ``real/frame.jpg`` the same
    file, exactly as a case-insensitive or Unicode-normalising volume does for
    names that differ only in case or normal form.
    """
    batch_id = "upload_1700000000_0a1b2c3d"
    monkeypatch.setattr(upload_staging, "build_batch_id", lambda _now=None: batch_id)
    input_dir = tmp_path / "uploads" / batch_id / "input"
    (input_dir / "real").mkdir(parents=True)
    (input_dir / "alias").symlink_to("real", target_is_directory=True)

    with pytest.raises(UploadStagingError) as exc_info:
        _stage(tmp_path, [_upload("real/frame.jpg", b"first"), _upload("alias/frame.jpg", b"second")])

    assert exc_info.value.reason == "duplicate_relative_path"
    assert exc_info.value.status_code == 400
    assert _batches_on_disk(tmp_path) == []


# ---------------------------------------------------------------------------
# Relative paths the staging filesystem cannot hold
# ---------------------------------------------------------------------------


def test_filename_at_the_filesystem_name_limit_is_staged(tmp_path: Path) -> None:
    name = "x" * (_name_max(tmp_path) - len(".jpg")) + ".jpg"

    result = _stage(tmp_path, [_upload(name, b"pixels")])

    assert (result.input_dir / name).read_bytes() == b"pixels"
    assert [path.name for path in result.input_dir.iterdir()] == [name]


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_name_longer_than_the_filesystem_allows_is_a_client_error(tmp_path: Path, kind: str) -> None:
    too_long = "y" * (_name_max(tmp_path) + 1)
    name = too_long if kind == "file" else f"{too_long}/frame.jpg"

    with pytest.raises(UploadStagingError) as exc_info:
        _stage(tmp_path, [_upload(name)])

    assert exc_info.value.reason == "invalid_relative_path"
    assert exc_info.value.status_code == 400
    assert _batches_on_disk(tmp_path) == []


@pytest.mark.parametrize("depth", [1200, 2000])
def test_absurdly_nested_relative_path_is_a_client_error(tmp_path: Path, depth: int) -> None:
    """1200 levels exhaust the recursive directory creation; 2000 exceed PATH_MAX."""
    name = "/".join(["d"] * depth) + "/frame.jpg"

    with pytest.raises(UploadStagingError) as exc_info:
        _stage(tmp_path, [_upload(name)])

    assert exc_info.value.reason == "invalid_relative_path"
    assert exc_info.value.status_code == 400
    assert _batches_on_disk(tmp_path) == []


@pytest.mark.parametrize("name", ["frame\udc80.jpg", "frame\ud800.jpg"], ids=["surrogateescape", "lone-high-surrogate"])
def test_relative_path_that_is_not_valid_unicode_is_a_client_error(tmp_path: Path, name: str) -> None:
    """JSON escapes in client_manifest can smuggle lone surrogates into a path."""
    manifest_paths = upload_staging.parse_client_manifest_relative_paths(
        '[{"relative_path": "' + name.encode("unicode_escape").decode("ascii") + '"}]',
        expected_count=1,
    )
    assert manifest_paths == [name]

    with pytest.raises(UploadStagingError) as exc_info:
        _stage(tmp_path, [_upload("")], client_manifest_paths=manifest_paths)

    assert exc_info.value.reason == "invalid_relative_path"
    assert exc_info.value.message == "relative path is not valid Unicode"
    assert _batches_on_disk(tmp_path) == []


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (OSError(errno.ENAMETOOLONG, "File name too long"), "invalid_relative_path"),
        (OSError(errno.EILSEQ, "Illegal byte sequence"), "invalid_relative_path"),
        (RecursionError("maximum recursion depth exceeded"), "invalid_relative_path"),
        (FileExistsError(errno.EEXIST, "File exists"), "conflicting_relative_path"),
        (IsADirectoryError(errno.EISDIR, "Is a directory"), "conflicting_relative_path"),
        (NotADirectoryError(errno.ENOTDIR, "Not a directory"), "conflicting_relative_path"),
    ],
)
def test_filesystem_refusal_of_a_relative_path_is_a_client_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException, reason: str
) -> None:
    def _refuse(_source: object, _destination: Path) -> tuple[str, int]:
        raise error

    monkeypatch.setattr(upload_staging, "_stream_upload_to_path", _refuse)

    with pytest.raises(UploadStagingError) as exc_info:
        _stage(tmp_path, [_upload("frame.jpg")])

    assert exc_info.value.reason == reason
    assert exc_info.value.status_code == 400
    assert exc_info.value.__cause__ is error
    assert _batches_on_disk(tmp_path) == []


@pytest.mark.parametrize(
    "error",
    [
        OSError(errno.ENOSPC, "No space left on device"),
        PermissionError(errno.EACCES, "Permission denied"),
        OSError(errno.EIO, "Input/output error"),
    ],
    ids=["no-space", "permission", "io-error"],
)
def test_server_side_storage_fault_is_not_reported_as_a_client_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: OSError
) -> None:
    def _fail(_source: object, _destination: Path) -> tuple[str, int]:
        raise error

    monkeypatch.setattr(upload_staging, "_stream_upload_to_path", _fail)

    with pytest.raises(OSError) as exc_info:
        _stage(tmp_path, [_upload("frame.jpg")])

    assert exc_info.value is error
    assert _batches_on_disk(tmp_path) == []


# ---------------------------------------------------------------------------
# client_manifest parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_manifest",
    [
        "[" * 100_000 + "]" * 100_000,
        "[" + "1" * 5000 + "]",
        '[{"relative_path": ' + "1" * 5000 + "}]",
    ],
    ids=["deeply-nested", "huge-integer", "huge-integer-relative-path"],
)
def test_client_manifest_that_overwhelms_the_json_decoder_is_a_client_error(raw_manifest: str) -> None:
    with pytest.raises(UploadStagingError) as exc_info:
        upload_staging.parse_client_manifest_relative_paths(raw_manifest, expected_count=1)

    assert exc_info.value.reason == "invalid_client_manifest"
    assert exc_info.value.field == "client_manifest"
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Expired batch cleanup
# ---------------------------------------------------------------------------

TTL_SECONDS = 86_400.0


def _expired_batch(tmp_path: Path, names: list[str]) -> upload_staging.StagedUploadResult:
    result = _stage(tmp_path, [_upload(name) for name in names])
    os.utime(result.batch_root, (NOW, NOW))
    return result


def _cleanup(tmp_path: Path, retained_input_dirs: list[str], *, age_seconds: float = 2 * TTL_SECONDS) -> list[str]:
    return upload_staging.cleanup_expired_batches(
        tmp_path / "uploads",
        now=NOW + age_seconds,
        ttl_seconds=TTL_SECONDS,
        retained_input_dirs=retained_input_dirs,
    )


@pytest.mark.parametrize(
    "job_input_dir",
    ["input", "input/shoot", "input/shoot/day", "."],
    ids=["staged-input", "top-level-root", "nested-directory", "batch-root"],
)
def test_expired_batch_is_retained_while_a_job_reads_from_inside_it(tmp_path: Path, job_input_dir: str) -> None:
    result = _expired_batch(tmp_path, ["shoot/day/frame.jpg"])
    retained = str(result.batch_root / job_input_dir)

    assert _cleanup(tmp_path, [retained]) == []
    assert (result.input_dir / "shoot" / "day" / "frame.jpg").is_file()


def test_expired_batch_is_removed_when_jobs_only_read_other_locations(tmp_path: Path) -> None:
    in_use = _expired_batch(tmp_path, ["shoot/frame.jpg"])
    unused = _expired_batch(tmp_path, ["shoot/frame.jpg"])
    retained = [
        str(in_use.input_dir / "shoot"),
        # Neither an unrelated directory nor a sibling that merely shares the
        # batch id as a string prefix keeps it.
        str(tmp_path / "elsewhere" / "input" / "shoot"),
        str(unused.batch_root) + "_sibling",
        "",
    ]

    assert _cleanup(tmp_path, retained) == [unused.batch_id]
    assert in_use.batch_root.is_dir()
    assert not unused.batch_root.exists()


def test_expired_batch_is_retained_while_a_job_reads_an_ancestor_of_it(tmp_path: Path) -> None:
    # A job that reads the upload root (or any directory above it) reads every
    # batch inside it, so no batch below it may be removed.
    first = _expired_batch(tmp_path, ["shoot/frame.jpg"])
    second = _expired_batch(tmp_path, ["shoot/frame.jpg"])

    assert _cleanup(tmp_path, [str(tmp_path / "uploads")]) == []
    assert _cleanup(tmp_path, [str(tmp_path)]) == []
    assert first.batch_root.is_dir()
    assert second.batch_root.is_dir()


def _interrupted_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Leave a batch behind the way a process killed mid-stage does.

    ``KeyboardInterrupt`` is not an ``Exception``, so it skips the staging
    cleanup handler just as SIGKILL would skip every handler.
    """

    def _killed(_source: object, _destination: Path) -> tuple[str, int]:
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(upload_staging, "_stream_upload_to_path", _killed)
        with pytest.raises(KeyboardInterrupt):
            _stage(tmp_path, [_upload("shoot/frame.jpg")])

    (batch_root,) = (tmp_path / "uploads").iterdir()
    assert not (batch_root / "_portal" / upload_staging.UPLOAD_RECEIPT_FILENAME).exists()
    (batch_root / "input" / ".upload-0a1b2c3d.part").write_bytes(b"partial")
    os.utime(batch_root, (NOW, NOW))
    return batch_root


def test_generated_batch_ids_match_the_interrupted_batch_pattern() -> None:
    assert upload_staging._BATCH_ID_PATTERN.fullmatch(upload_staging.build_batch_id(NOW)) is not None


def test_interrupted_batch_is_reaped_once_past_the_ttl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch_root = _interrupted_batch(tmp_path, monkeypatch)

    assert _cleanup(tmp_path, [], age_seconds=TTL_SECONDS / 2) == []
    assert batch_root.is_dir()

    assert _cleanup(tmp_path, [], age_seconds=2 * TTL_SECONDS) == [batch_root.name]
    assert not batch_root.exists()


def test_interrupted_batch_outlives_a_short_ttl_until_it_cannot_be_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt-less batch may still be staging, so a short TTL must not reap it."""
    batch_root = _interrupted_batch(tmp_path, monkeypatch)
    finished = _expired_batch(tmp_path, ["shoot/frame.jpg"])
    grace = upload_staging.INTERRUPTED_BATCH_GRACE_SECONDS

    def _cleanup_with_short_ttl(age_seconds: float) -> list[str]:
        return upload_staging.cleanup_expired_batches(
            tmp_path / "uploads", now=NOW + age_seconds, ttl_seconds=60.0, retained_input_dirs=[]
        )

    # The finished batch follows the TTL alone; the receipt-less one waits.
    assert _cleanup_with_short_ttl(grace / 2) == [finished.batch_id]
    assert batch_root.is_dir()
    assert _cleanup_with_short_ttl(grace + 1) == [batch_root.name]


def test_interrupted_batch_is_retained_while_a_job_reads_from_inside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_root = _interrupted_batch(tmp_path, monkeypatch)

    assert _cleanup(tmp_path, [str(batch_root / "input" / "shoot")]) == []
    assert batch_root.is_dir()


@pytest.mark.parametrize("identity", ["request_marker", "lock_entry", "dangling_lock_link", "linked_lock_directory"])
def test_interrupted_batch_owned_by_the_upload_ledger_is_never_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity: str
) -> None:
    # The receipt ledger fences and deletes its own batches; legacy TTL cleanup
    # must not reap a receipt-less batch that carries its marker or lock identity.
    batch_root = _interrupted_batch(tmp_path, monkeypatch)
    lock_directory = batch_root.parent / ".upload-locks"
    if identity == "request_marker":
        (batch_root / ".upload-request.v1.json").write_text("{}", encoding="utf-8")
    elif identity == "lock_entry":
        lock_directory.mkdir()
        (lock_directory / (batch_root.name + ".lock")).write_bytes(b"")
    elif identity == "dangling_lock_link":
        lock_directory.mkdir()
        (lock_directory / (batch_root.name + ".lock")).symlink_to(tmp_path / "missing-lock-target")
    else:
        (tmp_path / "lock-target").mkdir()
        lock_directory.symlink_to(tmp_path / "lock-target", target_is_directory=True)
    os.utime(batch_root, (NOW, NOW))

    assert _cleanup(tmp_path, [], age_seconds=2 * TTL_SECONDS + upload_staging.INTERRUPTED_BATCH_GRACE_SECONDS) == []
    assert batch_root.is_dir()


def test_cleanup_never_touches_receiptless_directories_it_did_not_create(tmp_path: Path) -> None:
    upload_root = tmp_path / "uploads"
    kept = {
        # Not a generated batch id.
        "upload_123": ("input", "_portal"),
        "upload_1700000000_0A1B2C3D": ("input", "_portal"),
        "manual_batch": ("input", "_portal"),
        # Generated-looking id without the managed layout.
        "upload_1700000000_0a1b2c3d": ("input",),
        "upload_1700000000_1a2b3c4d": ("_portal",),
    }
    for name, children in kept.items():
        for child in children:
            (upload_root / name / child).mkdir(parents=True)
        (upload_root / name / "notes.txt").write_text("keep", encoding="utf-8")
        os.utime(upload_root / name, (NOW, NOW))
    orphan = upload_root / "upload_1700000000_9f8e7d6c"
    for child in ("input", "_portal", "tmp"):
        (orphan / child).mkdir(parents=True)
    os.utime(orphan, (NOW, NOW))

    assert _cleanup(tmp_path, []) == [orphan.name]
    assert sorted(path.name for path in upload_root.iterdir()) == sorted(kept)
