from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from transformation_portal.core.security.path import PathValidator, is_safe_path, safe_resolve_path
from transformation_portal.core.security.sanitization import SanitizationPolicy, sanitize_filename, validate_input_file
from transformation_portal.core.security.validation import ValidationError

pytestmark = [pytest.mark.unit, pytest.mark.security]


def test_path_validator_accepts_paths_within_any_allowed_root(tmp_path: Path) -> None:
    root_a = tmp_path / "allowed-a"
    root_b = tmp_path / "allowed-b"
    nested = root_b / "nested" / "file.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("ok", encoding="utf-8")

    validator = PathValidator([root_a, root_b])

    assert validator.is_safe(nested)
    assert not validator.is_safe(tmp_path / "outside.txt")


def test_path_validator_handles_resolution_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    validator = PathValidator([tmp_path])
    original_resolve = Path.resolve

    def _raise_resolve(self: Path, *args, **kwargs):  # noqa: ANN001, ANN003
        if self == Path("bad-path"):
            raise OSError("boom")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", _raise_resolve)

    assert validator.is_safe("bad-path") is False


def test_safe_resolve_path_accepts_path_inside_allowed_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "root"
    allowed_root.mkdir()
    target = allowed_root / "subdir" / "asset.txt"
    target.parent.mkdir(parents=True)
    target.write_text("ok", encoding="utf-8")

    resolved = safe_resolve_path(target, allowed_root=allowed_root)

    assert resolved == target.resolve()


def test_safe_resolve_path_blocks_traversal_outside_allowed_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "root"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()
    outside_file = outside_root / "escape.txt"
    outside_file.write_text("blocked", encoding="utf-8")

    with pytest.raises(ValidationError, match="Path traversal detected"):
        safe_resolve_path(outside_file, allowed_root=allowed_root)


def test_is_safe_path_uses_import_time_default_root(tmp_path: Path) -> None:
    repo_local_file = Path(__file__).resolve()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside", encoding="utf-8")

    assert is_safe_path(repo_local_file)
    assert not is_safe_path(outside_file)


def test_sanitization_policy_sets_default_extensions() -> None:
    policy = SanitizationPolicy()
    assert policy.allowed_extensions == [".jpg", ".jpeg", ".png", ".tiff", ".exr"]


def test_sanitize_filename_strips_path_dangerous_chars_hidden_prefix_and_length() -> None:
    unsafe = "../.bad<name>|with spaces?.png"
    result = sanitize_filename(unsafe)

    assert "/" not in result
    assert "<" not in result and ">" not in result and "|" not in result and "?" not in result
    assert not result.startswith(".")
    assert len(sanitize_filename("a" * 300 + ".png")) == 255


def test_validate_input_file_raises_for_missing_path(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="File not found"):
        validate_input_file(tmp_path / "missing.jpg")


def test_validate_input_file_raises_for_non_file_path(tmp_path: Path) -> None:
    directory = tmp_path / "as-directory.jpg"
    directory.mkdir()

    with pytest.raises(ValidationError, match="Not a file"):
        validate_input_file(directory)


def test_validate_input_file_rejects_disallowed_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("text", encoding="utf-8")

    with pytest.raises(ValidationError, match="File type '\\.txt' not allowed"):
        validate_input_file(file_path, policy=SanitizationPolicy(allowed_extensions=[".jpg"]))


def test_validate_input_file_accepts_supported_signatures(tmp_path: Path) -> None:
    jpeg = tmp_path / "image.jpg"
    jpeg.write_bytes(b"\xff\xd8\xff\x00extra")

    png = tmp_path / "image.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nextra")

    validate_input_file(jpeg)
    validate_input_file(png)


def test_validate_input_file_rejects_signature_mismatch(tmp_path: Path) -> None:
    fake_jpeg = tmp_path / "fake.jpg"
    fake_jpeg.write_bytes(b"not-a-jpeg")

    with pytest.raises(ValidationError, match="File signature mismatch"):
        validate_input_file(fake_jpeg)


def test_validate_input_file_wraps_io_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"\xff\xd8\xff")
    original_open = builtins.open

    def _broken_open(*args, **kwargs):  # noqa: ANN002, ANN003
        target = args[0] if args else kwargs.get("file")
        if target == image:
            raise OSError("read denied")
        return original_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", _broken_open)

    with pytest.raises(ValidationError, match="Could not read file header"):
        validate_input_file(image)
