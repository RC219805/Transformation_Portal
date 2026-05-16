"""Unit coverage for depth tools source and mask discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from transformation_portal.depth import tools

pytestmark = pytest.mark.unit


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture", encoding="utf-8")


def test_find_file_for_base_prefers_priority_tags_recursively(tmp_path) -> None:
    _touch(tmp_path / "nested" / "villa_golden.png")
    _touch(tmp_path / "villa_enh.PNG")
    _touch(tmp_path / "villa_punchy.jpg")

    match = tools.find_file_for_base(str(tmp_path), "villa")

    assert match == str(tmp_path / "villa_enh.PNG")


def test_find_file_for_base_honors_custom_priority_and_extension_case(tmp_path) -> None:
    _touch(tmp_path / "villa_view.TIFF")
    _touch(tmp_path / "villa_punchy.png")

    match = tools.find_file_for_base(
        str(tmp_path),
        "villa",
        priority_tags=("_view", "_punchy"),
        extensions=(".tiff", ".png"),
    )

    assert match == str(tmp_path / "villa_view.TIFF")


def test_find_file_for_base_returns_none_when_no_candidate_matches(tmp_path) -> None:
    _touch(tmp_path / "other_enh.png")

    assert tools.find_file_for_base(str(tmp_path), "villa") is None


def test_find_file_for_base_filters_unsupported_extensions(tmp_path) -> None:
    _touch(tmp_path / "villa_enh.txt")

    assert tools.find_file_for_base(str(tmp_path), "villa") is None


def test_find_mask_for_base_finds_exact_mask_by_kind(tmp_path) -> None:
    exact = tmp_path / "villa_mask_sky.png"
    generic = tmp_path / "villa_mask_sky.webp"
    _touch(generic)
    _touch(exact)

    assert tools.find_mask_for_base(str(tmp_path), "villa", "sky") == str(exact)


def test_find_mask_for_base_falls_back_to_generic_extension(tmp_path) -> None:
    mask = tmp_path / "villa_mask_building.webp"
    _touch(mask)

    assert tools.find_mask_for_base(str(tmp_path), "villa", "building") == str(mask)


def test_find_mask_for_base_returns_none_without_mask_root() -> None:
    assert tools.find_mask_for_base(None, "villa", "sky") is None


def test_find_file_for_base_falls_through_to_no_tag_score(tmp_path) -> None:
    # Candidate matches by extension but contains none of PRIORITY_TAGS;
    # the score function must return its large fallback and still produce a
    # result rather than silently dropping the file.
    target = tmp_path / "villa_plain.png"
    target.write_text("fixture", encoding="utf-8")

    match = tools.find_file_for_base(str(tmp_path), "villa")

    assert match == str(target)


def test_find_mask_for_base_iterates_exact_extension_list(tmp_path) -> None:
    # Place a non-PNG exact-extension match alongside the generic glob so the
    # explicit extension loop wins over the fallback glob.
    exact = tmp_path / "villa_mask_sky.tif"
    distractor = tmp_path / "villa_mask_sky.xyz"
    exact.write_text("fixture", encoding="utf-8")
    distractor.write_text("fixture", encoding="utf-8")

    assert tools.find_mask_for_base(str(tmp_path), "villa", "sky") == str(exact)
