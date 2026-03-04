from pathlib import Path

from transformation_portal.lux_depth_v3.scene_groups import build_scene_groups


def test_scene_groups_default_behavior():
    images = [Path("foo/a.jpg"), Path("bar/a.png")]
    groups = build_scene_groups(images, dataset_root=Path("."), grouping_mode="single")

    assert len(groups) == 2

    assert groups[0].images == (Path("foo/a.jpg"),)
    assert groups[1].images == (Path("bar/a.png"),)

    assert len(groups[0].scene_id) == 12
    assert len(groups[1].scene_id) == 12
    assert groups[0].scene_id != groups[1].scene_id


def test_scene_groups_parent_dir_mode_groups_deterministically():
    images = [
        Path("scene_b/img2.png"),
        Path("scene_a/img2.png"),
        Path("scene_a/img1.png"),
        Path("scene_b/img1.png"),
    ]

    groups = build_scene_groups(images, dataset_root=Path("."), grouping_mode="parent_dir")

    assert len(groups) == 2
    assert groups[0].images == (Path("scene_a/img1.png"), Path("scene_a/img2.png"))
    assert groups[1].images == (Path("scene_b/img1.png"), Path("scene_b/img2.png"))
    assert len(groups[0].scene_id) == 12
    assert len(groups[1].scene_id) == 12
