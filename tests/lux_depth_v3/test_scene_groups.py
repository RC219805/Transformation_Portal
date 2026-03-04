from pathlib import Path

from transformation_portal.lux_depth_v3.scene_groups import build_scene_groups


def test_scene_groups_default_behavior():
    images = [Path("foo/a.jpg"), Path("bar/a.png")]
    groups = build_scene_groups(images)

    assert len(groups) == 2

    assert groups[0].images == (Path("foo/a.jpg"),)
    assert groups[1].images == (Path("bar/a.png"),)

    assert groups[0].scene_id == "foo/a"
    assert groups[1].scene_id == "bar/a"
