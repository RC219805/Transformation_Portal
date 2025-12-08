from pathlib import Path

from lux_depth_v2.hardening.stamping import config_hash, stamp_report


def test_config_hash_is_stable():
    cfg = {"a": 1, "b": [2, 3]}
    h1 = config_hash(cfg)
    h2 = config_hash(cfg)
    assert h1 == h2


def test_stamp_report_adds_meta(tmp_path: Path):
    report = {"ai_color_diff": 0.1, "ai_luma_diff": 0.2}
    stamped = stamp_report(report, config={"preset": "x"}, input_path=tmp_path / "in.tif", output_dir=tmp_path)
    assert "meta" in stamped
    assert "config_hash" in stamped["meta"]
    assert stamped["ai_color_diff"] == 0.1
