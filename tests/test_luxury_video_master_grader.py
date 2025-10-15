import argparse
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from .documentation import documents


def load_module() -> ModuleType:
    module_path = Path(__file__).resolve().parent.parent / "luxury_video_master_grader.py"
    spec = importlib.util.spec_from_file_location("luxury_video_master_grader", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader  # for mypy
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module()
assess_frame_rate = MODULE.assess_frame_rate
build_command = MODULE.build_command
build_filter_graph = MODULE.build_filter_graph
determine_color_metadata = MODULE.determine_color_metadata
plan_tone_mapping = MODULE.plan_tone_mapping
summarize_probe = MODULE.summarize_probe
parse_arguments = MODULE.parse_arguments


@pytest.fixture
def probe_with_unknown_duration() -> dict:
    return {
        "format": {"duration": "N/A"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "30000/1001",
            }
        ],
    }


@documents("Operator judgment can override sensing when continuity demands")
def test_assess_frame_rate_respects_user_override():
    probe = {"streams": [{"codec_type": "video"}]}

    plan = assess_frame_rate(probe, "25", tolerance=0.05)

    assert plan.target == "25/1"
    assert "override" in plan.note


@documents("Frame cadence analysis flags variability before delivery")
def test_assess_frame_rate_detects_vfr():
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "24000/1001",
            }
        ]
    }

    plan = assess_frame_rate(probe, None, tolerance=0.05)

    assert plan.target == "30000/1001"
    assert "variable frame-rate" in plan.note


@documents("System normalizes near-standards for interchange resilience")
def test_assess_frame_rate_conforms_off_standard():
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "avg_frame_rate": "48/1",
                "r_frame_rate": "48/1",
            }
        ]
    }

    plan = assess_frame_rate(probe, None, tolerance=0.05)

    assert plan.target == "50/1"
    assert "off-standard" in plan.note


@documents("Small deviations stay untouched to honor original motion")
def test_assess_frame_rate_preserves_within_tolerance():
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "30000/1001",
            }
        ]
    }

    plan = assess_frame_rate(probe, None, tolerance=0.05)

    assert plan.target is None
    assert "preserving timing" in plan.note


@documents("Filter graph composes cinematic treatments modularly")
def test_build_filter_graph_includes_optional_nodes(tmp_path):
    lut_path = tmp_path / "dummy.cube"
    lut_path.write_text("# dummy LUT\n")

    config = {
        "lut": lut_path,
        "contrast": 1.05,
        "saturation": 1.02,
        "gamma": 0.98,
        "brightness": 0.01,
        "warmth": 0.9,
        "cool": -0.7,
        "lut_strength": 0.6,
        "denoise": "medium",
        "sharpen": "soft",
        "grain": 3.5,
        "target_fps": "24000/1001",
        "tone_map": "hable",
        "tone_map_peak": 1200.0,
        "tone_map_desat": 0.25,
        "deband": "medium",
        "halation_intensity": 0.4,
        "halation_radius": 20.0,
    }

    graph, output_label = build_filter_graph(config)

    assert output_label == "vout"
    assert f"lut3d=file={lut_path}:interp=tetrahedral" in graph
    assert "hqdn3d=luma_spatial=2.8" in graph
    assert "eq=contrast=1.0500:saturation=1.0200:gamma=0.9800:brightness=0.0100" in graph
    assert "colorbalance=rm=0.5000:gm=0.0000:bm=-0.5000" in graph
    assert "blend=all_expr='A*(1-0.6000)+B*0.6000'" in graph
    assert "unsharp=luma_msize_x=7" in graph
    assert "noise=alls=3.50:allf=t+u" in graph
    assert "zscale=transfer=linear:npl=1200.0000" in graph
    assert "tonemap=hable:peak=1200.0000:desat=0.2500" in graph
    assert graph.count("zscale=primaries=bt709:transfer=bt709:matrix=bt709:range=tv") == 1
    assert "tonemap_desat=" not in graph
    assert "tonemap_param=" not in graph
    assert "gradfun=strength=0.70:radius=16" in graph
    assert "split=2" in graph
    assert "colorlevels=rimin=0.600:gimin=0.600:bimin=0.600" in graph
    assert "gblur=sigma=20.00:steps=2" in graph
    assert "colorbalance=rm=0.2200:gm=0.1000:bm=-0.0600" in graph
    assert "blend=all_expr='A+(0.400*B)'" in graph
    assert "fps=fps=24000/1001" in graph


@documents("Invalid recipe ingredients are rejected before encoding")
def test_build_filter_graph_rejects_unknown_denoise(tmp_path):
    lut_path = tmp_path / "dummy.cube"
    lut_path.write_text("# dummy LUT\n")

    config = {
        "lut": lut_path,
        "denoise": "extreme",
    }

    with pytest.raises(ValueError):
        build_filter_graph(config)


@documents("Tone pipeline only accepts curated debanding dialects")
def test_build_filter_graph_rejects_unknown_deband(tmp_path):
    lut_path = tmp_path / "dummy.cube"
    lut_path.write_text("# dummy LUT\n")

    config = {
        "lut": lut_path,
        "deband": "impossible",
    }

    with pytest.raises(ValueError):
        build_filter_graph(config)


@documents("Blend logic respects grading order when attenuating LUT influence")
def test_build_filter_graph_blends_post_eq_when_lut_strength_lt_one(tmp_path):
    lut_path = tmp_path / "dummy.cube"
    lut_path.write_text("# dummy LUT\n")

    config = {
        "lut": lut_path,
        "contrast": 1.1,
        "saturation": 0.9,
        "gamma": 0.95,
        "brightness": 0.02,
        "warmth": 0.1,
        "cool": -0.05,
        "lut_strength": 0.5,
    }

    graph, _ = build_filter_graph(config)
    nodes = graph.split(";")

    eq_node = next(node for node in nodes if "eq=" in node)
    eq_match = re.search(r"^\[(v\d+)\]eq=[^\[]+\[(v\d+)\]$", eq_node)
    assert eq_match, "Failed to parse EQ node"
    pre_eq_label, post_eq_label = eq_match.groups()

    color_node = next((node for node in nodes if "colorbalance=" in node), None)
    if color_node:
        color_match = re.search(r"^\[(v\d+)\]colorbalance=[^\[]+\[(v\d+)\]$", color_node)
        assert color_match, "Failed to parse color balance node"
        assert color_match.group(1) == post_eq_label
        post_grade_label = color_match.group(2)
    else:
        post_grade_label = post_eq_label

    lut_node = next(node for node in nodes if "lut3d=" in node)
    lut_input_match = re.search(r"\[(v\d+)\]lut3d", lut_node)
    assert lut_input_match, "Failed to parse LUT input label"
    pre_lut_label = lut_input_match.group(1)
    assert pre_lut_label == post_grade_label

    blend_node = next(node for node in nodes if "blend=all_expr" in node)
    blend_match = re.search(r"^\[(v\d+)\]\[(v\d+)\]blend=all_expr", blend_node)
    assert blend_match, "Failed to parse blend node"

    assert (
        blend_match.group(1) == pre_lut_label
    ), "Blend should use the post-EQ label as the base input"
    assert (
        blend_match.group(1) != pre_eq_label
    ), "Blend should not use the pre-EQ label as the base input"


def make_tone_args(**overrides):
    defaults = {
        "tone_map": "auto",
        "tone_map_peak": 1000.0,
        "tone_map_desat": 0.1,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@documents("Tone mapping activates when HDR metadata signals spectral risk")
def test_plan_tone_mapping_detects_hdr():
    args = make_tone_args()
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "color_trc": "smpte2084",
                "color_primaries": "bt2020",
                "color_space": "bt2020nc",
            }
        ]
    }

    plan = plan_tone_mapping(args, probe)

    assert plan.enabled
    assert plan.config["tone_map"] == "hable"
    assert plan.metadata == ("bt709", "bt709", "bt709")
    assert "detected" in plan.note


@documents("Legacy metadata keys remain compatible with HDR heuristics")
def test_plan_tone_mapping_honours_legacy_colorspace_key():
    args = make_tone_args()
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "color_trc": "smpte2084",
                "color_primaries": "bt2020",
                "colorspace": "bt2020nc",
            }
        ]
    }

    plan = plan_tone_mapping(args, probe)

    assert plan.enabled
    assert "detected" in plan.note


@documents("Explicit opt-out disables automated tone orchestration")
def test_plan_tone_mapping_respects_off():
    args = make_tone_args(tone_map="off")
    probe = {"streams": [{"codec_type": "video"}]}

    plan = plan_tone_mapping(args, probe)

    assert not plan.enabled
    assert "disabled" in plan.note.lower()


@documents("Creative overrides can force tone strategy even on SDR footage")
def test_plan_tone_mapping_forced_override_on_sdr():
    args = make_tone_args(tone_map="mobius", tone_map_peak=1500.0, tone_map_desat=0.2)
    probe = {"streams": [{"codec_type": "video", "color_trc": "bt709"}]}

    plan = plan_tone_mapping(args, probe)

    assert plan.enabled
    assert plan.config["tone_map"] == "mobius"
    assert "forced" in plan.note.lower()


@documents("Explicit color metadata takes precedence over detection")
def test_determine_color_metadata_prioritises_explicit():
    args = argparse.Namespace(
        color_primaries="bt709",
        color_transfer="smpte2084",
        color_space="bt2020nc",
        color_from_source=False,
    )
    probe = {"streams": []}

    assert determine_color_metadata(args, probe) == ("bt709", "smpte2084", "bt2020nc")


@documents("Source metadata is trusted selectively, filtering unknown entries")
def test_determine_color_metadata_from_source_filters_unknown():
    args = argparse.Namespace(
        color_primaries=None,
        color_transfer=None,
        color_space=None,
        color_from_source=True,
    )
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "color_primaries": "bt2020",
                "color_trc": "unknown",
                "color_space": "bt2020nc",
            }
        ]
    }

    assert determine_color_metadata(args, probe) == ("bt2020", None, "bt2020nc")


@documents("Historical ffprobe keypath remains supported for metadata copy")
def test_determine_color_metadata_legacy_colorspace_key():
    args = argparse.Namespace(
        color_primaries=None,
        color_transfer=None,
        color_space=None,
        color_from_source=True,
    )
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "color_primaries": "bt2020",
                "color_trc": "smpte2084",
                "colorspace": "bt2020nc",
            }
        ]
    }

    assert determine_color_metadata(args, probe) == ("bt2020", "smpte2084", "bt2020nc")


@documents("Absent metadata yields neutral defaults for pipeline safety")
def test_determine_color_metadata_defaults_to_none_when_missing():
    args = argparse.Namespace(
        color_primaries=None,
        color_transfer=None,
        color_space=None,
        color_from_source=True,
    )
    probe = {"streams": []}

    assert determine_color_metadata(args, probe) == (None, None, None)


@documents("Command builder emits ffmpeg contract with color guarantees")
def test_build_command_includes_expected_arguments(tmp_path):
    input_path = tmp_path / "input.mov"
    output_path = tmp_path / "output.mov"

    cmd = build_command(
        input_path,
        output_path,
        "graph",
        "vout",
        overwrite=True,
        video_codec="prores_ks",
        prores_profile=3,
        bitrate="500M",
        audio_codec="pcm_s24le",
        audio_bitrate="320k",
        threads=8,
        log_level="warning",
        preview_frames=10,
        vsync="cfr",
        color_primaries="bt2020",
        color_transfer="smpte2084",
        color_space="bt2020nc",
    )

    assert cmd[0] == "ffmpeg"
    assert "-y" in cmd
    assert cmd.count("-color_primaries") == 1
    assert cmd[-1] == str(output_path)
    assert "-filter_complex" in cmd and "graph" in cmd
    assert "-profile:v" in cmd
    assert "-pix_fmt" in cmd
    assert "-b:v" in cmd
    assert "-b:a" in cmd
    assert "-threads" in cmd
    assert "-frames:v" in cmd
    assert "-vsync" in cmd and "cfr" in cmd
    assert "-color_trc" in cmd and "smpte2084" in cmd
    assert "-colorspace" in cmd and "bt2020nc" in cmd


@documents("Summaries omit meaningless tags to highlight actionable data")
def test_summarize_probe_ignores_non_descriptive_color_tags():
    probe = {
        "format": {"duration": "10.0"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "prores",
                "width": 3840,
                "height": 2160,
                "avg_frame_rate": "24/1",
                "color_primaries": "unknown",
                "color_trc": "unspecified",
                "color_space": "BT709",
            }
        ],
    }

    summary = summarize_probe(probe)

    assert "space=bt709" in summary
    assert "primaries" not in summary
    assert "trc=" not in summary


@documents("Legacy color_space keypath is summarised for operator visibility")
def test_summarize_probe_supports_legacy_colorspace_key():
    probe = {
        "format": {"duration": "5.0"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "prores",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "colorspace": "BT2020NC",
            }
        ],
    }

    summary = summarize_probe(probe)

    assert "space=bt2020nc" in summary


@documents("Metadata summaries gracefully skip unusable duration fields")
def test_summarize_probe_skips_non_numeric_duration(probe_with_unknown_duration):
    summary = summarize_probe(probe_with_unknown_duration)

    assert "duration" not in summary
    assert "video h264 1920x1080" in summary


@documents("CLI enforces required IO to prevent silent misfires")
def test_parse_arguments_requires_input_and_output(capsys):
    with pytest.raises(SystemExit) as exc:
        parse_arguments([])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "the following arguments are required: input_video, output_video" in captured.err


@documents("Preset catalog can be listed without invoking processing")
def test_parse_arguments_list_presets_exits_early(capsys):
    with pytest.raises(SystemExit) as exc:
        parse_arguments(["--list-presets"])

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "Available presets:" in captured.out
    for preset_key in MODULE.PRESETS:
        assert f"- {preset_key}:" in captured.out
