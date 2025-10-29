# tests/test_board_material_aerial_enhancer.py
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

import pytest

# Import the module under test
import board_material_aerial_enhancer as bma

# Utilities used across tests
def _rgb_array_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert float [0..1] HxWx3 to PIL RGB."""
    arr_u8 = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype("uint8")
    return Image.fromarray(arr_u8, mode="RGB")


def test_kmeans_simple_deterministic():
    # three clearly separated 2D points -> 2 clusters
    x = np.array([[0.0, 0.0], [0.01, 0.0], [0.99, 0.99], [1.0, 1.0]], dtype=np.float32)
    labels_a = bma._kmeans(x, k=2, max_iter=100, seed=42)
    labels_b = bma._kmeans(x, k=2, max_iter=100, seed=42)
    # deterministic: same labels for same seed
    assert np.array_equal(labels_a, labels_b)
    # labels should include exactly two values
    assert set(int(x) for x in np.unique(labels_a)) == {0, 1}


def test_compute_cluster_stats_basic():
    # Build 2x2 image with two colors
    rgb = np.array([
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    ], dtype=np.float32)
    labels = np.array([[0, 0], [1, 1]], dtype=np.int32)
    stats = bma.compute_cluster_stats(labels, rgb)
    # Two clusters returned
    assert len(stats) == 2
    # Check centroids roughly match colors
    cent0 = next(s for s in stats if s.label == 0)
    cent1 = next(s for s in stats if s.label == 1)
    # centroid values are floats in [0,1]
    assert pytest.approx(cent0.centroid[0], rel=1e-3) == 1.0
    assert pytest.approx(cent1.centroid[1], rel=1e-3) == 1.0


def test_relabel_and_relabel_safe_behavior():
    labels = np.array([[2, 2], [7, 7]], dtype=np.int32)
    assignments = {2: bma.MaterialRule(name="plaster")}
    # relabel (legacy compress) should map existing key 2 -> index 0
    relabeled = bma.relabel(assignments, labels)
    assert 0 in set(np.unique(relabeled))
    # relabel_safe with 'none' returns unchanged
    same = bma.relabel_safe(assignments, labels, mode="none", strict=False)
    assert np.array_equal(same, labels)
    # strict mode should raise due to missing palette key for label 7
    with pytest.raises(ValueError):
        bma.relabel_safe(assignments, labels, mode="none", strict=True)


def test_build_material_rules_and_assign_alias():
    rules = bma.build_material_rules({"plaster": Path("textures/plaster.png")})
    assert isinstance(rules, list)
    assert any(r.name == "plaster" for r in rules)
    # assign_materials alias should be available and callable
    alias = bma.assign_materials
    alt = alias({"plaster": Path("textures/plaster.png")})
    assert isinstance(alt, list)


def test_save_and_load_palette_assignments(tmp_path):
    # Create sample assignments mapping and save
    rule_a = bma.MaterialRule(name="plaster")
    assignments = {0: rule_a, 2: bma.MaterialRule(name="stone")}
    json_path = tmp_path / "palette.json"
    bma.save_palette_assignments(assignments, json_path)
    assert json_path.exists()
    # load back with candidate materials provided
    candidates = bma.build_material_rules(bma.DEFAULT_TEXTURES)
    loaded = bma.load_palette_assignments(json_path, candidates)
    assert isinstance(loaded, dict)
    # Keys should be ints and map to MaterialRule-like objects
    assert 0 in loaded and 2 in loaded
    assert loaded[0].name == "plaster"


def test_relabel_safe_verbose_warning(capfd):
    # Ensure verbose prints warnings for missing labels but does not raise when strict=False
    labels = np.array([[0, 1], [2, 2]], dtype=np.int32)
    assignments = {0: bma.MaterialRule(name="plaster")}  # missing keys for 1,2
    out = bma.relabel_safe(assignments, labels, mode="none", strict=False, verbose=True)
    # should be returned unchanged
    assert np.array_equal(out, labels)
    captured = capfd.readouterr()
    assert "Warning" in captured.err or "Note" in captured.err


def test_apply_materials_tint_and_missing_texture(tmp_path):
    # Small base image: 2x2, two labels
    base = np.array([
        [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
        [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8]],
    ], dtype=np.float32)
    labels = np.array([[0, 0], [1, 1]], dtype=np.int32)
    # Setup materials: label 0 tinted red, label 1 has missing texture
    mat0 = bma.MaterialRule(name="red", tint=(255, 0, 0), tint_strength=0.5, blend=0.0)
    mat1 = bma.MaterialRule(name="no_tex", tint=None, tint_strength=0.0, blend=0.0, texture="nonexistent.png")
    materials = {0: mat0, 1: mat1}
    out = bma.apply_materials(base, labels, materials)
    # label 0 pixels moved toward red
    assert out[0, 0, 0] > base[0, 0, 0]  # increased red channel
    # label 1 unchanged (no tint, missing texture)
    assert np.allclose(out[1, 0], base[1, 0])


def test_auto_assign_materials_by_stats_basic(tmp_path):
    # Create a simple 2-color image and labels
    rgb = np.zeros((4, 4, 3), dtype=np.float32)
    rgb[:2, :] = [1.0, 1.0, 1.0]  # bright region
    rgb[2:, :] = [0.1, 0.1, 0.1]  # dark region
    img = _rgb_array_to_pil(rgb)
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[2:, :] = 1
    # textures_map contains neutral names so auto_assign picks best matches deterministically
    tex_map = {"plaster": Path("textures/plaster.png"), "roof": Path("textures/roof.png")}
    assigned = bma.auto_assign_materials_by_stats(labels, img, tex_map)
    assert isinstance(assigned, dict)
    # must have entries for both labels 0 and 1
    assert 0 in assigned and 1 in assigned


def test_cluster_full_labels_and_cli_dry_run(tmp_path, capsys):
    # Build a tiny test image and write to disk
    img_path = tmp_path / "in_test.jpg"
    arr = np.zeros((20, 30, 3), dtype=np.uint8)
    arr[:10, :] = [255, 0, 0]
    arr[10:, :] = [0, 255, 0]
    Image.fromarray(arr, "RGB").save(img_path)
    out_path = tmp_path / "out.jpg"
    # Call CLI main in dry-run mode
    rc = bma.main([str(img_path), str(out_path), "--dry-run", "--clusters", "2"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Dry run" in captured.out or "Dry run — cluster counts" in captured.out


def test_enhance_aerial_simple_save_palette(tmp_path):
    # Create a tiny input image
    in_path = tmp_path / "in.png"
    img = Image.new("RGB", (32, 32), color=(120, 100, 80))
    img.save(in_path)
    out_path = tmp_path / "out.png"
    pal_path = tmp_path / "palette_out.json"
    # Run enhance_aerial with minimal settings; k=1 to be fast
    res = bma.enhance_aerial(
        input_path=in_path,
        output_path=out_path,
        k=1,
        analysis_max_dim=64,
        max_analysis_pixels=4096,
        enable_materials=False,
        save_palette=pal_path,
        seed=42,
        max_iter=10,
    )
    assert Path(res).exists()
    # palette file must exist and be valid JSON
    assert pal_path.exists()
    data = json.loads(pal_path.read_text(encoding="utf-8"))
    # keys are stringified ints
    assert all(k.isdigit() for k in data.keys())


def test__load_textures_map_defaults(tmp_path):
    # When no inputs provided, defaults must be present as keys
    mapping = bma._load_textures_map(None, None, bma.DEFAULT_TEXTURES, verbose=False)
    for k in bma.DEFAULT_TEXTURES:
        assert k in mapping


# End of file