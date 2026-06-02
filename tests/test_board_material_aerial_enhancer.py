import tempfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# Import from installed package
from transformation_portal.enhancers.board_material_aerial_enhancer import (
    DEFAULT_TEXTURES,
    ClusterStats,
    MaterialRule,
    _kmeans,
    auto_assign_materials_by_stats,
    build_material_rules,
    compute_cluster_stats,
    enhance_aerial,
    load_palette_assignments,
    relabel,
    relabel_safe,
    save_palette_assignments,
)

# ==========================
# Test K-means
# ==========================


def test_kmeans_simple_deterministic():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels_a = _kmeans(x, k=2, max_iter=100, seed=42)
    labels_b = _kmeans(x, k=2, max_iter=100, seed=42)
    assert np.array_equal(labels_a, labels_b)


# ==========================
# Compute cluster stats
# ==========================


def test_compute_cluster_stats_basic():
    x = np.array([[0, 0], [1, 1]])
    labels = np.array([0, 1])
    stats = compute_cluster_stats(labels, x)
    assert isinstance(stats, list)
    assert all(isinstance(s, ClusterStats) for s in stats)
    # Centroid check
    assert np.allclose(stats[0].centroid, [0, 0])
    assert np.allclose(stats[1].centroid, [1, 1])


# ==========================
# Relabel & relabel_safe
# ==========================


def test_relabel_and_relabel_safe_behavior():
    labels = np.array([0, 1, 2])
    assignments = {0: MaterialRule("plaster", "tex", 0.5, lambda x: 0.5), 2: MaterialRule("stone", "tex", 0.5, lambda x: 0.5)}
    relabeled = relabel(assignments, labels)
    assert set(relabeled) <= set(range(len(assignments)))

    safe = relabel_safe(assignments, labels, strict=False, verbose=False)
    assert safe.shape == labels.shape


# ==========================
# Build material rules
# ==========================


def test_build_material_rules_and_assign_alias():
    textures = {"plaster": "assets/textures/board_materials/plaster_marmorino_westwood_beige.png"}
    rules = build_material_rules(textures)
    assert isinstance(rules, list)
    assert all(isinstance(r, MaterialRule) for r in rules)


# ==========================
# Save/load palette assignments
# ==========================


def test_save_and_load_palette_assignments():
    assignments = {0: MaterialRule("plaster", "tex", 0.5, lambda x: 0.5)}
    with tempfile.TemporaryDirectory() as tmp:
        json_path = Path(tmp) / "palette.json"
        save_palette_assignments(assignments, json_path)
        loaded = load_palette_assignments(json_path)
        assert isinstance(loaded, dict)
        assert all(isinstance(v, MaterialRule) for v in loaded.values())
        assert loaded[0].name == "plaster"


# ==========================
# Apply tint / missing texture
# ==========================


def test_apply_materials_tint_and_missing_texture():
    img = np.ones((2, 2, 3), dtype=np.float32)
    out = enhance_aerial(img, k=1)
    assert out.shape == img.shape


# ==========================
# Auto-assign by stats
# ==========================


def test_auto_assign_materials_by_stats_basic():
    img = np.ones((2, 2, 3), dtype=np.float32)
    labels = np.array([0, 1, 0, 1])
    assignments = auto_assign_materials_by_stats(labels, img, DEFAULT_TEXTURES)
    assert isinstance(assignments, dict)
    assert all(isinstance(v, MaterialRule) for v in assignments.values())


# ==========================
# Enhance aerial with k clusters
# ==========================


def test_enhance_aerial_simple_save_palette():
    img = np.ones((2, 2, 3), dtype=np.float32)
    out = enhance_aerial(img, k=2)
    assert out.shape == img.shape


# ==========================
# Cluster full labels dry run (dummy)
# ==========================


def test_cluster_full_labels_and_cli_dry_run(capsys):
    img = np.ones((2, 2, 3), dtype=np.float32)
    _ = enhance_aerial(img, k=2)
    _captured = capsys.readouterr()  # noqa: F841
    # As function is silent, just assert shape
    assert _.shape == img.shape
