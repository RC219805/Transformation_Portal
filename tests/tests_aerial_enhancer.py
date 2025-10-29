import numpy as np
from PIL import Image
from your_module import (
    _kmeans,
    _cluster_stats,
    assign_materials,
    _soft_mask,
    enhance_aerial,
    ClusterStats,
    MaterialRule,
)

# --------------------------
# Helpers for tests
# --------------------------


def create_test_image(width=32, height=32, colors=((1, 0, 0), (0, 1, 0))):
    """Generate small RGB test image array with two colors in stripes."""
    arr = np.zeros((height, width, 3), dtype=np.float32)
    half = height // 2
    arr[:half, :, :] = colors[0]
    arr[half:, :, :] = colors[1]
    return arr


def dummy_rule(name: str):
    """Create dummy material rule scoring red/green."""
    def score_fn(stats: ClusterStats):
        return float(stats.mean_rgb[0] > 0.5) if name == "red" else float(stats.mean_rgb[1] > 0.5)
    return MaterialRule(name=name, texture=None, blend=0.5, score_fn=score_fn, min_score=0.0)

# --------------------------
# 1. _kmeans reproducibility
# --------------------------


def test_kmeans_reproducible():
    arr = np.random.rand(100, 3)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    centroids1 = _kmeans(arr, k=3, rng=rng1, iterations=10)
    centroids2 = _kmeans(arr, k=3, rng=rng2, iterations=10)
    assert np.allclose(centroids1, centroids2)

# --------------------------
# 2. _cluster_stats correctness
# --------------------------


def test_cluster_stats():
    img = create_test_image()
    labels = np.zeros((32, 32), dtype=int)
    labels[16:, :] = 1
    stats = _cluster_stats(img, labels)
    assert len(stats) == 2
    assert stats[0].count == 16 * 32
    assert stats[1].count == 16 * 32
    np.testing.assert_array_almost_equal(stats[0].mean_rgb, [1, 0, 0])
    np.testing.assert_array_almost_equal(stats[1].mean_rgb, [0, 1, 0])

# --------------------------
# 3. assign_materials logic
# --------------------------


def test_assign_materials():
    img = create_test_image()
    labels = np.zeros((32, 32), dtype=int)
    labels[16:, :] = 1
    stats = _cluster_stats(img, labels)
    rules = [dummy_rule("red"), dummy_rule("green")]
    assignments = assign_materials(stats, rules)
    assert 0 in assignments and assignments[0].name == "red"
    assert 1 in assignments and assignments[1].name == "green"

# --------------------------
# 4. _soft_mask Gaussian blending
# --------------------------


def test_soft_mask_blur():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1
    soft = _soft_mask(mask, radius=1.0)
    assert soft.min() >= 0.0 and soft.max() <= 1.0
    assert soft[4, 4] < 1.0  # edges blurred
    assert soft[7, 7] > 0.9  # center remains high

# --------------------------
# 5. end-to-end enhance_aerial
# --------------------------


def test_enhance_aerial(tmp_path):
    # Create tiny test input
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    arr = create_test_image()
    img = Image.fromarray((arr * 255).astype("uint8"))
    img.save(input_path)

    # Use default textures (None) to avoid file dependencies
    textures = {name: None for name in ["plaster", "stone", "cladding", "screens", "equitone", "roof", "bronze", "shade"]}

    out = enhance_aerial(
        input_path=input_path,
        output_path=output_path,
        analysis_max_dim=16,
        k=2,
        seed=1,
        target_width=16,
        textures=textures,
        palette_path=None,
        save_palette=None
    )
    assert out.exists()
    out_img = Image.open(out)
    assert out_img.size[0] == 16
