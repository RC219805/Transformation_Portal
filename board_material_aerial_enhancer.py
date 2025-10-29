# Palette helpers: use real ones if available; otherwise provide simple JSON fallbacks.
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Mapping, Callable, Dict, MutableMapping
import numpy as np
import argparse
import math
from PIL import Image, ImageFilter
try:
    from .palette_assignments import (  # type: ignore
        load_palette_assignments,
        save_palette_assignments,
    )
except Exception:
    def load_palette_assignments(
        path: str | Path,
        rules: Sequence[MaterialRule] | Mapping[str, MaterialRule] | None = None,
    ) -> dict[int, MaterialRule]:
        """Load a palette JSON of { "<label>": "<rule_name>" } and map back to MaterialRule."""
        p = Path(path)
        if not p.exists():
            return {}
        data = json.loads(p.read_text())

        # Build lookup: rule name -> MaterialRule
        lookup: dict[str, MaterialRule] = {}
        if isinstance(rules, Mapping):
            lookup.update(rules)  # assume already name->rule
        elif isinstance(rules, Sequence):
            lookup.update({r.name: r for r in rules})
        # else: rules None => leave lookup empty

        assignments: dict[int, MaterialRule] = {}
        for k, v in data.items():
            try:
                label = int(k)
            except Exception:
                # allow ints or numeric strings as keys
                label = int(k)  # will raise if truly invalid
            rule = lookup.get(v)
            if rule is not None:
                assignments[label] = rule
        return assignments

    def save_palette_assignments(assignments: Mapping[int, MaterialRule], path: str | Path) -> None:
        """Save palette as { "<label>": "<rule_name>" } for portability."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        serializable = {str(label): rule.name for label, rule in assignments.items()}
        p.write_text(json.dumps(serializable, indent=2, sort_keys=True))

@dataclass(frozen=True)
class ClusterStats:
    """Statistics for a single color cluster in the aerial image.

    Attributes:
        label: Cluster identifier (0-indexed).
        count: Number of pixels assigned to this cluster.
        mean_rgb: Mean RGB color (normalized 0-1).
        mean_hsv: Mean HSV color (hue 0-1, saturation 0-1, value 0-1).
        std_rgb: Standard deviation of RGB values.
    """

    label: int
    count: int
    mean_rgb: np.ndarray
    mean_hsv: np.ndarray
    std_rgb: np.ndarray


@dataclass(frozen=True)
class MaterialRule:
    """Defines how a board material is identified and applied.

    Attributes:
        name: Material identifier (e.g., "plaster", "stone").
        texture: Path to texture image file, or None for no texture.
        blend: Texture blend strength (0.0 = original, 1.0 = full texture).
        score_fn: Function mapping ClusterStats to material affinity score.
        min_score: Minimum score threshold for assignment.
        tint: Optional RGB tint color (0-1 range).
        tint_strength: Tint application strength (0.0 = none, 1.0 = full).
    """

    name: str
    texture: str | None
    blend: float
    score_fn: Callable[[ClusterStats], float]
    min_score: float = 0.0
    tint: tuple[float, float, float] | None = None
    tint_strength: float = 0.0


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TEXTURES: Mapping[str, Path] = {
    "plaster": BASE_DIR / "textures" / "board_materials" / "plaster_marmorino_westwood_beige.png",
    "stone": BASE_DIR / "textures" / "board_materials" / "stone_bokara_coastal.png",
    "cladding": BASE_DIR / "textures" / "board_materials" / "cladding_sculptform_warm.png",
    "screens": BASE_DIR / "textures" / "board_materials" / "screens_grey_gum.png",
    "equitone": BASE_DIR / "textures" / "board_materials" / "equitone_lt85.png",
    "roof": BASE_DIR / "textures" / "board_materials" / "bison_weathered_ipe.png",
    "bronze": BASE_DIR / "textures" / "board_materials" / "dark_bronze_anodized.png",
    "shade": BASE_DIR / "textures" / "board_materials" / "louvretec_powder_white.png",
}


def _validate_texture(path: Path | None) -> Path | None:
    """Validate that a texture file exists.

    Args:
        path: Path to texture file or None.

    Returns:
        Validated path or None.

    Raises:
        FileNotFoundError: If path is not None and file does not exist.
    """
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB color array to HSV color space.

    Args:
        rgb: Array of RGB values (0-1 range), shape (..., 3).

    Returns:
        Array of HSV values (all 0-1 range), shape (..., 3).
    """
    rgb = np.clip(rgb, 0.0, 1.0)
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    mask = delta > 1e-5

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    idx = (maxc == r) & mask
    hue[idx] = (g[idx] - b[idx]) / delta[idx]
    idx = (maxc == g) & mask
    hue[idx] = 2.0 + (b[idx] - r[idx]) / delta[idx]
    idx = (maxc == b) & mask
    hue[idx] = 4.0 + (r[idx] - g[idx]) / delta[idx]
    hue = (hue / 6.0) % 1.0

    saturation = np.zeros_like(maxc)
    nonzero = maxc > 1e-5
    saturation[nonzero] = delta[nonzero] / maxc[nonzero]

    return np.stack([hue, saturation, maxc], axis=-1)


def _downsample_image(image: Image.Image, max_dim: int) -> Image.Image:
    """Downsample image for faster clustering analysis.

    Args:
        image: Source PIL Image.
        max_dim: Maximum dimension (width or height) for output.

    Returns:
        Downsampled PIL Image.
    """
    w, h = image.size
    scale = max(1, max(w, h) // max_dim)
    if scale <= 1:
        return image.copy()
    size = (max(1, w // scale), max(1, h // scale))
    return image.resize(size, Image.Resampling.BILINEAR)


def _initial_centroids(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Select initial k-means centroids via random sampling.

    Args:
        data: Data points, shape (n_samples, n_features).
        k: Number of clusters.
        rng: NumPy random generator.

    Returns:
        Initial centroids, shape (k, n_features).

    Raises:
        ValueError: If k exceeds number of data points.
    """
    if k > len(data):  # pragma: no cover - defensive
        raise ValueError("k cannot exceed number of data points")
    indices = rng.choice(len(data), size=k, replace=False)
    return data[indices]


def _kmeans(data: np.ndarray, k: int, rng: np.random.Generator, iterations: int = 20) -> np.ndarray:
    """Perform k-means clustering on RGB pixel data.

    Uses Lloyd's algorithm with Euclidean distance for cluster assignment.
    Iterates until convergence or maximum iterations reached.

    Args:
        data: Pixel data, shape (n_pixels, 3) with RGB values 0-1.
        k: Number of clusters.
        rng: NumPy random generator for reproducible initialization.
        iterations: Maximum number of iterations.

    Returns:
        Final cluster centroids, shape (k, 3).
    """
    centroids = _initial_centroids(data, k, rng)
    for _ in range(iterations):
        distances = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            data[labels == idx].mean(axis=0) if np.any(labels == idx) else centroids[idx]
            for idx in range(k)
        ])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids


def _assign_full_image(image: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each pixel to nearest cluster centroid.

    Args:
        image: RGB image array, shape (h, w, 3).
        centroids: Cluster centroids, shape (k, 3).

    Returns:
        Label array, shape (h, w) with values 0 to k-1.
    """
    pixels = image.reshape(-1, 3)
    distances = np.sum((pixels[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels.reshape(image.shape[:2])


def _cluster_stats(image: np.ndarray, labels: np.ndarray) -> Sequence[ClusterStats]:
    """Compute statistics for each cluster.

    Args:
        image: RGB image array, shape (h, w, 3).
        labels: Label array, shape (h, w).

    Returns:
        List of ClusterStats for each cluster with pixels.
    """
    stats: list[ClusterStats] = []
    hsv = _rgb_to_hsv(image)
    for label in range(labels.max() + 1):
        mask = labels == label
        count = int(mask.sum())
        if count == 0:
            continue
        mean_rgb = image[mask].mean(axis=0)
        std_rgb = image[mask].std(axis=0)
        mean_hsv = hsv[mask].mean(axis=0)
        stats.append(ClusterStats(label=label, count=count, mean_rgb=mean_rgb, mean_hsv=mean_hsv, std_rgb=std_rgb))
    return stats


def _gaussian(x: float, mu: float, sigma: float) -> float:
    """Compute Gaussian probability density (unnormalized).

    Args:
        x: Input value.
        mu: Mean.
        sigma: Standard deviation.

    Returns:
        Gaussian value at x.
    """
    return math.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


def build_material_rules(textures: Mapping[str, Path]) -> Sequence[MaterialRule]:
    """Build MBAR material identification rules with scoring functions.

    Each rule uses HSV-based heuristics to identify clusters that match
    the expected color characteristics of board-approved materials.

    Args:
        textures: Mapping from material name to texture file path.

    Returns:
        Tuple of MaterialRule objects in priority order.
    """
    def plaster_score(stats: ClusterStats) -> float:
        _, s, v = stats.mean_hsv
        return max(0.0, (1.0 - s) * v)

    def stone_score(stats: ClusterStats) -> float:
        h, s, v = stats.mean_hsv
        warmth = _gaussian(h, 0.09, 0.05)
        return warmth * max(0.0, 1.0 - abs(v - 0.62) / 0.4) * max(0.0, 1.0 - abs(s - 0.22) / 0.4)

    def cladding_score(stats: ClusterStats) -> float:
        h, s, v = stats.mean_hsv
        warmth = _gaussian(h, 0.08, 0.04)
        return warmth * max(0.0, 1.0 - abs(v - 0.5) / 0.4) * max(0.0, 1.0 - abs(s - 0.28) / 0.4)

    def screen_score(stats: ClusterStats) -> float:
        _, s, v = stats.mean_hsv
        neutral = 1.0 - s
        return neutral * max(0.0, 1.0 - abs(v - 0.45) / 0.3)

    def equitone_score(stats: ClusterStats) -> float:
        _, s, v = stats.mean_hsv
        neutral = 1.0 - s
        return neutral * max(0.0, 1.0 - abs(v - 0.35) / 0.25)

    def roof_score(stats: ClusterStats) -> float:
        _, s, v = stats.mean_hsv
        neutral = 1.0 - s
        return neutral * max(0.0, 1.0 - abs(v - 0.65) / 0.2)

    def bronze_score(stats: ClusterStats) -> float:
        h, s, v = stats.mean_hsv
        warmth = _gaussian(h, 0.06, 0.06)
        return warmth * max(0.0, 1.0 - abs(v - 0.28) / 0.25) * max(0.0, 1.0 - abs(s - 0.24) / 0.3)

    def shade_score(stats: ClusterStats) -> float:
        _, s, v = stats.mean_hsv
        return (1.0 - s) * max(0.0, 1.0 - abs(v - 0.88) / 0.15)

    return (
        MaterialRule("plaster", str(textures["plaster"]), 0.6, plaster_score, min_score=0.45),
        MaterialRule("stone", str(textures["stone"]), 0.65, stone_score, min_score=0.2),
        MaterialRule("cladding", str(textures["cladding"]), 0.6, cladding_score, min_score=0.18),
        MaterialRule("screens", str(textures["screens"]), 0.55, screen_score, min_score=0.3),
        MaterialRule("equitone", str(textures["equitone"]), 0.55, equitone_score, min_score=0.28),
        MaterialRule("roof", str(textures["roof"]), 0.6, roof_score, min_score=0.25),
        MaterialRule("bronze", str(textures["bronze"]), 0.5, bronze_score, min_score=0.3),
        MaterialRule(
            "shade",
            str(textures["shade"]),
            0.45,
            shade_score,
            min_score=0.4,
            tint=(1.0, 1.0, 1.0),
            tint_strength=0.15,
        ),
    )


def _load_texture(path: str) -> np.ndarray:
    """Load texture image as normalized RGB array.

    Args:
        path: Path to texture image file.

    Returns:
        RGB array, shape (h, w, 3) with values 0-1.

    Raises:
        FileNotFoundError: If texture file does not exist.
        IOError: If texture cannot be loaded.
    """
    try:
        image = Image.open(path).convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Texture file not found: {path}") from exc
    except (OSError, IOError) as e:
        raise IOError(f"Failed to load texture {path}: {e}") from e


def _tile_texture(texture: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Tile texture to cover target image size.

    Args:
        texture: Texture array, shape (h, w, 3).
        size: Target size (width, height) in pixels.

    Returns:
        Tiled texture array, shape (target_height, target_width, 3).
    """
    h, w = size[1], size[0]
    tile_y = math.ceil(h / texture.shape[0])
    tile_x = math.ceil(w / texture.shape[1])
    tiled = np.tile(texture, (tile_y, tile_x, 1))
    return tiled[:h, :w, :]


def _soft_mask(mask: np.ndarray, radius: float = 1.5) -> np.ndarray:
    """Apply Gaussian blur to cluster mask for smooth transitions.

    Args:
        mask: Binary mask array, shape (h, w) with values 0 or 1.
        radius: Gaussian blur radius in pixels.

    Returns:
        Soft mask array, shape (h, w) with values 0-1.
    """
    img = Image.fromarray((mask * 255).astype("uint8"))
    img = img.convert("L")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    arr = np.asarray(blurred, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def assign_materials(stats: Sequence[ClusterStats], rules: Sequence[MaterialRule]) -> Dict[int, MaterialRule]:
    """Assign material rules to clusters using greedy best-match strategy.

    Each rule is assigned to the cluster with the highest score above the
    minimum threshold. Clusters can only be assigned once (no overlap).

    Args:
        stats: Cluster statistics from color analysis.
        rules: Material identification rules in priority order.

    Returns:
        Mapping from cluster label to assigned MaterialRule.
    """
    assignments: dict[int, MaterialRule] = {}
    used_labels: set[int] = set()
    for rule in rules:
        best_label = None
        best_score = rule.min_score
        for stat in stats:
            if stat.label in used_labels:
                continue
            score = rule.score_fn(stat)
            if score > best_score:
                best_score = score
                best_label = stat.label
        if best_label is not None:
            assignments[best_label] = rule
            used_labels.add(best_label)
    return assignments


def apply_materials(  # pylint: disable=too-many-locals
    image: np.ndarray, labels: np.ndarray, assignments: Mapping[int, MaterialRule]
) -> np.ndarray:
    """Apply material textures to image based on cluster assignments.

    Blends high-resolution texture plates with base image using soft masks
    for smooth transitions. Applies optional tinting for material refinement.

    Args:
        image: Base RGB image array, shape (h, w, 3).
        labels: Cluster label array, shape (h, w).
        assignments: Mapping from cluster label to MaterialRule.

    Returns:
        Enhanced RGB image array, shape (h, w, 3).
    """
    output = image.copy()
    h, w = image.shape[:2]
    cached_textures: MutableMapping[str, np.ndarray] = {}

    for label, rule in assignments.items():
        mask = labels == label
        if not np.any(mask):
            continue
        soft = _soft_mask(mask.astype("uint8"))
        if rule.texture:
            if rule.texture not in cached_textures:
                tex = _load_texture(rule.texture)
                cached_textures[rule.texture] = _tile_texture(tex, (w, h))
            texture = cached_textures[rule.texture]
        else:
            texture = np.zeros_like(output)
        blend = rule.blend
        mixed = output * (1.0 - blend) + texture * blend
        if rule.tint and rule.tint_strength > 0:
            tint = np.array(rule.tint, dtype=np.float32)
            mixed = np.clip(mixed * (1.0 - rule.tint_strength) + tint * rule.tint_strength, 0.0, 1.0)
        output = output * (1.0 - soft[..., None]) + mixed * soft[..., None]
    return np.clip(output, 0.0, 1.0)


def relabel(assignments: Mapping[int, MaterialRule], labels: np.ndarray) -> np.ndarray:
    """Relabel clusters based on material assignments (currently identity).

    Args:
        assignments: Mapping from cluster label to MaterialRule.
        labels: Original cluster label array.

    Returns:
        Relabeled array (currently unchanged).
    """
    renamed = labels.copy()
    for label, _ in assignments.items():
        renamed[labels == label] = label
    return renamed


def enhance_aerial(  # pylint: disable=too-many-arguments,too-many-locals
    input_path: Path,
    output_path: Path,
    *,
    analysis_max_dim: int = 1280,
    k: int = 8,
    seed: int = 22,
    target_width: int = 4096,
    textures: Mapping[str, Path] | None = None,
    palette_path: Optional[Path | str] = None,
    save_palette: Optional[Path | str] = None,
) -> Path:
    """Enhance an aerial image by clustering colors, assigning MBAR board materials,
    and blending high-resolution textures to approximate the approved palette.

    Workflow:
    1. Downsample image for fast k-means color clustering
    2. Compute cluster statistics (mean RGB/HSV, variance)
    3. Assign each cluster to best-matching MBAR material via scoring (or load from palette)
    4. Blend material textures with soft masks for natural transitions
    5. Scale result to 4K deliverable resolution

    Args:
        input_path: Path to input aerial image.
        output_path: Path to save enhanced image.
        analysis_max_dim: Maximum dimension for clustering analysis (smaller = faster).
        k: Number of color clusters (typically 6-12 for architectural aerials).
        seed: Random seed for reproducible clustering.
        target_width: Output width in pixels (height scaled proportionally).
        textures: Optional mapping of material names to texture paths.
                  Defaults to DEFAULT_TEXTURES if not provided.
        palette_path: Optional path to JSON palette file with cluster-to-material mappings.
                      When provided, uses these assignments instead of heuristic scoring.
        save_palette: Optional path to save the computed/loaded assignments as a palette JSON.

    Returns:
        Path to saved enhanced image.

    Raises:
        FileNotFoundError: If input image, texture files, or palette file do not exist.
        IOError: If image cannot be loaded or saved.
        ValueError: If palette references unknown materials.
    """
    source_textures = textures or DEFAULT_TEXTURES
    validated: dict[str, Path] = {}
    for key, path in source_textures.items():
        resolved = _validate_texture(path)
        if resolved is None:  # pragma: no cover - defensive
            continue
        validated[key] = resolved

    image = Image.open(input_path).convert("RGB")
    base_array = np.asarray(image, dtype=np.float32) / 255.0

    analysis_image = _downsample_image(image, analysis_max_dim)
    analysis_array = np.asarray(analysis_image, dtype=np.float32) / 255.0
    pixels = analysis_array.reshape(-1, 3)

    rng = np.random.default_rng(seed)
    sample_size = min(len(pixels), 200_000)
    if sample_size < len(pixels):
        indices = rng.choice(len(pixels), size=sample_size, replace=False)
        sample = pixels[indices]
    else:
        sample = pixels

    centroids = _kmeans(sample, k, rng)
    labels_small = _assign_full_image(analysis_array, centroids)
    labels_small_img = Image.fromarray(labels_small.astype("uint8"))
    labels_small_img = labels_small_img.convert("L")
    labels_full = labels_small_img.resize(image.size, Image.Resampling.NEAREST)
    labels = np.asarray(labels_full, dtype=np.uint8)

    stats = _cluster_stats(base_array, labels)
    rules = build_material_rules(validated)
    # Use palette assignments if provided, otherwise compute via heuristics
    if palette_path is not None:
        assignments = load_palette_assignments(palette_path, rules)
    else:
        assignments = assign_materials(stats, rules)
    # Optionally save the assignments for future use
    if save_palette is not None:
        save_palette_assignments(assignments, save_palette)
    
    enhanced = apply_materials(base_array, labels, assignments)

    enhanced_image = Image.fromarray((np.clip(enhanced, 0.0, 1.0) * 255.0 + 0.5).astype("uint8"))
    enhanced_image = enhanced_image.convert("RGB")
    if enhanced_image.width != target_width:
        scale = target_width / enhanced_image.width
        target_height = int(round(enhanced_image.height * scale))
        enhanced_image = enhanced_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced_image.save(output_path)
    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the base aerial image")
    parser.add_argument("output", type=Path, help="Destination image path")
    parser.add_argument("--analysis-max", type=int, default=1280, help="Maximum dimension for clustering analysis image")
    parser.add_argument("--k", type=int, default=8, help="Number of K-means clusters")
    parser.add_argument("--seed", type=int, default=22, help="Random seed for clustering")
    parser.add_argument("--target-width", type=int, default=4096, help="Output width in pixels")
    parser.add_argument("--palette", type=Path, default=None, help="Load cluster-to-material assignments from JSON palette file")
    parser.add_argument("--save-palette", type=Path, default=None, help="Save computed assignments to JSON palette file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    """Command-line entry point for aerial enhancement.

    Args:
        argv: Command-line arguments (defaults to sys.argv).

    Returns:
        Path to output image.
    """
    args = _parse_args(argv)
    return enhance_aerial(
        args.input,
        args.output,
        analysis_max_dim=args.analysis_max,
        k=args.k,
        seed=args.seed,
        target_width=args.target_width,
        palette_path=args.palette,
        save_palette=args.save_palette,
    )


if __name__ == "__main__":
    main()
