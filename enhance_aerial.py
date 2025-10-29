"""
enhance_aerial.py

Aerial image enhancement using color clustering, material assignment,
and high-resolution texture blending.

Usage:
    python enhance_aerial.py input.jpg output.jpg [options]
"""

import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Mapping, Callable, Dict, Optional, MutableMapping

import numpy as np
from PIL import Image, ImageFilter

# ------------------------------
# Data Classes
# ------------------------------

@dataclass(frozen=True)
class ClusterStats:
    label: int
    count: int
    mean_rgb: np.ndarray
    mean_hsv: np.ndarray
    std_rgb: np.ndarray


@dataclass(frozen=True)
class MaterialRule:
    name: str
    texture: Optional[str]
    blend: float
    score_fn: Callable[[ClusterStats], float]
    min_score: float = 0.0
    tint: Optional[tuple[float, float, float]] = None
    tint_strength: float = 0.0

# ------------------------------
# Palette Loading / Saving
# ------------------------------

try:
    from .palette_assignments import load_palette_assignments, save_palette_assignments
except ImportError:
    def load_palette_assignments(
        path: Path | str,
        rules: Sequence[MaterialRule] | Mapping[str, MaterialRule] | None = None
    ) -> dict[int, MaterialRule]:
        p = Path(path)
        if not p.exists():
            return {}
        data = json.loads(p.read_text())
        lookup = {r.name: r for r in rules} if isinstance(rules, Sequence) else dict(rules or {})
        assignments = {}
        for k, v in data.items():
            label = int(k)
            if v in lookup:
                assignments[label] = lookup[v]
        return assignments

    def save_palette_assignments(assignments: Mapping[int, MaterialRule], path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        serializable = {str(k): v.name for k, v in assignments.items()}
        p.write_text(json.dumps(serializable, indent=2, sort_keys=True))

# ------------------------------
# Utilities
# ------------------------------

def _validate_texture(path: Optional[Path]) -> Optional[Path]:
    if path is None: return None
    if not Path(path).exists(): raise FileNotFoundError(path)
    return Path(path)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
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
    w, h = image.size
    scale = max(1, max(w, h) // max_dim)
    if scale <= 1: return image.copy()
    return image.resize((max(1, w // scale), max(1, h // scale)), Image.Resampling.BILINEAR)


def _initial_centroids(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if k > len(data): raise ValueError("k cannot exceed number of data points")
    return data[rng.choice(len(data), size=k, replace=False)]


def _kmeans(data: np.ndarray, k: int, rng: np.random.Generator, iterations: int = 20) -> np.ndarray:
    centroids = _initial_centroids(data, k, rng)
    for _ in range(iterations):
        distances = np.sum((data[:, None] - centroids[None, :])**2, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels==i].mean(axis=0) if np.any(labels==i) else centroids[i] for i in range(k)])
        if np.allclose(new_centroids, centroids): break
        centroids = new_centroids
    return centroids


def _assign_full_image(image: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    pixels = image.reshape(-1, 3)
    distances = np.sum((pixels[:, None] - centroids[None, :])**2, axis=2)
    return np.argmin(distances, axis=1).reshape(image.shape[:2])


def _cluster_stats(image: np.ndarray, labels: np.ndarray) -> Sequence[ClusterStats]:
    stats = []
    hsv = _rgb_to_hsv(image)
    for label in range(labels.max() + 1):
        mask = labels == label
        count = int(mask.sum())
        if count == 0: continue
        stats.append(
            ClusterStats(
                label=label,
                count=count,
                mean_rgb=image[mask].mean(axis=0),
                mean_hsv=hsv[mask].mean(axis=0),
                std_rgb=image[mask].std(axis=0)
            )
        )
    return stats


def _gaussian(x: float, mu: float, sigma: float) -> float:
    return math.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


# ------------------------------
# Material Rules
# ------------------------------

def build_material_rules(textures: Mapping[str, Path]) -> Sequence[MaterialRule]:
    def plaster_score(s: ClusterStats) -> float:
        _, s_, v = s.mean_hsv
        return max(0.0, (1.0 - s_) * v)

    def stone_score(s: ClusterStats) -> float:
        h, s_, v = s.mean_hsv
        return _gaussian(h, 0.09, 0.05) * max(0.0, 1 - abs(v - 0.62)/0.4) * max(0.0, 1 - abs(s_ - 0.22)/0.4)

    # Add other scoring functions similarly...
    return (
        MaterialRule("plaster", str(textures["plaster"]), 0.6, plaster_score, 0.45),
        MaterialRule("stone", str(textures["stone"]), 0.65, stone_score, 0.2),
        # Add remaining rules here...
    )


def assign_materials(stats: Sequence[ClusterStats], rules: Sequence[MaterialRule]) -> Dict[int, MaterialRule]:
    assignments: Dict[int, MaterialRule] = {}
    used: set[int] = set()
    for rule in rules:
        best_label, best_score = None, rule.min_score
        for stat in stats:
            if stat.label in used: continue
            score = rule.score_fn(stat)
            if score > best_score:
                best_label, best_score = stat.label, score
        if best_label is not None:
            assignments[best_label] = rule
            used.add(best_label)
    return assignments


def _load_texture(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def _tile_texture(texture: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size[1], size[0]
    tile_y = math.ceil(h / texture.shape[0])
    tile_x = math.ceil(w / texture.shape[1])
    tiled = np.tile(texture, (tile_y, tile_x, 1))
    return tiled[:h, :w, :]


def _soft_mask(mask: np.ndarray, radius: float = 1.5) -> np.ndarray:
    img = Image.fromarray((mask*255).astype("uint8")).convert("L")
    img = img.filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(img, dtype=np.float32)/255.0


def apply_materials(
    image: np.ndarray,
    labels: np.ndarray,
    assignments: Mapping[int, MaterialRule]
) -> np.ndarray:
    output = image.copy()
    h, w = image.shape[:2]
    cache: MutableMapping[str, np.ndarray] = {}

    for label, rule in assignments.items():
        mask = labels == label
        if not np.any(mask): continue
        soft = _soft_mask(mask.astype("uint8"))

        if rule.texture:
            if rule.texture not in cache:
                cache[rule.texture] = _tile_texture(_load_texture(rule.texture), (w, h))
            texture = cache[rule.texture]
        else:
            texture = np.zeros_like(output)

