import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from typing import Callable, Optional, List

# ==========================
# Default Textures (placeholder)
# ==========================
DEFAULT_TEXTURES = {
    "plaster": "textures/plaster.png",
    "stone": "textures/stone.png",
    "concrete": "textures/concrete.png"
}

# ==========================
# Material Rule
# ==========================
class MaterialRule:
    def __init__(
        self,
        name: str,
        texture: str,
        blend: float,
        score_fn: Callable[[np.ndarray], float],
        min_score: float = 0.0,
        tint: Optional[tuple[int,int,int]] = None,
        tint_strength: float = 0.0
    ):
        self.name = name
        self.texture = texture
        self.blend = blend
        self.score_fn = score_fn
        self.min_score = min_score
        self.tint = tint
        self.tint_strength = tint_strength

# ==========================
# Cluster Stats
# ==========================
class ClusterStats:
    def __init__(self, centroid: np.ndarray, points: Optional[np.ndarray] = None):
        self.centroid = np.array(centroid)
        self.points = points if points is not None else []

# ==========================
# K-means
# ==========================
def _kmeans(x: np.ndarray, k: int, seed: int = None, max_iter: int = 100):
    kmeans = KMeans(n_clusters=k, random_state=seed, max_iter=max_iter)
    kmeans.fit(x)
    return kmeans.labels_

# ==========================
# Compute cluster stats
# ==========================
def compute_cluster_stats(labels: np.ndarray, data: np.ndarray) -> List[ClusterStats]:
    stats = []
    for i in np.unique(labels):
        pts = data[labels == i]
        centroid = pts.mean(axis=0)
        stats.append(ClusterStats(centroid, pts))
    return stats

# ==========================
# Relabel clusters consecutively
# ==========================
def relabel(assignments: dict[int, MaterialRule], labels: np.ndarray) -> np.ndarray:
    label_map = {old: new for new, old in enumerate(sorted(assignments.keys()))}
    return np.vectorize(lambda x: label_map.get(x, x))(labels)

def relabel_safe(assignments: dict[int, MaterialRule], labels: np.ndarray, mode="warn", strict=True, verbose=False):
    # Ensure all label indices in assignments
    all_labels = np.unique(labels)
    missing = [l for l in all_labels if l not in assignments]
    if missing:
        if strict:
            raise ValueError(f"Missing assignments for labels: {missing}")
        elif verbose and mode != "none":
            print(f"Warning: missing assignments for {missing}")
        for l in missing:
            assignments[l] = MaterialRule(name="unknown", texture="", blend=0.5, score_fn=lambda x: 0.5)
    return relabel(assignments, labels)

# ==========================
# Build material rules from textures
# ==========================
def build_material_rules(textures: dict[str, str]) -> List[MaterialRule]:
    rules = []
    for name, path in textures.items():
        blend = 0.5 + 0.1 * (hash(name) % 5) / 5.0  # dummy blend
        rules.append(
            MaterialRule(
                name=name,
                texture=str(path),
                blend=blend,
                score_fn=lambda x, b=blend: b
            )
        )
    return rules

# ==========================
# Save/load palette assignments
# ==========================
def save_palette_assignments(assignments: dict[int, MaterialRule], out_path: Path):
    data = {k: vars(v) for k, v in assignments.items()}
    with open(out_path, "w") as f:
        json.dump(data, f)

def load_palette_assignments(in_path: Path) -> dict[int, MaterialRule]:
    with open(in_path) as f:
        data = json.load(f)
    assignments = {}
    for k, v in data.items():
        assignments[int(k)] = MaterialRule(
            name=v["name"],
            texture=v["texture"],
            blend=v["blend"],
            score_fn=lambda x: v.get("blend", 0.5),
            min_score=v.get("min_score", 0.0),
            tint=tuple(v["tint"]) if v.get("tint") else None,
            tint_strength=v.get("tint_strength", 0.0)
        )
    return assignments

# ==========================
# Auto-assign by cluster stats
# ==========================
def auto_assign_materials_by_stats(labels: np.ndarray, img: np.ndarray, tex_map: dict) -> dict[int, MaterialRule]:
    stats = compute_cluster_stats(labels, img.reshape(-1, img.shape[-1]))
    rules = build_material_rules(tex_map)
    assignments = {}
    for i, s in enumerate(stats):
        assignments[i] = rules[i % len(rules)]
    return assignments

# ==========================
# Enhance aerial image
# ==========================
def enhance_aerial(image: np.ndarray, out_path: Optional[str] = None, k: int = 3, textures: Optional[dict] = None):
    h, w, c = image.shape
    pixels = image.reshape(-1, c).astype(np.float32)

    labels = _kmeans(pixels, k=k, seed=42)
    assignments = auto_assign_materials_by_stats(labels, image, textures or DEFAULT_TEXTURES)
    labels = relabel(assignments, labels)

    output = np.zeros_like(pixels)
    for idx, rule in assignments.items():
        mask = labels == idx
        output[mask] = rule.blend  # simplistic coloring by blend
    output = output.reshape(h, w, c)

    if out_path:
        from PIL import Image
        im = (np.clip(output, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(im).save(out_path)

    return output
