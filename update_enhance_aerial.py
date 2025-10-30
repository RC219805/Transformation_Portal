#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
board_material_aerial_enhancer.py — MBAR Material Application Engine

High-performance aerial image enhancement via color clustering,
material rule assignment, and linear-light texture blending.

Usage:
    python board_material_aerial_enhancer.py input.jpg output.jpg
        --tilesize 1024 --clusters 6 --workers 4

Features:
    • Color clustering (k-means)
    • Tile-based material application with optional parallel execution
    • Linear-light blending (normal, multiply, screen, overlay)
    • Memory-safe texture streaming
    • Per-texture LRU cache with hit/miss/eviction stats
    • Optional progress bar and benchmarking
    • Deterministic results via RNG seed

Author: Richie Cheetham
Date: October 2025
"""

import sys
import math
import json
import argparse
import threading
import collections
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Mapping, Callable, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

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
    blend_mode: Optional[str] = "normal"
    texture_gamma: float = 1.0

# ------------------------------
# Palette Loading / Saving
# ------------------------------


def load_palette_assignments(
    path: Path | str,
    rules: Sequence[MaterialRule] | Mapping[str, MaterialRule] | None = None
) -> dict[int, MaterialRule]:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    lookup = (
        {r.name: r for r in rules} if isinstance(rules, Sequence)
        else dict(rules or {})
    )
    assignments = {}
    for k, v in data.items():
        label = int(k)
        if v in lookup:
            assignments[label] = lookup[v]
    return assignments


def save_palette_assignments(
    assignments: Mapping[int, MaterialRule], path: Path | str
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(k): v.name for k, v in assignments.items()}
    p.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")

# ------------------------------
# Utilities
# ------------------------------


def _srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    srgb = np.clip(srgb, 0.0, 1.0)
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055)**2.4)


def _linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


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


def _gaussian(x: float, mu: float, sigma: float) -> float:
    return math.exp(-((x - mu)**2) / (2.0 * sigma**2))


def _initial_centroids(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if k > len(data):
        raise ValueError("k cannot exceed number of data points")
    return data[rng.choice(len(data), size=k, replace=False)]


def _kmeans(
    data: np.ndarray, k: int, rng: np.random.Generator, iterations: int = 20
) -> np.ndarray:
    centroids = _initial_centroids(data, k, rng)
    for _ in range(iterations):
        distances = np.sum((data[:, None] - centroids[None, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            data[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        if np.allclose(new_centroids, centroids):
            break
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
        if count == 0:
            continue
        stats.append(ClusterStats(
            label=label,
            count=count,
            mean_rgb=image[mask].mean(axis=0),
            mean_hsv=hsv[mask].mean(axis=0),
            std_rgb=image[mask].std(axis=0)
        ))
    return stats

# ------------------------------
# Material Rules
# ------------------------------


def build_material_rules(textures: Mapping[str, Path]) -> Sequence[MaterialRule]:
    def plaster_score(s: ClusterStats) -> float:
        _, s_, v = s.mean_hsv
        return max(0.0, (1.0 - s_) * v)

    def stone_score(s: ClusterStats) -> float:
        h, s_, v = s.mean_hsv
        return _gaussian(h, 0.09, 0.05) * max(0.0, 1 - abs(v - 0.62) / 0.4) * max(0.0, 1 - abs(s_ - 0.22) / 0.4)
    return (
        MaterialRule("plaster", str(textures["plaster"]), 0.6, plaster_score, 0.45),
        MaterialRule("stone", str(textures["stone"]), 0.65, stone_score, 0.2)
    )


def assign_materials(
    stats: Sequence[ClusterStats], rules: Sequence[MaterialRule]
) -> Dict[int, MaterialRule]:
    assignments: Dict[int, MaterialRule] = {}
    used: set[int] = set()
    for rule in rules:
        best_label, best_score = None, rule.min_score
        for stat in stats:
            if stat.label in used:
                continue
            score = rule.score_fn(stat)
            if score > best_score:
                best_label, best_score = stat.label, score
        if best_label is not None:
            assignments[best_label] = rule
            used.add(best_label)
    return assignments

# ------------------------------
# Tiling and Material Application
# ------------------------------


def apply_materials_tiled(
    base: np.ndarray,
    labels: np.ndarray,
    materials: Mapping[int, MaterialRule],
    *,
    tile_size: int = 1024,
    texture_cache_limit: int = 8,
    verbose: bool = False,
    workers: int = 0,
    stream_large_texture_threshold: int = 2_000_000,
    progress: bool = True,
    eviction_callback: Optional[Callable[[str, dict], None]] = None
) -> np.ndarray:

    height, width, _channels = base.shape
    out_linear = _srgb_to_linear(base.copy())

    texture_cache: "collections.OrderedDict[str, Optional[Image.Image]]" = (
        collections.OrderedDict()
    )
    cache_lock = threading.Lock()
    cache_stats: dict[str, dict] = {}

    def _ensure_stats(path: str):
        if path not in cache_stats:
            cache_stats[path] = {"hits": 0, "misses": 0, "opens": 0, "evictions": 0}

    def _evict_one_if_needed():
        with cache_lock:
            while len(texture_cache) > max(1, int(texture_cache_limit)):
                old_path, _ = texture_cache.popitem(last=False)
                _ensure_stats(old_path)
                cache_stats[old_path]["evictions"] += 1
                if eviction_callback:
                    try:
                        eviction_callback(old_path, dict(cache_stats.get(old_path, {})))
                    except (IOError, OSError):
                        if verbose:
                            print(
                                f"[cache] eviction callback failed for {old_path}",
                                file=sys.stderr
                            )
                if verbose:
                    print(f"[cache] evicted {old_path}", file=sys.stderr)

    def _cache_get_image(path: str) -> Optional[Image.Image]:
        with cache_lock:
            if path in texture_cache:
                texture_cache.move_to_end(path)
                _ensure_stats(path)
                cache_stats[path]["hits"] += 1
                return texture_cache[path]
            _ensure_stats(path)
            cache_stats[path]["misses"] += 1
        try:
            pil = Image.open(path).convert("RGB")
        except (IOError, OSError):
            with cache_lock:
                texture_cache[path] = None
                _ensure_stats(path)
                cache_stats[path]["opens"] += 1
                _evict_one_if_needed()
            return None
        with cache_lock:
            texture_cache[path] = pil
            _ensure_stats(path)
            cache_stats[path]["opens"] += 1
            _evict_one_if_needed()
        return pil

    def _make_texture_patch(
        pil_img: Image.Image, tile_w: int, tile_h: int, gamma: float = 1.0
    ) -> np.ndarray:
        tw, th = pil_img.size
        src_pixels = tw * th
        if src_pixels * 3 <= stream_large_texture_threshold or src_pixels <= tile_w * tile_h:
            tex_arr = np.asarray(pil_img, dtype=np.float32) / 255.0
            if gamma != 1.0:
                tex_arr = np.clip(tex_arr**gamma, 0.0, 1.0)
            reps_y = (tile_h + th - 1) // th
            reps_x = (tile_w + tw - 1) // tw
            tiled = np.tile(tex_arr, (reps_y, reps_x, 1))
            return tiled[:tile_h, :tile_w, :]

        patch = np.empty((tile_h, tile_w, 3), dtype=np.float32)
        reps_y = (tile_h + th - 1) // th
        reps_x = (tile_w + tw - 1) // tw
        y = 0
        for _ in range(reps_y):
            ph = min(th, tile_h - y)
            x = 0
            for _ in range(reps_x):
                pw = min(tw, tile_w - x)
                crop = pil_img.crop((0, 0, pw, ph))
                arr = np.asarray(crop, dtype=np.float32) / 255.0
                if gamma != 1.0:
                    arr = np.clip(arr**gamma, 0.0, 1.0)
                patch[y:y+ph, x:x+pw, :] = arr[:ph, :pw, :]
                x += pw
            y += ph
        return patch

    def _soft_mask(mask: np.ndarray, radius: float = 1.5) -> np.ndarray:
        img = Image.fromarray((mask * 255).astype("uint8")).convert("L")
        img = img.filter(ImageFilter.GaussianBlur(radius))
        return np.asarray(img, dtype=np.float32) / 255.0

    def _blend_linear(
        base_l: np.ndarray, tex_l: np.ndarray, mode: str = "normal"
    ) -> np.ndarray:
        if mode == "normal":
            return tex_l
        if mode == "multiply":
            return base_l * tex_l
        if mode == "screen":
            return 1.0 - (1.0 - base_l) * (1.0 - tex_l)
        if mode == "overlay":
            mask = base_l <= 0.5
            result = np.empty_like(base_l)
            result[mask] = 2 * base_l[mask] * tex_l[mask]
            result[~mask] = 1 - 2 * (1 - base_l[~mask]) * (1 - tex_l[~mask])
            return result
        return tex_l

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
    jobs = [
        (ty * tile_size, min(height, (ty + 1) * tile_size),
         tx * tile_size, min(width, (tx + 1) * tile_size))
        for ty in range(tiles_y) for tx in range(tiles_x)
    ]
    results = []

    iterator = tqdm(jobs, desc="[tiles]") if progress else jobs

    def _process_tile(job):
        y0, y1, x0, x1 = job
        tile_lab = labels[y0:y1, x0:x1]
        uniq = np.unique(tile_lab)
        tile_base = out_linear[y0:y1, x0:x1, :].copy()
        for label in uniq:
            if label not in materials:
                continue
            rule = materials[label]
            mask = tile_lab == label
            if not np.any(mask):
                continue
            base_masked = tile_base[mask]

            tstrength = getattr(rule, "tint_strength", 0.0) or 0.0
            if getattr(rule, "tint", None) and tstrength > 0.0:
                tint_rgb = np.asarray(rule.tint, dtype=np.float32)
                if tint_rgb.max() > 1.5:
                    tint_rgb /= 255.0
                tint_l = _srgb_to_linear(tint_rgb[None, :])
                base_masked = (1.0 - tstrength) * base_masked + tstrength * tint_l

            blend_val = getattr(rule, "blend", 0.0) or 0.0
            tex_path = getattr(rule, "texture", None)
            if tex_path and blend_val > 0.0:
                pil_img = _cache_get_image(tex_path)
                if pil_img is None:
                    tile_base[mask] = base_masked
                    continue
                tex_patch = _make_texture_patch(
                    pil_img,
                    x1 - x0,
                    y1 - y0,
                    gamma=getattr(rule, "texture_gamma", 1.0)
                )
                tex_patch_l = _srgb_to_linear(tex_patch)
                tex_masked = tex_patch_l[mask]
                blended = _blend_linear(
                    base_masked, tex_masked, getattr(rule, "blend_mode", "normal")
                )
                tile_base[mask] = (1.0 - blend_val) * base_masked + blend_val * blended
            else:
                tile_base[mask] = base_masked
        return (y0, y1, x0, x1, tile_base)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process_tile, job) for job in iterator]
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for job in iterator:
            results.append(_process_tile(job))

    for y0, y1, x0, x1, tile in results:
        out_linear[y0:y1, x0:x1, :] = tile

    # Print cache stats
    if verbose:
        print("[cache stats]")
        for path, stats in cache_stats.items():
            print(f"{path}: {stats}")

    return np.clip(_linear_to_srgb(out_linear), 0.0, 1.0)

# ------------------------------
# CLI
# ------------------------------


def main():
    parser = argparse.ArgumentParser(description="MBAR Aerial Material Enhancer")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tilesize", type=int, default=1024)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    img = Image.open(args.input).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    height, width, _ = arr.shape
    print(f"[info] image size: {width}x{height}")

    rng = np.random.default_rng(42)
    pixels = arr.reshape(-1, 3)
    centroids = _kmeans(pixels, args.clusters, rng)
    labels = _assign_full_image(arr, centroids)
    stats = _cluster_stats(arr, labels)

    # example textures
    textures = {
        "plaster": Path("textures/plaster.jpg"),
        "stone": Path("textures/stone.jpg")
    }
    rules = build_material_rules(textures)
    assignments = assign_materials(stats, rules)

    out = apply_materials_tiled(
        arr,
        labels,
        assignments,
        tile_size=args.tilesize,
        workers=args.workers,
        verbose=args.verbose,
        progress=args.progress
    )

    Image.fromarray((out * 255).astype("uint8")).save(args.output)
    print(f"[done] saved to {args.output}")


if __name__ == "__main__":
    main()
