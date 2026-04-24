from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional

from transformation_portal.spatial_ai.segmentation.backends.tiled_backend import TiledSegmentationBackend
from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput, SegmentationResult
from transformation_portal.spatial_ai.segmentation.tiling.config import SegmentationTilingConfig
from transformation_portal.spatial_ai.segmentation.tiling.interfaces import MergeValidator, TileMerger, TilingPlanner
from transformation_portal.spatial_ai.segmentation.tiling.types import GlobalSeedHints, TileSegmentationResult


@dataclass
class TiledSegmentationEngine:
    planner: TilingPlanner
    merger: TileMerger
    validator: Optional[MergeValidator] = None

    def run(
        self,
        *,
        backend: TiledSegmentationBackend,
        seg_input: SegmentationInput,
        image_hash: str,
        config: SegmentationTilingConfig,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> SegmentationResult:
        if seg_input.image is None:
            raise ValueError("TiledSegmentationEngine requires seg_input.image (non-video mode)")

        H, W = int(seg_input.image.shape[0]), int(seg_input.image.shape[1])
        rng_seed = config.seed

        global_hints: Optional[GlobalSeedHints] = None
        if config.global_pass.enabled:
            global_hints = backend.global_seed_pass(
                image_linear=seg_input.image,
                image_hash=image_hash,
                longest_side=config.global_pass.longest_side,
                rng_seed=rng_seed,
            )

        manifest = self.planner.plan(
            image_hash=image_hash,
            W=W,
            H=H,
            config=config,
            global_hints=global_hints,
            prompts=seg_input.prompts,
            mode=seg_input.mode,
        )

        n_tiles = max(1, len(manifest.tiles))

        def process_tile(index: int) -> tuple[int, TileSegmentationResult]:
            tile = manifest.tiles[index]
            t0 = time.time()
            x0, y0, x1, y1 = tile.bbox.x0, tile.bbox.y0, tile.bbox.x1, tile.bbox.y1
            tile_img = seg_input.image[y0:y1, x0:x1, :]

            instances = backend.segment_tile(
                tile_linear=tile_img,
                image_hash=image_hash,
                tile_spec=tile,
                mode=seg_input.mode,
                prompts=seg_input.prompts,
                global_hints=global_hints,
                rng_seed=rng_seed,
            )

            return (
                index,
                TileSegmentationResult(
                    image_hash=image_hash,
                    tile_id=tile.tile_id,
                    tile_spec=tile,
                    instances=tuple(instances),
                    runtime_s=time.time() - t0,
                ),
            )

        tile_results_by_index: list[Optional[TileSegmentationResult]] = [None] * len(manifest.tiles)
        if config.max_concurrency == 1 or len(manifest.tiles) <= 1:
            for i in range(len(manifest.tiles)):
                idx, tile_result = process_tile(i)
                tile_results_by_index[idx] = tile_result
                if on_progress:
                    on_progress(((i + 1) / n_tiles) * 100.0)
        else:
            completed = 0
            with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
                futures = {executor.submit(process_tile, i): i for i in range(len(manifest.tiles))}
                for future in as_completed(futures):
                    idx, tile_result = future.result()
                    tile_results_by_index[idx] = tile_result
                    completed += 1
                    if on_progress:
                        on_progress((completed / n_tiles) * 100.0)

        tile_results = [tile_result for tile_result in tile_results_by_index if tile_result is not None]

        masks, scores, metadata, merge_stats = self.merger.merge(
            image_hash=image_hash,
            W=W,
            H=H,
            manifest=manifest,
            tile_results=tile_results,
            global_hints=global_hints,
            merge_config=config.merge,
        )

        if self.validator and config.validation.enabled:
            ok, details = self.validator.validate(manifest=manifest, merge_stats=merge_stats, config=config)
            if not ok:
                reason = details.get("warning") or details
                raise RuntimeError(f"SAM2 tiling validation failed: {reason}")

        return SegmentationResult(
            masks=masks,
            scores=scores,
            metadata=metadata,
            temporal_ids=None,
        )
