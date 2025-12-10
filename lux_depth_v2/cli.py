from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig, Preset
from .logging_utils import setup_logging
from .pipeline import LuxPipelineV2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gold Standard Lux Depth Pipeline V2 (GPU-accelerated, modular).")

    # IO
    p.add_argument("--input", type=str, default=None, help="Single input image path.")
    p.add_argument("--input-dir", type=str, default=None, help="Input directory of images.")
    p.add_argument("--depth-dir", type=str, default=None, help="Depth directory (depth TIFFs and optional zone masks).")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory.")

    # Look
    p.add_argument("--preset", type=str, default=Preset.PHOTO_REALISTIC.value, choices=[e.value for e in Preset])

    # Device
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32"])

    # Upscaling
    p.add_argument("--upscale", type=int, default=4, choices=[2,4])
    p.add_argument("--upscaler-backend", type=str, default="torch", choices=["torch", "realesrgan", "onnx", "none"])
    p.add_argument("--model-path", type=str, default=None, help="Upscaler model path (.pth or .onnx).")
    p.add_argument("--model-sha256", type=str, default=None, help="Optional SHA256 for verifying model file.")
    p.add_argument("--tile", type=int, default=512, help="Real-ESRGAN tile size (0 disables).")
    p.add_argument("--tile-pad", type=int, default=16)
    p.add_argument("--half", action="store_true", help="Use half precision inside Real-ESRGAN (GPU only).")

    # Segmentation
    p.add_argument("--seg-backend", type=str, default="auto", choices=["auto","onnx","segformer","heuristic","none"])
    p.add_argument("--seg-onnx-model", type=str, default=None, help="Material segmentation ONNX model path.")
    p.add_argument("--seg-onnx-labels", type=str, default=None, help="Optional JSON label mapping for ONNX model.")
    p.add_argument("--seg-segformer-model", type=str, default=None, help="Local dir or HF model id for segformer backend.")
    p.add_argument("--seg-allow-downloads", action="store_true", help="Allow downloading pretrained segformer weights.")
    p.add_argument("--seg-long-side", type=int, default=768)
    p.add_argument("--seg-min-conf", type=float, default=0.25)

    # Materials v2 (Phase 1 Integration Pack)
    p.add_argument("--materials-v2", action="store_true", 
                   help="Enable Materials v2 confidence-gated material response.")
    p.add_argument("--confidence-threshold", type=float, default=0.6, 
                   help="Confidence threshold for Materials v2 gating (default: 0.6).")
    p.add_argument("--confidence-blend-range", type=float, default=0.1,
                   help="Blend range for soft confidence falloff (default: 0.1).")
    p.add_argument("--confidence-blend-mode", type=str, default="soft", choices=["soft", "hard"],
                   help="Confidence blending mode (default: soft).")
    p.add_argument("--cache-masks", action="store_true", 
                   help="Enable mask caching for Materials v2.")
    p.add_argument("--cache-dir", type=str, default=".mask_cache",
                   help="Mask cache directory (default: .mask_cache).")
    p.add_argument("--max-segmentation-side", type=int, default=1024,
                   help="Max segmentation resolution for Materials v2 (default: 1024).")

    # Service mode (optional)
    p.add_argument("--service", action="store_true", help="Run as HTTP service (FastAPI).")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8088)

    # Phase 1 Stability Architecture
    p.add_argument("--enable-orchestrator", action="store_true", default=True, 
                   help="Enable process orchestrator (default: True).")
    p.add_argument("--disable-orchestrator", action="store_true", 
                   help="Disable process orchestrator (legacy mode).")
    p.add_argument("--checkpoint-dir", type=str, default=".checkpoints", 
                   help="Checkpoint directory for resume capability.")
    p.add_argument("--max-retries", type=int, default=3, 
                   help="Maximum retry attempts for failed tasks.")
    p.add_argument("--memory-budget", type=float, default=None, 
                   help="Memory budget per task in GB (None=no limit).")
    p.add_argument("--pre-flight-check", action="store_true", default=True,
                   help="Enable pre-flight validation (default: True).")
    p.add_argument("--skip-pre-flight", action="store_true",
                   help="Skip pre-flight validation checks.")

    # Phase 2 Performance Optimizations
    phase2_group = p.add_argument_group('Phase 2 Performance Optimizations')
    
    # Master toggle
    phase2_group.add_argument("--phase2-optimizations", action="store_true",
                             help="Enable all Phase 2 performance optimizations.")
    
    # I/O Optimization
    phase2_group.add_argument("--async-io", action="store_true",
                             help="Enable asynchronous TIFF writing (5-7× I/O speedup).")
    phase2_group.add_argument("--streaming-upscale", action="store_true",
                             help="Stream upscaled output progressively (reduces memory).")
    phase2_group.add_argument("--tiff-compression", type=str, default="lzw",
                             choices=["lzw", "deflate", "none"],
                             help="TIFF compression method (default: lzw).")
    
    # Storage Management
    phase2_group.add_argument("--storage-external", type=str, default=None,
                             help="External storage path (T9 SSD) for large files.")
    phase2_group.add_argument("--auto-migrate", action="store_true",
                             help="Auto-migrate large files (>2GB) to external storage.")
    phase2_group.add_argument("--migrate-threshold", type=float, default=2.0,
                             help="File size threshold (GB) for auto-migration (default: 2.0).")
    
    # Parallel Processing
    phase2_group.add_argument("--parallel-workers", type=int, default=1,
                             help="Number of concurrent workers (1-4, default: 1).")
    phase2_group.add_argument("--memory-per-worker", type=float, default=25.0,
                             help="Memory budget per worker in GB (default: 25.0).")
    
    # Caching
    phase2_group.add_argument("--model-cache", action="store_true",
                             help="Cache ML models across batch (saves 18-30s).")
    phase2_group.add_argument("--depth-cache", action="store_true",
                             help="Cache depth maps (avoid regeneration).")
    phase2_group.add_argument("--phase2-cache-dir", type=str, default=".cache",
                             help="Phase 2 cache directory (default: .cache).")
    
    # Upscaling Optimization
    phase2_group.add_argument("--tile-based-upscale", action="store_true",
                             help="Use tile-based upscaling (memory efficient).")
    phase2_group.add_argument("--upscale-tile-size", type=int, default=512,
                             help="Tile size for upscaling in pixels (default: 512).")
    
    # Export Autotune (Phase 2 Slice 3)
    phase2_group.add_argument("--autotune-export", action="store_true",
                             help="Enable adaptive export configuration (autotune based on image stats).")
    phase2_group.add_argument("--autotune-complexity", action="store_true", default=True,
                             help="Use scene complexity in autotune decisions (default: True).")

    return p


def main() -> None:
    args = build_parser().parse_args()
    logger = setup_logging("INFO")

    cfg = PipelineConfig(
        input_dir=Path(args.input_dir) if args.input_dir else None,
        depth_dir=Path(args.depth_dir) if args.depth_dir else None,
        output_dir=Path(args.output_dir),
        preset=Preset(args.preset),
        device=args.device,
        precision=args.precision,
        upscale=int(args.upscale),
        upscaler_backend=args.upscaler_backend,
        model_path=Path(args.model_path) if args.model_path else None,
        model_sha256=args.model_sha256,
        tile=int(args.tile),
        tile_pad=int(args.tile_pad),
        half=bool(args.half),
    )
    cfg.segmentation.backend = args.seg_backend
    cfg.segmentation.onnx_model_path = Path(args.seg_onnx_model) if args.seg_onnx_model else None
    cfg.segmentation.onnx_labels_path = Path(args.seg_onnx_labels) if args.seg_onnx_labels else None
    cfg.segmentation.segformer_model = args.seg_segformer_model
    cfg.segmentation.allow_downloads = bool(args.seg_allow_downloads)
    cfg.segmentation.input_long_side = int(args.seg_long_side)
    cfg.segmentation.min_confidence = float(args.seg_min_conf)
    
    # Phase 1 Stability: Configure orchestrator
    cfg.orchestrator.enabled = args.enable_orchestrator and not args.disable_orchestrator
    cfg.orchestrator.checkpoint_dir = args.checkpoint_dir
    cfg.orchestrator.max_retries = args.max_retries
    cfg.orchestrator.memory_budget_gb = args.memory_budget
    
    # Materials v2: Configure confidence gating and caching
    if hasattr(args, 'materials_v2') and args.materials_v2:
        from .materials_v2 import MaterialsV2Config, ConfidenceConfig, SegmentationConfig as Mat2SegConfig
        
        # Initialize Materials v2 config
        cfg.materials_v2 = MaterialsV2Config(
            enabled=True,
            confidence=ConfidenceConfig(
                confidence_threshold=args.confidence_threshold,
                blend_range=args.confidence_blend_range,
                blend_mode=args.confidence_blend_mode,
            ),
            segmentation=Mat2SegConfig(
                max_segmentation_side=args.max_segmentation_side,
            ),
            cache_enabled=args.cache_masks,
            cache_dir=args.cache_dir if args.cache_masks else None,
            backend='heuristic',  # Default backend
        )
    cfg.orchestrator.pre_flight_check = args.pre_flight_check and not args.skip_pre_flight
    
    # Phase 2 Performance Optimizations
    phase2_enabled = args.phase2_optimizations if hasattr(args, 'phase2_optimizations') else False
    
    if phase2_enabled or (hasattr(args, 'parallel_workers') and args.parallel_workers > 1):
        from .config import Phase2Config
        
        # Build Phase2Config from CLI options
        phase2_config = Phase2Config(
            # I/O
            async_io_enabled=getattr(args, 'async_io', False) or phase2_enabled,
            streaming_upscale=getattr(args, 'streaming_upscale', False) or phase2_enabled,
            tiff_compression=getattr(args, 'tiff_compression', 'lzw'),
            
            # Storage
            storage_external_t9=getattr(args, 'storage_external', None),
            auto_migrate_large_files=(getattr(args, 'auto_migrate', False) or 
                                     (phase2_enabled and getattr(args, 'storage_external', None))),
            migrate_threshold_gb=getattr(args, 'migrate_threshold', 2.0),
            
            # Parallel
            max_concurrent_workers=max(getattr(args, 'parallel_workers', 1), 1),
            memory_budget_per_worker_gb=getattr(args, 'memory_per_worker', 25.0),
            
            # Caching
            model_cache_enabled=getattr(args, 'model_cache', False) or phase2_enabled,
            depth_map_cache_enabled=getattr(args, 'depth_cache', False) or phase2_enabled,
            cache_dir=getattr(args, 'phase2_cache_dir', '.cache'),
            
            # Upscaling
            tile_based_upscaling=getattr(args, 'tile_based_upscale', False) or phase2_enabled,
            upscale_tile_size=getattr(args, 'upscale_tile_size', 512),
            
            # Autotune (Phase 2 Slice 3)
            autotune_export=getattr(args, 'autotune_export', False),
            autotune_use_complexity=getattr(args, 'autotune_complexity', True),
        )
        
        # Store in config for pipeline access
        cfg.phase2 = phase2_config
        
        # Display Phase 2 status
        if phase2_config.async_io_enabled or phase2_config.max_concurrent_workers > 1:
            logger.info("🚀 Phase 2 Performance Optimizations Enabled:")
            if phase2_config.async_io_enabled:
                logger.info("  ✓ Async I/O (5-7× speedup)")
            if phase2_config.streaming_upscale:
                logger.info("  ✓ Streaming upscale (memory efficient)")
            if phase2_config.max_concurrent_workers > 1:
                logger.info(f"  ✓ Parallel processing ({phase2_config.max_concurrent_workers} workers)")
            if phase2_config.storage_external_t9:
                logger.info(f"  ✓ External storage: {phase2_config.storage_external_t9}")
            if phase2_config.model_cache_enabled:
                logger.info("  ✓ Model caching (batch optimization)")
            if phase2_config.depth_map_cache_enabled:
                logger.info("  ✓ Depth map caching")
    else:
        cfg.phase2 = None

    if args.service:
        # Defer imports so batch-only users don't need server deps installed.
        from .service import run_service
        run_service(cfg, host=args.host, port=int(args.port), logger=logger)
        return

    # Phase 1 Stability: Pre-flight validation
    if cfg.orchestrator.pre_flight_check and args.input:
        from .preflight import PreFlightValidator
        
        validator = PreFlightValidator(logger=logger)
        report = validator.validate_all(
            input_path=Path(args.input),
            depth_dir=cfg.depth_dir,
            device=cfg.device,
            upscale=cfg.upscale
        )
        validator.log_report(report)
        
        if not report.passed:
            logger.error("Pre-flight validation failed. Use --skip-pre-flight to bypass.")
            return

    pipe = LuxPipelineV2(cfg, logger=logger)

    if args.input:
        rep = pipe.process_one(Path(args.input))
        logger.info(f"Done: {rep.get('status')} | {Path(args.input).name}")
    else:
        pipe.process_directory()


if __name__ == "__main__":
    main()
