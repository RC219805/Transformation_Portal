"""Real-time streaming and progress tracking for Transformation Portal.

Provides live progress updates, streaming results, checkpoint/resume
capabilities, and async pipeline infrastructure for high-throughput
image processing.

Features:
- Progress tracking with tqdm integration
- Checkpoint/resume for long-running operations
- Async/await pipeline stages with backpressure handling
- Queue-based execution with worker pool management
- Streaming I/O to reduce memory footprint by 50%

Performance targets:
- Sequential (100 4K images): ~6.9 hours
- With async pipeline: ~1.5-2 hours (3-5x faster)

Example - Basic streaming:
    >>> from transformation_portal.streaming import ProgressTracker, StreamingProcessor
    >>>
    >>> tracker = ProgressTracker(total=100)
    >>> for i in range(100):
    ...     result = process_item(i)
    ...     tracker.update(1, message=f"Processed {i}")

Example - Async pipeline:
    >>> from transformation_portal.streaming import (
    ...     AsyncPipeline, ImageLoadStage, DepthEstimationStage, ImageSaveStage
    ... )
    >>>
    >>> async def process():
    ...     pipeline = AsyncPipeline(max_queue_size=10)
    ...     pipeline.add_stage(ImageLoadStage(max_concurrent=4))
    ...     pipeline.add_stage(DepthEstimationStage())
    ...     pipeline.add_stage(ImageSaveStage(output_dir="./output"))
    ...
    ...     async with pipeline:
    ...         async for result in pipeline.process_batch(image_paths):
    ...             print(f"Processed: {result.data.path}")
"""

from .checkpoint import (
    Checkpoint,
    CheckpointManager,
    checkpoint,
)
from .progress import (
    MultiProgress,
    ProgressBar,
    ProgressTracker,
    create_progress,
)
from .streaming import (
    StreamingProcessor,
    batch_stream,
    stream_results,
)

# Async pipeline components
from .async_pipeline import (
    AsyncBatchProcessor,
    AsyncPipeline,
    AsyncStage,
    AsyncStageProtocol,
    BackpressureQueue,
    DeviceType,
    PipelineMetrics,
    StageResult,
    StageStatus,
    StreamingImageLoader,
    WorkerPool,
    WorkItem,
    run_async_pipeline,
)

# Concrete pipeline stages
from .stages import (
    ColorGradingStage,
    DenoiseStage,
    DepthEstimationStage,
    ImageData,
    ImageLoadStage,
    ImageSaveStage,
    MaterialResponseStage,
    ResizeStage,
    create_luxury_pipeline_stages,
)

__all__ = [
    # Progress tracking
    'ProgressTracker',
    'ProgressBar',
    'MultiProgress',
    'create_progress',
    # Checkpoint/resume
    'Checkpoint',
    'CheckpointManager',
    'checkpoint',
    # Basic streaming
    'StreamingProcessor',
    'stream_results',
    'batch_stream',
    # Async pipeline infrastructure
    'AsyncPipeline',
    'AsyncBatchProcessor',
    'AsyncStage',
    'AsyncStageProtocol',
    'BackpressureQueue',
    'DeviceType',
    'PipelineMetrics',
    'StageResult',
    'StageStatus',
    'StreamingImageLoader',
    'WorkerPool',
    'WorkItem',
    'run_async_pipeline',
    # Concrete stages
    'ImageData',
    'ImageLoadStage',
    'ImageSaveStage',
    'DepthEstimationStage',
    'MaterialResponseStage',
    'ColorGradingStage',
    'ResizeStage',
    'DenoiseStage',
    'create_luxury_pipeline_stages',
]

__version__ = '1.1.0'
