"""Process Orchestrator for Lux Depth V2 Pipeline.

Provides fault-tolerant batch processing with:
- Subprocess isolation (one failure doesn't kill batch)
- Task queue management with priority support
- Resource allocation per task
- Graceful shutdown with cleanup
- Progress tracking and reporting
- Phase 2: Parallel processing with 2-4 concurrent workers
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import signal
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .logging_utils import setup_logging


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class TaskConfig:
    """Configuration for a single processing task."""
    task_id: str
    input_path: Path
    output_dir: Path
    depth_path: Optional[Path] = None
    preset: str = "photo_realistic"
    device: str = "auto"
    upscale: int = 4
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    status: TaskStatus
    input_path: Path
    output_path: Optional[Path] = None
    error: Optional[str] = None
    elapsed_time: float = 0.0
    retry_count: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)


class ProcessOrchestrator:
    """Orchestrates batch processing with fault isolation and resource management.
    
    Features:
    - One task per worker process (fault isolation)
    - Task queue with priority support
    - Graceful shutdown handling
    - Progress tracking and callbacks
    - Resource budget enforcement
    
    Args:
        max_workers: Maximum concurrent workers (default: 1 for GPU safety)
        memory_budget_gb: Memory budget per task in GB (None = no limit)
        device: Device assignment ('auto', 'cuda', 'cpu', 'mps')
        logger: Optional logger instance
    """
    
    def __init__(
        self,
        max_workers: int = 1,
        memory_budget_gb: Optional[float] = None,
        device: str = "auto",
        logger=None
    ):
        self.max_workers = max_workers
        self.memory_budget_gb = memory_budget_gb
        self.device = device
        self.logger = logger or setup_logging("INFO")
        
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.results: List[TaskResult] = []
        self.active_workers: Dict[str, mp.Process] = {}
        self.shutdown_requested = False
        
        # Statistics
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.cancelled_tasks = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info(
            f"ProcessOrchestrator initialized | workers={max_workers} "
            f"memory_budget={memory_budget_gb}GB device={device}"
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def submit_task(self, task_config: TaskConfig, priority: int = 0) -> str:
        """Submit a task to the processing queue.
        
        Args:
            task_config: Task configuration
            priority: Task priority (lower = higher priority)
            
        Returns:
            Task ID
        """
        # Priority queue uses (priority, counter, item) to maintain order
        self.task_queue.put((priority, self.total_tasks, task_config))
        self.total_tasks += 1
        
        self.logger.debug(
            f"Task submitted | id={task_config.task_id} priority={priority} "
            f"input={task_config.input_path.name}"
        )
        
        return task_config.task_id
    
    def process_batch(
        self,
        tasks: List[TaskConfig],
        progress_callback: Optional[Callable[[TaskResult], None]] = None
    ) -> List[TaskResult]:
        """Process a batch of tasks with fault isolation.
        
        Args:
            tasks: List of task configurations
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of task results
        """
        # Submit all tasks
        for i, task in enumerate(tasks):
            self.submit_task(task, priority=task.priority)
        
        self.logger.info(f"Processing batch | total_tasks={len(tasks)} workers={self.max_workers}")
        
        start_time = time.time()
        
        # Process queue until empty or shutdown requested
        while not self.task_queue.empty() and not self.shutdown_requested:
            # Wait for available worker slot
            while len(self.active_workers) >= self.max_workers and not self.shutdown_requested:
                self._reap_finished_workers()
                time.sleep(0.1)
            
            if self.shutdown_requested:
                break
            
            try:
                # Get next task (with timeout to check for shutdown)
                priority, counter, task_config = self.task_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # Start worker process for this task
            self._start_worker(task_config, progress_callback)
        
        # Wait for all active workers to complete
        self.logger.info("Waiting for active workers to complete...")
        while self.active_workers and not self.shutdown_requested:
            self._reap_finished_workers()
            time.sleep(0.5)
        
        elapsed = time.time() - start_time
        
        # Log summary
        self.logger.info(
            f"Batch processing complete | "
            f"total={self.total_tasks} "
            f"completed={self.completed_tasks} "
            f"failed={self.failed_tasks} "
            f"cancelled={self.cancelled_tasks} "
            f"elapsed={elapsed:.1f}s"
        )
        
        return self.results
    
    def _start_worker(
        self,
        task_config: TaskConfig,
        progress_callback: Optional[Callable[[TaskResult], None]]
    ):
        """Start a worker process for a task."""
        # Create process for task execution
        process = mp.Process(
            target=_worker_task,
            args=(task_config, self.device, self.logger),
            name=f"worker-{task_config.task_id}"
        )
        process.start()
        
        self.active_workers[task_config.task_id] = process
        
        self.logger.info(
            f"Worker started | task_id={task_config.task_id} pid={process.pid} "
            f"input={task_config.input_path.name}"
        )
    
    def _reap_finished_workers(self):
        """Check for and clean up finished worker processes."""
        finished = []
        
        for task_id, process in self.active_workers.items():
            if not process.is_alive():
                finished.append(task_id)
                
                # Get exit code
                process.join(timeout=1.0)
                exit_code = process.exitcode
                
                if exit_code == 0:
                    self.completed_tasks += 1
                    status = TaskStatus.SUCCESS
                    self.logger.info(f"Worker completed | task_id={task_id}")
                else:
                    self.failed_tasks += 1
                    status = TaskStatus.FAILED
                    self.logger.error(
                        f"Worker failed | task_id={task_id} exit_code={exit_code}"
                    )
                
                # Create result (detailed result would come from checkpoint)
                result = TaskResult(
                    task_id=task_id,
                    status=status,
                    input_path=Path("unknown"),  # Would be loaded from checkpoint
                )
                self.results.append(result)
        
        # Remove finished workers
        for task_id in finished:
            del self.active_workers[task_id]
    
    def shutdown(self, graceful: bool = True, timeout: float = 30.0):
        """Shutdown the orchestrator.
        
        Args:
            graceful: If True, wait for current tasks to complete
            timeout: Maximum time to wait for graceful shutdown
        """
        self.logger.info(f"Shutdown initiated | graceful={graceful} timeout={timeout}s")
        
        self.shutdown_requested = True
        
        if graceful:
            # Wait for active workers with timeout
            start = time.time()
            while self.active_workers and (time.time() - start) < timeout:
                self._reap_finished_workers()
                time.sleep(0.5)
        
        # Terminate remaining workers
        for task_id, process in self.active_workers.items():
            if process.is_alive():
                self.logger.warning(f"Terminating worker | task_id={task_id}")
                process.terminate()
                process.join(timeout=5.0)
                
                if process.is_alive():
                    self.logger.error(f"Force killing worker | task_id={task_id}")
                    process.kill()
                
                self.cancelled_tasks += 1
        
        self.active_workers.clear()
        
        self.logger.info("Shutdown complete")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress statistics.
        
        Returns:
            Dictionary with progress metrics
        """
        return {
            "total_tasks": self.total_tasks,
            "completed": self.completed_tasks,
            "failed": self.failed_tasks,
            "cancelled": self.cancelled_tasks,
            "active": len(self.active_workers),
            "queued": self.task_queue.qsize(),
            "success_rate": (
                self.completed_tasks / max(1, self.completed_tasks + self.failed_tasks)
            ),
        }


def _worker_task(task_config: TaskConfig, device: str, logger):
    """Worker function to execute a single task in subprocess.
    
    This runs in a separate process for fault isolation.
    """
    try:
        # Import inside worker to avoid serialization issues
        from .pipeline import LuxPipelineV2
        from .config import PipelineConfig, Preset
        
        logger.info(f"Worker executing | task_id={task_config.task_id}")
        
        # Build pipeline config
        cfg = PipelineConfig(
            input_dir=None,  # Single file mode
            depth_dir=task_config.depth_path.parent if task_config.depth_path else None,
            output_dir=task_config.output_dir,
            preset=Preset(task_config.preset),
            device=device,
            upscale=task_config.upscale,
        )
        
        # Create pipeline
        pipe = LuxPipelineV2(cfg, logger=logger)
        
        # Process single image
        result = pipe.process_one(task_config.input_path)
        
        logger.info(
            f"Worker complete | task_id={task_config.task_id} status={result.get('status')}"
        )
        
        return 0  # Success
        
    except Exception as e:
        logger.error(f"Worker exception | task_id={task_config.task_id} error={e}")
        return 1  # Failure


@dataclass
class WorkerState:
    """Track state of a parallel worker."""
    worker_id: int
    task_id: str
    process: mp.Process
    start_time: float
    memory_budget_gb: float


@dataclass
class ParallelCapacityCheck:
    """Result of parallel capacity checking."""
    can_support_requested: bool
    recommended_workers: int
    memory_per_worker_gb: float
    total_memory_required_gb: float
    available_memory_gb: float
    available_disk_gb: float
    warnings: List[str] = field(default_factory=list)


class ParallelOrchestrator(ProcessOrchestrator):
    """Enhanced orchestrator with parallel processing support (Phase 2).
    
    Extends ProcessOrchestrator with:
    - 2-4 concurrent workers (configurable)
    - Resource-aware worker scheduling
    - Memory budget per worker
    - Dynamic worker allocation based on available resources
    
    Features:
    - Backward compatible (enable_parallel=False for Phase 1 mode)
    - Resource monitoring before starting workers
    - Graceful degradation when resources constrained
    - Progress tracking across parallel workers
    
    Args:
        max_workers: Maximum concurrent workers (1-4)
        memory_budget_per_worker: GB of memory per worker (default: 25GB)
        enable_parallel: Enable parallel processing (default: False for Phase 1 compat)
        resource_monitor: Optional ResourceMonitor instance
        device: Device assignment per worker
        logger: Optional logger instance
    """
    
    def __init__(
        self,
        max_workers: int = 1,
        memory_budget_per_worker: float = 25.0,
        enable_parallel: bool = False,
        resource_monitor=None,
        device: str = "auto",
        logger=None
    ):
        super().__init__(
            max_workers=1 if not enable_parallel else max_workers,
            memory_budget_gb=memory_budget_per_worker,
            device=device,
            logger=logger
        )
        
        self.enable_parallel = enable_parallel
        self.max_parallel_workers = max_workers if enable_parallel else 1
        self.memory_budget_per_worker = memory_budget_per_worker
        
        # Import resource monitor only if needed
        self.resource_monitor = resource_monitor
        if self.resource_monitor is None and enable_parallel:
            try:
                from .resource_monitor import ResourceMonitor
                self.resource_monitor = ResourceMonitor()
            except ImportError:
                self.logger.warning("ResourceMonitor not available, disabling resource checks")
                self.resource_monitor = None
        
        self.worker_states: List[WorkerState] = []
        self.next_worker_id = 0
        
        if enable_parallel:
            self.logger.info(
                f"ParallelOrchestrator initialized | "
                f"max_workers={max_workers} "
                f"memory_per_worker={memory_budget_per_worker}GB "
                f"parallel_enabled={enable_parallel}"
            )
    
    def get_available_worker_slots(self) -> int:
        """Check how many workers can run concurrently based on resources.
        
        Returns:
            Number of available worker slots (0 to max_workers)
        """
        if not self.enable_parallel:
            return 1 if len(self.worker_states) == 0 else 0
        
        # Count active workers
        active_count = len(self.worker_states)
        if active_count >= self.max_parallel_workers:
            return 0
        
        # Check memory availability if resource monitor available
        if self.resource_monitor:
            try:
                # Get available memory
                metrics = self.resource_monitor.get_metrics()
                available_memory_gb = metrics.ram_total_gb - metrics.ram_used_gb
                
                # Calculate how many workers can fit
                possible_workers = int(available_memory_gb / self.memory_budget_per_worker)
                available_slots = min(
                    possible_workers - active_count,
                    self.max_parallel_workers - active_count
                )
                
                return max(0, available_slots)
            except Exception as e:
                self.logger.warning(f"Resource check failed: {e}, allowing slots")
                # Fallback to simple counting
                return self.max_parallel_workers - active_count
        
        # No resource monitor, use simple counting
        return self.max_parallel_workers - active_count
    
    def check_parallel_capacity(
        self,
        required_workers: int
    ) -> ParallelCapacityCheck:
        """Check if system can support requested parallel workers.
        
        Args:
            required_workers: Number of workers requested
            
        Returns:
            ParallelCapacityCheck with recommendations
        """
        if not self.resource_monitor:
            # No monitoring, optimistically allow requested workers
            return ParallelCapacityCheck(
                can_support_requested=True,
                recommended_workers=min(required_workers, self.max_parallel_workers, 4),
                memory_per_worker_gb=self.memory_budget_per_worker,
                total_memory_required_gb=required_workers * self.memory_budget_per_worker,
                available_memory_gb=0.0,
                available_disk_gb=0.0,
                warnings=["Resource monitoring not available"]
            )
        
        try:
            metrics = self.resource_monitor.get_metrics()
            available_memory_gb = metrics.ram_total_gb - metrics.ram_used_gb
            
            # Calculate maximum possible workers
            max_workers_by_memory = int(available_memory_gb / self.memory_budget_per_worker)
            recommended_workers = min(required_workers, max_workers_by_memory, 4)
            
            # Check disk space
            available_disk_gb = 0.0
            for path, disk_metrics in metrics.disk_metrics.items():
                available_disk_gb = max(available_disk_gb, disk_metrics.get('available_gb', 0))
            
            # Generate warnings
            warnings = []
            if recommended_workers < required_workers:
                warnings.append(
                    f"Insufficient memory for {required_workers} workers. "
                    f"Recommended: {recommended_workers}"
                )
            if available_disk_gb < 20.0:
                warnings.append(
                    f"Low disk space: {available_disk_gb:.1f}GB available"
                )
            
            return ParallelCapacityCheck(
                can_support_requested=recommended_workers >= required_workers,
                recommended_workers=recommended_workers,
                memory_per_worker_gb=self.memory_budget_per_worker,
                total_memory_required_gb=recommended_workers * self.memory_budget_per_worker,
                available_memory_gb=available_memory_gb,
                available_disk_gb=available_disk_gb,
                warnings=warnings
            )
        except Exception as e:
            self.logger.error(f"Capacity check failed: {e}")
            return ParallelCapacityCheck(
                can_support_requested=False,
                recommended_workers=1,
                memory_per_worker_gb=self.memory_budget_per_worker,
                total_memory_required_gb=0.0,
                available_memory_gb=0.0,
                available_disk_gb=0.0,
                warnings=[f"Capacity check error: {e}"]
            )
    
    def process_batch_with_parallelism(
        self,
        tasks: List[TaskConfig],
        max_concurrent: Optional[int] = None,
        progress_callback: Optional[Callable[[TaskResult], None]] = None
    ) -> List[TaskResult]:
        """Process batch with parallel workers.
        
        Args:
            tasks: List of task configurations
            max_concurrent: Override max workers (None = use configured max)
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of task results
        """
        if not self.enable_parallel or self.max_parallel_workers == 1:
            # Fallback to sequential Phase 1 processing
            self.logger.info("Parallel disabled, using sequential processing")
            return self.process_batch(tasks, progress_callback)
        
        # Override max workers if specified
        original_max = self.max_parallel_workers
        if max_concurrent:
            self.max_parallel_workers = min(max_concurrent, 4)
        
        try:
            # Check capacity before starting
            capacity = self.check_parallel_capacity(self.max_parallel_workers)
            if capacity.warnings:
                for warning in capacity.warnings:
                    self.logger.warning(f"Capacity check: {warning}")
            
            self.logger.info(
                f"Starting parallel batch | "
                f"total_tasks={len(tasks)} "
                f"max_workers={self.max_parallel_workers} "
                f"available_memory={capacity.available_memory_gb:.1f}GB"
            )
            
            # Submit all tasks to queue
            for task in tasks:
                self.submit_task(task, priority=task.priority)
            
            start_time = time.time()
            
            # Process queue with parallel workers
            while not self.task_queue.empty() and not self.shutdown_requested:
                # Start new workers if slots available
                available_slots = self.get_available_worker_slots()
                
                while available_slots > 0 and not self.task_queue.empty():
                    try:
                        # Get next task
                        priority, counter, task_config = self.task_queue.get(timeout=0.1)
                        
                        # Start worker
                        self._start_parallel_worker(task_config, progress_callback)
                        available_slots -= 1
                        
                    except queue.Empty:
                        break
                
                # Reap finished workers
                self._reap_finished_workers()
                time.sleep(0.1)
            
            # Wait for all workers to complete
            self.logger.info("Waiting for parallel workers to complete...")
            while self.worker_states and not self.shutdown_requested:
                self._reap_finished_workers()
                time.sleep(0.5)
            
            elapsed = time.time() - start_time
            
            self.logger.info(
                f"Parallel batch complete | "
                f"total={self.total_tasks} "
                f"completed={self.completed_tasks} "
                f"failed={self.failed_tasks} "
                f"elapsed={elapsed:.1f}s"
            )
            
            return self.results
            
        finally:
            # Restore original max workers
            self.max_parallel_workers = original_max
    
    def _start_parallel_worker(
        self,
        task_config: TaskConfig,
        progress_callback: Optional[Callable[[TaskResult], None]]
    ):
        """Start a parallel worker process."""
        worker_id = self.next_worker_id
        self.next_worker_id += 1
        
        # Create worker process
        process = mp.Process(
            target=_worker_task,
            args=(task_config, self.device, self.logger),
            name=f"parallel-worker-{worker_id}"
        )
        process.start()
        
        # Track worker state
        worker_state = WorkerState(
            worker_id=worker_id,
            task_id=task_config.task_id,
            process=process,
            start_time=time.time(),
            memory_budget_gb=self.memory_budget_per_worker
        )
        self.worker_states.append(worker_state)
        self.active_workers[task_config.task_id] = process
        
        self.logger.info(
            f"Parallel worker started | "
            f"worker_id={worker_id} "
            f"task_id={task_config.task_id} "
            f"pid={process.pid} "
            f"active_workers={len(self.worker_states)}"
        )
    
    def _reap_finished_workers(self):
        """Check for and clean up finished worker processes."""
        finished = []
        
        for worker_state in self.worker_states:
            if not worker_state.process.is_alive():
                finished.append(worker_state)
                
                # Get exit code
                worker_state.process.join(timeout=1.0)
                exit_code = worker_state.process.exitcode
                
                elapsed = time.time() - worker_state.start_time
                
                if exit_code == 0:
                    self.completed_tasks += 1
                    status = TaskStatus.SUCCESS
                    self.logger.info(
                        f"Parallel worker completed | "
                        f"worker_id={worker_state.worker_id} "
                        f"task_id={worker_state.task_id} "
                        f"elapsed={elapsed:.1f}s"
                    )
                else:
                    self.failed_tasks += 1
                    status = TaskStatus.FAILED
                    self.logger.error(
                        f"Parallel worker failed | "
                        f"worker_id={worker_state.worker_id} "
                        f"task_id={worker_state.task_id} "
                        f"exit_code={exit_code}"
                    )
                
                # Create result
                result = TaskResult(
                    task_id=worker_state.task_id,
                    status=status,
                    input_path=Path("unknown"),
                    elapsed_time=elapsed
                )
                self.results.append(result)
        
        # Remove finished workers
        for worker_state in finished:
            self.worker_states.remove(worker_state)
            if worker_state.task_id in self.active_workers:
                del self.active_workers[worker_state.task_id]
