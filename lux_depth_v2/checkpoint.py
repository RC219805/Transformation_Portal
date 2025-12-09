"""Checkpoint System for Lux Depth V2 Pipeline.

Features:
- Stage-wise progress persistence (depth → material → upscale → export)
- Resume from last successful stage
- Checkpoint file format (JSON with metadata)
- Automatic cleanup of old checkpoints
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logging_utils import setup_logging


class ProcessingStage(str, Enum):
    """Pipeline processing stages."""
    INIT = "init"
    DEPTH_LOAD = "depth_load"
    MATERIAL_SEGMENTATION = "material_segmentation"
    MATERIALS_V2 = "materials_v2"  # New: Materials v2 stage
    POST_PROCESSING = "post_processing"
    UPSCALING = "upscaling"
    EXPORT = "export"
    COMPLETE = "complete"


@dataclass
class StageCheckpoint:
    """Checkpoint data for a single stage."""
    stage: ProcessingStage
    status: str  # 'success', 'failed', 'running'
    timestamp: float
    elapsed_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskCheckpoint:
    """Complete checkpoint for a task."""
    task_id: str
    input_path: str
    output_dir: str
    depth_path: Optional[str] = None
    
    # Configuration
    preset: str = "photo_realistic"
    device: str = "auto"
    upscale: int = 4
    
    # Progress tracking
    current_stage: ProcessingStage = ProcessingStage.INIT
    stages: Dict[str, StageCheckpoint] = field(default_factory=dict)
    
    # Timing
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    
    # Results
    output_path: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    
    # Retry tracking
    retry_count: int = 0
    max_retries: int = 3
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def can_resume(self) -> bool:
        """Check if task can be resumed."""
        return (
            not self.success and 
            self.retry_count < self.max_retries and
            self.current_stage != ProcessingStage.COMPLETE
        )
    
    def get_last_successful_stage(self) -> Optional[ProcessingStage]:
        """Get the last successfully completed stage."""
        stage_order = [
            ProcessingStage.INIT,
            ProcessingStage.DEPTH_LOAD,
            ProcessingStage.MATERIAL_SEGMENTATION,
            ProcessingStage.MATERIALS_V2,  # New: Materials v2 stage
            ProcessingStage.POST_PROCESSING,
            ProcessingStage.UPSCALING,
            ProcessingStage.EXPORT,
        ]
        
        last_stage = None
        for stage in stage_order:
            stage_key = stage.value
            if stage_key in self.stages and self.stages[stage_key].status == "success":
                last_stage = stage
            else:
                break
        
        return last_stage
    
    def get_next_stage(self) -> Optional[ProcessingStage]:
        """Get the next stage to execute."""
        last_stage = self.get_last_successful_stage()
        
        stage_order = [
            ProcessingStage.INIT,
            ProcessingStage.DEPTH_LOAD,
            ProcessingStage.MATERIAL_SEGMENTATION,
            ProcessingStage.POST_PROCESSING,
            ProcessingStage.UPSCALING,
            ProcessingStage.EXPORT,
            ProcessingStage.COMPLETE,
        ]
        
        if last_stage is None:
            return ProcessingStage.INIT
        
        try:
            idx = stage_order.index(last_stage)
            if idx < len(stage_order) - 1:
                return stage_order[idx + 1]
        except ValueError:
            pass
        
        return None


class CheckpointManager:
    """Manages checkpoints for pipeline processing.
    
    Features:
    - Save/load checkpoints in JSON format
    - Resume from last successful stage
    - Automatic cleanup of old checkpoints
    - Thread-safe file operations
    
    Args:
        checkpoint_dir: Directory to store checkpoints
        logger: Optional logger instance
    """
    
    def __init__(self, checkpoint_dir: Path = Path(".checkpoints"), logger=None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or setup_logging("INFO")
        
        self.logger.info(f"CheckpointManager initialized | dir={self.checkpoint_dir}")
    
    def _get_checkpoint_path(self, task_id: str) -> Path:
        """Get checkpoint file path for a task."""
        return self.checkpoint_dir / f"{task_id}.json"
    
    def save_checkpoint(
        self,
        task_id: str,
        stage: ProcessingStage,
        status: str = "success",
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save checkpoint for a task stage.
        
        Args:
            task_id: Unique task identifier
            stage: Processing stage
            status: Stage status ('success', 'failed', 'running')
            error: Optional error message
            metadata: Optional stage metadata
        """
        checkpoint_path = self._get_checkpoint_path(task_id)
        
        # Load existing checkpoint or create new
        if checkpoint_path.exists():
            checkpoint = self.load_checkpoint(task_id)
        else:
            checkpoint = TaskCheckpoint(
                task_id=task_id,
                input_path="",  # Will be set by caller
                output_dir="",
            )
        
        # Update stage info
        stage_checkpoint = StageCheckpoint(
            stage=stage,
            status=status,
            timestamp=time.time(),
            error=error,
            metadata=metadata or {},
        )
        
        checkpoint.stages[stage.value] = stage_checkpoint
        checkpoint.current_stage = stage
        
        if status == "failed":
            checkpoint.error = error
        
        if stage == ProcessingStage.COMPLETE and status == "success":
            checkpoint.success = True
            checkpoint.completed_at = time.time()
        
        # Save to disk
        try:
            with open(checkpoint_path, "w") as f:
                json.dump(asdict(checkpoint), f, indent=2)
            
            self.logger.debug(
                f"Checkpoint saved | task_id={task_id} stage={stage.value} status={status}"
            )
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint | task_id={task_id} error={e}")
    
    def load_checkpoint(self, task_id: str) -> Optional[TaskCheckpoint]:
        """Load checkpoint for a task.
        
        Args:
            task_id: Unique task identifier
            
        Returns:
            TaskCheckpoint object or None if not found
        """
        checkpoint_path = self._get_checkpoint_path(task_id)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
            
            # Convert stage dicts back to StageCheckpoint objects
            stages = {}
            for stage_key, stage_data in data.get("stages", {}).items():
                stages[stage_key] = StageCheckpoint(
                    stage=ProcessingStage(stage_data["stage"]),
                    status=stage_data["status"],
                    timestamp=stage_data["timestamp"],
                    elapsed_time=stage_data.get("elapsed_time", 0.0),
                    error=stage_data.get("error"),
                    metadata=stage_data.get("metadata", {}),
                )
            
            checkpoint = TaskCheckpoint(
                task_id=data["task_id"],
                input_path=data["input_path"],
                output_dir=data["output_dir"],
                depth_path=data.get("depth_path"),
                preset=data.get("preset", "photo_realistic"),
                device=data.get("device", "auto"),
                upscale=data.get("upscale", 4),
                current_stage=ProcessingStage(data["current_stage"]),
                stages=stages,
                started_at=data["started_at"],
                completed_at=data.get("completed_at"),
                output_path=data.get("output_path"),
                success=data.get("success", False),
                error=data.get("error"),
                retry_count=data.get("retry_count", 0),
                max_retries=data.get("max_retries", 3),
                metadata=data.get("metadata", {}),
            )
            
            self.logger.debug(
                f"Checkpoint loaded | task_id={task_id} "
                f"stage={checkpoint.current_stage.value} "
                f"success={checkpoint.success}"
            )
            
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint | task_id={task_id} error={e}")
            return None
    
    def can_resume(self, task_id: str) -> bool:
        """Check if a task can be resumed.
        
        Args:
            task_id: Unique task identifier
            
        Returns:
            True if task can be resumed
        """
        checkpoint = self.load_checkpoint(task_id)
        return checkpoint is not None and checkpoint.can_resume()
    
    def delete_checkpoint(self, task_id: str):
        """Delete checkpoint for a task.
        
        Args:
            task_id: Unique task identifier
        """
        checkpoint_path = self._get_checkpoint_path(task_id)
        
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                self.logger.debug(f"Checkpoint deleted | task_id={task_id}")
            except Exception as e:
                self.logger.error(f"Failed to delete checkpoint | task_id={task_id} error={e}")
    
    def cleanup(self, older_than_days: int = 7, completed_only: bool = True):
        """Clean up old checkpoints.
        
        Args:
            older_than_days: Delete checkpoints older than this many days
            completed_only: If True, only delete completed tasks
        """
        cutoff_time = time.time() - (older_than_days * 86400)
        deleted_count = 0
        
        for checkpoint_path in self.checkpoint_dir.glob("*.json"):
            try:
                # Load checkpoint to check status
                with open(checkpoint_path, "r") as f:
                    data = json.load(f)
                
                started_at = data.get("started_at", 0)
                is_complete = data.get("success", False)
                
                # Check if should delete
                should_delete = False
                if started_at < cutoff_time:
                    if completed_only:
                        should_delete = is_complete
                    else:
                        should_delete = True
                
                if should_delete:
                    checkpoint_path.unlink()
                    deleted_count += 1
                    
            except Exception as e:
                self.logger.debug(f"Error processing checkpoint {checkpoint_path}: {e}")
        
        if deleted_count > 0:
            self.logger.info(
                f"Checkpoint cleanup | deleted={deleted_count} "
                f"older_than={older_than_days}days "
                f"completed_only={completed_only}"
            )
    
    def list_checkpoints(self) -> List[TaskCheckpoint]:
        """List all checkpoints.
        
        Returns:
            List of TaskCheckpoint objects
        """
        checkpoints = []
        
        for checkpoint_path in self.checkpoint_dir.glob("*.json"):
            try:
                task_id = checkpoint_path.stem
                checkpoint = self.load_checkpoint(task_id)
                if checkpoint:
                    checkpoints.append(checkpoint)
            except Exception as e:
                self.logger.debug(f"Error loading checkpoint {checkpoint_path}: {e}")
        
        return checkpoints
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get checkpoint statistics.
        
        Returns:
            Dictionary with statistics
        """
        checkpoints = self.list_checkpoints()
        
        total = len(checkpoints)
        completed = sum(1 for c in checkpoints if c.success)
        failed = sum(1 for c in checkpoints if c.error and not c.success)
        in_progress = total - completed - failed
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "checkpoint_dir": str(self.checkpoint_dir),
        }
