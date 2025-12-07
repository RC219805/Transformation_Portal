"""
Pipeline Interface

Base contract for multi-stage processing pipelines.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class PipelineError(Exception):
    """Raised when pipeline execution fails."""
    pass


class PipelineStage(ABC):
    """Base interface for pipeline stages."""
    
    @abstractmethod
    def execute(self, data: Any, context: Dict[str, Any]) -> Any:
        """Execute stage operation."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return stage name for logging/debugging."""
        pass


class Pipeline(ABC):
    """Base interface for multi-stage processing pipelines."""
    
    @abstractmethod
    def add_stage(self, stage: PipelineStage, name: Optional[str] = None) -> None:
        """Add processing stage to pipeline."""
        pass
    
    @abstractmethod
    def execute(self, input_path: Path, output_path: Optional[Path] = None, **kwargs) -> Dict[str, Any]:
        """Execute complete pipeline."""
        pass
    
    @abstractmethod
    def get_stages(self) -> List[PipelineStage]:
        """Return list of pipeline stages in execution order."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return complete pipeline configuration."""
        pass


class BatchPipeline(Pipeline):
    """Extended pipeline interface for batch processing."""
    
    @abstractmethod
    def execute_batch(
        self,
        input_paths: List[Path],
        output_dir: Path,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute pipeline on multiple inputs."""
        pass
