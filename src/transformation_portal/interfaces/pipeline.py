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
        """
        Execute stage operation on input data.

        Args:
            data: Input data for the stage (type varies by stage implementation).
            context: Shared context dictionary for passing data between stages.

        Returns:
            Processed data to pass to the next stage.

        Raises:
            PipelineError: If stage execution fails.
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Return stage name for logging/debugging.

        Returns:
            str: Human-readable stage name.
        """
        pass


class Pipeline(ABC):
    """Base interface for multi-stage processing pipelines."""
    
    @abstractmethod
    def add_stage(self, stage: PipelineStage, name: Optional[str] = None) -> None:
        """
        Add a processing stage to the pipeline.

        Args:
            stage: PipelineStage instance to add.
            name: Optional name override for the stage (defaults to stage.get_name()).

        Raises:
            PipelineError: If the stage cannot be added.
        """
        pass
    
    @abstractmethod
    def execute(self, input_path: Path, output_path: Optional[Path] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute complete pipeline on input file.

        Args:
            input_path: Path to input file.
            output_path: Optional path for output file (None = auto-generate).
            **kwargs: Pipeline-specific parameters.

        Returns:
            Dictionary with execution results (e.g., output_path, metrics, timings).

        Raises:
            PipelineError: If pipeline execution fails.
        """
        pass
    
    @abstractmethod
    def get_stages(self) -> List[PipelineStage]:
        """
        Return list of pipeline stages in execution order.

        Returns:
            List[PipelineStage]: Ordered list of all stages in the pipeline.
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return complete pipeline configuration.

        Returns:
            Dictionary containing full pipeline configuration including all stages
            (must be JSON-serializable for reproducibility)
        """
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
        """
        Execute pipeline on multiple inputs in batch mode.

        Args:
            input_paths (List[Path]): List of input file paths to process.
            output_dir (Path): Directory for output files.
            parallel (bool, optional): Whether to process files in parallel. Defaults to True.
            max_workers (Optional[int], optional): Maximum number of parallel workers. None for auto-detect.
            **kwargs: Pipeline-specific parameters passed to each execution.

        Returns:
            List[Dict[str, Any]]: List of execution result dictionaries (one per input file).

        Raises:
            PipelineError: If batch execution fails.
        """
        pass
