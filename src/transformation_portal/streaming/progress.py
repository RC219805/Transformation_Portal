"""Progress tracking with real-time updates."""

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Optional


@dataclass
class ProgressState:
    """Current state of a progress tracker."""
    current: int = 0
    total: Optional[int] = None
    message: str = ""
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    completed: bool = False
    
    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def percentage(self) -> Optional[float]:
        """Progress percentage (0-100)."""
        if self.total and self.total > 0:
            return (self.current / self.total) * 100
        return None
    
    @property
    def eta(self) -> Optional[float]:
        """Estimated time remaining in seconds."""
        if self.total and self.current > 0:
            elapsed = self.elapsed
            rate = self.current / elapsed
            remaining = self.total - self.current
            return remaining / rate
        return None


class ProgressTracker:
    """Track progress with real-time updates and callbacks.
    
    Provides thread-safe progress tracking with optional callbacks for
    live updates (e.g., WebSocket notifications, UI updates).
    
    Example:
        >>> tracker = ProgressTracker(total=100, update_interval=0.1)
        >>> tracker.on_update(lambda state: print(f"{state.percentage:.1f}%"))
        >>> 
        >>> for i in range(100):
        ...     process_item(i)
        ...     tracker.update(1)
    """
    
    def __init__(
        self,
        total: Optional[int] = None,
        description: str = "",
        update_interval: float = 0.1
    ):
        """Initialize progress tracker.
        
        Args:
            total: Total number of items (None for indeterminate)
            description: Description of the task
            update_interval: Minimum interval between callbacks (seconds)
        """
        self.state = ProgressState(total=total, message=description)
        self._update_interval = update_interval
        self._callbacks: list[Callable[[ProgressState], None]] = []
        self._lock = Lock()
    
    def update(self, n: int = 1, message: Optional[str] = None) -> None:
        """Update progress.
        
        Args:
            n: Number of items completed
            message: Optional status message
        """
        with self._lock:
            self.state.current += n
            if message is not None:
                self.state.message = message
            
            # Check if we should trigger callbacks
            now = time.time()
            if now - self.state.last_update >= self._update_interval:
                self.state.last_update = now
                self._trigger_callbacks()
            
            # Check completion
            if self.state.total and self.state.current >= self.state.total:
                self.complete()
    
    def complete(self, message: Optional[str] = None) -> None:
        """Mark progress as completed.
        
        Args:
            message: Final completion message
        """
        with self._lock:
            if message is not None:
                self.state.message = message
            self.state.completed = True
            self._trigger_callbacks()
    
    def on_update(self, callback: Callable[[ProgressState], None]) -> None:
        """Register callback for progress updates.
        
        Args:
            callback: Function called with ProgressState on updates
        """
        with self._lock:
            self._callbacks.append(callback)
    
    def _trigger_callbacks(self) -> None:
        """Trigger all registered callbacks (assumes lock held)."""
        for callback in self._callbacks:
            try:
                callback(self.state)
            except Exception as e:
                # Don't let callback errors break progress tracking
                print(f"Progress callback error: {e}")
    
    def get_state(self) -> ProgressState:
        """Get current progress state (thread-safe).
        
        Returns:
            Copy of current ProgressState
        """
        with self._lock:
            return ProgressState(
                current=self.state.current,
                total=self.state.total,
                message=self.state.message,
                start_time=self.state.start_time,
                last_update=self.state.last_update,
                completed=self.state.completed,
            )


class ProgressBar:
    """Terminal progress bar with rich formatting.
    
    Example:
        >>> with ProgressBar(total=100, description="Processing") as pbar:
        ...     for i in range(100):
        ...         process_item(i)
        ...         pbar.update(1)
    """
    
    def __init__(
        self,
        total: Optional[int] = None,
        description: str = "",
        width: int = 50
    ):
        """Initialize progress bar.
        
        Args:
            total: Total items
            description: Task description
            width: Width of progress bar in characters
        """
        self.tracker = ProgressTracker(total=total, description=description)
        self.width = width
        self._last_render = ""
    
    def update(self, n: int = 1, message: Optional[str] = None) -> None:
        """Update progress and render bar.
        
        Args:
            n: Number of items completed
            message: Status message
        """
        self.tracker.update(n, message)
        self._render()
    
    def _render(self) -> None:
        """Render progress bar to terminal."""
        state = self.tracker.get_state()
        
        if state.total:
            filled = int(self.width * state.current / state.total)
            bar = "█" * filled + "░" * (self.width - filled)
            percentage = state.percentage or 0
            
            line = f"\r{state.message}: |{bar}| {percentage:.1f}% "
            
            if state.eta:
                line += f"ETA: {state.eta:.1f}s"
        else:
            # Indeterminate progress
            line = f"\r{state.message}: {state.current} items ({state.elapsed:.1f}s)"
        
        # Only update if changed
        if line != self._last_render:
            print(line, end='', flush=True)
            self._last_render = line
        
        if state.completed:
            print()  # New line on completion
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if not self.tracker.state.completed:
            self.tracker.complete()
        self._render()


class MultiProgress:
    """Track multiple concurrent progress operations.
    
    Example:
        >>> multi = MultiProgress()
        >>> task1 = multi.add_task("Task 1", total=100)
        >>> task2 = multi.add_task("Task 2", total=50)
        >>> 
        >>> multi.update(task1, 10)
        >>> multi.update(task2, 5)
    """
    
    def __init__(self):
        """Initialize multi-progress tracker."""
        self.tasks: dict[str, ProgressTracker] = {}
        self._lock = Lock()
    
    def add_task(
        self,
        description: str,
        total: Optional[int] = None,
        task_id: Optional[str] = None
    ) -> str:
        """Add a new task to track.
        
        Args:
            description: Task description
            total: Total items
            task_id: Optional task ID (auto-generated if None)
            
        Returns:
            Task ID
        """
        if task_id is None:
            task_id = f"task_{len(self.tasks)}"
        
        with self._lock:
            self.tasks[task_id] = ProgressTracker(total=total, description=description)
        
        return task_id
    
    def update(self, task_id: str, n: int = 1, message: Optional[str] = None) -> None:
        """Update specific task progress.
        
        Args:
            task_id: Task identifier
            n: Number of items completed
            message: Status message
        """
        if task_id in self.tasks:
            self.tasks[task_id].update(n, message)
    
    def get_summary(self) -> dict[str, ProgressState]:
        """Get summary of all tasks.
        
        Returns:
            Dictionary mapping task IDs to ProgressState
        """
        with self._lock:
            return {
                task_id: tracker.get_state()
                for task_id, tracker in self.tasks.items()
            }


def create_progress(
    total: Optional[int] = None,
    description: str = "",
    use_rich: bool = True
) -> ProgressTracker:
    """Create a progress tracker (with optional rich formatting).
    
    Args:
        total: Total items
        description: Task description
        use_rich: Use rich terminal formatting if available
        
    Returns:
        ProgressTracker instance
    """
    if use_rich:
        try:
            # Try to use rich library if available
            from rich.progress import Progress

            # Could wrap rich.Progress here
            pass
        except ImportError:
            pass
    
    return ProgressTracker(total=total, description=description)
