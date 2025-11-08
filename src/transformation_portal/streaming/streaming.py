"""Streaming processing utilities."""

import time
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional


class StreamingProcessor:
    """Process items in a streaming fashion without loading all into memory.
    
    Example:
        >>> processor = StreamingProcessor(
        ...     process_func=enhance_image,
        ...     batch_size=4
        ... )
        >>> 
        >>> for result in processor.stream(image_paths):
        ...     save_result(result)
    """
    
    def __init__(
        self,
        process_func: Callable[[Any], Any],
        batch_size: int = 1,
        max_workers: Optional[int] = None
    ):
        """Initialize streaming processor.
        
        Args:
            process_func: Function to process each item
            batch_size: Number of items to process in parallel
            max_workers: Maximum worker threads (defaults to batch_size)
        """
        self.process_func = process_func
        self.batch_size = batch_size
        self.max_workers = max_workers or batch_size
    
    def stream(self, items: Iterator[Any]) -> Iterator[Any]:
        """Stream process items.
        
        Args:
            items: Iterator of items to process
            
        Yields:
            Processed results
        """
        if self.batch_size == 1:
            # Sequential processing
            for item in items:
                yield self.process_func(item)
        else:
            # Batch processing with threads
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                batch = []
                
                for item in items:
                    batch.append(item)
                    
                    if len(batch) >= self.batch_size:
                        # Process batch
                        futures = [executor.submit(self.process_func, x) for x in batch]
                        for future in futures:
                            yield future.result()
                        batch = []
                
                # Process remaining items
                if batch:
                    futures = [executor.submit(self.process_func, x) for x in batch]
                    for future in futures:
                        yield future.result()


def stream_results(
    items: List[Any],
    process_func: Callable[[Any], Any],
    callback: Optional[Callable[[Any], None]] = None
) -> Iterator[Any]:
    """Stream processing results with optional callback.
    
    Args:
        items: Items to process
        process_func: Processing function
        callback: Optional callback for each result
        
    Yields:
        Processing results
        
    Example:
        >>> def on_result(result):
        ...     print(f"Completed: {result}")
        ...
        >>> for result in stream_results(images, enhance, on_result):
        ...     results.append(result)
    """
    for item in items:
        result = process_func(item)
        
        if callback:
            callback(result)
        
        yield result


def batch_stream(
    items: Iterator[Any],
    batch_size: int
) -> Iterator[List[Any]]:
    """Stream items in batches.
    
    Args:
        items: Iterator of items
        batch_size: Size of each batch
        
    Yields:
        Batches of items
        
    Example:
        >>> for batch in batch_stream(all_images, batch_size=32):
        ...     results = model.predict_batch(batch)
        ...     save_batch(results)
    """
    batch = []
    
    for item in items:
        batch.append(item)
        
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    # Yield remaining items
    if batch:
        yield batch


class RealTimeMonitor:
    """Monitor processing in real-time with statistics.
    
    Example:
        >>> monitor = RealTimeMonitor(window_size=100)
        >>> 
        >>> for item in items:
        ...     result = process(item)
        ...     monitor.record(item, result)
        ...     print(f"Throughput: {monitor.throughput:.1f} items/sec")
    """
    
    def __init__(self, window_size: int = 100):
        """Initialize monitor.
        
        Args:
            window_size: Number of recent items to track for statistics
        """
        self.window_size = window_size
        self.timestamps: List[float] = []
        self.processing_times: List[float] = []
        self._start_time = time.time()
    
    def record(self, processing_time: Optional[float] = None) -> None:
        """Record a processing event.
        
        Args:
            processing_time: Time taken to process (auto-calculated if None)
        """
        now = time.time()
        self.timestamps.append(now)
        
        if processing_time is not None:
            self.processing_times.append(processing_time)
        
        # Keep only recent window
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
            if self.processing_times:
                self.processing_times.pop(0)
    
    @property
    def throughput(self) -> float:
        """Current throughput (items per second).
        
        Returns:
            Items processed per second (windowed average)
        """
        if len(self.timestamps) < 2:
            return 0.0
        
        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span > 0:
            return len(self.timestamps) / time_span
        return 0.0
    
    @property
    def avg_processing_time(self) -> float:
        """Average processing time per item.
        
        Returns:
            Average time in seconds
        """
        if self.processing_times:
            return sum(self.processing_times) / len(self.processing_times)
        return 0.0
    
    @property
    def total_elapsed(self) -> float:
        """Total elapsed time since monitoring started.
        
        Returns:
            Elapsed time in seconds
        """
        return time.time() - self._start_time
    
    def get_stats(self) -> dict:
        """Get current statistics.
        
        Returns:
            Dictionary with monitoring statistics
        """
        return {
            'throughput': self.throughput,
            'avg_processing_time': self.avg_processing_time,
            'total_elapsed': self.total_elapsed,
            'items_processed': len(self.timestamps),
        }
