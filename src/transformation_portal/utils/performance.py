#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performance monitoring and optimization utilities.

This module provides decorators and utilities for performance profiling,
caching, and optimization of image/video processing pipelines.
"""
import functools
import time
import logging
from typing import Any, Callable, Optional, TypeVar, cast

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


def timing_decorator(func: F) -> F:
    """Decorator to measure and log function execution time.

    Args:
        func: Function to time

    Returns:
        Wrapped function that logs execution time

    Example:
        @timing_decorator
        def process_image(path):
            # ... processing ...
            return result
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    return cast(F, wrapper)


def profile_memory(func: F) -> F:
    """Decorator to profile memory usage (requires memory-profiler).

    Args:
        func: Function to profile

    Returns:
        Wrapped function that logs memory usage if profiler available

    Example:
        @profile_memory
        def batch_process(images):
            # ... processing ...
            return results
    """
    try:
        from memory_profiler import memory_usage
        has_profiler = True
    except ImportError:
        has_profiler = False

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not has_profiler:
            return func(*args, **kwargs)

        mem_before = memory_usage()[0]
        result = func(*args, **kwargs)
        mem_after = memory_usage()[0]
        mem_delta = mem_after - mem_before

        logger.info(
            f"{func.__name__} memory: {mem_delta:+.1f}MB "
            f"(peak: {mem_after:.1f}MB)"
        )
        return result
    return cast(F, wrapper)


def cache_result(maxsize: int = 128, typed: bool = False) -> Callable[[F], F]:
    """Enhanced LRU cache with logging support.

    This is a wrapper around functools.lru_cache that adds logging for
    cache hits/misses. Useful for performance-critical functions.

    Args:
        maxsize: Maximum cache size
        typed: If True, arguments of different types are cached separately

    Returns:
        Decorator function

    Example:
        @cache_result(maxsize=256)
        def estimate_depth(image_hash):
            # ... expensive computation ...
            return depth_map
    """
    def decorator(func: F) -> F:
        cached_func = functools.lru_cache(maxsize=maxsize, typed=typed)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache info before call
            info_before = cached_func.cache_info()
            result = cached_func(*args, **kwargs)
            info_after = cached_func.cache_info()

            # Log if this was a cache hit
            if info_after.hits > info_before.hits:
                logger.debug(
                    f"{func.__name__} cache hit "
                    f"(hits: {info_after.hits}, misses: {info_after.misses})"
                )

            return result

        # Expose cache_info and cache_clear
        wrapper.cache_info = cached_func.cache_info  # type: ignore
        wrapper.cache_clear = cached_func.cache_clear  # type: ignore

        return cast(F, wrapper)
    return decorator


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable[[F], F]:
    """Decorator to retry failed operations with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each attempt
        exceptions: Tuple of exception types to catch

    Returns:
        Decorator function

    Example:
        @retry_on_failure(max_attempts=3, delay=0.5)
        def download_model(url):
            # ... network operation ...
            return model
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay

            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise

                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1

        return cast(F, wrapper)
    return decorator


def make_batch_processor(
    func: F, batch_size: Optional[int] = None
) -> Callable[[list, Any], list]:
    """Create a batch-processing version of a single-item function.

    This function returns a new function that processes a list of items in batches,
    calling the original function on each item and returning a list of results.

    Args:
        func: Function that processes a single item.
        batch_size: Number of items per batch (None = process all at once).

    Returns:
        A function that takes a list of items and returns a list of results.

    Example:
        def process_image(image_path):
            # ... processing ...
            return result

        batch_process_images = make_batch_processor(process_image, batch_size=32)
        results = batch_process_images([path1, path2, path3])

    Note:
        This function generator is preferred over a decorator, as it does not
        violate the decorator pattern or change the original function's signature.
    """
    @functools.wraps(func)
    def batch_func(items, *args, **kwargs):
        if not items:
            return []
        if batch_size is None or len(items) <= batch_size:
            return [func(item, *args, **kwargs) for item in items]
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            logger.debug(
                f"Processing batch {i//batch_size + 1} "
                f"({len(batch)} items)"
            )
            batch_results = [func(item, *args, **kwargs) for item in batch]
            results.extend(batch_results)
        return results
    return batch_func


class PerformanceMonitor:
    """Context manager for monitoring performance of code blocks.

    Example:
        with PerformanceMonitor("batch_processing") as monitor:
            results = process_images(image_paths)

        print(f"Processed {monitor.item_count} items in {monitor.elapsed:.2f}s")
        print(f"Throughput: {monitor.throughput:.1f} items/sec")
    """

    def __init__(self, name: str, item_count: Optional[int] = None):
        """Initialize performance monitor.

        Args:
            name: Name of the operation being monitored
            item_count: Optional number of items being processed
        """
        self.name = name
        self.item_count = item_count
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        logger.info(f"Starting {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log results."""
        self.end_time = time.perf_counter()

        if exc_type is not None:
            logger.error(f"{self.name} failed after {self.elapsed:.2f}s")
            return False

        logger.info(f"{self.name} completed in {self.elapsed:.2f}s")
        if self.item_count:
            logger.info(f"Throughput: {self.throughput:.1f} items/sec")

        return False

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.perf_counter()
        return end - self.start_time

    @property
    def throughput(self) -> float:
        """Get throughput in items per second."""
        if self.item_count is None or self.elapsed == 0:
            return 0.0
        return self.item_count / self.elapsed


# Convenient aliases
timed = timing_decorator
cached = cache_result
retry = retry_on_failure
monitor = PerformanceMonitor
