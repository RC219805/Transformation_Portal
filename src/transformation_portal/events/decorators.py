"""Event tracking decorators."""

import functools
import time
import uuid
from typing import Callable, Any, Optional

from .store import Event, get_global_store


def event(event_type: str, include_result: bool = False):
    """Decorator to automatically track function calls as events.
    
    Args:
        event_type: Type of event (e.g., "image.enhanced")
        include_result: Include function result in event data
        
    Example:
        >>> @event("image.enhanced")
        ... def enhance_image(image_path, preset="default"):
        ...     return process(image_path, preset)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            event_id = str(uuid.uuid4())
            start_time = time.time()
            
            # Capture function arguments
            event_data = {
                'function': func.__name__,
                'args': [str(arg) for arg in args],  # Convert to strings for JSON
                'kwargs': {k: str(v) for k, v in kwargs.items()},
            }
            
            try:
                result = func(*args, **kwargs)
                
                # Add result if requested
                if include_result:
                    event_data['result'] = str(result)
                
                event_data['status'] = 'success'
                event_data['duration'] = time.time() - start_time
                
                # Create and store event
                evt = Event(
                    id=event_id,
                    type=event_type,
                    timestamp=start_time,
                    data=event_data
                )
                get_global_store().append(evt)
                
                return result
                
            except Exception as e:
                event_data['status'] = 'error'
                event_data['error'] = str(e)
                event_data['duration'] = time.time() - start_time
                
                # Create error event
                evt = Event(
                    id=event_id,
                    type=f"{event_type}.error",
                    timestamp=start_time,
                    data=event_data
                )
                get_global_store().append(evt)
                
                raise
        
        return wrapper
    
    return decorator


def tracked(correlation_id: Optional[str] = None):
    """Decorator to track related operations with correlation ID.
    
    Args:
        correlation_id: Correlation ID for grouping related events
        
    Example:
        >>> batch_id = str(uuid.uuid4())
        >>> 
        >>> @tracked(correlation_id=batch_id)
        ... def process_batch(items):
        ...     for item in items:
        ...         process_item(item)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            corr_id = correlation_id or str(uuid.uuid4())
            
            # Store correlation ID in function for nested calls
            wrapper._correlation_id = corr_id
            
            # Create start event
            start_event = Event(
                id=str(uuid.uuid4()),
                type=f"{func.__name__}.started",
                timestamp=time.time(),
                data={'function': func.__name__},
                correlation_id=corr_id
            )
            get_global_store().append(start_event)
            
            try:
                result = func(*args, **kwargs)
                
                # Create completion event
                end_event = Event(
                    id=str(uuid.uuid4()),
                    type=f"{func.__name__}.completed",
                    timestamp=time.time(),
                    data={'function': func.__name__, 'status': 'success'},
                    correlation_id=corr_id
                )
                get_global_store().append(end_event)
                
                return result
                
            except Exception as e:
                # Create error event
                error_event = Event(
                    id=str(uuid.uuid4()),
                    type=f"{func.__name__}.failed",
                    timestamp=time.time(),
                    data={'function': func.__name__, 'error': str(e)},
                    correlation_id=corr_id
                )
                get_global_store().append(error_event)
                
                raise
        
        return wrapper
    
    return decorator
