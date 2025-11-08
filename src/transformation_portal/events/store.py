"""Event store for tracking operations."""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


@dataclass
class Event:
    """Represents an operation event.
    
    Attributes:
        id: Unique event identifier
        type: Event type (e.g., "image.enhanced", "depth.estimated")
        timestamp: When event occurred
        data: Event payload data
        metadata: Additional metadata
        user: Optional user identifier
        correlation_id: Optional correlation ID for related events
    """
    id: str
    type: str
    timestamp: float
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    user: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        return cls(**data)


class EventStore:
    """Store and query events for audit and replay.
    
    Example:
        >>> store = EventStore()
        >>> 
        >>> # Record event
        >>> event = Event(
        ...     id=str(uuid.uuid4()),
        ...     type="image.processed",
        ...     timestamp=time.time(),
        ...     data={"path": "image.jpg", "preset": "golden_hour"}
        ... )
        >>> store.append(event)
        >>> 
        >>> # Query events
        >>> recent = store.get_events(limit=10)
        >>> by_type = store.get_events_by_type("image.processed")
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize event store.
        
        Args:
            storage_path: Path to store events (defaults to .events/)
        """
        self.storage_path = storage_path or Path('.events')
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._events: List[Event] = []
        self._lock = Lock()
        self._load_events()
    
    def append(self, event: Event) -> None:
        """Append event to store.
        
        Args:
            event: Event to append
        """
        with self._lock:
            self._events.append(event)
            self._persist_event(event)
    
    def get_events(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        reverse: bool = True
    ) -> List[Event]:
        """Get events from store.
        
        Args:
            limit: Maximum number of events to return
            offset: Number of events to skip
            reverse: Return most recent first
            
        Returns:
            List of events
        """
        with self._lock:
            events = self._events.copy()
        
        if reverse:
            events = list(reversed(events))
        
        if offset:
            events = events[offset:]
        
        if limit:
            events = events[:limit]
        
        return events
    
    def get_events_by_type(
        self,
        event_type: str,
        limit: Optional[int] = None
    ) -> List[Event]:
        """Get events of a specific type.
        
        Args:
            event_type: Event type to filter
            limit: Maximum number of events
            
        Returns:
            List of matching events
        """
        with self._lock:
            filtered = [e for e in self._events if e.type == event_type]
        
        if limit:
            filtered = filtered[-limit:]
        
        return filtered
    
    def get_events_by_correlation(self, correlation_id: str) -> List[Event]:
        """Get all events with a specific correlation ID.
        
        Args:
            correlation_id: Correlation identifier
            
        Returns:
            List of correlated events
        """
        with self._lock:
            return [e for e in self._events if e.correlation_id == correlation_id]
    
    def get_events_in_range(
        self,
        start_time: float,
        end_time: float
    ) -> List[Event]:
        """Get events within a time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of events in range
        """
        with self._lock:
            return [
                e for e in self._events
                if start_time <= e.timestamp <= end_time
            ]
    
    def clear(self) -> None:
        """Clear all events from store."""
        with self._lock:
            self._events.clear()
            # Clear persisted events
            for event_file in self.storage_path.glob('*.json'):
                event_file.unlink()
    
    def _persist_event(self, event: Event) -> None:
        """Persist event to disk (assumes lock held).
        
        Args:
            event: Event to persist
        """
        # Store events by date for easier management
        date_dir = self.storage_path / time.strftime('%Y-%m-%d', time.localtime(event.timestamp))
        date_dir.mkdir(exist_ok=True)
        
        event_file = date_dir / f"{event.id}.json"
        with open(event_file, 'w') as f:
            json.dump(event.to_dict(), f, indent=2)
    
    def _load_events(self) -> None:
        """Load persisted events from disk."""
        for event_file in self.storage_path.rglob('*.json'):
            try:
                with open(event_file) as f:
                    event_data = json.load(f)
                self._events.append(Event.from_dict(event_data))
            except Exception as e:
                print(f"Failed to load event from {event_file}: {e}")
        
        # Sort by timestamp
        self._events.sort(key=lambda e: e.timestamp)


# Global event store
_global_store: Optional[EventStore] = None
_store_lock = Lock()


def get_global_store() -> EventStore:
    """Get the global event store singleton.
    
    Returns:
        Global EventStore instance
    """
    global _global_store
    
    if _global_store is None:
        with _store_lock:
            if _global_store is None:
                _global_store = EventStore()
    
    return _global_store
