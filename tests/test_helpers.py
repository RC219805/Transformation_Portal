#!/usr/bin/env python3
"""Shared mock classes and utilities for testing.

This module provides common mock implementations used across multiple test files
to avoid code duplication and ensure consistency.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class MockRoomContext:
    """Mock room context for testing."""
    name: str
    dimensions: Optional[Tuple[float, float]] = None
    floor_level: Optional[str] = None
    ceiling_height: Optional[float] = None
    materials: Optional[List[str]] = None
    features: Optional[List[str]] = None
    adjacent_rooms: Optional[List[str]] = None

    def __post_init__(self):
        if self.materials is None:
            self.materials = []
        if self.features is None:
            self.features = []
        if self.adjacent_rooms is None:
            self.adjacent_rooms = []


@dataclass
class MockProjectContext:
    """Mock project context for testing."""
    project_name: str
    project_number: Optional[str] = None
    address: Optional[str] = None
    architect: Optional[str] = None
    total_sqft: Optional[float] = None
    floors: Optional[List[str]] = None
    rooms: Optional[Dict[str, MockRoomContext]] = None
    materials_palette: Optional[List[str]] = None
    design_style: Optional[str] = None
    extracted_images: Optional[List[str]] = None
    raw_text: Optional[str] = None

    def __post_init__(self):
        if self.floors is None:
            self.floors = []
        if self.rooms is None:
            self.rooms = {}
        if self.materials_palette is None:
            self.materials_palette = []
        if self.extracted_images is None:
            self.extracted_images = []
