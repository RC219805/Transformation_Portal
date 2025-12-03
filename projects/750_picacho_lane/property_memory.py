#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Property Memory System for 750 Picacho Lane
============================================

Implements persistent memory and learning system for property-specific
processing knowledge, including room configurations, optimal parameters,
and historical results.

Features:
- Room configuration memory (materials, lighting, optimal settings)
- Parameter learning from feedback loops
- Historical results tracking and trend analysis
- Scene-type pattern recognition
- Quality metric persistence
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("property_memory")


class SceneType(str, Enum):
    """Scene types for 750 Picacho Lane."""
    POOL = "pool"
    GREAT_ROOM = "great_room"
    KITCHEN = "kitchen"
    PRIMARY_BEDROOM = "primary_bedroom"
    PRIMARY_BATHROOM = "primary_bathroom"
    AERIAL = "aerial"
    AERIAL_2 = "aerial_2"
    EXTERIOR = "exterior"


class MaterialType(str, Enum):
    """Material types detected in scenes."""
    WATER = "water"
    STONE = "stone"
    WOOD = "wood"
    METAL = "metal"
    GLASS = "glass"
    FABRIC = "fabric"
    CONCRETE = "concrete"
    VEGETATION = "vegetation"
    ROOF = "roof"


@dataclass
class ProcessingResult:
    """Record of a processing run result."""
    timestamp: str
    scene_type: str
    input_path: str
    output_path: str
    parameters: Dict[str, Any]
    quality_score: float
    processing_time: float
    success: bool
    notes: Optional[str] = None
    user_feedback: Optional[str] = None


@dataclass
class RoomConfiguration:
    """Configuration and learned parameters for a specific room/scene."""
    scene_type: SceneType
    materials: List[MaterialType]
    optimal_parameters: Dict[str, Any] = field(default_factory=dict)
    quality_baseline: float = 0.0
    processing_history: List[ProcessingResult] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'scene_type': self.scene_type.value,
            'materials': [m.value for m in self.materials],
            'optimal_parameters': self.optimal_parameters,
            'quality_baseline': self.quality_baseline,
            'processing_history': [asdict(p) for p in self.processing_history],
            'notes': self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'RoomConfiguration':
        """Create from dictionary."""
        history = [
            ProcessingResult(**h) for h in data.get('processing_history', [])
        ]
        return cls(
            scene_type=SceneType(data['scene_type']),
            materials=[MaterialType(m) for m in data.get('materials', [])],
            optimal_parameters=data.get('optimal_parameters', {}),
            quality_baseline=data.get('quality_baseline', 0.0),
            processing_history=history,
            notes=data.get('notes', ''),
        )


@dataclass
class PropertyKnowledge:
    """Aggregated knowledge about the property."""
    property_name: str
    location: str
    total_scenes: int
    average_quality: float
    best_performing_scene: str
    common_materials: List[str]
    optimal_global_parameters: Dict[str, Any]
    processing_trends: Dict[str, str]  # "improving", "stable", "degrading"
    last_updated: str


class PropertyMemory:
    """
    Persistent memory system for 750 Picacho Lane property.

    Stores and retrieves:
    - Room configurations with learned optimal parameters
    - Processing history and quality metrics
    - Material compositions per scene
    - Learning feedback from user inputs
    """

    # Default configurations for 750 Picacho Lane rooms
    DEFAULT_ROOM_CONFIGS = {
        SceneType.POOL: {
            'materials': [MaterialType.WATER, MaterialType.STONE, MaterialType.CONCRETE],
            'optimal_parameters': {
                'water_enhance': True,
                'water_saturation': 1.25,
                'contrast': 1.08,
                'saturation': 1.05,
                'temperature': 5,
                'clarity': 1.15,
                'atmospheric_haze': True,
                'haze_density': 0.02,
            },
            'notes': 'Pool & Aquatic Features - emphasize water clarity and reflections',
        },
        SceneType.GREAT_ROOM: {
            'materials': [MaterialType.WOOD, MaterialType.FABRIC, MaterialType.GLASS, MaterialType.STONE],
            'optimal_parameters': {
                'wood_enhance': True,
                'fabric_enhance': True,
                'contrast': 1.10,
                'saturation': 1.03,
                'temperature': 3,
                'warmth': 8,
            },
            'notes': 'Great Room - warm interior lighting, wood grain detail',
        },
        SceneType.KITCHEN: {
            'materials': [MaterialType.METAL, MaterialType.STONE, MaterialType.GLASS, MaterialType.WOOD],
            'optimal_parameters': {
                'metal_enhance': True,
                'stone_enhance': True,
                'contrast': 1.12,
                'saturation': 1.02,
                'temperature': 2,
                'clarity': 1.15,
            },
            'notes': 'Kitchen - crisp metal surfaces, stone counters',
        },
        SceneType.PRIMARY_BEDROOM: {
            'materials': [MaterialType.FABRIC, MaterialType.WOOD, MaterialType.GLASS],
            'optimal_parameters': {
                'fabric_enhance': True,
                'wood_enhance': True,
                'contrast': 1.05,
                'saturation': 1.02,
                'temperature': 6,
                'warmth': 10,
                'softness': 0.95,
            },
            'notes': 'Primary Bedroom - soft, luxurious feel',
        },
        SceneType.PRIMARY_BATHROOM: {
            'materials': [MaterialType.STONE, MaterialType.GLASS, MaterialType.METAL, MaterialType.WATER],
            'optimal_parameters': {
                'stone_enhance': True,
                'glass_enhance': True,
                'contrast': 1.08,
                'saturation': 1.04,
                'temperature': 4,
            },
            'notes': 'Primary Bathroom - spa-like atmosphere',
        },
        SceneType.AERIAL: {
            'materials': [MaterialType.WATER, MaterialType.STONE, MaterialType.VEGETATION, MaterialType.ROOF],
            'optimal_parameters': {
                'water_enhance': True,
                'landscape_enhance': True,
                'contrast': 1.15,
                'saturation': 1.08,
                'temperature': 7,
                'clarity': 1.20,
                'atmospheric_depth': True,
                'atmospheric_haze': True,
                'haze_density': 0.03,
            },
            'notes': 'Aerial View - estate overview with depth',
        },
        SceneType.AERIAL_2: {
            'materials': [MaterialType.WATER, MaterialType.STONE, MaterialType.VEGETATION, MaterialType.ROOF],
            'optimal_parameters': {
                'water_enhance': True,
                'landscape_enhance': True,
                'contrast': 1.15,
                'saturation': 1.08,
                'temperature': 7,
                'clarity': 1.20,
                'atmospheric_depth': True,
                'atmospheric_haze': True,
                'haze_density': 0.03,
            },
            'notes': 'Aerial View 2 - neighborhood context',
        },
    }

    def __init__(self, memory_path: Optional[Path] = None):
        """
        Initialize PropertyMemory.

        Args:
            memory_path: Path to persistent storage file
        """
        if memory_path is None:
            memory_path = Path(__file__).parent / 'memory' / 'property_memory.json'

        self.memory_path = Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

        self.room_configs: Dict[SceneType, RoomConfiguration] = {}
        self.global_learnings: Dict[str, Any] = {}
        self.feedback_records: List[Dict] = []

        self._load_or_initialize()

    def _load_or_initialize(self):
        """Load existing memory or initialize with defaults."""
        if self.memory_path.exists():
            try:
                self._load_from_file()
                logger.info(f"Loaded property memory from {self.memory_path}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load memory, initializing defaults: {e}")
                self._initialize_defaults()
        else:
            self._initialize_defaults()
            logger.info("Initialized new property memory with defaults")

    def _initialize_defaults(self):
        """Initialize with default 750 Picacho Lane configurations."""
        for scene_type, config in self.DEFAULT_ROOM_CONFIGS.items():
            self.room_configs[scene_type] = RoomConfiguration(
                scene_type=scene_type,
                materials=config['materials'],
                optimal_parameters=config['optimal_parameters'].copy(),
                notes=config['notes'],
            )

        self.global_learnings = {
            'property_name': '750 Picacho Lane',
            'location': 'Santa Barbara, CA 93103',
            'style': 'Mediterranean Coastal Estate',
            'lighting_preference': 'golden_hour',
            'color_profile': 'Montecito_Golden_Hour_HDR',
            'lut_strength': 0.70,
            'global_saturation_boost': 1.05,
        }

        self._save_to_file()

    def _load_from_file(self):
        """Load memory from JSON file."""
        with open(self.memory_path, 'r') as f:
            data = json.load(f)

        # Load room configurations
        for scene_key, config_data in data.get('room_configs', {}).items():
            scene_type = SceneType(scene_key)
            self.room_configs[scene_type] = RoomConfiguration.from_dict(config_data)

        # Load global learnings
        self.global_learnings = data.get('global_learnings', {})

        # Load feedback records
        self.feedback_records = data.get('feedback_records', [])

    def _save_to_file(self):
        """Save memory to JSON file."""
        data = {
            'room_configs': {
                st.value: cfg.to_dict() for st, cfg in self.room_configs.items()
            },
            'global_learnings': self.global_learnings,
            'feedback_records': self.feedback_records,
            'last_updated': datetime.now().isoformat(),
        }

        with open(self.memory_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved property memory to {self.memory_path}")

    def get_room_config(self, scene_type: SceneType) -> RoomConfiguration:
        """
        Get configuration for a specific room/scene.

        Args:
            scene_type: The scene type to retrieve

        Returns:
            RoomConfiguration for the scene
        """
        if scene_type not in self.room_configs:
            # Initialize with defaults if not found
            default = self.DEFAULT_ROOM_CONFIGS.get(scene_type, {
                'materials': [],
                'optimal_parameters': {},
                'notes': 'New scene type',
            })
            self.room_configs[scene_type] = RoomConfiguration(
                scene_type=scene_type,
                materials=default.get('materials', []),
                optimal_parameters=default.get('optimal_parameters', {}),
                notes=default.get('notes', ''),
            )

        return self.room_configs[scene_type]

    def get_optimal_parameters(self, scene_type: SceneType) -> Dict[str, Any]:
        """
        Get optimal processing parameters for a scene type.

        Args:
            scene_type: The scene type

        Returns:
            Dictionary of optimal parameters
        """
        config = self.get_room_config(scene_type)
        return config.optimal_parameters.copy()

    def get_materials(self, scene_type: SceneType) -> List[MaterialType]:
        """
        Get list of materials present in a scene.

        Args:
            scene_type: The scene type

        Returns:
            List of MaterialType present in scene
        """
        config = self.get_room_config(scene_type)
        return config.materials.copy()

    def add_processing_result(
        self,
        scene_type: SceneType,
        input_path: str,
        output_path: str,
        parameters: Dict[str, Any],
        quality_score: float,
        processing_time: float,
        success: bool,
        notes: Optional[str] = None,
    ):
        """
        Record a processing result for learning.

        Args:
            scene_type: The scene type processed
            input_path: Path to input image
            output_path: Path to output image
            parameters: Parameters used for processing
            quality_score: Quality score (0.0 - 1.0)
            processing_time: Processing time in seconds
            success: Whether processing succeeded
            notes: Optional notes about the result
        """
        result = ProcessingResult(
            timestamp=datetime.now().isoformat(),
            scene_type=scene_type.value,
            input_path=str(input_path),
            output_path=str(output_path),
            parameters=parameters,
            quality_score=quality_score,
            processing_time=processing_time,
            success=success,
            notes=notes,
        )

        config = self.get_room_config(scene_type)
        config.processing_history.append(result)

        # Update quality baseline if this is better
        if success and quality_score > config.quality_baseline:
            config.quality_baseline = quality_score
            config.optimal_parameters.update(parameters)
            logger.info(
                f"Updated optimal parameters for {scene_type.value} "
                f"(quality: {quality_score:.2%})"
            )

        self._save_to_file()

    def add_user_feedback(
        self,
        scene_type: SceneType,
        feedback: str,
        rating: Optional[float] = None,
        suggested_parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Add user feedback for learning.

        Args:
            scene_type: The scene type
            feedback: User feedback text
            rating: Optional rating (0.0 - 1.0)
            suggested_parameters: Optional suggested parameter changes
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'scene_type': scene_type.value,
            'feedback': feedback,
            'rating': rating,
            'suggested_parameters': suggested_parameters,
        }

        self.feedback_records.append(record)

        # Apply suggested parameter updates if provided and rated highly
        if suggested_parameters and rating and rating >= 0.8:
            config = self.get_room_config(scene_type)
            config.optimal_parameters.update(suggested_parameters)
            logger.info(f"Applied user-suggested parameters for {scene_type.value}")

        self._save_to_file()

    def learn_from_results(
        self,
        scene_type: SceneType,
        min_samples: int = 3,
    ) -> Dict[str, Any]:
        """
        Analyze processing history and learn optimal parameters.

        Args:
            scene_type: The scene type to analyze
            min_samples: Minimum samples needed for learning

        Returns:
            Dictionary with learning insights
        """
        config = self.get_room_config(scene_type)
        history = config.processing_history

        if len(history) < min_samples:
            return {
                'status': 'insufficient_data',
                'samples': len(history),
                'needed': min_samples,
            }

        # Filter successful runs
        successful_runs = [r for r in history if r.success]

        if not successful_runs:
            return {
                'status': 'no_successful_runs',
                'total_runs': len(history),
            }

        # Find highest quality runs
        successful_runs.sort(key=lambda x: x.quality_score, reverse=True)
        top_runs = successful_runs[:max(1, len(successful_runs) // 3)]

        # Aggregate parameters from top runs
        learned_params = {}
        param_counts = {}

        for run in top_runs:
            for key, value in run.parameters.items():
                if isinstance(value, (int, float)):
                    if key not in learned_params:
                        learned_params[key] = 0
                        param_counts[key] = 0
                    learned_params[key] += value
                    param_counts[key] += 1
                else:
                    # For non-numeric, use most common value
                    learned_params[key] = value

        # Average numeric parameters
        for key in learned_params:
            if key in param_counts and param_counts[key] > 0:
                learned_params[key] /= param_counts[key]

        # Update optimal parameters
        config.optimal_parameters.update(learned_params)

        # Calculate quality trend
        recent = history[-min(5, len(history)):]
        recent_successful = [r for r in recent if r.success]
        recent_avg = (
            sum(r.quality_score for r in recent_successful) / len(recent_successful)
            if recent_successful else 0.0
        )

        older = history[:-len(recent)] if len(history) > len(recent) else []
        older_successful = [r for r in older if r.success]
        older_avg = (
            sum(r.quality_score for r in older_successful) / len(older_successful)
            if older_successful else recent_avg
        )

        if recent_avg > older_avg * 1.05:
            trend = 'improving'
        elif recent_avg < older_avg * 0.95:
            trend = 'degrading'
        else:
            trend = 'stable'

        self._save_to_file()

        return {
            'status': 'success',
            'samples_analyzed': len(successful_runs),
            'top_quality_samples': len(top_runs),
            'learned_parameters': learned_params,
            'quality_trend': trend,
            'average_quality': sum(r.quality_score for r in successful_runs) / len(successful_runs),
            'best_quality': successful_runs[0].quality_score if successful_runs else 0,
        }

    def get_property_knowledge(self) -> PropertyKnowledge:
        """
        Get aggregated knowledge about the property.

        Returns:
            PropertyKnowledge with aggregated insights
        """
        # Calculate aggregate statistics
        all_quality_scores = []
        all_materials = []
        scene_qualities = {}

        for scene_type, config in self.room_configs.items():
            successful = [r for r in config.processing_history if r.success]
            if successful:
                avg_quality = sum(r.quality_score for r in successful) / len(successful)
                scene_qualities[scene_type.value] = avg_quality
                all_quality_scores.extend([r.quality_score for r in successful])

            all_materials.extend([m.value for m in config.materials])

        # Find common materials
        material_counts = {}
        for m in all_materials:
            material_counts[m] = material_counts.get(m, 0) + 1
        common_materials = sorted(
            material_counts.keys(),
            key=lambda x: material_counts[x],
            reverse=True
        )[:5]

        # Find best performing scene
        best_scene = max(scene_qualities.items(), key=lambda x: x[1])[0] if scene_qualities else 'N/A'

        # Calculate trends
        trends = {}
        for scene_type, config in self.room_configs.items():
            learning = self.learn_from_results(scene_type, min_samples=2)
            if learning.get('status') == 'success':
                trends[scene_type.value] = learning.get('quality_trend', 'stable')

        return PropertyKnowledge(
            property_name=self.global_learnings.get('property_name', '750 Picacho Lane'),
            location=self.global_learnings.get('location', 'Santa Barbara, CA'),
            total_scenes=len(self.room_configs),
            average_quality=sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else 0.0,
            best_performing_scene=best_scene,
            common_materials=common_materials,
            optimal_global_parameters=self.global_learnings.copy(),
            processing_trends=trends,
            last_updated=datetime.now().isoformat(),
        )

    def export_knowledge(self, output_path: Path) -> None:
        """
        Export property knowledge to a JSON file.

        Args:
            output_path: Path for output file
        """
        knowledge = self.get_property_knowledge()

        export_data = {
            'property_knowledge': {
                'property_name': knowledge.property_name,
                'location': knowledge.location,
                'total_scenes': knowledge.total_scenes,
                'average_quality': knowledge.average_quality,
                'best_performing_scene': knowledge.best_performing_scene,
                'common_materials': knowledge.common_materials,
                'optimal_global_parameters': knowledge.optimal_global_parameters,
                'processing_trends': knowledge.processing_trends,
                'last_updated': knowledge.last_updated,
            },
            'room_configurations': {
                st.value: cfg.to_dict() for st, cfg in self.room_configs.items()
            },
            'export_timestamp': datetime.now().isoformat(),
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported property knowledge to {output_path}")

    def get_scene_type_from_filename(self, filename: str) -> Optional[SceneType]:
        """
        Infer scene type from filename.

        Args:
            filename: Image filename

        Returns:
            SceneType if recognized, None otherwise
        """
        filename_lower = filename.lower()

        # Scene detection patterns
        patterns = {
            SceneType.POOL: ['pool', 'aquatic'],
            SceneType.GREAT_ROOM: ['greatroom', 'great_room', 'living'],
            SceneType.KITCHEN: ['kitchen', 'culinary'],
            SceneType.PRIMARY_BEDROOM: ['bedroom', 'primarybed', 'master_bed'],
            SceneType.PRIMARY_BATHROOM: ['bathroom', 'primarybath', 'master_bath', 'spa'],
            SceneType.AERIAL: ['aerial', 'drone'],
            SceneType.AERIAL_2: ['aerial-2', 'aerial_2', 'drone2'],
            SceneType.EXTERIOR: ['exterior', 'facade', 'front'],
        }

        for scene_type, keywords in patterns.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return scene_type

        return None


def main():
    """Demonstrate PropertyMemory usage."""
    print("=" * 70)
    print("750 Picacho Lane - Property Memory System")
    print("=" * 70)

    # Initialize memory
    memory = PropertyMemory()

    # Display room configurations
    print("\nRoom Configurations:")
    print("-" * 50)

    for scene_type in SceneType:
        config = memory.get_room_config(scene_type)
        materials = [m.value for m in config.materials]
        print(f"\n{scene_type.value}:")
        print(f"  Materials: {', '.join(materials)}")
        print("  Key parameters:")
        for key, value in list(config.optimal_parameters.items())[:4]:
            print(f"    - {key}: {value}")

    # Display global learnings
    print("\nGlobal Learnings:")
    print("-" * 50)
    for key, value in memory.global_learnings.items():
        print(f"  {key}: {value}")

    # Get property knowledge summary
    print("\nProperty Knowledge Summary:")
    print("-" * 50)
    knowledge = memory.get_property_knowledge()
    print(f"  Property: {knowledge.property_name}")
    print(f"  Location: {knowledge.location}")
    print(f"  Scenes: {knowledge.total_scenes}")
    print(f"  Common Materials: {', '.join(knowledge.common_materials)}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
