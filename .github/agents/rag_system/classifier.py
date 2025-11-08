"""
Artifact Classification System for Image Processing Pipelines

Automatically classifies and organizes image processing artifacts with
metadata extraction and hierarchical organization.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set


class ArtifactType(Enum):
    """Types of image processing artifacts."""

    ANALYSIS = "analysis"
    DEPTH_MAP = "depth_map"
    COLOR_GRADE = "color_grade"
    HDR_OUTPUT = "hdr_output"
    METRIC = "metric"
    LOG = "log"
    PROFILE = "profile"
    RENDER = "render"
    MATERIAL_RESPONSE = "material_response"
    LUT_APPLICATION = "lut_application"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"


class PipelineType(Enum):
    """Types of processing pipelines."""

    DEPTH_PIPELINE = "depth_pipeline"
    LUX_RENDER = "lux_render"
    MATERIAL_RESPONSE = "material_response"
    VIDEO_GRADER = "video_grader"
    TIFF_PROCESSOR = "tiff_processor"
    HDR_PRODUCTION = "hdr_production"
    AGXFILMIC = "agx_filmic"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


@dataclass
class ProcessingMetadata:
    """Metadata extracted from processing artifacts."""

    pipeline: PipelineType
    artifact_type: ArtifactType
    timestamp: Optional[datetime] = None
    parameters: Dict = field(default_factory=dict)
    hardware: Optional[str] = None
    success: Optional[bool] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None

    # Image-specific metadata
    resolution: Optional[tuple] = None  # (width, height)
    color_space: Optional[str] = None
    bit_depth: Optional[int] = None

    # AI model information
    ai_model: Optional[str] = None
    model_version: Optional[str] = None

    # Performance metrics
    memory_usage: Optional[float] = None  # MB
    gpu_utilization: Optional[float] = None  # percentage

    # Quality metrics
    quality_score: Optional[float] = None
    similarity_score: Optional[float] = None


@dataclass
class ArtifactNode:
    """Node in the artifact hierarchy with relational links."""

    artifact_id: str
    file_path: str
    artifact_type: ArtifactType
    metadata: ProcessingMetadata

    # Relational links
    parent_id: Optional[str] = None  # Original/source artifact
    children_ids: List[str] = field(default_factory=list)  # Derived artifacts
    related_ids: List[str] = field(default_factory=list)  # Related artifacts (e.g., analysis of same image)

    # Tags for retrieval
    tags: Set[str] = field(default_factory=set)

    # Version tracking
    version: str = "1.0"
    previous_version_id: Optional[str] = None


class ArtifactClassifier:
    """
    Classifies image processing artifacts and extracts metadata.

    Features:
    - Auto-classifies analyses, depth maps, color grades, HDR outputs, etc.
    - Extracts metadata from filenames, logs, and EXIF data
    - Organizes artifacts hierarchically
    - Tags for efficient retrieval
    """

    # Filename patterns for classification
    PATTERNS = {
        ArtifactType.DEPTH_MAP: [
            r'depth_map', r'depth.*\.png', r'depth.*\.tiff', r'.*_depth\.',
        ],
        ArtifactType.COLOR_GRADE: [
            r'color_grade', r'graded_', r'.*_lut_applied', r'tone_mapped',
        ],
        ArtifactType.HDR_OUTPUT: [
            r'hdr_', r'.*_hdr\.', r'tonemapped', r'high_dynamic_range',
        ],
        ArtifactType.ANALYSIS: [
            r'analysis', r'report', r'stats', r'histogram', r'comparison',
        ],
        ArtifactType.METRIC: [
            r'metrics?\.json', r'performance', r'benchmark', r'timing',
        ],
        ArtifactType.LOG: [
            r'\.log$', r'debug', r'trace', r'error_log',
        ],
        ArtifactType.PROFILE: [
            r'profile', r'memory', r'cpu_usage', r'gpu_usage',
        ],
        ArtifactType.MATERIAL_RESPONSE: [
            r'material_response', r'material_enhanced', r'surface_enhanced',
        ],
        ArtifactType.LUT_APPLICATION: [
            r'lut_', r'color_transform', r'film_emulation',
        ],
    }

    # Pipeline detection patterns
    PIPELINE_PATTERNS = {
        PipelineType.DEPTH_PIPELINE: [
            r'depth_pipeline', r'depth_processing', r'architectural_depth',
        ],
        PipelineType.LUX_RENDER: [
            r'lux_render', r'luxury_render', r'ai_enhanced',
        ],
        PipelineType.MATERIAL_RESPONSE: [
            r'material_response', r'surface_processing',
        ],
        PipelineType.VIDEO_GRADER: [
            r'video_grade', r'video_master', r'color_grade',
        ],
        PipelineType.TIFF_PROCESSOR: [
            r'tiff_batch', r'tiff_process', r'16bit_processing',
        ],
        PipelineType.HDR_PRODUCTION: [
            r'hdr_production', r'hdr_pipeline', r'tone_mapping',
        ],
    }

    def __init__(self):
        """Initialize the classifier."""
        self.artifacts: Dict[str, ArtifactNode] = {}
        self.artifact_counter = 0

    def classify_artifact(
        self,
        file_path: str,
        content: Optional[str] = None,
    ) -> ArtifactType:
        """
        Classify an artifact based on file path and content.

        Args:
            file_path: Path to the artifact file
            content: Optional content for additional classification hints

        Returns:
            Classified artifact type
        """
        file_path_lower = file_path.lower()

        # Check patterns for each artifact type
        for artifact_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, file_path_lower):
                    return artifact_type

        # Content-based classification if available
        if content:
            if 'depth' in content.lower() and ('map' in content.lower() or 'estimation' in content.lower()):
                return ArtifactType.DEPTH_MAP
            if 'lut' in content.lower() or 'color grade' in content.lower():
                return ArtifactType.COLOR_GRADE
            if 'hdr' in content.lower() or 'tone map' in content.lower():
                return ArtifactType.HDR_OUTPUT

        return ArtifactType.UNKNOWN

    def detect_pipeline(
        self,
        file_path: str,
        content: Optional[str] = None,
    ) -> PipelineType:
        """
        Detect which pipeline produced the artifact.

        Args:
            file_path: Path to the artifact file
            content: Optional content for additional detection hints

        Returns:
            Detected pipeline type
        """
        file_path_lower = file_path.lower()

        # Check patterns for each pipeline type
        for pipeline_type, patterns in self.PIPELINE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, file_path_lower):
                    return pipeline_type

        # Content-based detection
        if content:
            content_lower = content.lower()
            if 'architectural' in content_lower and 'depth' in content_lower:
                return PipelineType.DEPTH_PIPELINE
            if 'stable diffusion' in content_lower or 'controlnet' in content_lower:
                return PipelineType.LUX_RENDER
            if 'material' in content_lower and 'response' in content_lower:
                return PipelineType.MATERIAL_RESPONSE

        return PipelineType.UNKNOWN

    def extract_metadata(
        self,
        file_path: str,
        artifact_type: ArtifactType,
        pipeline_type: PipelineType,
        content: Optional[str] = None,
    ) -> ProcessingMetadata:
        """
        Extract metadata from artifact.

        Args:
            file_path: Path to the artifact
            artifact_type: Classified artifact type
            pipeline_type: Detected pipeline type
            content: Optional content to parse

        Returns:
            Extracted metadata
        """
        metadata = ProcessingMetadata(
            pipeline=pipeline_type,
            artifact_type=artifact_type,
        )

        # Extract timestamp from filename if present
        timestamp_match = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2})', file_path)
        if timestamp_match:
            try:
                date_str = timestamp_match.group(1).replace('_', '-')
                metadata.timestamp = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                # Ignore invalid or missing date formats in filename; timestamp is optional metadata.
                pass

        # Extract resolution from filename
        resolution_match = re.search(r'(\d{3,4})x(\d{3,4})', file_path)
        if resolution_match:
            metadata.resolution = (int(resolution_match.group(1)), int(resolution_match.group(2)))

        # Extract parameters from JSON content
        if content and artifact_type == ArtifactType.METRIC:
            try:
                data = json.loads(content)
                metadata.parameters = data.get('parameters', {})
                metadata.processing_time = data.get('processing_time')
                metadata.memory_usage = data.get('memory_usage')
                metadata.gpu_utilization = data.get('gpu_utilization')
                metadata.success = data.get('success', True)
            except json.JSONDecodeError:
                # Content may not always be valid JSON; ignore and proceed with empty/default parameters.
                pass

        # Extract error information from logs
        if content and artifact_type == ArtifactType.LOG:
            if 'error' in content.lower() or 'exception' in content.lower():
                metadata.success = False
                # Extract first error message
                error_match = re.search(r'(error|exception)[:\s]+([^\n]+)', content, re.IGNORECASE)
                if error_match:
                    metadata.error_message = error_match.group(2)[:200]  # First 200 chars

        # Extract AI model info
        ai_model_patterns = [
            r'depth_anything_v2',
            r'stable_diffusion',
            r'controlnet',
            r'real_esrgan',
        ]
        for pattern in ai_model_patterns:
            if re.search(pattern, file_path.lower()):
                metadata.ai_model = pattern.replace('_', ' ').title()
                break

        # Extract color space info
        color_space_patterns = [
            r'srgb', r'adobe_rgb', r'prophoto', r'aces', r'rec709', r'rec2020',
        ]
        for pattern in color_space_patterns:
            if re.search(pattern, file_path.lower()):
                metadata.color_space = pattern.upper()
                break

        # Extract bit depth
        bit_depth_match = re.search(r'(\d+)bit', file_path.lower())
        if bit_depth_match:
            metadata.bit_depth = int(bit_depth_match.group(1))

        return metadata

    def generate_tags(
        self,
        artifact_type: ArtifactType,
        metadata: ProcessingMetadata,
        file_path: str,
    ) -> Set[str]:
        """
        Generate tags for efficient retrieval.

        Args:
            artifact_type: Type of artifact
            metadata: Extracted metadata
            file_path: File path

        Returns:
            Set of tags
        """
        tags = set()

        # Type tags
        tags.add(artifact_type.value)
        tags.add(metadata.pipeline.value)

        # Status tags
        if metadata.success is not None:
            tags.add('success' if metadata.success else 'failure')

        # Hardware tags
        if metadata.hardware:
            tags.add(f'hardware:{metadata.hardware}')

        # AI model tags
        if metadata.ai_model:
            tags.add(f'ai_model:{metadata.ai_model.lower().replace(" ", "_")}')

        # Color space tags
        if metadata.color_space:
            tags.add(f'color_space:{metadata.color_space.lower()}')

        # Resolution tags
        if metadata.resolution:
            width, height = metadata.resolution
            tags.add(f'resolution:{width}x{height}')
            # Add general resolution category
            if width >= 3840:
                tags.add('4k_plus')
            elif width >= 1920:
                tags.add('full_hd')
            elif width >= 1280:
                tags.add('hd')

        # Performance tags
        if metadata.processing_time:
            if metadata.processing_time < 1.0:
                tags.add('fast_processing')
            elif metadata.processing_time > 10.0:
                tags.add('slow_processing')

        # Error tags
        if metadata.error_message:
            tags.add('has_error')
            # Extract error type
            error_type_match = re.search(r'(\w+Error|\w+Exception)', metadata.error_message)
            if error_type_match:
                tags.add(f'error_type:{error_type_match.group(1)}')

        return tags

    def add_artifact(
        self,
        file_path: str,
        content: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> ArtifactNode:
        """
        Add and classify an artifact.

        Args:
            file_path: Path to artifact
            content: Optional content
            parent_id: Optional parent artifact ID

        Returns:
            Created artifact node
        """
        # Classify
        artifact_type = self.classify_artifact(file_path, content)
        pipeline_type = self.detect_pipeline(file_path, content)

        # Extract metadata
        metadata = self.extract_metadata(file_path, artifact_type, pipeline_type, content)

        # Generate tags
        tags = self.generate_tags(artifact_type, metadata, file_path)

        # Create artifact node
        artifact_id = f"artifact_{self.artifact_counter:06d}"
        self.artifact_counter += 1

        node = ArtifactNode(
            artifact_id=artifact_id,
            file_path=file_path,
            artifact_type=artifact_type,
            metadata=metadata,
            parent_id=parent_id,
            tags=tags,
        )

        # Update parent's children
        if parent_id and parent_id in self.artifacts:
            self.artifacts[parent_id].children_ids.append(artifact_id)

        # Store
        self.artifacts[artifact_id] = node

        return node

    def link_related_artifacts(self, artifact_id1: str, artifact_id2: str):
        """Link two related artifacts."""
        if artifact_id1 in self.artifacts and artifact_id2 in self.artifacts:
            self.artifacts[artifact_id1].related_ids.append(artifact_id2)
            self.artifacts[artifact_id2].related_ids.append(artifact_id1)

    def get_transformation_chain(self, artifact_id: str) -> List[ArtifactNode]:
        """
        Get the full transformation chain for an artifact.

        Returns list of artifacts from original to final, including the given artifact.
        """
        chain = []
        current_id = artifact_id

        # Walk up to root
        while current_id:
            if current_id not in self.artifacts:
                break
            chain.insert(0, self.artifacts[current_id])
            current_id = self.artifacts[current_id].parent_id

        # Walk down from given artifact
        def collect_children(node_id: str):
            if node_id not in self.artifacts:
                return []
            node = self.artifacts[node_id]
            result = []
            for child_id in node.children_ids:
                result.append(self.artifacts[child_id])
                result.extend(collect_children(child_id))
            return result

        chain.extend(collect_children(artifact_id))

        return chain

    def search_by_tags(self, tags: Set[str], require_all: bool = False) -> List[ArtifactNode]:
        """
        Search artifacts by tags.

        Args:
            tags: Set of tags to search for
            require_all: If True, artifact must have all tags; if False, any tag matches

        Returns:
            List of matching artifacts
        """
        results = []

        for artifact in self.artifacts.values():
            if require_all:
                if tags.issubset(artifact.tags):
                    results.append(artifact)
            else:
                if tags.intersection(artifact.tags):
                    results.append(artifact)

        return results

    def get_statistics(self) -> Dict:
        """Get classification statistics."""
        stats = {
            'total_artifacts': len(self.artifacts),
            'by_type': {},
            'by_pipeline': {},
            'success_rate': 0.0,
            'avg_processing_time': 0.0,
            'artifacts_with_errors': 0,
        }

        total_time = 0.0
        time_count = 0
        success_count = 0
        total_with_status = 0

        for artifact in self.artifacts.values():
            # Count by type
            type_name = artifact.artifact_type.value
            stats['by_type'][type_name] = stats['by_type'].get(type_name, 0) + 1

            # Count by pipeline
            pipeline_name = artifact.metadata.pipeline.value
            stats['by_pipeline'][pipeline_name] = stats['by_pipeline'].get(pipeline_name, 0) + 1

            # Processing time
            if artifact.metadata.processing_time:
                total_time += artifact.metadata.processing_time
                time_count += 1

            # Success rate
            if artifact.metadata.success is not None:
                total_with_status += 1
                if artifact.metadata.success:
                    success_count += 1

            # Errors
            if artifact.metadata.error_message:
                stats['artifacts_with_errors'] += 1

        # Calculate averages
        if time_count > 0:
            stats['avg_processing_time'] = total_time / time_count

        if total_with_status > 0:
            stats['success_rate'] = success_count / total_with_status

        return stats

    def export_to_json(self, output_path: str):
        """Export artifacts to JSON."""
        data = {
            'artifacts': {},
            'statistics': self.get_statistics(),
            'export_time': datetime.now().isoformat(),
        }

        for artifact_id, node in self.artifacts.items():
            data['artifacts'][artifact_id] = {
                'artifact_id': node.artifact_id,
                'file_path': node.file_path,
                'artifact_type': node.artifact_type.value,
                'pipeline': node.metadata.pipeline.value,
                'parent_id': node.parent_id,
                'children_ids': node.children_ids,
                'related_ids': node.related_ids,
                'tags': list(node.tags),
                'version': node.version,
                'metadata': {
                    'timestamp': node.metadata.timestamp.isoformat() if node.metadata.timestamp else None,
                    'parameters': node.metadata.parameters,
                    'hardware': node.metadata.hardware,
                    'success': node.metadata.success,
                    'processing_time': node.metadata.processing_time,
                    'error_message': node.metadata.error_message,
                    'resolution': node.metadata.resolution,
                    'color_space': node.metadata.color_space,
                    'bit_depth': node.metadata.bit_depth,
                    'ai_model': node.metadata.ai_model,
                    'quality_score': node.metadata.quality_score,
                },
            }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    """CLI for artifact classification."""
    import argparse

    parser = argparse.ArgumentParser(description='Classify image processing artifacts')
    parser.add_argument('--input-dir', required=True, help='Directory containing artifacts')
    parser.add_argument('--output', default='artifacts.json', help='Output JSON file')
    parser.add_argument('--tags', nargs='+', help='Search by tags')
    parser.add_argument('--require-all-tags', action='store_true', help='Require all tags')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    classifier = ArtifactClassifier()

    # Scan directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        return

    print(f"Scanning {input_dir}...")

    for file_path in input_dir.rglob('*'):
        if file_path.is_file():
            # Try to read content if it's a text file
            content = None
            if file_path.suffix in {'.json', '.log', '.txt', '.md'}:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                except Exception as e:
                    if args.verbose:
                        print(f"  [WARN] Could not read {file_path}: {e}")

            artifact = classifier.add_artifact(str(file_path), content)

            if args.verbose:
                print(f"  {artifact.artifact_id}: {artifact.artifact_type.value} ({artifact.metadata.pipeline.value})")

    # Get statistics
    stats = classifier.get_statistics()
    print(f"\nClassified {stats['total_artifacts']} artifacts")
    print("\nBy type:")
    for artifact_type, count in sorted(stats['by_type'].items()):
        print(f"  {artifact_type}: {count}")
    print("\nBy pipeline:")
    for pipeline, count in sorted(stats['by_pipeline'].items()):
        print(f"  {pipeline}: {count}")

    if stats['total_artifacts'] > 0:
        print(f"\nSuccess rate: {stats['success_rate']:.1%}")
        print(f"Average processing time: {stats['avg_processing_time']:.2f}s")
        print(f"Artifacts with errors: {stats['artifacts_with_errors']}")

    # Search by tags if provided
    if args.tags:
        print(f"\nSearching for tags: {args.tags}")
        results = classifier.search_by_tags(set(args.tags), require_all=args.require_all_tags)
        print(f"Found {len(results)} matching artifacts")
        for result in results[:10]:  # Show first 10
            print(f"  {result.artifact_id}: {result.file_path}")

    # Export
    classifier.export_to_json(args.output)
    print(f"\nExported to {args.output}")


if __name__ == '__main__':
    main()
