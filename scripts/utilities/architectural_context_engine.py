#!/usr/bin/env python3
"""
Architectural Context Engine
Transformation Portal - Context-Aware Rendering System

Integrates architectural documentation (floor plans, elevations, dimensions)
to enhance AI-powered rendering with spatial and design awareness.

Features:
- PDF parsing for floor plans, elevations, dimensions
- Spatial relationship extraction
- Material specification parsing
- Context-aware prompt enrichment
- Architectural knowledge base integration
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Architectural document types."""

    FLOOR_PLAN = "floor_plan"
    ELEVATION = "elevation"
    SECTION = "section"
    DETAIL = "detail"
    SCHEDULE = "schedule"
    SPECIFICATION = "specification"
    RENDERING = "rendering"
    SITE_PLAN = "site_plan"


class SpaceType(Enum):
    """Architectural space classifications."""

    KITCHEN = "kitchen"
    LIVING = "living_room"
    DINING = "dining_room"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    OFFICE = "office"
    ENTRY = "entry"
    HALLWAY = "hallway"
    EXTERIOR = "exterior"
    POOL_AREA = "pool_area"
    COURTYARD = "courtyard"
    TERRACE = "terrace"


@dataclass
class DimensionInfo:
    """Dimensional information extracted from drawings."""

    width: Optional[float] = None
    length: Optional[float] = None
    height: Optional[float] = None
    area: Optional[float] = None
    ceiling_height: Optional[float] = None
    unit: str = "feet"

    def to_prompt_fragment(self) -> str:
        """Generate prompt context from dimensions."""
        parts = []
        if self.width and self.length:
            parts.append(f"{self.width}' x {self.length}' space")
        if self.ceiling_height:
            parts.append(f"{self.ceiling_height}' ceiling height")
        if self.area:
            parts.append(f"{self.area} sq ft")
        return ", ".join(parts) if parts else ""


@dataclass
class MaterialSpec:
    """Material specification from architectural documents."""

    material_type: str
    location: str
    finish: Optional[str] = None
    color: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None

    def to_prompt_fragment(self) -> str:
        """Generate prompt context from material spec."""
        parts = [self.material_type]
        if self.finish:
            parts.append(self.finish)
        if self.color:
            parts.append(self.color)
        return " ".join(parts)


@dataclass
class SpatialContext:
    """Spatial relationship and adjacency information."""

    space_name: str
    space_type: SpaceType
    adjacent_spaces: List[str] = field(default_factory=list)
    windows: List[str] = field(default_factory=list)
    doors: List[str] = field(default_factory=list)
    ceiling_type: Optional[str] = None
    flooring_type: Optional[str] = None

    def to_prompt_fragment(self) -> str:
        """Generate prompt context from spatial info."""
        parts = [f"{self.space_type.value}"]
        if self.ceiling_type:
            parts.append(f"{self.ceiling_type} ceiling")
        if self.flooring_type:
            parts.append(f"{self.flooring_type} flooring")
        if self.windows:
            parts.append(f"natural light from {len(self.windows)} window(s)")
        return ", ".join(parts)


@dataclass
class ArchitecturalContext:
    """Complete architectural context for a rendering."""

    project_name: str
    project_address: Optional[str] = None
    space_name: Optional[str] = None
    space_type: Optional[SpaceType] = None
    dimensions: Optional[DimensionInfo] = None
    materials: List[MaterialSpec] = field(default_factory=list)
    spatial_context: Optional[SpatialContext] = None
    design_intent: List[str] = field(default_factory=list)
    style_notes: List[str] = field(default_factory=list)
    source_documents: List[Path] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_enhanced_prompt(self, base_prompt: str = "") -> str:
        """Generate enhanced prompt with architectural context."""
        fragments = []

        # Project context
        if self.project_name:
            fragments.append(f"Luxury residence: {self.project_name}")

        # Space context
        if self.space_type:
            fragments.append(self.space_type.value.replace("_", " "))

        # Dimensions
        if self.dimensions:
            dim_text = self.dimensions.to_prompt_fragment()
            if dim_text:
                fragments.append(dim_text)

        # Spatial context
        if self.spatial_context:
            spatial_text = self.spatial_context.to_prompt_fragment()
            if spatial_text:
                fragments.append(spatial_text)

        # Materials (top 3 most important)
        if self.materials:
            mat_texts = [m.to_prompt_fragment() for m in self.materials[:3]]
            fragments.append("materials: " + ", ".join(mat_texts))

        # Design intent
        if self.design_intent:
            fragments.extend(self.design_intent[:2])

        # Style notes
        if self.style_notes:
            fragments.extend(self.style_notes[:2])

        # Combine with base prompt
        context_prompt = ", ".join(fragments)
        if base_prompt:
            return f"{base_prompt}, {context_prompt}"
        return context_prompt

    def save(self, output_path: Path):
        """Save context to JSON file."""
        data = {
            "project_name": self.project_name,
            "project_address": self.project_address,
            "space_name": self.space_name,
            "space_type": self.space_type.value if self.space_type else None,
            "dimensions": self.dimensions.__dict__ if self.dimensions else None,
            "materials": [m.__dict__ for m in self.materials],
            "spatial_context": self.spatial_context.__dict__ if self.spatial_context else None,
            "design_intent": self.design_intent,
            "style_notes": self.style_notes,
            "source_documents": [str(p) for p in self.source_documents],
            "metadata": self.metadata,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved architectural context to {output_path}")

    @classmethod
    def load(cls, input_path: Path) -> "ArchitecturalContext":
        """Load context from JSON file."""
        with open(input_path) as f:
            data = json.load(f)

        # Reconstruct objects
        dimensions = DimensionInfo(**data["dimensions"]) if data.get("dimensions") else None
        materials = [MaterialSpec(**m) for m in data.get("materials", [])]
        spatial_context = SpatialContext(**data["spatial_context"]) if data.get("spatial_context") else None
        space_type = SpaceType(data["space_type"]) if data.get("space_type") else None

        return cls(
            project_name=data["project_name"],
            project_address=data.get("project_address"),
            space_name=data.get("space_name"),
            space_type=space_type,
            dimensions=dimensions,
            materials=materials,
            spatial_context=spatial_context,
            design_intent=data.get("design_intent", []),
            style_notes=data.get("style_notes", []),
            source_documents=[Path(p) for p in data.get("source_documents", [])],
            metadata=data.get("metadata", {}),
        )


class ArchitecturalContextExtractor:
    """Extract architectural context from documents and file names."""

    def __init__(self):
        self.context_cache = {}

    def extract_from_filename(self, filename: str) -> ArchitecturalContext:
        """Extract context from structured filename."""
        # Example: "Giga-V2_750Picacho_Kitchen_compatible_kitchen-bright"
        # Example: "250930_MBAR_SUBMITTAL_2.pd"

        parts = Path(filename).stem.split("_")

        context = ArchitecturalContext(project_name="Unknown Project", metadata={"source_filename": filename})

        # Pattern matching for common naming conventions
        filename_lower = filename.lower()

        # Extract project address/name
        if "picacho" in filename_lower:
            context.project_name = "750 Picacho Lane"
            context.project_address = "750 Picacho Lane"
        elif "mbar" in filename_lower:
            context.project_name = "MBAR Project"

        # Extract space type
        space_keywords = {
            "kitchen": SpaceType.KITCHEN,
            "living": SpaceType.LIVING,
            "greatroom": SpaceType.LIVING,
            "dining": SpaceType.DINING,
            "bedroom": SpaceType.BEDROOM,
            "master": SpaceType.BEDROOM,
            "bath": SpaceType.BATHROOM,
            "pool": SpaceType.POOL_AREA,
            "courtyard": SpaceType.COURTYARD,
            "exterior": SpaceType.EXTERIOR,
        }

        for keyword, space_type in space_keywords.items():
            if keyword in filename_lower:
                context.space_type = space_type
                context.space_name = keyword.capitalize()
                break

        # Extract style hints from filename
        style_keywords = {
            "bright": "bright, airy atmosphere",
            "moody": "dramatic, moody lighting",
            "twilight": "twilight ambiance",
            "golden": "golden hour lighting",
            "coastal": "coastal contemporary style",
            "modern": "modern minimalist design",
            "luxury": "luxury finishes",
        }

        for keyword, style_note in style_keywords.items():
            if keyword in filename_lower:
                context.style_notes.append(style_note)

        return context

    def extract_from_pdf(self, pdf_path: Path) -> ArchitecturalContext:
        """Extract context from architectural PDF documents."""
        logger.info(f"Extracting context from PDF: {pdf_path}")

        # Start with filename context
        context = self.extract_from_filename(pdf_path.name)
        context.source_documents.append(pdf_path)

        # Try to import PDF parsing libraries
        try:
            from pypdf import PdfReader

            with open(pdf_path, "rb") as f:
                pdf = PdfReader(f)

                # Extract text from first few pages
                text = ""
                for page_num in range(min(5, len(pdf.pages))):
                    text += pdf.pages[page_num].extract_text()

                # Parse dimensions
                dimensions = self._parse_dimensions(text)
                if dimensions:
                    context.dimensions = dimensions

                # Parse materials
                materials = self._parse_materials(text)
                context.materials.extend(materials)

                # Extract design intent
                design_intent = self._parse_design_intent(text)
                context.design_intent.extend(design_intent)

        except ImportError:
            logger.warning("pypdf not installed. Install with: pip install pypdf")
        except Exception as e:
            logger.warning(f"Error parsing PDF: {e}")

        return context

    def _parse_dimensions(self, text: str) -> Optional[DimensionInfo]:
        """Parse dimensional information from text."""
        dims = DimensionInfo()

        # Pattern: "12' x 14'" or "12'-0" x 14'-6""
        pattern = r"(\d+)[''-]?\s*(?:\d+)?\s*[xX×]\s*(\d+)[''-]?"
        matches = re.findall(pattern, text)

        if matches:
            width, length = matches[0]
            dims.width = float(width)
            dims.length = float(length)
            dims.area = dims.width * dims.length

        # Pattern: "ceiling height: 10'-0""
        ceiling_pattern = r"ceiling\s+height[:\s]+(\d+)[''-]"
        ceiling_match = re.search(ceiling_pattern, text, re.IGNORECASE)
        if ceiling_match:
            dims.ceiling_height = float(ceiling_match.group(1))

        return dims if (dims.width or dims.ceiling_height) else None

    def _parse_materials(self, text: str) -> List[MaterialSpec]:
        """Parse material specifications from text."""
        materials = []

        # Common material keywords
        material_patterns = {
            r"(white oak|oak|walnut|maple)\s+(flooring|floor|cabinetry)": "wood",
            r"(quartz|marble|granite|stone)\s+(counter|countertop)": "stone",
            r"(stainless|brass|bronze|copper)\s+(hardware|fixture)": "metal",
            r"(glass|glazing)\s+(panel|door|window)": "glass",
        }

        for pattern, mat_type in material_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                materials.append(
                    MaterialSpec(
                        material_type=mat_type,
                        location="unspecified",
                        finish=match.group(0),
                    )
                )

        return materials[:10]  # Limit to top 10

    def _parse_design_intent(self, text: str) -> List[str]:
        """Extract design intent phrases from text."""
        intent_keywords = [
            "open concept",
            "natural light",
            "indoor-outdoor",
            "luxury finishes",
            "contemporary",
            "modern",
            "traditional",
            "transitional",
            "coastal",
            "desert modern",
            "mediterranean",
            "craftsman",
        ]

        found_intents = []
        text_lower = text.lower()

        for keyword in intent_keywords:
            if keyword in text_lower:
                found_intents.append(keyword)

        return found_intents[:5]  # Top 5


class ContextAwareRenderingPipeline:
    """Pipeline that integrates architectural context into rendering."""

    def __init__(self, context_dir: Path = Path("extracted_context")):
        self.context_dir = context_dir
        self.context_dir.mkdir(exist_ok=True)
        self.extractor = ArchitecturalContextExtractor()

    def prepare_context(self, image_path: Path, pdf_documents: Optional[List[Path]] = None) -> ArchitecturalContext:
        """Prepare architectural context for an image."""

        # Check cache
        cache_path = self.context_dir / f"{image_path.stem}_context.json"
        if cache_path.exists():
            logger.info(f"Loading cached context from {cache_path}")
            return ArchitecturalContext.load(cache_path)

        # Extract from filename
        context = self.extractor.extract_from_filename(image_path.name)

        # Enhance with PDF documents
        if pdf_documents:
            for pdf_path in pdf_documents:
                pdf_context = self.extractor.extract_from_pdf(pdf_path)

                # Merge contexts
                if pdf_context.dimensions:
                    context.dimensions = pdf_context.dimensions
                context.materials.extend(pdf_context.materials)
                context.design_intent.extend(pdf_context.design_intent)
                context.source_documents.append(pdf_path)

        # Save to cache
        context.save(cache_path)

        return context

    def enhance_prompt(self, base_prompt: str, image_path: Path, pdf_documents: Optional[List[Path]] = None) -> str:
        """Generate context-enhanced prompt."""

        context = self.prepare_context(image_path, pdf_documents)
        enhanced = context.to_enhanced_prompt(base_prompt)

        logger.info(f"Enhanced prompt: {enhanced}")
        return enhanced


def main():
    """Demonstration of architectural context extraction."""

    print("=" * 80)
    print("ARCHITECTURAL CONTEXT ENGINE - DEMONSTRATION")
    print("=" * 80)

    # Example 1: Extract from filename
    print("\n1. FILENAME CONTEXT EXTRACTION")
    print("-" * 80)

    extractor = ArchitecturalContextExtractor()

    test_filenames = [
        "Giga-V2_750Picacho_Kitchen_compatible_kitchen-bright.jpg",
        "250930_MBAR_SUBMITTAL_2.pd",
        "Coastal_Estate_Greatroom_Twilight.tif",
    ]

    for filename in test_filenames:
        context = extractor.extract_from_filename(filename)
        print(f"\nFilename: {filename}")
        print(f"  Project: {context.project_name}")
        print(f"  Space: {context.space_type}")
        print(f"  Style: {', '.join(context.style_notes)}")

    # Example 2: Enhanced prompt generation
    print("\n" + "=" * 80)
    print("2. ENHANCED PROMPT GENERATION")
    print("-" * 80)

    pipeline = ContextAwareRenderingPipeline()

    image_path = Path("input_images/Giga-V2_750Picacho_Kitchen_compatible_kitchen-bright.jpg")
    base_prompt = "photorealistic architectural rendering"

    enhanced_prompt = pipeline.enhance_prompt(base_prompt=base_prompt, image_path=image_path)

    print(f"\nBase prompt: {base_prompt}")
    print(f"Enhanced prompt: {enhanced_prompt}")

    # Example 3: PDF context extraction (if available)
    print("\n" + "=" * 80)
    print("3. PDF DOCUMENT ANALYSIS")
    print("-" * 80)

    pdf_path = Path.home() / "24098.00_750 PICACHO LANE.pd"
    if pdf_path.exists():
        pdf_context = extractor.extract_from_pdf(pdf_path)
        print(f"\nPDF: {pdf_path.name}")
        print(f"  Dimensions: {pdf_context.dimensions}")
        print(f"  Materials: {len(pdf_context.materials)} found")
        print(f"  Design Intent: {', '.join(pdf_context.design_intent)}")
    else:
        print(f"\nPDF not found: {pdf_path}")
        print("  (This is optional - context still works without PDFs)")

    print("\n" + "=" * 80)
    print("✅ Demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
