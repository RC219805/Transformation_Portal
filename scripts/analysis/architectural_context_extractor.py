#!/usr/bin/env python3
"""
Architectural Context Extractor
Transformation Portal - Intelligent Document Analysis

Extracts architectural intelligence from PDFs:
- Floor plans and dimensions
- Elevation drawings
- Material specifications
- Room identifications
- Design intent annotations
- Project metadata

Enriches rendering pipeline with contextual understanding.
"""

import io
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - depends on optional local runtime
    fitz = None


@dataclass
class RoomContext:
    """Room-specific context from plans."""

    name: str
    dimensions: Optional[Tuple[float, float]] = None  # (width, depth) in feet
    floor_level: Optional[str] = None
    ceiling_height: Optional[float] = None
    materials: List[str] = None
    features: List[str] = None
    adjacent_rooms: List[str] = None

    def __post_init__(self):
        if self.materials is None:
            self.materials = []
        if self.features is None:
            self.features = []
        if self.adjacent_rooms is None:
            self.adjacent_rooms = []


@dataclass
class ProjectContext:
    """Complete project architectural context."""

    project_name: str
    project_number: Optional[str] = None
    address: Optional[str] = None
    architect: Optional[str] = None
    total_sqft: Optional[float] = None
    floors: List[str] = None
    rooms: Dict[str, RoomContext] = None
    materials_palette: List[str] = None
    design_style: Optional[str] = None
    extracted_images: List[str] = None
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


class ArchitecturalContextExtractor:
    """Extract architectural intelligence from construction documents."""

    # Pattern recognition for common architectural elements
    ROOM_PATTERNS = {
        "kitchen": r"kitchen|kitch\b",
        "bathroom": r"bath(?:room)?|powder\s+room",
        "bedroom": r"bed(?:room)?|master\s+(?:bed|suite)|primary\s+(?:bed|suite)",
        "living": r"living|great\s+room|family\s+room",
        "dining": r"dining",
        "office": r"office|study|den",
        "garage": r"garage|carport",
        "entry": r"entry|foyer|vestibule",
        "laundry": r"laundry|utility",
        "outdoor": r"pool|patio|deck|terrace|courtyard|veranda",
    }

    MATERIAL_PATTERNS = {
        "wood": r"wood|oak|walnut|maple|cherry|timber|veneer",
        "stone": r"stone|granite|marble|limestone|travertine|slate",
        "metal": r"metal|steel|bronze|brass|copper|aluminum",
        "glass": r"glass|glazing|window|skylight",
        "concrete": r"concrete|cement",
        "tile": r"tile|porcelain|ceramic",
        "fabric": r"fabric|textile|upholstery|linen",
        "leather": r"leather",
    }

    DIMENSION_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*[\'\"]?\s*(?:x|×|by)\s*(\d+(?:\.\d+)?)\s*[\'\"]?", re.IGNORECASE)

    def __init__(self, output_dir: Path = None):
        """Initialize extractor."""
        self.output_dir = output_dir or Path("extracted_context")
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def extract_from_pdf(self, pdf_path: Path) -> ProjectContext:
        """
        Extract comprehensive architectural context from PDF.

        Args:
            pdf_path: Path to architectural PDF (plans, elevations, specs)

        Returns:
            ProjectContext with extracted intelligence
        """
        if fitz is None:
            raise RuntimeError("PyMuPDF is required for PDF extraction. Install with: pip install PyMuPDF")

        print(f"\n{'='*70}")
        print("EXTRACTING ARCHITECTURAL CONTEXT")
        print(f"{'='*70}")
        print(f"PDF: {pdf_path.name}")

        doc = fitz.open(pdf_path)

        # Initialize context
        context = ProjectContext(
            project_name=self._extract_project_name(pdf_path.stem),
            project_number=None,
            address=None,
        )

        # Extract text from all pages
        all_text = []
        page_texts = []

        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            all_text.append(text)
            page_texts.append((page_num, text))

        context.raw_text = "\n\n".join(all_text)

        # Extract metadata
        self._extract_metadata(context, doc.metadata)

        # Extract project information
        self._extract_project_info(context, all_text[0] if all_text else "")

        # Extract room information
        self._extract_rooms(context, page_texts)

        # Extract materials palette
        self._extract_materials(context, context.raw_text)

        # Extract design style
        self._infer_design_style(context)

        # Extract images (floor plans, elevations, sections)
        self._extract_images(context, doc, pdf_path.stem)

        # Save context
        self._save_context(context, pdf_path.stem)

        doc.close()

        print("\n✓ Extraction complete")
        print(f"  Rooms identified: {len(context.rooms)}")
        print(f"  Materials found: {len(context.materials_palette)}")
        print(f"  Images extracted: {len(context.extracted_images)}")

        return context

    def _extract_project_name(self, filename: str) -> str:
        """Extract project name from filename."""
        # Remove common prefixes/suffixes
        name = re.sub(r"^\d+[\._-]\s*", "", filename)  # Remove leading numbers
        name = re.sub(r"[-_]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def _extract_metadata(self, context: ProjectContext, metadata: dict):
        """Extract PDF metadata."""
        if metadata:
            if "title" in metadata and metadata["title"]:
                context.project_name = metadata["title"]
            if "subject" in metadata and metadata["subject"]:
                # Often contains project number or address
                subject = metadata["subject"]
                project_num_match = re.search(r"\b(\d{5,}(?:\.\d+)?)\b", subject)
                if project_num_match:
                    context.project_number = project_num_match.group(1)

    def _extract_project_info(self, context: ProjectContext, first_page_text: str):
        """Extract project information from title block."""
        lines = first_page_text.split("\n")

        # Look for project number
        for line in lines[:30]:  # Check first 30 lines
            if not context.project_number:
                proj_match = re.search(r"(?:project|job)\s*#?\s*:?\s*(\d{5,}(?:\.\d+)?)", line, re.I)
                if proj_match:
                    context.project_number = proj_match.group(1)

            # Look for address
            if not context.address:
                # Match street addresses
                addr_match = re.search(
                    r"\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Street|St|Avenue|Ave|Lane|Ln|Road|Rd|Drive|Dr|Way|Court|Ct))",
                    line,
                )
                if addr_match:
                    context.address = addr_match.group(0)

            # Look for architect
            if not context.architect:
                arch_match = re.search(r"(?:architect|designer)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", line, re.I)
                if arch_match:
                    context.architect = arch_match.group(1)

    def _extract_rooms(self, context: ProjectContext, page_texts: List[Tuple[int, str]]):
        """Extract room information from floor plans."""
        for page_num, text in page_texts:
            # Detect floor level
            floor_level = None
            if re.search(r"(?:1st|first)\s+floor", text, re.I):
                floor_level = "1st Floor"
            elif re.search(r"(?:2nd|second)\s+floor", text, re.I):
                floor_level = "2nd Floor"
            elif re.search(r"(?:ground|main)\s+floor", text, re.I):
                floor_level = "Ground Floor"

            if floor_level and floor_level not in context.floors:
                context.floors.append(floor_level)

            # Find rooms
            for room_type, pattern in self.ROOM_PATTERNS.items():
                matches = re.finditer(pattern, text, re.I)
                for match in matches:
                    # Extract context around match
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    room_context_text = text[start:end]

                    # Create room key
                    room_name = match.group(0).title()
                    room_key = f"{room_type}_{len([r for r in context.rooms.keys() if r.startswith(room_type)])}"

                    if room_key not in context.rooms:
                        room = RoomContext(name=room_name, floor_level=floor_level)

                        # Extract dimensions
                        dim_match = self.DIMENSION_PATTERN.search(room_context_text)
                        if dim_match:
                            try:
                                width = float(dim_match.group(1))
                                depth = float(dim_match.group(2))
                                room.dimensions = (width, depth)
                            except ValueError:
                                pass

                        # Extract ceiling height
                        height_match = re.search(r"(\d+(?:\.\d+)?)\s*[\'\"]?\s*(?:ceiling|clg|ht)", room_context_text, re.I)
                        if height_match:
                            try:
                                room.ceiling_height = float(height_match.group(1))
                            except ValueError:
                                pass

                        context.rooms[room_key] = room

    def _extract_materials(self, context: ProjectContext, text: str):
        """Extract materials palette from specifications."""
        material_counts = {}

        for material_type, pattern in self.MATERIAL_PATTERNS.items():
            matches = list(re.finditer(pattern, text, re.I))
            if matches:
                material_counts[material_type] = len(matches)

        # Add materials sorted by frequency
        sorted_materials = sorted(material_counts.items(), key=lambda x: x[1], reverse=True)
        context.materials_palette = [mat for mat, _ in sorted_materials]

    def _infer_design_style(self, context: ProjectContext):
        """Infer design style from context clues."""
        text_lower = context.raw_text.lower()

        style_indicators = {
            "Modern": ["modern", "contemporary", "minimalist", "clean lines"],
            "Traditional": ["traditional", "classic", "colonial", "crown molding"],
            "Transitional": ["transitional", "blend", "timeless"],
            "Mediterranean": ["mediterranean", "spanish", "tile roo", "stucco"],
            "Craftsman": ["craftsman", "bungalow", "exposed beams"],
            "Industrial": ["industrial", "exposed", "concrete", "metal"],
            "Luxury Estate": ["estate", "luxury", "grand", "palatial"],
        }

        style_scores = {}
        for style, keywords in style_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                style_scores[style] = score

        if style_scores:
            context.design_style = max(style_scores, key=style_scores.get)

    def _extract_images(self, context: ProjectContext, doc, pdf_stem: str):
        """Extract images (plans, elevations) from PDF."""
        images_dir = self.output_dir / f"{pdf_stem}_images"
        images_dir.mkdir(exist_ok=True)

        for page_num, page in enumerate(doc, 1):
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Save image
                image_filename = f"page{page_num:02d}_img{img_index:02d}.{image_ext}"
                image_path = images_dir / image_filename

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                context.extracted_images.append(str(image_path))

                # Try to classify image (floor plan vs elevation vs detail)
                # This could be enhanced with ML-based image classification

    def _save_context(self, context: ProjectContext, pdf_stem: str):
        """Save extracted context to JSON."""
        output_path = self.output_dir / f"{pdf_stem}_context.json"

        # Convert to dict (handle RoomContext objects)
        context_dict = {
            "project_name": context.project_name,
            "project_number": context.project_number,
            "address": context.address,
            "architect": context.architect,
            "total_sqft": context.total_sqft,
            "floors": context.floors,
            "rooms": {k: asdict(v) for k, v in context.rooms.items()},
            "materials_palette": context.materials_palette,
            "design_style": context.design_style,
            "extracted_images": context.extracted_images,
        }

        with open(output_path, "w") as f:
            json.dump(context_dict, f, indent=2)

        print(f"\n✓ Context saved: {output_path}")

    def load_context(self, pdf_stem: str) -> Optional[ProjectContext]:
        """Load previously extracted context."""
        context_path = self.output_dir / f"{pdf_stem}_context.json"

        if not context_path.exists():
            return None

        with open(context_path, "r") as f:
            data = json.load(f)

        # Reconstruct RoomContext objects
        rooms = {}
        for room_key, room_data in data.get("rooms", {}).items():
            rooms[room_key] = RoomContext(**room_data)

        context = ProjectContext(
            project_name=data["project_name"],
            project_number=data.get("project_number"),
            address=data.get("address"),
            architect=data.get("architect"),
            total_sqft=data.get("total_sqft"),
            floors=data.get("floors", []),
            rooms=rooms,
            materials_palette=data.get("materials_palette", []),
            design_style=data.get("design_style"),
            extracted_images=data.get("extracted_images", []),
        )

        return context


def main():
    """CLI for architectural context extraction."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract architectural context from construction documents")
    parser.add_argument("pdf", type=Path, help="PDF file to analyze")
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("extracted_context"), help="Output directory for extracted context"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"✗ PDF not found: {args.pdf}")
        return 1

    extractor = ArchitecturalContextExtractor(output_dir=args.output)
    try:
        context = extractor.extract_from_pdf(args.pdf)
    except RuntimeError as exc:
        print(f"✗ {exc}")
        return 1

    if args.verbose:
        print(f"\n{'='*70}")
        print("EXTRACTED CONTEXT SUMMARY")
        print(f"{'='*70}")
        print(f"\nProject: {context.project_name}")
        if context.project_number:
            print(f"Number: {context.project_number}")
        if context.address:
            print(f"Address: {context.address}")
        if context.design_style:
            print(f"Style: {context.design_style}")

        print(f"\nFloors: {', '.join(context.floors) if context.floors else 'Unknown'}")

        print(f"\nRooms ({len(context.rooms)}):")
        for room_key, room in context.rooms.items():
            dims = f"{room.dimensions[0]}' x {room.dimensions[1]}'" if room.dimensions else "Unknown"
            floor = f" [{room.floor_level}]" if room.floor_level else ""
            print(f"  • {room.name}: {dims}{floor}")

        print("\nMaterials Palette:")
        for material in context.materials_palette[:10]:  # Top 10
            print(f"  • {material.title()}")

    return 0


if __name__ == "__main__":
    exit(main())
