#!/usr/bin/env python3
"""
Extract Architectural Context from PDFs
Transformation Portal - Context-Aware Rendering

Extracts:
- Floor plans and elevations
- Dimensions and measurements
- Material specifications
- Room relationships
- Design intent

Enhances rendering pipeline with architectural knowledge.
"""

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF

    HAS_PYMUPDF = True
except ImportError:
    fitz = None
    HAS_PYMUPDF = False


@dataclass
class ArchitecturalContext:
    """Structured architectural context from documents."""

    property_address: Optional[str] = None
    project_number: Optional[str] = None
    rooms: List[str] = None
    dimensions: Dict[str, str] = None
    materials: Dict[str, List[str]] = None
    floor_plan_pages: List[int] = None
    elevation_pages: List[int] = None
    detail_pages: List[int] = None
    extracted_text: List[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.rooms is None:
            self.rooms = []
        if self.dimensions is None:
            self.dimensions = {}
        if self.materials is None:
            self.materials = {}
        if self.floor_plan_pages is None:
            self.floor_plan_pages = []
        if self.elevation_pages is None:
            self.elevation_pages = []
        if self.detail_pages is None:
            self.detail_pages = []
        if self.extracted_text is None:
            self.extracted_text = []
        if self.metadata is None:
            self.metadata = {}


class ArchitecturalContextExtractor:
    """Extract architectural context from PDF documents."""

    # Pattern recognition for architectural drawings
    ROOM_PATTERNS = [
        r"kitchen",
        r"bedroom",
        r"bath",
        r"living",
        r"dining",
        r"entry",
        r"foyer",
        r"office",
        r"studio",
        r"gallery",
        r"great\s*room",
        r"master\s*bed",
        r"primary\s*bed",
    ]

    DIMENSION_PATTERN = r"(\d+(?:\.\d+)?)\s*[\'\"x×]\s*(\d+(?:\.\d+)?)"

    MATERIAL_KEYWORDS = {
        "wood": ["oak", "walnut", "maple", "cherry", "wood", "timber"],
        "stone": ["granite", "marble", "limestone", "travertine", "stone"],
        "metal": ["steel", "bronze", "brass", "aluminum", "metal"],
        "glass": ["glass", "glazing", "window"],
        "tile": ["tile", "ceramic", "porcelain"],
    }

    DRAWING_TYPE_KEYWORDS = {
        "floor_plan": ["floor plan", "plan view", "layout"],
        "elevation": ["elevation", "front view", "side view"],
        "section": ["section", "cross section"],
        "detail": ["detail", "enlarged"],
    }

    def __init__(self):
        self.context = ArchitecturalContext()

    def extract_from_pdf(self, pdf_path: Path) -> ArchitecturalContext:
        """Extract architectural context from PDF."""
        if not HAS_PYMUPDF:
            print("✗ PyMuPDF required for PDF extraction")
            return self.context

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"✗ PDF not found: {pdf_path}")
            return self.context

        print(f"\n{'='*70}")
        print("EXTRACTING ARCHITECTURAL CONTEXT")
        print(f"{'='*70}")
        print(f"Source: {pdf_path.name}")

        try:
            doc = fitz.open(pdf_path)
            self.context.metadata = {
                "filename": pdf_path.name,
                "pages": len(doc),
                "title": doc.metadata.get("title", ""),
                "subject": doc.metadata.get("subject", ""),
            }

            print(f"Pages: {len(doc)}")

            # Extract from filename
            self._extract_from_filename(pdf_path.name)

            # Process each page
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                self.context.extracted_text.append(text)

                # Classify page type
                page_type = self._classify_page(text)
                if "floor" in page_type or "plan" in page_type:
                    self.context.floor_plan_pages.append(page_num)
                elif "elevation" in page_type:
                    self.context.elevation_pages.append(page_num)
                elif "detail" in page_type:
                    self.context.detail_pages.append(page_num)

                # Extract rooms
                rooms = self._extract_rooms(text)
                self.context.rooms.extend(rooms)

                # Extract dimensions
                dimensions = self._extract_dimensions(text)
                self.context.dimensions.update(dimensions)

                # Extract materials
                materials = self._extract_materials(text)
                for material_type, items in materials.items():
                    if material_type not in self.context.materials:
                        self.context.materials[material_type] = []
                    self.context.materials[material_type].extend(items)

            doc.close()

            # Deduplicate
            self.context.rooms = sorted(set([r.lower() for r in self.context.rooms]))
            for mat_type in self.context.materials:
                self.context.materials[mat_type] = sorted(set(self.context.materials[mat_type]))

            self._print_summary()

        except Exception as e:
            print(f"✗ Error extracting context: {e}")

        return self.context

    def _extract_from_filename(self, filename: str):
        """Extract context from filename."""
        # Project number (e.g., "24098.00")
        proj_match = re.search(r"(\d{5}\.\d{2})", filename)
        if proj_match:
            self.context.project_number = proj_match.group(1)

        # Address (e.g., "750 PICACHO LANE")
        addr_match = re.search(r"(\d+\s+[A-Z\s]+(?:LANE|ROAD|DRIVE|STREET|WAY|COURT))", filename)
        if addr_match:
            self.context.property_address = addr_match.group(1).strip()

    def _classify_page(self, text: str) -> str:
        """Classify page as floor plan, elevation, etc."""
        text_lower = text.lower()
        for page_type, keywords in self.DRAWING_TYPE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return page_type
        return "general"

    def _extract_rooms(self, text: str) -> List[str]:
        """Extract room names from text."""
        rooms = []
        text_lower = text.lower()
        for pattern in self.ROOM_PATTERNS:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                rooms.append(match.group())
        return rooms

    def _extract_dimensions(self, text: str) -> Dict[str, str]:
        """Extract dimensions from text."""
        dimensions = {}
        matches = re.finditer(self.DIMENSION_PATTERN, text)
        for i, match in enumerate(matches):
            dim_str = f"{match.group(1)} x {match.group(2)}"
            dimensions[f"dim_{i}"] = dim_str
        return dimensions

    def _extract_materials(self, text: str) -> Dict[str, List[str]]:
        """Extract material specifications."""
        materials = {}
        text_lower = text.lower()

        for material_type, keywords in self.MATERIAL_KEYWORDS.items():
            found = []
            for keyword in keywords:
                if keyword in text_lower:
                    found.append(keyword)
            if found:
                materials[material_type] = found

        return materials

    def _print_summary(self):
        """Print extraction summary."""
        print(f"\n{'='*70}")
        print("EXTRACTED CONTEXT")
        print(f"{'='*70}")

        if self.context.project_number:
            print(f"\n📋 Project: {self.context.project_number}")
        if self.context.property_address:
            print(f"📍 Address: {self.context.property_address}")

        if self.context.rooms:
            print(f"\n🏠 Rooms ({len(self.context.rooms)}):")
            for room in self.context.rooms[:10]:
                print(f"   • {room}")
            if len(self.context.rooms) > 10:
                print(f"   ... and {len(self.context.rooms) - 10} more")

        if self.context.materials:
            print("\n🎨 Materials:")
            for mat_type, items in self.context.materials.items():
                print(f"   • {mat_type}: {', '.join(items[:3])}")

        if self.context.floor_plan_pages:
            print(f"\n📐 Floor Plans: Pages {self.context.floor_plan_pages}")
        if self.context.elevation_pages:
            print(f"📐 Elevations: Pages {self.context.elevation_pages}")
        if self.context.detail_pages:
            print(f"📐 Details: Pages {self.context.detail_pages}")

    def save_json(self, output_path: Path):
        """Save context as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(asdict(self.context), f, indent=2)

        print(f"\n✓ Context saved: {output_path}")


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: extract_architectural_context.py <pdf_file> [output.json]")
        print("\nExample:")
        print("  python extract_architectural_context.py floor_plans.pdf context.json")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else pdf_path.with_suffix(".json")

    extractor = ArchitecturalContextExtractor()
    context = extractor.extract_from_pdf(pdf_path)
    extractor.save_json(output_path)

    print("\n✅ Extraction complete!")


if __name__ == "__main__":
    main()
