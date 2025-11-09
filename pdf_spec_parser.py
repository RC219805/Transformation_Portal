#!/usr/bin/env python3
"""
PDF Specification Parser for Architectural Submittals
Extracts material specs, color palettes, and design intent from PDF documents

Uses PyPDF2 for text extraction with pattern matching for common architectural specs
"""

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import PyPDF2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ColorSpec:
    """Color specification from architectural documents."""
    name: str
    category: str  # primary, accent, neutral
    hex_code: Optional[str] = None
    rgb: Optional[Tuple[int, int, int]] = None
    application: Optional[str] = None  # walls, trim, accent, etc.

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.rgb:
            data['rgb'] = list(self.rgb)
        return data


@dataclass
class FinishSpec:
    """Finish specification from architectural documents."""
    material: str
    material_type: str  # wood, stone, metal, glass, etc.
    finish_type: str = 'natural'  # matte, glossy, brushed, honed, etc.
    manufacturer: Optional[str] = None
    product_name: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DimensionSpec:
    """Dimension specification from architectural documents."""
    element: str
    dimension: float
    unit: str = 'feet'
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PDFSpecParser:
    """
    Parse architectural specification PDFs for rendering-relevant data.

    Extracts:
    - Material specifications and finishes
    - Color palettes and paint schedules
    - Dimension schedules
    - Design intent descriptions
    - Technical specifications for surfaces
    """

    def __init__(self, pdf_path: Path):
        """Initialize with PDF path."""
        self.pdf_path = Path(pdf_path)
        self.text_content: str = ""
        self.pages: List[str] = []

    def extract_text(self) -> str:
        """Extract all text from PDF."""
        logger.info(f"Extracting text from: {self.pdf_path}")

        all_text = []
        try:
            with open(self.pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)
                logger.info(f"Processing {total_pages} pages")

                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    self.pages.append(text)
                    all_text.append(text)

                    if (i + 1) % 10 == 0:
                        logger.debug(f"Processed {i + 1}/{total_pages} pages")

            self.text_content = "\n".join(all_text)
            logger.info(f"Extracted {len(self.text_content)} characters from PDF")

        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")

        return self.text_content

    def extract_material_specs(self) -> List[FinishSpec]:
        """
        Extract material and finish specifications.
        Look for common architectural material keywords.
        """
        finishes = []

        # Material patterns for luxury estates
        material_patterns = [
            # Wood
            (r'(white oak|walnut|maple|cherry|mahogany|teak)\s+(\w+\s+)?finish', 'wood'),
            (r'custom\s+cabinetry.*?(\w+\s+oak|\w+\s+walnut)', 'wood'),
            (r'flooring.*?(oak|walnut|maple|hardwood)', 'wood'),

            # Stone
            (r'(carrara|calacatta|statuario)\s+marble', 'stone'),
            (r'(granite|limestone|travertine|onyx)\s+(\w+)', 'stone'),
            (r'(honed|polished|flamed)\s+(marble|granite|stone)', 'stone'),

            # Metal
            (r'(brushed|polished|satin)\s+(nickel|brass|bronze|steel)', 'metal'),
            (r'stainless\s+steel.*?(brushed|polished)', 'metal'),

            # Glass
            (r'(frameless|tempered|low-e|laminated)\s+glass', 'glass'),
            (r'glazing.*?(clear|tinted|frosted)', 'glass'),

            # Tile
            (r'(porcelain|ceramic|mosaic)\s+tile', 'tile'),
            (r'tile.*?(subway|hexagon|penny)', 'tile'),
        ]

        for pattern, category in material_patterns:
            matches = re.finditer(pattern, self.text_content, re.IGNORECASE)
            for match in matches:
                finishes.append(FinishSpec(
                    material=match.group(0),
                    material_type=category,
                    finish_type='standard',
                    location=self._find_location_context(match.start())
                ))

        # Deduplicate and clean
        unique_finishes = self._deduplicate_finishes(finishes)
        logger.info(f"Extracted {len(unique_finishes)} material specifications")

        return unique_finishes

    def extract_color_palette(self) -> List[ColorSpec]:
        """
        Extract color palette specifications.
        Look for paint schedules, color names, and hex/RGB codes.
        """
        colors = []

        # Color name patterns
        color_patterns = [
            # Sherwin Williams / Benjamin Moore style
            (r'(SW|BM)\s+\d{4}[-\s]*([\w\s]+)', 'paint'),

            # Generic color names with categories
            (r'(warm white|soft white|bright white|off-white)', 'neutral'),
            (r'(gray|grey|charcoal|slate)', 'neutral'),
            (r'(beige|taupe|greige|sand)', 'neutral'),
            (r'(navy|ocean blue|coastal blue)', 'accent'),
            (r'(sage|olive|moss|forest green)', 'accent'),
            (r'(terracotta|rust|clay)', 'accent'),

            # With hex codes
            (r'#([0-9A-Fa-f]{6})', 'hex'),

            # RGB values
            (r'RGB\s*\(?\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)?', 'rgb'),
        ]

        for pattern, category in color_patterns:
            matches = re.finditer(pattern, self.text_content, re.IGNORECASE)
            for match in matches:
                if category == 'hex':
                    colors.append(ColorSpec(
                        name=f"Color #{match.group(1)}",
                        category='accent',
                        hex_code=f"#{match.group(1)}"
                    ))
                elif category == 'rgb':
                    r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    colors.append(ColorSpec(
                        name=f"RGB Color",
                        category='custom',
                        rgb=(r, g, b)
                    ))
                else:
                    colors.append(ColorSpec(
                        name=match.group(0),
                        category=category,
                        application=self._find_location_context(match.start())
                    ))

        # Deduplicate
        unique_colors = self._deduplicate_colors(colors)
        logger.info(f"Extracted {len(unique_colors)} color specifications")

        return unique_colors

    def extract_dimensions(self) -> List[DimensionSpec]:
        """
        Extract dimension specifications.
        Look for room dimensions, ceiling heights, window sizes, etc.
        """
        dimensions = []

        # Dimension patterns
        dimension_patterns = [
            # Ceiling height
            (r'ceiling\s+height.*?(\d+)[\s\-]*(?:ft|feet|\')', 'ceiling_height'),
            (r'(\d+)[\s\-]*(?:ft|feet|\')\s+ceiling', 'ceiling_height'),

            # Room dimensions
            (r'(\d+)[\s\-]*x\s*(\d+)[\s\-]*(?:ft|feet)', 'room_dimension'),

            # Window/door sizes
            (r'(window|door).*?(\d+)[\s\-]*x\s*(\d+)', 'opening'),

            # General dimensions
            (r'(\d+)[\s\-]*(?:ft|feet|\')\s+(\w+)', 'general'),
        ]

        for pattern, dim_type in dimension_patterns:
            matches = re.finditer(pattern, self.text_content, re.IGNORECASE)
            for match in matches:
                try:
                    if dim_type == 'room_dimension':
                        dimensions.append(DimensionSpec(
                            element='room',
                            dimension=float(match.group(1)),
                            unit='feet',
                            notes=f"{match.group(1)} x {match.group(2)} feet"
                        ))
                    elif dim_type == 'ceiling_height':
                        dimensions.append(DimensionSpec(
                            element='ceiling_height',
                            dimension=float(match.group(1)),
                            unit='feet'
                        ))
                except (ValueError, IndexError):
                    continue

        logger.info(f"Extracted {len(dimensions)} dimension specifications")
        return dimensions

    def extract_design_intent(self) -> Dict[str, str]:
        """
        Extract design intent and descriptive keywords.
        Look for sections with design philosophy, style descriptions.
        """
        intent = {
            'style_keywords': [],
            'ambiance': [],
            'key_features': [],
        }

        # Style keywords
        style_patterns = [
            'mediterranean', 'modern', 'contemporary', 'traditional', 'transitional',
            'coastal', 'luxury', 'estate', 'villa', 'resort-style',
            'minimalist', 'elegant', 'sophisticated', 'timeless', 'classic',
        ]

        for keyword in style_patterns:
            if re.search(r'\b' + keyword + r'\b', self.text_content, re.IGNORECASE):
                intent['style_keywords'].append(keyword)

        # Ambiance keywords
        ambiance_patterns = [
            'bright', 'airy', 'open', 'spacious', 'intimate', 'cozy',
            'warm', 'cool', 'inviting', 'serene', 'dramatic', 'tranquil',
        ]

        for keyword in ambiance_patterns:
            if re.search(r'\b' + keyword + r'\b', self.text_content, re.IGNORECASE):
                intent['ambiance'].append(keyword)

        # Key features
        feature_patterns = [
            'floor-to-ceiling', 'vaulted ceiling', 'skylights', 'french doors',
            'custom cabinetry', 'built-in', 'fireplace', 'wine cellar',
            'pool', 'spa', 'outdoor living', 'ocean view', 'mountain view',
        ]

        for keyword in feature_patterns:
            if re.search(r'\b' + keyword + r'\b', self.text_content, re.IGNORECASE):
                intent['key_features'].append(keyword)

        logger.info(f"Extracted design intent: {len(intent['style_keywords'])} style keywords")
        return intent

    def _find_location_context(self, position: int, window: int = 100) -> Optional[str]:
        """Find location context around a match position."""
        start = max(0, position - window)
        end = min(len(self.text_content), position + window)
        context = self.text_content[start:end]

        # Look for room names
        room_patterns = ['kitchen', 'bath', 'bedroom', 'living', 'dining', 'entry', 'pool']
        for room in room_patterns:
            if room in context.lower():
                return room

        return None

    def _deduplicate_finishes(self, finishes: List[FinishSpec]) -> List[FinishSpec]:
        """Remove duplicate finish specifications."""
        seen = set()
        unique = []
        for finish in finishes:
            key = (finish.material.lower(), finish.finish_type)
            if key not in seen:
                seen.add(key)
                unique.append(finish)
        return unique

    def _deduplicate_colors(self, colors: List[ColorSpec]) -> List[ColorSpec]:
        """Remove duplicate color specifications."""
        seen = set()
        unique = []
        for color in colors:
            key = color.name.lower()
            if key not in seen:
                seen.add(key)
                unique.append(color)
        return unique

    def parse_all(self) -> Dict[str, Any]:
        """
        Parse all specifications from PDF.

        Returns:
            Complete specification dictionary
        """
        logger.info(f"Parsing PDF specifications from: {self.pdf_path}")

        # Extract text
        self.extract_text()

        if not self.text_content:
            logger.warning("No text content extracted from PDF")
            return self._get_fallback_specs()

        # Extract all specs
        materials = self.extract_material_specs()
        colors = self.extract_color_palette()
        dimensions = self.extract_dimensions()
        design_intent = self.extract_design_intent()

        result = {
            'pdf_file': str(self.pdf_path),
            'pdf_file_size_mb': self.pdf_path.stat().st_size / (1024 * 1024),
            'total_pages': len(self.pages),
            'text_length': len(self.text_content),
            'material_specifications': [m.to_dict() for m in materials],
            'color_palette': [c.to_dict() for c in colors],
            'dimensions': [d.to_dict() for d in dimensions],
            'design_intent': design_intent,
            'extraction_summary': {
                'materials_count': len(materials),
                'colors_count': len(colors),
                'dimensions_count': len(dimensions),
                'style_keywords': len(design_intent['style_keywords']),
            }
        }

        logger.info(f"Parsed {len(materials)} materials, {len(colors)} colors, {len(dimensions)} dimensions")

        return result

    def _get_fallback_specs(self) -> Dict[str, Any]:
        """
        Provide fallback specifications for 750 Picacho Lane.
        Based on typical Montecito luxury estate standards.
        """
        logger.info("Using fallback specifications for 750 Picacho Lane")

        return {
            'pdf_file': str(self.pdf_path),
            'extraction_method': 'fallback_montecito_luxury_standards',
            'material_specifications': [
                FinishSpec('White Oak Flooring', 'wood', 'matte',
                           location='living areas').to_dict(),
                FinishSpec('Carrara Marble', 'stone', 'honed',
                           location='kitchen/bath').to_dict(),
                FinishSpec('Stainless Steel', 'metal', 'brushed',
                           location='kitchen').to_dict(),
                FinishSpec('Floor-to-Ceiling Glass', 'glass', 'clear',
                           location='living areas').to_dict(),
            ],
            'color_palette': [
                ColorSpec('Warm White', 'primary', application='walls').to_dict(),
                ColorSpec('Soft Gray', 'neutral', application='trim').to_dict(),
                ColorSpec('Ocean Blue', 'accent', application='accents').to_dict(),
                ColorSpec('Natural Wood', 'primary', application='floors/cabinetry').to_dict(),
            ],
            'dimensions': [
                DimensionSpec('ceiling_height', 12.0, 'feet', 'living areas').to_dict(),
                DimensionSpec('ceiling_height', 10.0, 'feet', 'bedrooms').to_dict(),
            ],
            'design_intent': {
                'style_keywords': ['mediterranean', 'luxury', 'coastal', 'elegant'],
                'ambiance': ['bright', 'airy', 'serene', 'sophisticated'],
                'key_features': ['floor-to-ceiling', 'pool', 'ocean view', 'custom cabinetry'],
            },
            'extraction_summary': {
                'materials_count': 4,
                'colors_count': 4,
                'dimensions_count': 2,
                'style_keywords': 4,
            }
        }

    def save_specs(self, output_path: Path, specs: Dict[str, Any]) -> None:
        """Save extracted specifications to JSON."""
        with open(output_path, 'w') as f:
            json.dump(specs, f, indent=2)
        logger.info(f"Saved specifications to: {output_path}")


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Parse architectural specifications from PDF')
    parser.add_argument('pdf_file', type=Path, help='Path to PDF file')
    parser.add_argument('--output', '-o', type=Path, default=Path('pdf_specs.json'),
                       help='Output JSON file')

    args = parser.parse_args()

    parser_obj = PDFSpecParser(args.pdf_file)
    specs = parser_obj.parse_all()
    parser_obj.save_specs(args.output, specs)

    print(f"\nExtracted specifications:")
    print(f"  PDF: {specs['pdf_file']}")
    print(f"  Pages: {specs.get('total_pages', 'N/A')}")
    print(f"  Materials: {specs['extraction_summary']['materials_count']}")
    print(f"  Colors: {specs['extraction_summary']['colors_count']}")
    print(f"  Dimensions: {specs['extraction_summary']['dimensions_count']}")
    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
