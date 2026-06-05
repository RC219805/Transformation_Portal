#!/usr/bin/env python3
"""
Example: Context-Aware Processing
Demonstrates architectural context integration with 750 Picacho Kitchen rendering
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utilities.architectural_context_engine import (
    ArchitecturalContext,
    ArchitecturalContextExtractor,
    DimensionInfo,
    MaterialSpec,
    SpaceType,
    SpatialContext,
)

print("=" * 80)
print("EXAMPLE: CONTEXT-AWARE PROCESSING")
print("750 Picacho Kitchen Rendering with Architectural Intelligence")
print("=" * 80)

# Step 1: Extract context from filename
print("\n[1/4] Extracting context from filename...")
print("-" * 80)

extractor = ArchitecturalContextExtractor()

image_path = Path("input_images/Giga-V2_750Picacho_Kitchen_compatible_kitchen-bright.jpg")
context = extractor.extract_from_filename(image_path.name)

print(f"Image: {image_path.name}")
print(f"  → Project: {context.project_name}")
print(f"  → Space: {context.space_type.value if context.space_type else 'Unknown'}")
print(f"  → Style: {', '.join(context.style_notes)}")

# Step 2: Enrich with PDF documents (if available)
print("\n[2/4] Enriching with architectural documents...")
print("-" * 80)

pdf_paths = [
    Path.home() / "24098.00_750 PICACHO LANE.pd",
    REPO_ROOT / "input_images" / "250930_MBAR SUBMITTAL 2.pd",
]

pdf_found = []
for pdf_path in pdf_paths:
    if pdf_path.exists():
        print(f"  ✓ Found: {pdf_path.name}")
        pdf_found.append(pdf_path)

        # Extract context from PDF
        pdf_context = extractor.extract_from_pdf(pdf_path)

        # Merge into main context
        if pdf_context.dimensions:
            context.dimensions = pdf_context.dimensions
        context.materials.extend(pdf_context.materials)
        context.design_intent.extend(pdf_context.design_intent)
        context.source_documents.append(pdf_path)
    else:
        print(f"  ✗ Not found: {pdf_path.name}")

# Step 3: Manual context enrichment (based on known project details)
print("\n[3/4] Enriching with known project details...")
print("-" * 80)

# Add dimensions (from floor plans)
if not context.dimensions:
    context.dimensions = DimensionInfo(width=18.0, length=22.0, ceiling_height=10.0, area=396.0, unit="feet")
    print(f"  Added dimensions: {context.dimensions.to_prompt_fragment()}")

# Add material specifications
if not context.materials:
    context.materials = [
        MaterialSpec(material_type="wood", location="cabinetry", finish="white oak", color="natural"),
        MaterialSpec(material_type="stone", location="countertop", finish="quartz", color="white"),
        MaterialSpec(material_type="metal", location="hardware", finish="brushed brass"),
        MaterialSpec(material_type="stone", location="backsplash", finish="marble", color="carrara white"),
        MaterialSpec(material_type="metal", location="appliances", finish="stainless steel"),
    ]
    print(f"  Added {len(context.materials)} material specifications")
    for mat in context.materials:
        print(f"    - {mat.to_prompt_fragment()}")

# Add spatial context
if not context.spatial_context:
    context.spatial_context = SpatialContext(
        space_name="Kitchen",
        space_type=SpaceType.KITCHEN,
        adjacent_spaces=["Dining Room", "Living Room"],
        windows=["North wall (3)", "East wall (2)"],
        doors=["Entry from dining", "Pantry door"],
        ceiling_type="Coffered ceiling",
        flooring_type="White oak hardwood",
    )
    print(f"  Added spatial context: {context.spatial_context.to_prompt_fragment()}")

# Add design intent
if not context.design_intent:
    context.design_intent = ["open concept", "natural light", "luxury finishes", "contemporary design", "indoor-outdoor flow"]
    print(f"  Added design intent: {', '.join(context.design_intent)}")

# Step 4: Generate enhanced prompts
print("\n[4/4] Generating context-enhanced prompts...")
print("-" * 80)

base_prompts = [
    "photorealistic architectural rendering",
    "magazine-quality luxury kitchen photography",
    "professional real estate marketing image",
]

print("\nEnhanced Prompts:")
for i, base_prompt in enumerate(base_prompts, 1):
    enhanced = context.to_enhanced_prompt(base_prompt)
    print(f"\n{i}. Base: {base_prompt}")
    print(f"   Enhanced: {enhanced}")

# Save context
output_path = Path("extracted_context/750Picacho_Kitchen_enriched_context.json")
context.save(output_path)

print("\n" + "=" * 80)
print("CONTEXT SUMMARY")
print("=" * 80)

print(f"\nProject: {context.project_name}")
if context.project_address:
    print(f"Address: {context.project_address}")

print(f"\nSpace: {context.space_type.value if context.space_type else 'Unknown'}")
if context.dimensions:
    print(f"Dimensions: {context.dimensions.to_prompt_fragment()}")

print(f"\nMaterials ({len(context.materials)}):")
for mat in context.materials:
    print(f"  - {mat.location}: {mat.to_prompt_fragment()}")

print("\nDesign Intent:")
for intent in context.design_intent:
    print(f"  - {intent}")

print("\nStyle Notes:")
for note in context.style_notes:
    print(f"  - {note}")

print("\nSource Documents:")
for doc in context.source_documents:
    print(f"  - {doc}")

print(f"\n📄 Context saved to: {output_path}")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

print(
    """
1. Run context-aware pipeline:
   python context_aware_pro_pipeline.py \\
       input_images/Giga-V2_750Picacho_Kitchen_compatible_kitchen-bright.jpg \\
       --pdf "$HOME/24098.00_750 PICACHO LANE.pd"

2. Or use the enriched context directly:
   from scripts.utilities.architectural_context_engine import ArchitecturalContext
   context = ArchitecturalContext.load(
       "extracted_context/750Picacho_Kitchen_enriched_context.json"
   )

3. Batch process with shared context:
   for image in input_images/*.jpg; do
       python context_aware_pro_pipeline.py "$image" --pdf floor_plans.pdf
   done
"""
)

print("=" * 80)
print("✅ Example complete!")
print("=" * 80)
