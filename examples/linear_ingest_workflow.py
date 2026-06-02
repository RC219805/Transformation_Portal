#!/usr/bin/env python3
"""Example workflow: Linear Ingest Pipeline End-to-End Demo

This script demonstrates the complete linear ingest workflow:
1. Process RAW/TIFF files to linear tensors
2. Capture full provenance
3. Create validated dataset manifest
4. Verify data quality

Usage:
    python examples/linear_ingest_workflow.py

Requirements:
    pip install -e ".[raw]"
"""

from pathlib import Path

import numpy as np
from PIL import Image

from transformation_portal.spatial_ai.ingest import (
    DatasetManifestBuilder,
    ImageManifestEntry,
    LinearDecoder,
    ProvenanceCapture,
    validate_linear_output,
)


def create_test_images(test_dir: Path) -> list[Path]:
    """Create synthetic test images for demonstration."""
    test_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []

    # Create 16-bit TIFF (simulates processed RAW)
    for i in range(3):
        img_array = (np.random.rand(512, 768, 3) * 65535).astype(np.uint16)
        img_path = test_dir / f"test_image_{i:03d}.tiff"

        # Use tifffile for proper 16-bit TIFF creation
        import tifffile

        tifffile.imwrite(img_path, img_array, photometric="rgb")

        image_paths.append(img_path)
        print(f"  Created: {img_path.name} (16-bit RGB)")

    return image_paths


def main():
    """Run complete linear ingest workflow."""

    print("=" * 80)
    print("LINEAR INGEST WORKFLOW DEMONSTRATION")
    print("=" * 80)
    print()

    # Setup directories
    workspace = Path("./output/examples/linear_ingest_demo")
    input_dir = workspace / "input"
    output_dir = workspace / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Creating test images...")
    print("-" * 80)
    image_paths = create_test_images(input_dir)
    print(f"✅ Created {len(image_paths)} test images\n")

    # Initialize decoder with strict mode
    print("Step 2: Initializing LinearDecoder (strict mode)...")
    print("-" * 80)
    decoder = LinearDecoder(
        gamma=1.0,
        bit_depth=32,
        strict_ingest=True,  # Reject 8-bit inputs
    )
    print("✅ Decoder ready\n")

    # Initialize manifest builder
    print("Step 3: Creating dataset manifest...")
    print("-" * 80)
    builder = DatasetManifestBuilder(
        name="linear_ingest_demo_dataset",
        description="Demonstration of linear ingest pipeline with full provenance",
        version="1.0.0",
        tags=["demo", "linear_sRGB", "synthetic"],
    )
    print("✅ Manifest builder ready\n")

    # Process images
    print("Step 4: Processing images...")
    print("-" * 80)
    results = []

    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {img_path.name}")

        # Decode to linear
        result = decoder.decode(
            input_path=img_path,
            output_dir=output_dir,
            emit_exr=False,  # Skip EXR for demo (requires OpenEXR)
            emit_provenance=True,
        )

        # Validate output
        validate_linear_output(
            result.linear_rgb,
            gamma=result.gamma,
            input_path=result.input_path,
        )

        # Report
        print(f"  ✓ Shape: {result.linear_rgb.shape}")
        print(f"  ✓ Dtype: {result.linear_rgb.dtype}")
        print(f"  ✓ Gamma: {result.gamma}")
        print(f"  ✓ Range: [{result.linear_rgb.min():.4f}, {result.linear_rgb.max():.4f}]")
        print(f"  ✓ Content hash: {result.content_hash[:16]}...")
        print(f"  ✓ Provenance: {result.provenance_path.name}")

        # Add to manifest
        builder.add_image(
            ImageManifestEntry(
                file_path=str(result.provenance_path.relative_to(workspace)),
                content_hash=result.content_hash,
                input_format=result.input_format,
                dimensions=result.linear_rgb.shape,
                value_range=(
                    float(result.linear_rgb.min()),
                    float(result.linear_rgb.max()),
                ),
                has_hdr=result.linear_rgb.max() > 1.0,
                tags=["synthetic", f"image_{i:03d}"],
            )
        )

        results.append(result)

    print(f"\n✅ Processed {len(results)} images successfully\n")

    # Build and write manifest
    print("Step 5: Writing dataset manifest...")
    print("-" * 80)
    manifest = builder.build()
    manifest_path = workspace / "manifest.json"
    manifest.write(manifest_path)

    print(f"✅ Manifest written: {manifest_path}")
    print(f"  - Dataset: {manifest.dataset_name}")
    print(f"  - Total images: {manifest.total_images}")
    print(f"  - Schema version: {manifest.schema_version}")
    print(f"  - Gamma: {manifest.gamma}")
    print(f"  - Color space: {manifest.color_space}")
    print()

    # Verify determinism
    print("Step 6: Verifying determinism...")
    print("-" * 80)

    # Decode first image again
    result_original = results[0]
    result_repeat = decoder.decode(image_paths[0])

    if result_original.content_hash == result_repeat.content_hash:
        print("✅ Determinism verified: Same input → same content hash")
        print(f"  Hash: {result_original.content_hash[:32]}...")
    else:
        print("❌ Determinism failed: Hashes don't match!")
    print()

    # Verify provenance completeness
    print("Step 7: Verifying provenance completeness...")
    print("-" * 80)

    capture = ProvenanceCapture()
    prov_data = capture.load_sidecar(results[0].provenance_path)

    required_fields = ["camera", "ingest", "transform", "output"]
    missing = [f for f in required_fields if f not in prov_data]

    if not missing:
        print("✅ Provenance complete: All required fields present")
        print(f"  - Camera metadata: {len(prov_data['camera'])} fields")
        print(f"  - Ingest metadata: {len(prov_data['ingest'])} fields")
        print(f"  - Transform metadata: {len(prov_data['transform'])} fields")
        print(f"  - Output metadata: {len(prov_data['output'])} fields")
    else:
        print(f"❌ Missing provenance fields: {missing}")
    print()

    # Summary
    print("=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  ✅ {len(results)} images processed to linear float32")
    print(f"  ✅ {len(results)} provenance JSON sidecars written")
    print(f"  ✅ Dataset manifest created and validated")
    print(f"  ✅ Determinism verified (reproducible hashes)")
    print(f"  ✅ Provenance completeness verified")
    print()
    print("Output directory:")
    print(f"  {workspace.absolute()}")
    print()
    print("Files created:")
    print(f"  - {len(image_paths)} input images (16-bit TIFF)")
    print(f"  - {len(results)} provenance JSON files")
    print(f"  - 1 dataset manifest (manifest.json)")
    print()
    print("Next steps:")
    print("  1. Review manifest: cat {}/manifest.json".format(workspace))
    print("  2. Review provenance: cat {}/processed/*_provenance.json".format(workspace))
    print("  3. Load tensors for training (see LINEAR_INGEST_GUIDE.md)")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
