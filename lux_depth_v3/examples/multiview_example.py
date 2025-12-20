#!/usr/bin/env python3
"""
Multi-view reconstruction example for Lux Depth V3.

Demonstrates camera pose estimation and 3D reconstruction.
"""

from pathlib import Path
import numpy as np
from PIL import Image

from lux_depth_v3 import (
    DA3Config,
    Preset,
    InputManager,
    DA3InferenceEngine,
)
from lux_depth_v3.input_manager import CameraPose
from lux_depth_v3.config import InferenceMode, ExportFormat
from lux_depth_v3.postprocessing import Postprocessor
from lux_depth_v3.export import Exporter


def create_circular_camera_poses(num_views: int, radius: float = 2.0) -> list:
    """Create camera poses in a circular pattern around origin.
    
    Args:
        num_views: Number of camera views
        radius: Distance from origin
    
    Returns:
        List of CameraPose objects
    """
    poses = []
    
    for i in range(num_views):
        # Angle around circle
        angle = 2 * np.pi * i / num_views
        
        # Camera position (looking at origin)
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        y = 0.0
        
        # Rotation matrix (camera looking at origin)
        # Y-axis rotation
        rotation = np.array([
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)],
        ])
        
        translation = np.array([x, y, z])
        
        # Camera intrinsics (assuming 1024x1024 image, FOV ~60 degrees)
        focal_length = (1000.0, 1000.0)
        principal_point = (512.0, 512.0)
        
        pose = CameraPose(
            rotation=rotation,
            translation=translation,
            focal_length=focal_length,
            principal_point=principal_point,
        )
        
        poses.append(pose)
    
    return poses


def main():
    """Run multi-view reconstruction example."""
    
    print("=" * 60)
    print("Lux Depth V3 - Multi-View Reconstruction Example")
    print("=" * 60)
    
    # Configuration for multi-view
    print("\n1. Configuration")
    config = DA3Config.from_preset(Preset.ARCHITECTURAL_3D)
    config.inference_mode = InferenceMode.MULTI_VIEW
    config.export.output_dir = Path("multiview_output")
    config.export.formats = [ExportFormat.PLY, ExportFormat.NPZ]
    
    print(f"   Model: {config.model_variant.value}")
    print(f"   Mode: {config.inference_mode.value}")
    print(f"   Export: {config.export.formats}")
    
    # Create camera poses
    print("\n2. Camera Setup")
    num_views = 8
    poses = create_circular_camera_poses(num_views, radius=3.0)
    
    print(f"   Created {len(poses)} camera poses")
    print(f"   Pattern: Circular (radius=3.0m)")
    
    # Generate synthetic views (in practice, these would be real images)
    print("\n3. Creating Synthetic Views")
    sample_dir = Path("multiview_input")
    sample_dir.mkdir(exist_ok=True)
    
    manager = InputManager(inference_mode=InferenceMode.MULTI_VIEW)
    
    for i, pose in enumerate(poses):
        # Generate view-dependent image (in practice, from renderer/camera)
        # Here we create a simple pattern that varies with view angle
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # View-dependent pattern
        angle = 2 * np.pi * i / num_views
        r_value = int(127 + 127 * np.cos(angle))
        g_value = int(127 + 127 * np.sin(angle))
        
        img[:, :, 0] = r_value
        img[:, :, 1] = g_value
        img[:, :, 2] = 128
        
        # Add some structure
        img[200:300, 200:300] = [255, 255, 255]
        
        # Save and add to manager
        img_path = sample_dir / f"view_{i:02d}.jpg"
        Image.fromarray(img).save(img_path)
        
        manager.add_image(path=img_path, pose=pose)
        print(f"   ✓ View {i+1}/{num_views}")
    
    # Validate inputs
    print("\n4. Input Validation")
    manager.validate_inputs()
    print("   ✓ All inputs valid")
    
    # Multi-view inference
    print("\n5. Multi-View Inference")
    engine = DA3InferenceEngine(config)
    engine.load_model()
    
    print("   Running multi-view depth estimation...")
    results = engine.inference(manager.get_images())
    
    print(f"   ✓ Generated {len(results)} depth maps")
    for i, result in enumerate(results):
        depth_range = result.get_depth_range()
        print(f"     View {i+1}: depth range {depth_range}")
    
    # Fusion
    print("\n6. Multi-View Fusion")
    postprocessor = Postprocessor(config.postprocessing)
    fused_result = postprocessor.fuse_multiview(results)
    
    print(f"   ✓ Fused {len(results)} views")
    print(f"   Fusion mode: {config.postprocessing.fusion_mode}")
    print(f"   Final depth range: {fused_result.get_depth_range()}")
    
    # Export
    print("\n7. Export 3D Reconstruction")
    config.export.output_dir.mkdir(parents=True, exist_ok=True)
    exporter = Exporter(config.export)
    
    # Export fused result
    exported = exporter.export(fused_result, "fused_reconstruction")
    
    print("   ✓ Exported:")
    for fmt, path in exported.items():
        print(f"     - {fmt}: {path.name}")
    
    # Export individual views
    print("\n8. Export Individual Views")
    for i, result in enumerate(results):
        exported_view = exporter.export(result, f"view_{i:02d}")
        print(f"   ✓ View {i+1}: {list(exported_view.keys())}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Multi-View Reconstruction Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {config.export.output_dir}")
    print(f"\nGenerated outputs:")
    print(f"  - Fused 3D reconstruction (PLY)")
    print(f"  - Individual depth maps (NPZ)")
    print(f"  - {num_views} camera views")
    print("\nVisualization:")
    print("  - Use MeshLab or CloudCompare to view PLY files")
    print("  - Load NPZ files with: np.load('file.npz')['depth']")
    print("\nNext steps:")
    print("  - Use real camera-captured images")
    print("  - Calibrate camera intrinsics")
    print("  - Export to TSDF volume for meshing")


if __name__ == "__main__":
    main()
