#!/usr/bin/env python3
"""
Analyze and visualize depth zones for the luxury pool image.

Creates additional visualizations showing:
- Depth histogram
- Zone segmentation overlay
- Depth statistics per zone
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Configure matplotlib for non-interactive backend
plt.switch_backend('Agg')

def analyze_depth_zones(depth_map_path, output_dir):
    """Analyze and visualize depth zones."""
    
    # Load depth map
    depth = np.load(depth_map_path)
    
    print(f"Loaded depth map: {depth.shape}")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
    print()
    
    # Define zone boundaries
    zones = [
        ("Foreground (Pool)", 0.0, 0.33),
        ("Midground (Landscape)", 0.33, 0.67),
        ("Background (Sky)", 0.67, 1.0)
    ]
    
    # Analyze zones
    print("=" * 70)
    print("DEPTH ZONE ANALYSIS")
    print("=" * 70)
    print()
    
    for zone_name, z_min, z_max in zones:
        mask = (depth >= z_min) & (depth < z_max)
        zone_pixels = np.sum(mask)
        total_pixels = depth.size
        percentage = (zone_pixels / total_pixels) * 100
        
        zone_depth = depth[mask]
        if len(zone_depth) > 0:
            print(f"{zone_name}:")
            print(f"  Depth Range:   [{z_min:.2f} - {z_max:.2f}]")
            print(f"  Pixel Count:   {zone_pixels:,} ({percentage:.1f}% of image)")
            print(f"  Mean Depth:    {zone_depth.mean():.3f}")
            print(f"  Median Depth:  {np.median(zone_depth):.3f}")
            print(f"  Std Dev:       {zone_depth.std():.3f}")
            print()
    
    # Create visualizations
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Depth histogram with zone boundaries
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(depth.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add zone boundary lines
    colors = ['red', 'orange', 'green']
    for i, (zone_name, z_min, z_max) in enumerate(zones):
        if i < len(zones) - 1:
            ax.axvline(z_max, color=colors[i], linestyle='--', linewidth=2, 
                      label=f'{zone_name} boundary')
    
    ax.set_xlabel('Depth Value (0=near, 1=far)', fontsize=12)
    ax.set_ylabel('Pixel Count', fontsize=12)
    ax.set_title('Depth Distribution - 750 Picacho Luxury Pool', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    histogram_path = output_dir / 'depth_histogram_zones.png'
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved depth histogram: {histogram_path.name}")
    plt.close()
    
    # 2. Zone segmentation visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original depth map
    im0 = axes[0, 0].imshow(depth, cmap='turbo')
    axes[0, 0].set_title('Full Depth Map', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Zone masks
    zone_titles = ['Foreground Zone (Pool)', 'Midground Zone (Landscape)', 'Background Zone (Sky)']
    zone_cmaps = ['Blues', 'Greens', 'Reds']
    
    for i, (zone_name, z_min, z_max) in enumerate(zones):
        ax_idx = [(0, 1), (1, 0), (1, 1)][i]
        mask = (depth >= z_min) & (depth < z_max)
        zone_depth = np.where(mask, depth, np.nan)
        
        im = axes[ax_idx].imshow(zone_depth, cmap=zone_cmaps[i])
        axes[ax_idx].set_title(zone_titles[i], fontsize=12, fontweight='bold')
        axes[ax_idx].axis('off')
        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046)
    
    plt.suptitle('Depth Zone Segmentation - 750 Picacho Pool', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    segmentation_path = output_dir / 'depth_zone_segmentation.png'
    plt.savefig(segmentation_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved zone segmentation: {segmentation_path.name}")
    plt.close()
    
    # 3. Depth statistics summary
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Create statistics table
    stats_data = []
    for zone_name, z_min, z_max in zones:
        mask = (depth >= z_min) & (depth < z_max)
        zone_pixels = np.sum(mask)
        total_pixels = depth.size
        percentage = (zone_pixels / total_pixels) * 100
        zone_depth = depth[mask]
        
        if len(zone_depth) > 0:
            stats_data.append([
                zone_name,
                f"[{z_min:.2f}, {z_max:.2f}]",
                f"{percentage:.1f}%",
                f"{zone_depth.mean():.3f}",
                f"{zone_depth.std():.3f}"
            ])
    
    # Add overall statistics
    stats_data.append([
        "Full Image",
        "[0.00, 1.00]",
        "100.0%",
        f"{depth.mean():.3f}",
        f"{depth.std():.3f}"
    ])
    
    table = ax.table(
        cellText=stats_data,
        colLabels=['Zone', 'Depth Range', 'Coverage', 'Mean', 'Std Dev'],
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.2, 0.15, 0.15, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style the header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style the data rows
    colors_rows = ['#E7E6E6', '#F2F2F2', '#E7E6E6', '#D9D9D9']
    for i, color in enumerate(colors_rows, start=1):
        for j in range(5):
            table[(i, j)].set_facecolor(color)
    
    ax.set_title('Depth Zone Statistics - 750 Picacho Luxury Pool\n', 
                fontsize=14, fontweight='bold', pad=20)
    
    stats_path = output_dir / 'depth_statistics_table.png'
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved statistics table: {stats_path.name}")
    plt.close()
    
    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print()
    print(f"All visualizations saved to: {output_dir}")
    print()

if __name__ == "__main__":
    depth_map_path = "/Users/rc/Transformation_Portal/output_images/depth_processed/V2_V2_750Picacho_Pool_Luxury_Enhanced_depth.npy"
    output_dir = "/Users/rc/Transformation_Portal/output_images/depth_processed/analysis"
    
    print("=" * 70)
    print("DEPTH ZONE ANALYSIS - 750 PICACHO LUXURY POOL")
    print("=" * 70)
    print()
    
    analyze_depth_zones(depth_map_path, output_dir)
