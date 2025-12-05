#!/usr/bin/env python3
"""
HDR Statistics Visualization
=============================
Generate histograms and statistical visualizations for HDR processing analysis.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime

import numpy as np
from PIL import Image

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠ Warning: matplotlib not available, visualization features disabled")

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


class HDRVisualizer:
    """Visualize HDR statistics and tone mapping effects."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for HDR visualization")
    
    def load_image(self, path: Path) -> np.ndarray:
        """Load image as float32 normalized array."""
        if path.suffix.lower() in ['.tif', '.tiff']:
            if HAS_TIFFFILE:
                arr = tifffile.imread(path)
                # Normalize based on dtype
                if arr.dtype == np.uint16:
                    return arr.astype(np.float32) / 65535.0
                elif arr.dtype == np.uint8:
                    return arr.astype(np.float32) / 255.0
                elif arr.dtype in [np.float32, np.float64]:
                    # HDR data - keep as-is (may have values outside [0, 1])
                    return arr.astype(np.float32)
                else:
                    raise ValueError(f"Unsupported dtype: {arr.dtype}")
            else:
                img = Image.open(path)
                arr = np.array(img)
                if arr.dtype == np.uint8:
                    return arr.astype(np.float32) / 255.0
                else:
                    return arr.astype(np.float32) / 65535.0
        else:
            img = Image.open(path)
            arr = np.array(img)
            return arr.astype(np.float32) / 255.0
    
    def generate_histogram_comparison(
        self,
        before_path: Path,
        after_path: Path,
        scene_name: str,
        is_hdr: bool = False
    ) -> Path:
        """Generate before/after histogram comparison."""
        print(f"  Generating histogram comparison for {scene_name}...")
        
        # Load images
        before = self.load_image(before_path)
        after = self.load_image(after_path)
        
        # Handle alpha channel
        if before.shape[2] == 4:
            before = before[:, :, :3]
        if after.shape[2] == 4:
            after = after[:, :, :3]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"HDR Tone Mapping Analysis: {scene_name}", fontsize=16, fontweight='bold')
        
        colors = ['red', 'green', 'blue']
        channel_names = ['Red', 'Green', 'Blue']
        
        # Row 1: Before histograms
        for i, (color, name) in enumerate(zip(colors, channel_names)):
            ax = axes[0, i]
            channel_data = before[:, :, i].flatten()
            
            if is_hdr:
                # HDR data may have extreme values
                bins = np.linspace(channel_data.min(), min(channel_data.max(), 10), 256)
            else:
                bins = np.linspace(0, 1, 256)
            
            ax.hist(channel_data, bins=bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_title(f"{name} Channel - Before", fontweight='bold')
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f"Min: {channel_data.min():.3f}\n"
            stats_text += f"Max: {channel_data.max():.3f}\n"
            stats_text += f"Mean: {channel_data.mean():.3f}\n"
            stats_text += f"Std: {channel_data.std():.3f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Row 2: After histograms
        for i, (color, name) in enumerate(zip(colors, channel_names)):
            ax = axes[1, i]
            channel_data = after[:, :, i].flatten()
            
            bins = np.linspace(0, 1, 256)
            ax.hist(channel_data, bins=bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_title(f"{name} Channel - After", fontweight='bold')
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f"Min: {channel_data.min():.3f}\n"
            stats_text += f"Max: {channel_data.max():.3f}\n"
            stats_text += f"Mean: {channel_data.mean():.3f}\n"
            stats_text += f"Std: {channel_data.std():.3f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"histogram_{scene_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Histogram saved: {output_path.name}")
        return output_path
    
    def generate_luminance_distribution(
        self,
        before_path: Path,
        after_path: Path,
        scene_name: str
    ) -> Path:
        """Generate luminance distribution comparison."""
        print(f"  Generating luminance distribution for {scene_name}...")
        
        # Load images
        before = self.load_image(before_path)
        after = self.load_image(after_path)
        
        # Handle alpha
        if before.shape[2] == 4:
            before = before[:, :, :3]
        if after.shape[2] == 4:
            after = after[:, :, :3]
        
        # Compute luminance (Rec. 709)
        lum_before = 0.2126 * before[:, :, 0] + 0.7152 * before[:, :, 1] + 0.0722 * before[:, :, 2]
        lum_after = 0.2126 * after[:, :, 0] + 0.7152 * after[:, :, 1] + 0.0722 * after[:, :, 2]
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Luminance Distribution: {scene_name}", fontsize=14, fontweight='bold')
        
        # Before luminance
        ax = axes[0]
        bins = np.linspace(lum_before.min(), min(lum_before.max(), 5), 256)
        ax.hist(lum_before.flatten(), bins=bins, color='gray', alpha=0.8, edgecolor='black')
        ax.set_title("Before Tone Mapping", fontweight='bold')
        ax.set_xlabel("Luminance")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        # After luminance
        ax = axes[1]
        bins = np.linspace(0, 1, 256)
        ax.hist(lum_after.flatten(), bins=bins, color='gray', alpha=0.8, edgecolor='black')
        ax.set_title("After Tone Mapping", fontweight='bold')
        ax.set_xlabel("Luminance")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        # Overlay comparison
        ax = axes[2]
        # Normalize before histogram to [0, 1] range for comparison
        lum_before_norm = np.clip(lum_before, 0, 1)
        ax.hist(lum_before_norm.flatten(), bins=100, color='red', alpha=0.5, 
               label='Before (clipped)', edgecolor='none')
        ax.hist(lum_after.flatten(), bins=100, color='blue', alpha=0.5, 
               label='After', edgecolor='none')
        ax.set_title("Overlay Comparison", fontweight='bold')
        ax.set_xlabel("Luminance (normalized)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"luminance_{scene_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Luminance distribution saved: {output_path.name}")
        return output_path
    
    def generate_clipping_analysis(
        self,
        image_path: Path,
        scene_name: str,
        is_before: bool = False
    ) -> Path:
        """Analyze and visualize clipping zones."""
        print(f"  Generating clipping analysis for {scene_name}...")
        
        img = self.load_image(image_path)
        
        # Handle alpha
        if img.shape[2] == 4:
            img = img[:, :, :3]
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        stage = "Before" if is_before else "After"
        fig.suptitle(f"Clipping Analysis ({stage}): {scene_name}", fontsize=14, fontweight='bold')
        
        channel_names = ['Red', 'Green', 'Blue']
        
        # Analyze each channel
        for i, name in enumerate(channel_names):
            channel = img[:, :, i]
            
            # Top row: Clipping zones visualization
            ax = axes[0, i]
            
            # Create clipping mask
            if is_before:
                # For HDR input, highlight extreme values
                shadow_clip = channel < 0
                highlight_clip = channel > 2.0  # Beyond reasonable HDR range
                valid = ~(shadow_clip | highlight_clip)
            else:
                # For tone-mapped output
                shadow_clip = channel < 0.01
                highlight_clip = channel > 0.99
                valid = ~(shadow_clip | highlight_clip)
            
            # Create RGB visualization
            clip_viz = np.zeros((*channel.shape, 3), dtype=np.uint8)
            clip_viz[shadow_clip] = [0, 0, 255]  # Blue for shadows
            clip_viz[highlight_clip] = [255, 0, 0]  # Red for highlights
            clip_viz[valid] = [0, 255, 0]  # Green for valid range
            
            ax.imshow(clip_viz)
            ax.set_title(f"{name} - Clipping Zones", fontweight='bold')
            ax.axis('off')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', label='Shadow Clip'),
                Patch(facecolor='green', label='Valid Range'),
                Patch(facecolor='red', label='Highlight Clip')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
            # Bottom row: Statistics
            ax = axes[1, i]
            
            total_pixels = channel.size
            shadow_pct = shadow_clip.sum() / total_pixels * 100
            highlight_pct = highlight_clip.sum() / total_pixels * 100
            valid_pct = valid.sum() / total_pixels * 100
            
            categories = ['Shadow\nClip', 'Valid\nRange', 'Highlight\nClip']
            percentages = [shadow_pct, valid_pct, highlight_pct]
            colors_bar = ['blue', 'green', 'red']
            
            bars = ax.bar(categories, percentages, color=colors_bar, alpha=0.7, edgecolor='black')
            ax.set_ylabel("Percentage (%)")
            ax.set_title(f"{name} - Statistics", fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add percentage labels on bars
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{pct:.2f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save
        stage_suffix = "before" if is_before else "after"
        output_path = self.output_dir / f"clipping_{scene_name}_{stage_suffix}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Clipping analysis saved: {output_path.name}")
        return output_path
    
    def generate_dynamic_range_comparison(
        self,
        before_path: Path,
        after_path: Path,
        scene_name: str
    ) -> Path:
        """Generate dynamic range compression visualization."""
        print(f"  Generating dynamic range comparison for {scene_name}...")
        
        before = self.load_image(before_path)
        after = self.load_image(after_path)
        
        # Handle alpha
        if before.shape[2] == 4:
            before = before[:, :, :3]
        if after.shape[2] == 4:
            after = after[:, :, :3]
        
        # Compute per-channel ranges
        fig, ax = plt.subplots(figsize=(12, 6))
        
        channels = ['Red', 'Green', 'Blue', 'Luminance']
        colors = ['red', 'green', 'blue', 'gray']
        
        before_ranges = []
        after_ranges = []
        compression_ratios = []
        
        for i, (channel, color) in enumerate(zip(channels, colors)):
            if channel == 'Luminance':
                before_data = 0.2126 * before[:, :, 0] + 0.7152 * before[:, :, 1] + 0.0722 * before[:, :, 2]
                after_data = 0.2126 * after[:, :, 0] + 0.7152 * after[:, :, 1] + 0.0722 * after[:, :, 2]
            else:
                before_data = before[:, :, i]
                after_data = after[:, :, i]
            
            before_range = before_data.max() - before_data.min()
            after_range = after_data.max() - after_data.min()
            
            before_ranges.append(before_range)
            after_ranges.append(after_range)
            
            if after_range > 0:
                compression_ratios.append(before_range / after_range)
            else:
                compression_ratios.append(0)
        
        # Plot
        x = np.arange(len(channels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, before_ranges, width, label='Before', alpha=0.8, color='darkred')
        bars2 = ax.bar(x + width/2, after_ranges, width, label='After', alpha=0.8, color='darkblue')
        
        ax.set_ylabel('Dynamic Range', fontweight='bold')
        ax.set_title(f'Dynamic Range Compression: {scene_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(channels)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add compression ratio annotations
        for i, (ratio, bar1, bar2) in enumerate(zip(compression_ratios, bars1, bars2)):
            if ratio > 0:
                y_pos = max(bar1.get_height(), bar2.get_height())
                ax.text(i, y_pos + 0.05, f'{ratio:.1f}x', ha='center', va='bottom',
                       fontweight='bold', fontsize=10, color='black',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"dynamic_range_{scene_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Dynamic range comparison saved: {output_path.name}")
        return output_path


def main():
    """CLI for HDR visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate HDR statistics visualizations")
    parser.add_argument("--before", type=Path, required=True, help="HDR input image")
    parser.add_argument("--after", type=Path, required=True, help="Tone-mapped output image")
    parser.add_argument("--name", required=True, help="Scene name")
    parser.add_argument("--output", type=Path, default=Path("output_hdr_viz"), help="Output directory")
    parser.add_argument("--is-hdr", action="store_true", help="Input is HDR (may have values outside [0, 1])")
    
    args = parser.parse_args()
    
    print(f"🎨 Generating HDR visualizations for: {args.name}")
    
    viz = HDRVisualizer(args.output)
    
    # Generate all visualizations
    viz.generate_histogram_comparison(args.before, args.after, args.name, args.is_hdr)
    viz.generate_luminance_distribution(args.before, args.after, args.name)
    viz.generate_clipping_analysis(args.before, args.name, is_before=True)
    viz.generate_clipping_analysis(args.after, args.name, is_before=False)
    viz.generate_dynamic_range_comparison(args.before, args.after, args.name)
    
    print(f"\n✅ All visualizations generated in: {args.output}")


if __name__ == "__main__":
    main()
