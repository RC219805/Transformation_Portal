#!/usr/bin/env python3
"""
Batch Comparison Tool
=====================
Comprehensive comparison utility for 16-bit vs 32-bit HDR outputs.
Validates tone mapping, measures quality metrics, generates visual reports.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class ImageComparison:
    """Compare two images with comprehensive quality metrics."""
    
    def __init__(self, image1_path: Path, image2_path: Path, label1: str = "Image 1", label2: str = "Image 2"):
        self.image1_path = image1_path
        self.image2_path = image2_path
        self.label1 = label1
        self.label2 = label2
        self.metrics: Dict[str, Any] = {}
        
    def load_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load both images as normalized float arrays [0, 1]."""
        img1 = self._load_image(self.image1_path)
        img2 = self._load_image(self.image2_path)
        
        # Ensure same dimensions
        if img1.shape != img2.shape:
            print(f"⚠ Warning: Shape mismatch - {img1.shape} vs {img2.shape}")
            # Resize img2 to match img1
            from PIL import Image as PILImage
            img2_pil = PILImage.fromarray((np.clip(img2, 0, 1) * 255).astype(np.uint8))
            img2_pil = img2_pil.resize((img1.shape[1], img1.shape[0]), PILImage.Resampling.LANCZOS)
            img2 = np.array(img2_pil).astype(np.float32) / 255.0
            
        return img1, img2
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image file as float32 normalized array."""
        if path.suffix.lower() in ['.tif', '.tiff']:
            if HAS_TIFFFILE:
                arr = tifffile.imread(path)
                # Handle different bit depths
                if arr.dtype == np.uint16:
                    return arr.astype(np.float32) / 65535.0
                elif arr.dtype == np.uint8:
                    return arr.astype(np.float32) / 255.0
                elif arr.dtype in [np.float32, np.float64]:
                    return np.clip(arr, 0, 1).astype(np.float32)
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
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute comprehensive quality metrics."""
        img1, img2 = self.load_images()
        
        # Handle alpha channel if present
        if img1.shape[2] == 4:
            img1 = img1[:, :, :3]
        if img2.shape[2] == 4:
            img2 = img2[:, :, :3]
        
        metrics = {
            'resolution': img1.shape[:2],
            'megapixels': img1.shape[0] * img1.shape[1] / 1_000_000
        }
        
        # Basic statistics
        metrics['img1_stats'] = {
            'min': float(img1.min()),
            'max': float(img1.max()),
            'mean': float(img1.mean()),
            'std': float(img1.std())
        }
        
        metrics['img2_stats'] = {
            'min': float(img2.min()),
            'max': float(img2.max()),
            'mean': float(img2.mean()),
            'std': float(img2.std())
        }
        
        # Difference statistics
        diff = np.abs(img1 - img2)
        metrics['difference'] = {
            'mean_absolute_error': float(diff.mean()),
            'max_difference': float(diff.max()),
            'median_difference': float(np.median(diff)),
            'std_difference': float(diff.std())
        }
        
        # PSNR and SSIM (if available)
        if HAS_SKIMAGE:
            try:
                metrics['psnr'] = float(psnr(img1, img2, data_range=1.0))
            except Exception as e:
                metrics['psnr'] = None
                print(f"⚠ PSNR calculation failed: {e}")
            
            try:
                # SSIM per channel
                ssim_scores = []
                for c in range(3):
                    score = ssim(img1[:, :, c], img2[:, :, c], data_range=1.0)
                    ssim_scores.append(score)
                metrics['ssim'] = {
                    'mean': float(np.mean(ssim_scores)),
                    'r': float(ssim_scores[0]),
                    'g': float(ssim_scores[1]),
                    'b': float(ssim_scores[2])
                }
            except Exception as e:
                metrics['ssim'] = None
                print(f"⚠ SSIM calculation failed: {e}")
        else:
            metrics['psnr'] = None
            metrics['ssim'] = None
        
        # Histogram analysis
        metrics['histogram'] = self._compute_histogram_stats(img1, img2)
        
        # Perceptual difference zones
        metrics['difference_zones'] = self._analyze_difference_zones(diff)
        
        self.metrics = metrics
        return metrics
    
    def _compute_histogram_stats(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, Any]:
        """Compute histogram-based statistics."""
        hist_stats = {}
        
        for i, channel in enumerate(['r', 'g', 'b']):
            hist1, _ = np.histogram(img1[:, :, i], bins=256, range=(0, 1))
            hist2, _ = np.histogram(img2[:, :, i], bins=256, range=(0, 1))
            
            # Histogram intersection
            intersection = np.minimum(hist1, hist2).sum()
            union = np.maximum(hist1, hist2).sum()
            similarity = intersection / union if union > 0 else 0
            
            hist_stats[channel] = {
                'similarity': float(similarity),
                'correlation': float(np.corrcoef(hist1, hist2)[0, 1])
            }
        
        return hist_stats
    
    def _analyze_difference_zones(self, diff: np.ndarray) -> Dict[str, Any]:
        """Analyze regions of significant difference."""
        # Per-pixel max difference across channels
        max_diff = diff.max(axis=2)
        
        # Define thresholds
        low_threshold = 0.02  # 2% difference
        med_threshold = 0.05  # 5% difference
        high_threshold = 0.10  # 10% difference
        
        zones = {
            'negligible': float((max_diff < low_threshold).sum() / max_diff.size * 100),
            'low': float(((max_diff >= low_threshold) & (max_diff < med_threshold)).sum() / max_diff.size * 100),
            'medium': float(((max_diff >= med_threshold) & (max_diff < high_threshold)).sum() / max_diff.size * 100),
            'high': float((max_diff >= high_threshold).sum() / max_diff.size * 100)
        }
        
        return zones
    
    def generate_comparison_image(self, output_path: Path, max_width: int = 3840) -> Path:
        """Generate side-by-side comparison with difference overlay."""
        img1, img2 = self.load_images()
        
        # Handle alpha
        if img1.shape[2] == 4:
            img1 = img1[:, :, :3]
        if img2.shape[2] == 4:
            img2 = img2[:, :, :3]
        
        # Compute difference
        diff = np.abs(img1 - img2)
        diff_vis = (diff * 5).clip(0, 1)  # Amplify for visibility
        
        # Convert to 8-bit
        img1_8bit = (np.clip(img1, 0, 1) * 255).astype(np.uint8)
        img2_8bit = (np.clip(img2, 0, 1) * 255).astype(np.uint8)
        diff_8bit = (np.clip(diff_vis, 0, 1) * 255).astype(np.uint8)
        
        # Create PIL images
        pil1 = Image.fromarray(img1_8bit)
        pil2 = Image.fromarray(img2_8bit)
        pil_diff = Image.fromarray(diff_8bit)
        
        # Resize if too large
        h, w = img1.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w = max_width
            new_h = int(h * scale)
            pil1 = pil1.resize((new_w, new_h), Image.Resampling.LANCZOS)
            pil2 = pil2.resize((new_w, new_h), Image.Resampling.LANCZOS)
            pil_diff = pil_diff.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = new_w, new_h
        
        # Create composite (side-by-side with difference below)
        label_height = 60
        composite_width = w * 2
        composite_height = h * 2 + label_height * 3
        
        composite = Image.new('RGB', (composite_width, composite_height), color=(30, 30, 30))
        draw = ImageDraw.Draw(composite)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            font = ImageFont.load_default()
            font_small = font
        
        # Top row: original images
        composite.paste(pil1, (0, label_height))
        composite.paste(pil2, (w, label_height))
        
        # Bottom row: difference maps
        composite.paste(pil_diff, (0, h + label_height * 2))
        composite.paste(pil_diff, (w, h + label_height * 2))
        
        # Add labels
        draw.text((w // 2, label_height // 2), self.label1, fill=(255, 255, 255), anchor="mm", font=font)
        draw.text((w + w // 2, label_height // 2), self.label2, fill=(255, 255, 255), anchor="mm", font=font)
        draw.text((w, h + label_height + label_height // 2), "Difference (5x amplified)", 
                 fill=(255, 200, 100), anchor="mm", font=font)
        
        # Add metrics overlay
        if self.metrics:
            y_offset = h * 2 + label_height * 3 - 150
            metrics_text = []
            if self.metrics.get('psnr'):
                metrics_text.append(f"PSNR: {self.metrics['psnr']:.2f} dB")
            if self.metrics.get('ssim'):
                metrics_text.append(f"SSIM: {self.metrics['ssim']['mean']:.4f}")
            metrics_text.append(f"MAE: {self.metrics['difference']['mean_absolute_error']:.4f}")
            
            for i, text in enumerate(metrics_text):
                draw.text((20, y_offset + i * 35), text, fill=(100, 255, 100), font=font_small)
        
        # Save
        composite.save(output_path, quality=95, optimize=True)
        print(f"  ✓ Comparison image saved: {output_path.name}")
        
        return output_path


class BatchComparisonTool:
    """Batch comparison tool for multiple image pairs."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.comparisons: List[Dict[str, Any]] = []
        
    def add_comparison(
        self,
        image1: Path,
        image2: Path,
        name: str,
        label1: str = "16-bit",
        label2: str = "32-bit HDR"
    ) -> Dict[str, Any]:
        """Add and process a comparison."""
        print(f"\n🔍 Comparing: {name}")
        print(f"  {label1}: {image1.name}")
        print(f"  {label2}: {image2.name}")
        
        comp = ImageComparison(image1, image2, label1, label2)
        
        # Compute metrics
        metrics = comp.compute_metrics()
        
        # Generate comparison image
        comp_image_path = self.output_dir / f"comparison_{name}.jpg"
        comp.generate_comparison_image(comp_image_path)
        
        result = {
            'name': name,
            'image1': str(image1),
            'image2': str(image2),
            'label1': label1,
            'label2': label2,
            'metrics': metrics,
            'comparison_image': str(comp_image_path)
        }
        
        self.comparisons.append(result)
        
        # Print summary
        print(f"  Metrics:")
        if metrics.get('psnr'):
            print(f"    PSNR: {metrics['psnr']:.2f} dB")
        if metrics.get('ssim'):
            print(f"    SSIM: {metrics['ssim']['mean']:.4f}")
        print(f"    MAE: {metrics['difference']['mean_absolute_error']:.4f}")
        print(f"    Max Diff: {metrics['difference']['max_difference']:.4f}")
        
        return result
    
    def generate_report(self) -> Path:
        """Generate comprehensive markdown report."""
        report_path = self.output_dir / "comparison_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Image Comparison Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Comparisons:** {len(self.comparisons)}\n\n")
            
            # Summary statistics
            if self.comparisons:
                f.write("## Summary Statistics\n\n")
                
                psnr_values = [c['metrics']['psnr'] for c in self.comparisons if c['metrics'].get('psnr')]
                ssim_values = [c['metrics']['ssim']['mean'] for c in self.comparisons if c['metrics'].get('ssim')]
                mae_values = [c['metrics']['difference']['mean_absolute_error'] for c in self.comparisons]
                
                if psnr_values:
                    f.write(f"- **Average PSNR:** {np.mean(psnr_values):.2f} dB (range: {min(psnr_values):.2f} - {max(psnr_values):.2f})\n")
                if ssim_values:
                    f.write(f"- **Average SSIM:** {np.mean(ssim_values):.4f} (range: {min(ssim_values):.4f} - {max(ssim_values):.4f})\n")
                f.write(f"- **Average MAE:** {np.mean(mae_values):.4f} (range: {min(mae_values):.4f} - {max(mae_values):.4f})\n\n")
            
            # Individual comparisons
            f.write("## Individual Comparisons\n\n")
            
            for comp in self.comparisons:
                f.write(f"### {comp['name']}\n\n")
                f.write(f"![Comparison]({Path(comp['comparison_image']).name})\n\n")
                
                metrics = comp['metrics']
                f.write("#### Quality Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                
                if metrics.get('psnr'):
                    f.write(f"| PSNR | {metrics['psnr']:.2f} dB |\n")
                if metrics.get('ssim'):
                    f.write(f"| SSIM (mean) | {metrics['ssim']['mean']:.4f} |\n")
                    f.write(f"| SSIM (R) | {metrics['ssim']['r']:.4f} |\n")
                    f.write(f"| SSIM (G) | {metrics['ssim']['g']:.4f} |\n")
                    f.write(f"| SSIM (B) | {metrics['ssim']['b']:.4f} |\n")
                
                f.write(f"| Mean Absolute Error | {metrics['difference']['mean_absolute_error']:.4f} |\n")
                f.write(f"| Max Difference | {metrics['difference']['max_difference']:.4f} |\n")
                f.write(f"| Median Difference | {metrics['difference']['median_difference']:.4f} |\n\n")
                
                f.write("#### Difference Zones\n\n")
                zones = metrics['difference_zones']
                f.write(f"- **Negligible (<2%):** {zones['negligible']:.1f}% of pixels\n")
                f.write(f"- **Low (2-5%):** {zones['low']:.1f}% of pixels\n")
                f.write(f"- **Medium (5-10%):** {zones['medium']:.1f}% of pixels\n")
                f.write(f"- **High (>10%):** {zones['high']:.1f}% of pixels\n\n")
                
                f.write("---\n\n")
        
        print(f"\n📄 Report generated: {report_path}")
        
        # Also save JSON
        json_path = self.output_dir / "comparison_results.json"
        with open(json_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'comparisons': self.comparisons
            }, f, indent=2)
        
        print(f"📄 JSON data saved: {json_path}")
        
        return report_path


def main():
    """CLI for batch comparison tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare image processing outputs")
    parser.add_argument("--dir1", type=Path, required=True, help="First directory (e.g., 16-bit outputs)")
    parser.add_argument("--dir2", type=Path, required=True, help="Second directory (e.g., 32-bit HDR outputs)")
    parser.add_argument("--output", type=Path, default=Path("output_comparisons"), help="Output directory")
    parser.add_argument("--pattern", default="*.tif", help="File pattern to match")
    parser.add_argument("--label1", default="16-bit", help="Label for first set")
    parser.add_argument("--label2", default="32-bit HDR", help="Label for second set")
    
    args = parser.parse_args()
    
    # Find matching files
    files1 = sorted(args.dir1.glob(args.pattern))
    files2_dict = {f.stem: f for f in args.dir2.glob(args.pattern)}
    
    print(f"Found {len(files1)} files in {args.dir1}")
    print(f"Found {len(files2_dict)} files in {args.dir2}")
    
    # Create comparison tool
    tool = BatchComparisonTool(args.output)
    
    # Process matching pairs
    matched = 0
    for file1 in files1:
        stem = file1.stem
        if stem in files2_dict:
            tool.add_comparison(file1, files2_dict[stem], stem, args.label1, args.label2)
            matched += 1
        else:
            print(f"⚠ No match found for: {file1.name}")
    
    print(f"\n✅ Processed {matched} comparison pairs")
    
    # Generate report
    tool.generate_report()


if __name__ == "__main__":
    main()
