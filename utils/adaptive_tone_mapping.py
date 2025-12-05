#!/usr/bin/env python3
"""
Adaptive Tone Mapping
=====================
Intelligent tone mapping parameter selection based on image histogram analysis.
"""

from typing import Dict, Tuple, Any
import numpy as np


class AdaptiveToneMapper:
    """Intelligently select tone mapping parameters based on image content."""
    
    def __init__(self):
        # Default parameter ranges
        self.key_range = (0.10, 0.36)  # Low-key to high-key
        self.sat_range = (0.75, 0.95)  # Saturation preservation
        
    def analyze_scene(self, hdr_image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze HDR image to determine optimal tone mapping parameters.
        
        Parameters:
        -----------
        hdr_image : np.ndarray
            HDR image array (float, any range)
        
        Returns:
        --------
        analysis : Dict[str, Any]
            Scene analysis including brightness classification and recommended parameters
        """
        # Handle alpha channel
        if hdr_image.shape[2] == 4:
            rgb = hdr_image[:, :, :3]
        else:
            rgb = hdr_image
        
        # Compute luminance
        luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        
        # Clip negative values for analysis
        luminance_positive = np.maximum(luminance, 1e-6)
        
        # Log-average luminance
        log_lum = np.log(luminance_positive + 1e-6)
        lum_avg = np.exp(log_lum.mean())
        
        # Statistics
        lum_min = luminance.min()
        lum_max = luminance.max()
        lum_median = np.median(luminance)
        lum_std = np.std(luminance)
        
        # Histogram analysis
        hist, bin_edges = np.histogram(luminance_positive, bins=256, range=(1e-6, max(10, luminance.max())))
        
        # Find peak (mode)
        peak_idx = np.argmax(hist)
        peak_value = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
        
        # Cumulative distribution for percentiles
        cumsum = np.cumsum(hist)
        total = cumsum[-1]
        
        # Find percentiles
        p01_idx = np.searchsorted(cumsum, total * 0.01)
        p99_idx = np.searchsorted(cumsum, total * 0.99)
        p01_value = bin_edges[p01_idx]
        p99_value = bin_edges[p99_idx]
        
        # Dynamic range (1st to 99th percentile)
        dynamic_range = p99_value / max(p01_value, 1e-6)
        
        # Color saturation analysis
        color_diff = rgb - luminance[:, :, np.newaxis]
        saturation = np.sqrt(np.sum(color_diff ** 2, axis=2))
        avg_saturation = saturation.mean()
        
        # Classify scene brightness
        scene_type = self._classify_scene_brightness(lum_avg, lum_median, peak_value)
        
        # Determine optimal key value
        key = self._determine_key_value(scene_type, lum_avg, lum_median, dynamic_range)
        
        # Determine saturation preservation
        sat = self._determine_saturation(avg_saturation, scene_type)
        
        analysis = {
            'luminance_stats': {
                'min': float(lum_min),
                'max': float(lum_max),
                'mean': float(luminance.mean()),
                'median': float(lum_median),
                'std': float(lum_std),
                'log_avg': float(lum_avg)
            },
            'histogram_stats': {
                'peak_value': float(peak_value),
                'p01': float(p01_value),
                'p99': float(p99_value),
                'dynamic_range': float(dynamic_range)
            },
            'color_stats': {
                'avg_saturation': float(avg_saturation)
            },
            'scene_classification': scene_type,
            'recommended_params': {
                'key': key,
                'sat': sat,
                'epsilon': 1e-6
            },
            'reasoning': self._generate_reasoning(scene_type, key, sat, dynamic_range)
        }
        
        return analysis
    
    def _classify_scene_brightness(
        self,
        lum_avg: float,
        lum_median: float,
        peak_value: float
    ) -> str:
        """
        Classify scene as low-key, mid-key, or high-key.
        
        Low-key: Dark, moody scenes (lum_avg < 0.15)
        Mid-key: Balanced scenes (0.15 <= lum_avg <= 0.30)
        High-key: Bright, airy scenes (lum_avg > 0.30)
        """
        # Use combination of log-average and median
        brightness_score = (lum_avg + lum_median) / 2
        
        if brightness_score < 0.15:
            return 'low_key'
        elif brightness_score < 0.30:
            return 'mid_key'
        else:
            return 'high_key'
    
    def _determine_key_value(
        self,
        scene_type: str,
        lum_avg: float,
        lum_median: float,
        dynamic_range: float
    ) -> float:
        """
        Determine optimal key value (target middle gray).
        
        Key value maps the log-average luminance to middle gray.
        Lower key = darker output, Higher key = brighter output
        """
        if scene_type == 'low_key':
            # Dark scenes: use lower key to preserve moodiness
            # But not too low to avoid crushing shadows
            base_key = 0.14
            # Adjust based on dynamic range
            if dynamic_range > 100:
                base_key *= 1.1  # Slightly brighter for high DR scenes
        elif scene_type == 'high_key':
            # Bright scenes: use higher key to maintain airiness
            base_key = 0.28
            # Adjust based on dynamic range
            if dynamic_range > 100:
                base_key *= 0.95  # Slightly darker to avoid blown highlights
        else:
            # Mid-key: standard photographic 18% gray
            base_key = 0.18
            # Fine-tune based on actual luminance
            if lum_avg < 0.18:
                base_key *= 1.1  # Slightly brighter
            elif lum_avg > 0.25:
                base_key *= 0.95  # Slightly darker
        
        # Clamp to reasonable range
        return np.clip(base_key, self.key_range[0], self.key_range[1])
    
    def _determine_saturation(self, avg_saturation: float, scene_type: str) -> float:
        """
        Determine saturation preservation parameter.
        
        Higher saturation = more color preservation
        Lower saturation = more desaturation (closer to luminance)
        """
        # Base saturation on scene type
        if scene_type == 'low_key':
            # Dark scenes often benefit from slightly reduced saturation
            base_sat = 0.82
        elif scene_type == 'high_key':
            # Bright scenes often have vibrant colors
            base_sat = 0.90
        else:
            # Mid-key: balanced saturation
            base_sat = 0.85
        
        # Adjust based on actual color content
        if avg_saturation < 0.05:
            # Nearly monochrome
            base_sat *= 0.95
        elif avg_saturation > 0.15:
            # Highly saturated
            base_sat *= 1.05
        
        # Clamp to valid range
        return np.clip(base_sat, self.sat_range[0], self.sat_range[1])
    
    def _generate_reasoning(
        self,
        scene_type: str,
        key: float,
        sat: float,
        dynamic_range: float
    ) -> str:
        """Generate human-readable reasoning for parameter selection."""
        reasoning_parts = []
        
        # Scene type
        scene_descriptions = {
            'low_key': 'Low-key scene detected (dark, moody)',
            'mid_key': 'Mid-key scene detected (balanced exposure)',
            'high_key': 'High-key scene detected (bright, airy)'
        }
        reasoning_parts.append(scene_descriptions.get(scene_type, 'Unknown scene type'))
        
        # Key value
        if key < 0.16:
            reasoning_parts.append(f"Using low key value ({key:.2f}) to preserve dark atmosphere")
        elif key > 0.24:
            reasoning_parts.append(f"Using high key value ({key:.2f}) to maintain brightness")
        else:
            reasoning_parts.append(f"Using standard key value ({key:.2f}) for balanced tone mapping")
        
        # Saturation
        if sat < 0.82:
            reasoning_parts.append(f"Reduced saturation ({sat:.2f}) for subtle color treatment")
        elif sat > 0.88:
            reasoning_parts.append(f"High saturation preservation ({sat:.2f}) for vibrant colors")
        else:
            reasoning_parts.append(f"Balanced saturation ({sat:.2f}) for natural color rendition")
        
        # Dynamic range
        if dynamic_range > 100:
            reasoning_parts.append(f"High dynamic range ({dynamic_range:.1f}x) requires careful tone mapping")
        elif dynamic_range < 10:
            reasoning_parts.append(f"Low dynamic range ({dynamic_range:.1f}x) - minimal tone mapping needed")
        
        return "; ".join(reasoning_parts)
    
    def apply_adaptive_tone_mapping(
        self,
        hdr_image: np.ndarray,
        override_params: Dict[str, float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply tone mapping with automatically determined parameters.
        
        Parameters:
        -----------
        hdr_image : np.ndarray
            HDR image to tone map
        override_params : Dict[str, float], optional
            Manual overrides for any parameter (key, sat, epsilon)
        
        Returns:
        --------
        tone_mapped : np.ndarray
            Tone-mapped image [0, 1]
        metadata : Dict[str, Any]
            Analysis and parameters used
        """
        # Analyze scene
        analysis = self.analyze_scene(hdr_image)
        
        # Get recommended parameters
        params = analysis['recommended_params'].copy()
        
        # Apply overrides
        if override_params:
            params.update(override_params)
        
        # Apply Reinhard local tone mapping
        tone_mapped = self._reinhard_local(hdr_image, **params)
        
        # Package metadata
        metadata = {
            'analysis': analysis,
            'parameters_used': params,
            'override_applied': bool(override_params)
        }
        
        return tone_mapped, metadata
    
    def _reinhard_local(
        self,
        hdr_image: np.ndarray,
        key: float,
        sat: float,
        epsilon: float
    ) -> np.ndarray:
        """Apply Reinhard local tone mapping operator."""
        # Handle alpha
        has_alpha = hdr_image.shape[2] == 4
        if has_alpha:
            alpha = hdr_image[:, :, 3]
            rgb = hdr_image[:, :, :3]
        else:
            rgb = hdr_image
        
        # Compute luminance
        luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        luminance = np.maximum(luminance, epsilon)
        
        # Log-average luminance
        log_lum = np.log(luminance + epsilon)
        lum_avg = np.exp(log_lum.mean())
        
        # Scale luminance
        scaled_lum = (key / lum_avg) * luminance
        
        # Tone map luminance
        tone_mapped_lum = scaled_lum / (1.0 + scaled_lum)
        
        # Apply to color channels
        tone_mapped = np.zeros_like(rgb)
        
        for c in range(3):
            # Scale each channel by tone mapping ratio
            ratio = tone_mapped_lum / (luminance + epsilon)
            tone_mapped[:, :, c] = rgb[:, :, c] * ratio
            
            # Blend with luminance based on saturation
            tone_mapped[:, :, c] = (
                sat * tone_mapped[:, :, c] +
                (1 - sat) * tone_mapped_lum
            )
        
        # Clip to valid range
        tone_mapped = np.clip(tone_mapped, 0.0, 1.0)
        
        # Restore alpha if present
        if has_alpha:
            tone_mapped = np.dstack([tone_mapped, alpha])
        
        return tone_mapped
    
    def print_analysis(self, analysis: Dict[str, Any], image_name: str = "Image"):
        """Pretty-print scene analysis."""
        print(f"\n🎨 Adaptive Tone Mapping Analysis: {image_name}")
        print(f"{'='*80}")
        
        lum = analysis['luminance_stats']
        print(f"\n📊 Luminance Statistics:")
        print(f"  Range: [{lum['min']:.4f}, {lum['max']:.4f}]")
        print(f"  Mean: {lum['mean']:.4f}")
        print(f"  Median: {lum['median']:.4f}")
        print(f"  Log-average: {lum['log_avg']:.4f}")
        
        hist = analysis['histogram_stats']
        print(f"\n📈 Histogram Analysis:")
        print(f"  Peak value: {hist['peak_value']:.4f}")
        print(f"  1st-99th percentile: [{hist['p01']:.4f}, {hist['p99']:.4f}]")
        print(f"  Dynamic range: {hist['dynamic_range']:.1f}x")
        
        print(f"\n🎯 Scene Classification: {analysis['scene_classification'].upper().replace('_', '-')}")
        
        params = analysis['recommended_params']
        print(f"\n⚙️  Recommended Parameters:")
        print(f"  Key (target gray): {params['key']:.4f}")
        print(f"  Saturation: {params['sat']:.4f}")
        
        print(f"\n💡 Reasoning:")
        print(f"  {analysis['reasoning']}")
        
        print(f"{'='*80}")


def main():
    """CLI for adaptive tone mapping."""
    import argparse
    from pathlib import Path
    from PIL import Image
    
    try:
        import tifffile
        HAS_TIFFFILE = True
    except ImportError:
        HAS_TIFFFILE = False
    
    parser = argparse.ArgumentParser(description="Adaptive tone mapping with intelligent parameter selection")
    parser.add_argument("input", type=Path, help="HDR input image")
    parser.add_argument("output", type=Path, help="Output tone-mapped image")
    parser.add_argument("--analyze-only", action="store_true", help="Only show analysis, don't process")
    parser.add_argument("--key", type=float, help="Override key value")
    parser.add_argument("--sat", type=float, help="Override saturation")
    
    args = parser.parse_args()
    
    print(f"📷 Loading HDR image: {args.input}")
    
    # Load image
    if args.input.suffix.lower() in ['.tif', '.tiff'] and HAS_TIFFFILE:
        hdr_array = tifffile.imread(args.input)
    else:
        img = Image.open(args.input)
        hdr_array = np.array(img).astype(np.float32) / 255.0
    
    # Create tone mapper
    mapper = AdaptiveToneMapper()
    
    if args.analyze_only:
        # Just analyze
        analysis = mapper.analyze_scene(hdr_array)
        mapper.print_analysis(analysis, args.input.name)
    else:
        # Apply tone mapping
        overrides = {}
        if args.key is not None:
            overrides['key'] = args.key
        if args.sat is not None:
            overrides['sat'] = args.sat
        
        tone_mapped, metadata = mapper.apply_adaptive_tone_mapping(hdr_array, overrides)
        
        # Print analysis
        mapper.print_analysis(metadata['analysis'], args.input.name)
        
        # Save result
        result_8bit = (np.clip(tone_mapped[:, :, :3], 0, 1) * 255).astype(np.uint8)
        Image.fromarray(result_8bit).save(args.output, quality=98)
        
        print(f"\n✅ Tone-mapped image saved: {args.output}")


if __name__ == "__main__":
    main()
