"""Lighting Condition Detector for Lux Depth V2 Pipeline.

PHASE 2 - STUB for adaptive tone mapping and color grading.

Detects lighting conditions (time of day, sky characteristics) to enable
context-aware processing that adapts to the scene's natural lighting.

TODO - PHASE 2 IMPLEMENTATION (Task 4: 12-14h):
1. Implement sky region analysis
2. Design time-of-day classification algorithm
3. Create tone mapping adaptation rules
4. Design color grading adaptation rules
5. Integrate with processing pipeline
6. Benchmark on dawn/golden hour/twilight scenes
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from . import torch_ops


class TimeOfDay(str, Enum):
    """Time of day classification for lighting adaptation."""
    
    DAWN = "dawn"  # Pre-sunrise, cool blue tones
    SUNRISE = "sunrise"  # Golden hour morning
    MORNING = "morning"  # Bright, neutral daylight
    MIDDAY = "midday"  # Harsh overhead sun
    AFTERNOON = "afternoon"  # Warm, angled light
    GOLDEN_HOUR = "golden_hour"  # Sunset golden hour
    TWILIGHT = "twilight"  # Post-sunset, purple-blue tones
    NIGHT = "night"  # Artificial lighting
    OVERCAST = "overcast"  # Cloudy, diffuse light


@dataclass
class LightingCondition:
    """Detected lighting condition metadata."""
    
    time_of_day: TimeOfDay
    confidence: float  # [0, 1]
    
    # Sky characteristics
    sky_coverage: float  # Percentage of image that is sky [0, 1]
    sky_color_temp: float  # Estimated color temperature (K)
    sky_brightness: float  # Average sky luminance [0, 1]
    
    # Directional lighting
    has_strong_shadows: bool
    shadow_direction: Optional[str]  # "top", "left", "right", etc.
    
    # Color characteristics
    dominant_hue: float  # Dominant hue in degrees [0, 360]
    warmth: float  # Cool (-1) to warm (+1)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'time_of_day': self.time_of_day.value,
            'confidence': self.confidence,
            'sky_coverage': self.sky_coverage,
            'sky_color_temp': self.sky_color_temp,
            'sky_brightness': self.sky_brightness,
            'has_strong_shadows': self.has_strong_shadows,
            'shadow_direction': self.shadow_direction,
            'dominant_hue': self.dominant_hue,
            'warmth': self.warmth,
        }


class LightingConditionDetector:
    """Detect lighting conditions for adaptive processing (PHASE 2 - STUB).
    
    Analyzes scene lighting to enable context-aware tone mapping and
    color grading that respects the natural lighting conditions.
    
    Expected API:
        >>> detector = LightingConditionDetector(device)
        >>> condition = detector.detect(rgb_tensor, depth_map, sky_mask)
        >>> print(condition.time_of_day)  # TimeOfDay.GOLDEN_HOUR
        >>> print(condition.warmth)  # 0.8 (very warm)
        >>> 
        >>> # Adapt tone mapping
        >>> tone_config = detector.adapt_tone_mapping(condition, base_config)
        >>> # Increases highlight preservation for golden hour
        >>> 
        >>> # Adapt color grading
        >>> color_config = detector.adapt_color_grading(condition, base_config)
        >>> # Enhances warm tones for sunset scenes
    
    Integration Points:
        - pipeline.py: DepthAwarePipeline.process() pre-analysis
        - config.py: LightingConfig dataclass
        - materials_v2.py: Lighting-aware material response
    """
    
    def __init__(self, device: "torch_ops.torch.device"):
        """Initialize lighting condition detector.
        
        Args:
            device: Torch device
        """
        torch_ops.require_torch()
        self.device = device
    
    def detect(
        self,
        rgb: "torch_ops.torch.Tensor",
        depth_map: Optional["torch_ops.torch.Tensor"] = None,
        sky_mask: Optional["torch_ops.torch.Tensor"] = None
    ) -> LightingCondition:
        """Detect lighting conditions in scene.
        
        Args:
            rgb: RGB tensor (1x3xHxW), range [0, 1]
            depth_map: Optional depth map (1x1xHxW)
            sky_mask: Optional sky mask (1x1xHxW)
        
        Returns:
            LightingCondition with detected metadata
        """
        import torch
        
        # Create default sky mask if not provided
        if sky_mask is None:
            sky_mask = self._detect_sky(rgb)
        
        # Analyze sky region
        sky_coverage, sky_color_temp, sky_brightness = self._analyze_sky_region(rgb, sky_mask)
        
        # Compute color characteristics
        dominant_hue = self._compute_dominant_hue(rgb, sky_mask)
        warmth = self._compute_warmth(rgb)
        
        # Classify time of day
        time_of_day, confidence = self._classify_time_of_day(
            sky_color_temp,
            sky_brightness,
            dominant_hue,
            warmth
        )
        
        # Detect shadows
        has_shadows, shadow_dir = self._detect_shadows(rgb, depth_map)
        
        return LightingCondition(
            time_of_day=time_of_day,
            confidence=confidence,
            sky_coverage=sky_coverage,
            sky_color_temp=sky_color_temp,
            sky_brightness=sky_brightness,
            has_strong_shadows=has_shadows,
            shadow_direction=shadow_dir,
            dominant_hue=dominant_hue,
            warmth=warmth,
        )
    
    def adapt_tone_mapping(
        self,
        condition: LightingCondition,
        base_config: Dict
    ) -> Dict:
        """Adapt tone mapping parameters based on lighting condition.
        
        Rules:
            - Golden hour: Increase highlight preservation (prevent clipping)
            - Midday: Reduce shadow crushing, increase contrast
            - Dawn/Twilight: Preserve cool tones, gentle highlights
            - Overcast: Increase local contrast, reduce global contrast
            - Night: Preserve blacks, boost midtones carefully
        
        Args:
            condition: Detected lighting condition
            base_config: Base tone mapping configuration (dict)
        
        Returns:
            Adapted tone mapping configuration (dict)
        """
        # Clone base config
        adapted = base_config.copy()
        
        # Apply adaptations based on time of day
        if condition.time_of_day == TimeOfDay.GOLDEN_HOUR:
            # Preserve highlights, reduce contrast to avoid clipping
            adapted['highlight_preservation'] = adapted.get('highlight_preservation', 0.8) + 0.1
            adapted['contrast'] = adapted.get('contrast', 1.0) * 0.95
            adapted['saturation_boost'] = adapted.get('saturation_boost', 1.0) * 1.1
        
        elif condition.time_of_day == TimeOfDay.MIDDAY:
            # Increase contrast, reduce shadow crushing
            adapted['contrast'] = adapted.get('contrast', 1.0) * 1.08
            adapted['shadow_lift'] = adapted.get('shadow_lift', 0.0) + 0.05
            adapted['highlight_compression'] = adapted.get('highlight_compression', 0.0) + 0.03
        
        elif condition.time_of_day in (TimeOfDay.DAWN, TimeOfDay.TWILIGHT):
            # Gentle tone mapping, preserve cool tones
            adapted['highlight_preservation'] = adapted.get('highlight_preservation', 0.8) + 0.05
            adapted['contrast'] = adapted.get('contrast', 1.0) * 0.98
            adapted['cool_tone_preservation'] = adapted.get('cool_tone_preservation', 1.0) * 1.05
        
        elif condition.time_of_day == TimeOfDay.OVERCAST:
            # Increase local contrast, reduce global contrast
            adapted['local_contrast'] = adapted.get('local_contrast', 1.0) * 1.12
            adapted['contrast'] = adapted.get('contrast', 1.0) * 0.95
            adapted['saturation_boost'] = adapted.get('saturation_boost', 1.0) * 0.95
        
        elif condition.time_of_day == TimeOfDay.NIGHT:
            # Preserve blacks, careful midtone boost
            adapted['black_point'] = adapted.get('black_point', 0.0) - 0.02
            adapted['midtone_boost'] = adapted.get('midtone_boost', 0.0) + 0.08
            adapted['noise_reduction'] = adapted.get('noise_reduction', 0.0) + 0.05
        
        # Apply confidence weighting
        confidence = condition.confidence
        for key in adapted:
            if key in base_config:
                # Blend between base and adapted based on confidence
                adapted[key] = base_config[key] * (1 - confidence) + adapted[key] * confidence
        
        return adapted
    
    def adapt_color_grading(
        self,
        condition: LightingCondition,
        base_config: Dict
    ) -> Dict:
        """Adapt color grading parameters based on lighting condition.
        
        Rules:
            - Golden hour: Enhance warm tones, reduce cool tones
            - Dawn/Twilight: Enhance cool tones, preserve purple-blue
            - Midday: Neutral grading, avoid color casts
            - Overcast: Slight desaturation, cool tone bias
            - Night: Preserve warm artificial lighting, reduce cool casts
        
        Args:
            condition: Detected lighting condition
            base_config: Base color grading configuration (dict)
        
        Returns:
            Adapted color grading configuration (dict)
        """
        # Clone base config
        adapted = base_config.copy()
        
        # Apply adaptations based on time of day
        if condition.time_of_day == TimeOfDay.GOLDEN_HOUR:
            # Enhance warm tones
            adapted['warm_tone_boost'] = adapted.get('warm_tone_boost', 1.0) * 1.15
            adapted['cool_tone_reduction'] = adapted.get('cool_tone_reduction', 1.0) * 0.92
            adapted['orange_saturation'] = adapted.get('orange_saturation', 1.0) * 1.12
            adapted['yellow_saturation'] = adapted.get('yellow_saturation', 1.0) * 1.10
        
        elif condition.time_of_day in (TimeOfDay.DAWN, TimeOfDay.TWILIGHT):
            # Enhance cool tones
            adapted['cool_tone_boost'] = adapted.get('cool_tone_boost', 1.0) * 1.12
            adapted['warm_tone_reduction'] = adapted.get('warm_tone_reduction', 1.0) * 0.95
            adapted['blue_saturation'] = adapted.get('blue_saturation', 1.0) * 1.08
            adapted['purple_saturation'] = adapted.get('purple_saturation', 1.0) * 1.10
        
        elif condition.time_of_day == TimeOfDay.MIDDAY:
            # Neutral grading, avoid casts
            adapted['white_balance_shift'] = 0.0
            adapted['global_saturation'] = adapted.get('global_saturation', 1.0) * 1.02
        
        elif condition.time_of_day == TimeOfDay.OVERCAST:
            # Slight desaturation, cool bias
            adapted['global_saturation'] = adapted.get('global_saturation', 1.0) * 0.96
            adapted['cool_tone_boost'] = adapted.get('cool_tone_boost', 1.0) * 1.05
        
        elif condition.time_of_day == TimeOfDay.NIGHT:
            # Preserve warm artificial lighting
            adapted['warm_tone_boost'] = adapted.get('warm_tone_boost', 1.0) * 1.08
            adapted['cool_tone_reduction'] = adapted.get('cool_tone_reduction', 1.0) * 0.90
        
        # Apply confidence weighting
        confidence = condition.confidence
        for key in adapted:
            if key in base_config:
                adapted[key] = base_config[key] * (1 - confidence) + adapted[key] * confidence
        
        return adapted
    
    def _analyze_sky_region(
        self,
        rgb: "torch_ops.torch.Tensor",
        sky_mask: "torch_ops.torch.Tensor"
    ) -> Tuple[float, float, float]:
        """Analyze sky region for color temperature and brightness.
        
        Returns:
            (sky_coverage, color_temp_K, brightness)
        """
        import torch
        
        # Compute sky coverage
        sky_coverage = float(sky_mask.mean())
        
        if sky_coverage < 0.01:
            # No sky detected
            return (0.0, 5500.0, 0.5)  # Default values
        
        # Extract sky pixels
        sky_pixels = rgb * sky_mask  # 1x3xHxW
        
        # Compute average color in sky region
        sky_r = (sky_pixels[:, 0:1] * sky_mask).sum() / (sky_mask.sum() + 1e-6)
        sky_g = (sky_pixels[:, 1:2] * sky_mask).sum() / (sky_mask.sum() + 1e-6)
        sky_b = (sky_pixels[:, 2:3] * sky_mask).sum() / (sky_mask.sum() + 1e-6)
        
        # Compute brightness (average luma)
        sky_brightness = float(0.299 * sky_r + 0.587 * sky_g + 0.114 * sky_b)
        
        # Estimate color temperature from R/B ratio
        # Warmer sky (sunset) has higher R/B, cooler sky (dawn) has lower R/B
        rb_ratio = float((sky_r + 1e-6) / (sky_b + 1e-6))
        
        # Map R/B ratio to approximate color temperature (K)
        # Cool sky (dawn): rb_ratio ~0.8, temp ~6500K
        # Neutral sky: rb_ratio ~1.0, temp ~5500K
        # Warm sky (sunset): rb_ratio ~1.3, temp ~3500K
        if rb_ratio > 1.1:
            # Warm sky
            color_temp = 5500.0 - (rb_ratio - 1.0) * 3000.0
        elif rb_ratio < 0.9:
            # Cool sky
            color_temp = 5500.0 + (1.0 - rb_ratio) * 2000.0
        else:
            # Neutral sky
            color_temp = 5500.0
        
        # Clamp to reasonable range
        color_temp = max(2500.0, min(8000.0, color_temp))
        
        return (sky_coverage, color_temp, sky_brightness)
    
    def _classify_time_of_day(
        self,
        sky_color_temp: float,
        sky_brightness: float,
        dominant_hue: float,
        warmth: float
    ) -> Tuple[TimeOfDay, float]:
        """Classify time of day based on sky and scene characteristics.
        
        Returns:
            (time_of_day, confidence)
        """
        # Decision tree based on color temperature and brightness
        
        # Golden hour: warm (3500-4500K), moderate brightness
        if 3500 <= sky_color_temp <= 4500 and 0.4 <= sky_brightness <= 0.7 and warmth > 0.3:
            if 20 <= dominant_hue <= 60:  # Orange-yellow hues
                return (TimeOfDay.GOLDEN_HOUR, 0.85)
            else:
                return (TimeOfDay.AFTERNOON, 0.70)
        
        # Dawn: cool (6000-7500K), low-moderate brightness
        if sky_color_temp >= 6000 and sky_brightness < 0.5 and warmth < -0.2:
            if 200 <= dominant_hue <= 260:  # Blue-violet hues
                return (TimeOfDay.DAWN, 0.80)
            else:
                return (TimeOfDay.MORNING, 0.65)
        
        # Twilight: cool (6500-8000K), low brightness
        if sky_color_temp >= 6500 and sky_brightness < 0.3:
            if 240 <= dominant_hue <= 300:  # Purple-blue hues
                return (TimeOfDay.TWILIGHT, 0.75)
            else:
                return (TimeOfDay.NIGHT, 0.60)
        
        # Midday: neutral (5000-6000K), high brightness
        if 5000 <= sky_color_temp <= 6000 and sky_brightness >= 0.7:
            return (TimeOfDay.MIDDAY, 0.80)
        
        # Morning: neutral-cool, moderate-high brightness
        if 5500 <= sky_color_temp <= 6500 and 0.5 <= sky_brightness < 0.7:
            return (TimeOfDay.MORNING, 0.70)
        
        # Afternoon: neutral-warm, moderate brightness
        if 4500 <= sky_color_temp <= 5500 and 0.5 <= sky_brightness <= 0.7:
            return (TimeOfDay.AFTERNOON, 0.70)
        
        # Overcast: neutral, moderate brightness, low saturation
        if 5000 <= sky_color_temp <= 6000 and abs(warmth) < 0.2:
            return (TimeOfDay.OVERCAST, 0.65)
        
        # Night: low brightness
        if sky_brightness < 0.2:
            return (TimeOfDay.NIGHT, 0.75)
        
        # Default: midday with low confidence
        return (TimeOfDay.MIDDAY, 0.40)
    
    def _detect_sky(self, rgb: "torch_ops.torch.Tensor") -> "torch_ops.torch.Tensor":
        """Detect sky region using simple heuristics.
        
        Args:
            rgb: RGB tensor (1x3xHxW)
        
        Returns:
            Sky mask (1x1xHxW)
        """
        import torch
        
        # Extract channels
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        
        # Sky is typically: blue-dominant, upper portion of image
        _, _, h, w = rgb.shape
        
        # Blue dominance
        blue_dominant = (b > r + 0.08) & (b > g + 0.05)
        
        # Reasonable brightness
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        bright_enough = luma > 0.25
        
        # Upper portion bias (sky typically in upper 40% of image)
        y_coords = torch.arange(h, device=self.device).view(h, 1).expand(h, w)
        upper_bias = (y_coords < int(h * 0.4)).unsqueeze(0).unsqueeze(0).float()
        
        # Combine criteria
        sky_mask = (blue_dominant & bright_enough).float() * (0.5 + 0.5 * upper_bias)
        
        return sky_mask
    
    def _compute_dominant_hue(
        self,
        rgb: "torch_ops.torch.Tensor",
        sky_mask: "torch_ops.torch.Tensor"
    ) -> float:
        """Compute dominant hue in non-sky regions.
        
        Args:
            rgb: RGB tensor (1x3xHxW)
            sky_mask: Sky mask to exclude (1x1xHxW)
        
        Returns:
            Dominant hue in degrees [0, 360]
        """
        import torch
        
        # Exclude sky
        non_sky_mask = 1.0 - sky_mask
        
        # Extract RGB channels
        r = rgb[:, 0:1] * non_sky_mask
        g = rgb[:, 1:2] * non_sky_mask
        b = rgb[:, 2:3] * non_sky_mask
        
        # Compute hue using HSV conversion
        c_max = torch.maximum(torch.maximum(r, g), b)
        c_min = torch.minimum(torch.minimum(r, g), b)
        delta = c_max - c_min + 1e-6
        
        # Hue calculation
        hue = torch.zeros_like(c_max)
        
        # Red is max
        mask_r = (c_max == r) & (delta > 0.01)
        hue = torch.where(mask_r, 60.0 * (((g - b) / delta) % 6.0), hue)
        
        # Green is max
        mask_g = (c_max == g) & (delta > 0.01)
        hue = torch.where(mask_g, 60.0 * (((b - r) / delta) + 2.0), hue)
        
        # Blue is max
        mask_b = (c_max == b) & (delta > 0.01)
        hue = torch.where(mask_b, 60.0 * (((r - g) / delta) + 4.0), hue)
        
        # Ensure positive
        hue = hue % 360.0
        
        # Compute weighted average hue (weighted by saturation)
        saturation = delta / (c_max + 1e-6)
        weights = saturation * non_sky_mask
        
        # Circular mean for hue
        hue_rad = hue * (3.14159265 / 180.0)
        mean_sin = (torch.sin(hue_rad) * weights).sum() / (weights.sum() + 1e-6)
        mean_cos = (torch.cos(hue_rad) * weights).sum() / (weights.sum() + 1e-6)
        
        dominant_hue = float(torch.atan2(mean_sin, mean_cos) * (180.0 / 3.14159265))
        if dominant_hue < 0:
            dominant_hue += 360.0
        
        return dominant_hue
    
    def _compute_warmth(self, rgb: "torch_ops.torch.Tensor") -> float:
        """Compute overall warmth of image.
        
        Args:
            rgb: RGB tensor (1x3xHxW)
        
        Returns:
            Warmth score: -1 (cool) to +1 (warm)
        """
        import torch
        
        # Extract channels
        r = rgb[:, 0:1]
        g = rgb[:, 1:2]
        b = rgb[:, 2:3]
        
        # Warmth = (R - B) normalized
        warmth_map = (r - b)  # Range: [-1, 1]
        
        # Average warmth across image
        warmth = float(warmth_map.mean())
        
        # Clamp to [-1, 1]
        warmth = max(-1.0, min(1.0, warmth))
        
        return warmth
    
    def _detect_shadows(
        self,
        rgb: "torch_ops.torch.Tensor",
        depth_map: Optional["torch_ops.torch.Tensor"]
    ) -> Tuple[bool, Optional[str]]:
        """Detect strong shadows and estimate direction.
        
        Returns:
            (has_strong_shadows, shadow_direction)
        """
        import torch
        
        # Compute luminance
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Compute gradients
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device).float().view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=self.device).float().view(1, 1, 3, 3)
        
        # Pad for convolution
        luma_padded = torch.nn.functional.pad(luma, (1, 1, 1, 1), mode='replicate')
        
        grad_x = torch.nn.functional.conv2d(luma_padded, sobel_x)
        grad_y = torch.nn.functional.conv2d(luma_padded, sobel_y)
        
        # Gradient magnitude
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Detect strong gradients (potential shadow edges)
        shadow_threshold = 0.15
        strong_gradients = (grad_magnitude > shadow_threshold).float()
        
        # Check if we have significant shadow edges
        shadow_edge_ratio = strong_gradients.mean()
        has_strong_shadows = shadow_edge_ratio > 0.05  # More than 5% of pixels are shadow edges
        
        if not has_strong_shadows:
            return (False, None)
        
        # Estimate shadow direction from gradient orientation
        # Average gradient direction
        avg_grad_x = float(grad_x.mean())
        avg_grad_y = float(grad_y.mean())
        
        # Determine dominant direction
        if abs(avg_grad_x) > abs(avg_grad_y):
            shadow_dir = "left" if avg_grad_x > 0 else "right"
        else:
            shadow_dir = "top" if avg_grad_y > 0 else "bottom"
        
        return (has_strong_shadows, shadow_dir)
