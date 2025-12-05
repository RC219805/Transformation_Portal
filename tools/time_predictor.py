#!/usr/bin/env python3
"""
Processing Time Prediction Model
=================================
Intelligent prediction of processing time based on image metadata and historical data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime, timedelta

import numpy as np
from PIL import Image

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


class ImageMetadata:
    """Extract and analyze image metadata for time prediction."""
    
    def __init__(self, path: Path):
        self.path = path
        self.metadata = self._extract_metadata()
    
    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract comprehensive metadata from image."""
        meta = {
            'path': str(self.path),
            'filename': self.path.name,
            'extension': self.path.suffix.lower(),
            'file_size_mb': self.path.stat().st_size / (1024 * 1024)
        }
        
        # Try to load image info
        try:
            if self.path.suffix.lower() in ['.tif', '.tiff'] and HAS_TIFFFILE:
                arr = tifffile.imread(self.path)
                meta['width'] = arr.shape[1]
                meta['height'] = arr.shape[0]
                meta['channels'] = arr.shape[2] if len(arr.shape) > 2 else 1
                meta['dtype'] = str(arr.dtype)
                
                # Determine bit depth
                if arr.dtype == np.uint8:
                    meta['bit_depth'] = 8
                elif arr.dtype == np.uint16:
                    meta['bit_depth'] = 16
                elif arr.dtype in [np.float32, np.float64]:
                    meta['bit_depth'] = 32
                else:
                    meta['bit_depth'] = 0
                
                # Check for HDR data
                if arr.dtype in [np.float32, np.float64]:
                    meta['is_hdr'] = bool((arr < 0).any() or (arr > 1.0).any())
                else:
                    meta['is_hdr'] = False
                    
            else:
                # Use PIL
                img = Image.open(self.path)
                meta['width'], meta['height'] = img.size
                meta['channels'] = len(img.getbands())
                meta['bit_depth'] = 8  # PIL typically 8-bit
                meta['is_hdr'] = False
                
        except Exception as e:
            print(f"⚠ Warning: Could not extract full metadata from {self.path.name}: {e}")
            meta['width'] = 0
            meta['height'] = 0
            meta['channels'] = 0
            meta['bit_depth'] = 0
            meta['is_hdr'] = False
        
        # Computed metrics
        meta['megapixels'] = (meta['width'] * meta['height']) / 1_000_000
        meta['has_alpha'] = meta['channels'] == 4
        
        return meta
    
    def __repr__(self) -> str:
        return f"ImageMetadata({self.path.name}: {self.metadata['width']}x{self.metadata['height']}, {self.metadata['bit_depth']}-bit)"


class ProcessingTimePredictor:
    """Predict processing time based on image characteristics and historical data."""
    
    # Base processing times (seconds per megapixel)
    BASE_TIME_PER_MP = 0.5  # Baseline for 8-bit RGB
    
    # Multipliers
    BIT_DEPTH_MULTIPLIER = {
        8: 1.0,
        16: 1.5,
        32: 2.5
    }
    
    HDR_OVERHEAD = 1.8  # HDR tone mapping overhead
    ALPHA_OVERHEAD = 1.1  # Alpha channel handling
    
    # Stage-specific times (seconds per megapixel)
    STAGE_TIMES = {
        'load': 0.05,
        'tone_mapping': 0.15,
        'depth_estimation': 0.30,
        'material_response': 0.20,
        'clarity': 0.10,
        'color_grading': 0.08,
        'save': 0.12
    }
    
    def __init__(self, history_path: Optional[Path] = None):
        self.history_path = history_path
        self.history: List[Dict[str, Any]] = []
        
        if history_path and history_path.exists():
            self._load_history()
    
    def _load_history(self):
        """Load historical processing data."""
        try:
            with open(self.history_path, 'r') as f:
                data = json.load(f)
                self.history = data.get('records', [])
            print(f"✓ Loaded {len(self.history)} historical records")
        except Exception as e:
            print(f"⚠ Could not load history: {e}")
    
    def predict_time(self, metadata: ImageMetadata, include_depth: bool = True) -> Dict[str, Any]:
        """Predict processing time for an image."""
        meta = metadata.metadata
        
        # Base time calculation
        megapixels = meta['megapixels']
        bit_depth = meta['bit_depth']
        
        # Apply multipliers
        bit_multiplier = self.BIT_DEPTH_MULTIPLIER.get(bit_depth, 1.5)
        hdr_multiplier = self.HDR_OVERHEAD if meta['is_hdr'] else 1.0
        alpha_multiplier = self.ALPHA_OVERHEAD if meta['has_alpha'] else 1.0
        
        # Stage-by-stage prediction
        stage_predictions = {}
        total_time = 0
        
        for stage, time_per_mp in self.STAGE_TIMES.items():
            # Skip depth estimation if not needed
            if stage == 'depth_estimation' and not include_depth:
                continue
            
            stage_time = time_per_mp * megapixels * bit_multiplier
            
            if stage == 'tone_mapping' and meta['is_hdr']:
                stage_time *= hdr_multiplier
            
            if stage == 'load' and meta['has_alpha']:
                stage_time *= alpha_multiplier
            
            stage_predictions[stage] = stage_time
            total_time += stage_time
        
        # Check history for similar images
        historical_adjustment = 1.0
        confidence = 0.7  # Base confidence
        
        if self.history:
            similar = self._find_similar_images(meta)
            if similar:
                # Adjust based on actual vs predicted for similar images
                adjustments = []
                for record in similar:
                    if 'predicted_time' in record and 'actual_time' in record:
                        if record['predicted_time'] > 0:
                            adj = record['actual_time'] / record['predicted_time']
                            adjustments.append(adj)
                
                if adjustments:
                    historical_adjustment = np.median(adjustments)
                    confidence = min(0.95, 0.7 + 0.05 * len(adjustments))
        
        adjusted_time = total_time * historical_adjustment
        
        # Confidence interval (±15% for now)
        confidence_range = 0.15
        min_time = adjusted_time * (1 - confidence_range)
        max_time = adjusted_time * (1 + confidence_range)
        
        return {
            'predicted_time_sec': adjusted_time,
            'predicted_time_min': adjusted_time / 60,
            'confidence': confidence,
            'confidence_interval': {
                'min_sec': min_time,
                'max_sec': max_time,
                'min_min': min_time / 60,
                'max_min': max_time / 60
            },
            'stage_breakdown': stage_predictions,
            'factors': {
                'megapixels': megapixels,
                'bit_depth_multiplier': bit_multiplier,
                'hdr_multiplier': hdr_multiplier,
                'alpha_multiplier': alpha_multiplier,
                'historical_adjustment': historical_adjustment
            },
            'similar_records_count': len(self._find_similar_images(meta)) if self.history else 0
        }
    
    def _find_similar_images(self, meta: Dict[str, Any], tolerance: float = 0.3) -> List[Dict[str, Any]]:
        """Find historically processed images with similar characteristics."""
        similar = []
        
        for record in self.history:
            record_meta = record.get('metadata', {})
            
            # Check similarity
            mp_diff = abs(record_meta.get('megapixels', 0) - meta['megapixels'])
            mp_ratio = mp_diff / max(meta['megapixels'], 0.1)
            
            if mp_ratio < tolerance:
                if record_meta.get('bit_depth') == meta['bit_depth']:
                    if record_meta.get('is_hdr') == meta['is_hdr']:
                        similar.append(record)
        
        return similar
    
    def predict_batch(self, image_paths: List[Path], include_depth: bool = True) -> Dict[str, Any]:
        """Predict total time for batch processing."""
        print(f"📊 Analyzing batch of {len(image_paths)} images...")
        
        predictions = []
        total_predicted = 0
        total_megapixels = 0
        
        for path in image_paths:
            meta = ImageMetadata(path)
            pred = self.predict_time(meta, include_depth)
            predictions.append({
                'filename': path.name,
                'metadata': meta.metadata,
                'prediction': pred
            })
            total_predicted += pred['predicted_time_sec']
            total_megapixels += meta.metadata['megapixels']
        
        # Aggregate statistics
        predicted_times = [p['prediction']['predicted_time_sec'] for p in predictions]
        
        batch_result = {
            'total_images': len(image_paths),
            'total_megapixels': total_megapixels,
            'total_predicted_sec': total_predicted,
            'total_predicted_min': total_predicted / 60,
            'total_predicted_hours': total_predicted / 3600,
            'estimated_completion': (datetime.now() + timedelta(seconds=total_predicted)).isoformat(),
            'average_time_per_image_min': (total_predicted / len(image_paths)) / 60 if image_paths else 0,
            'predictions': predictions,
            'statistics': {
                'min_time_sec': min(predicted_times) if predicted_times else 0,
                'max_time_sec': max(predicted_times) if predicted_times else 0,
                'median_time_sec': np.median(predicted_times) if predicted_times else 0,
                'std_time_sec': np.std(predicted_times) if predicted_times else 0
            }
        }
        
        return batch_result
    
    def record_actual_time(
        self,
        metadata: ImageMetadata,
        actual_time_sec: float,
        predicted_time_sec: Optional[float] = None
    ):
        """Record actual processing time for future prediction improvement."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata.metadata,
            'actual_time': actual_time_sec,
            'predicted_time': predicted_time_sec
        }
        
        self.history.append(record)
        
        if self.history_path:
            self._save_history()
    
    def _save_history(self):
        """Save processing history to disk."""
        try:
            data = {
                'last_updated': datetime.now().isoformat(),
                'total_records': len(self.history),
                'records': self.history
            }
            
            with open(self.history_path, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            print(f"⚠ Could not save history: {e}")
    
    def print_prediction(self, prediction: Dict[str, Any], name: str = "Image"):
        """Pretty-print prediction results."""
        pred = prediction
        
        print(f"\n⏱️  Time Prediction: {name}")
        print(f"  Estimated time: {pred['predicted_time_min']:.2f} minutes ({pred['predicted_time_sec']:.1f} seconds)")
        print(f"  Confidence: {pred['confidence']*100:.0f}%")
        print(f"  Range: {pred['confidence_interval']['min_min']:.2f} - {pred['confidence_interval']['max_min']:.2f} minutes")
        
        if pred.get('stage_breakdown'):
            print(f"\n  Stage breakdown:")
            for stage, time in pred['stage_breakdown'].items():
                print(f"    {stage:20s}: {time:6.2f}s")
        
        factors = pred['factors']
        print(f"\n  Factors:")
        print(f"    Megapixels: {factors['megapixels']:.1f}")
        print(f"    Bit depth multiplier: {factors['bit_depth_multiplier']:.2f}x")
        if factors['hdr_multiplier'] > 1:
            print(f"    HDR overhead: {factors['hdr_multiplier']:.2f}x")
        if factors['historical_adjustment'] != 1.0:
            print(f"    Historical adjustment: {factors['historical_adjustment']:.2f}x")


def main():
    """CLI for time prediction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict image processing time")
    parser.add_argument("inputs", nargs='+', type=Path, help="Image files or directory")
    parser.add_argument("--no-depth", action="store_true", help="Skip depth estimation")
    parser.add_argument("--history", type=Path, help="Path to historical data JSON")
    parser.add_argument("--output", type=Path, help="Save predictions to JSON")
    
    args = parser.parse_args()
    
    # Collect image paths
    image_paths = []
    for input_path in args.inputs:
        if input_path.is_dir():
            image_paths.extend(input_path.glob("*.tif"))
            image_paths.extend(input_path.glob("*.tiff"))
            image_paths.extend(input_path.glob("*.jpg"))
            image_paths.extend(input_path.glob("*.png"))
        else:
            image_paths.append(input_path)
    
    if not image_paths:
        print("❌ No images found")
        return
    
    # Create predictor
    predictor = ProcessingTimePredictor(args.history)
    
    # Predict batch
    batch_pred = predictor.predict_batch(image_paths, include_depth=not args.no_depth)
    
    # Print summary
    print("\n" + "="*80)
    print("⏱️  BATCH PROCESSING TIME PREDICTION")
    print("="*80)
    print(f"Total images: {batch_pred['total_images']}")
    print(f"Total megapixels: {batch_pred['total_megapixels']:.1f} MP")
    print(f"\nEstimated total time: {batch_pred['total_predicted_min']:.1f} minutes ({batch_pred['total_predicted_hours']:.2f} hours)")
    print(f"Average per image: {batch_pred['average_time_per_image_min']:.2f} minutes")
    print(f"\nEstimated completion: {batch_pred['estimated_completion']}")
    print("="*80)
    
    # Individual predictions
    print("\n📋 Individual Predictions:")
    for pred_item in batch_pred['predictions']:
        print(f"\n  {pred_item['filename']}")
        meta = pred_item['metadata']
        pred = pred_item['prediction']
        print(f"    {meta['width']}x{meta['height']} ({meta['megapixels']:.1f} MP), {meta['bit_depth']}-bit")
        print(f"    Time: {pred['predicted_time_min']:.2f} min (±{pred['confidence']*100:.0f}%)")
    
    # Save if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(batch_pred, f, indent=2)
        print(f"\n💾 Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
