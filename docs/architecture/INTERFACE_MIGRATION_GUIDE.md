# Interface Migration Guide

**Version**: 1.0  
**Date**: December 7, 2025  
**Status**: Active  

This guide provides practical examples for migrating existing code to use the new interface contracts defined in ADR-001.

## Table of Contents

1. [Overview](#overview)
2. [Interface Hierarchy](#interface-hierarchy)
3. [Migration Examples](#migration-examples)
4. [Testing Interface Implementations](#testing-interface-implementations)
5. [Common Patterns](#common-patterns)
6. [Troubleshooting](#troubleshooting)

---

## Overview

The Transformation Portal now provides explicit interface contracts for all major architectural layers. Adopting these interfaces provides:

- **Type safety**: Better IDE autocomplete and static analysis
- **Testability**: Easy to mock dependencies
- **Documentation**: Interface docstrings serve as API contracts
- **Modularity**: Clear boundaries between components

### Available Interfaces

| Interface | Purpose | Module |
|-----------|---------|--------|
| `ImageProcessor` | Core image transformations | `interfaces.processor` |
| `VideoProcessor` | Video processing operations | `interfaces.processor` |
| `Pipeline` | Multi-stage processing orchestration | `interfaces.pipeline` |
| `PipelineStage` | Individual pipeline stages | `interfaces.pipeline` |
| `BatchPipeline` | Batch processing workflows | `interfaces.pipeline` |
| `Enhancer` | Image enhancement algorithms | `interfaces.enhancer` |
| `AdaptiveEnhancer` | Context-aware enhancement | `interfaces.enhancer` |
| `Segmenter` | Image segmentation | `interfaces.segmenter` |
| `MaterialSegmenter` | Material detection | `interfaces.segmenter` |
| `SemanticSegmenter` | Semantic scene understanding | `interfaces.segmenter` |
| `DepthEstimator` | Monocular depth estimation | `interfaces.estimator` |
| `NormalEstimator` | Surface normal estimation | `interfaces.estimator` |
| `UnifiedEstimator` | Combined depth + normals | `interfaces.estimator` |

---

## Interface Hierarchy

```
interfaces/
├── processor.py      # Base transformation operations
├── pipeline.py       # Multi-stage orchestration
├── enhancer.py       # Enhancement algorithms
├── segmenter.py      # Segmentation & classification
└── estimator.py      # Geometry estimation
```

**Dependency Rule**: Interfaces should be imported by implementations, not by other interfaces.

---

## Migration Examples

### Example 1: Migrating an Image Processor

**Before** (no interface):
```python
# src/transformation_portal/processors/my_processor.py
import numpy as np

class MyProcessor:
    """Process images."""
    
    def __init__(self, strength: float = 1.0):
        self.strength = strength
    
    def process(self, image: np.ndarray) -> np.ndarray:
        # Apply processing
        return image * self.strength
```

**After** (with interface):
```python
# src/transformation_portal/processors/my_processor.py
import numpy as np
from typing import Dict, Any
from transformation_portal.interfaces import ImageProcessor

class MyProcessor(ImageProcessor):
    """
    Process images with configurable strength.
    
    Implements ImageProcessor interface for type safety and testability.
    """
    
    def __init__(self, strength: float = 1.0):
        self.strength = strength
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply processing to image.
        
        Args:
            image: Input image (H, W, C) in [0, 1] float32 or [0, 255] uint8
            **kwargs: Additional parameters (unused)
            
        Returns:
            Processed image, same shape and dtype as input
        """
        # Validate input
        self.validate_input(image)
        
        # Apply processing
        return image * self.strength
    
    def get_config(self) -> Dict[str, Any]:
        """Return processor configuration."""
        return {
            "type": "MyProcessor",
            "strength": self.strength,
            "version": "1.0"
        }
```

**Benefits**:
- Type hints enforced by interface
- Validation provided by base class
- Configuration is serializable
- Easy to mock in tests

---

### Example 2: Migrating an Enhancer

**Before**:
```python
class ClarityEnhancer:
    def enhance(self, image: np.ndarray, amount: float) -> np.ndarray:
        # Apply clarity enhancement
        return enhanced_image
```

**After**:
```python
from transformation_portal.interfaces import Enhancer

class ClarityEnhancer(Enhancer):
    """
    Enhance image clarity via unsharp masking.
    
    Implements Enhancer interface with standardized strength parameter.
    """
    
    def __init__(self, radius: float = 1.0):
        self.radius = radius
    
    def enhance(
        self,
        image: np.ndarray,
        strength: float = 1.0,
        **kwargs
    ) -> np.ndarray:
        """
        Apply clarity enhancement.
        
        Args:
            image: Input image (H, W, C)
            strength: Enhancement strength in [0, 1]
            **kwargs: Additional parameters
            
        Returns:
            Enhanced image
        """
        # Validate strength
        self.validate_strength(strength)
        
        # Apply enhancement with strength scaling
        amount = strength * 2.0  # Internal scaling
        enhanced = self._apply_unsharp_mask(image, self.radius, amount)
        return enhanced
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "ClarityEnhancer",
            "radius": self.radius
        }
    
    def _apply_unsharp_mask(
        self, 
        image: np.ndarray, 
        radius: float, 
        amount: float
    ) -> np.ndarray:
        # Implementation details...
        pass
```

---

### Example 3: Migrating a Material Segmenter

**Before**:
```python
class MaterialDetector:
    def detect(self, image: np.ndarray) -> dict:
        # Return material masks
        return {"wood": wood_mask, "metal": metal_mask}
```

**After**:
```python
from transformation_portal.interfaces import MaterialSegmenter, MaterialType
from typing import List, Optional

class MaterialDetector(MaterialSegmenter):
    """
    Detect materials using heuristic color analysis.
    
    Implements MaterialSegmenter interface for standardized output.
    """
    
    def segment(
        self,
        image: np.ndarray,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Segment image into material regions."""
        return self.segment_materials(image)
    
    def segment_materials(
        self,
        image: np.ndarray,
        materials: Optional[List[MaterialType]] = None,
        **kwargs
    ) -> Dict[MaterialType, np.ndarray]:
        """
        Segment image by material types.
        
        Args:
            image: Input image (H, W, C)
            materials: Optional list of materials to detect
            
        Returns:
            Dictionary mapping MaterialType to boolean masks
        """
        if materials is None:
            materials = [MaterialType.WOOD, MaterialType.METAL, MaterialType.GLASS]
        
        results = {}
        for mat in materials:
            mask = self._detect_material(image, mat)
            results[mat] = mask
        
        return results
    
    def get_supported_categories(self) -> List[str]:
        return [mat.value for mat in MaterialType]
    
    def get_config(self) -> Dict[str, Any]:
        return {"type": "MaterialDetector", "method": "heuristic"}
    
    def get_material_properties(self, material: MaterialType) -> Dict[str, Any]:
        """Get physical properties for material type."""
        properties = {
            MaterialType.WOOD: {"roughness": 0.7, "metallic": 0.0},
            MaterialType.METAL: {"roughness": 0.1, "metallic": 1.0},
            MaterialType.GLASS: {"roughness": 0.0, "metallic": 0.0},
        }
        return properties.get(material, {})
```

---

### Example 4: Migrating a Depth Estimator

**Before**:
```python
class DepthPredictor:
    def predict(self, image: np.ndarray) -> np.ndarray:
        # Return depth map
        return depth
```

**After**:
```python
from transformation_portal.interfaces import DepthEstimator

class DepthPredictor(DepthEstimator):
    """
    Predict depth using Depth Anything V2 model.
    
    Implements DepthEstimator interface for standardized depth output.
    """
    
    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self._model = self._load_model(model_size)
    
    def estimate_depth(
        self,
        image: np.ndarray,
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Estimate depth map from RGB image.
        
        Args:
            image: Input RGB image (H, W, 3)
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            Depth map (H, W) where 0=far, 1=near
        """
        # Predict raw depth
        depth = self._model.predict(image)
        
        # Normalize if requested
        if normalize:
            depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        return depth.astype(np.float32)
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "Depth Anything V2",
            "size": self.model_size,
            "architecture": "ViT-based",
            "input_size": "variable"
        }
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "DepthPredictor",
            "model_size": self.model_size
        }
```

---

### Example 5: Migrating a Pipeline

**Before**:
```python
class MyPipeline:
    def run(self, input_path: Path) -> Path:
        # Process image
        return output_path
```

**After**:
```python
from transformation_portal.interfaces import Pipeline, PipelineStage
from pathlib import Path
from typing import Optional, List, Dict, Any

class MyPipeline(Pipeline):
    """
    Multi-stage image processing pipeline.
    
    Implements Pipeline interface for orchestration consistency.
    """
    
    def __init__(self):
        self._stages: List[PipelineStage] = []
    
    def add_stage(self, stage: PipelineStage, name: Optional[str] = None) -> None:
        """Add processing stage to pipeline."""
        self._stages.append(stage)
    
    def execute(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute complete pipeline.
        
        Args:
            input_path: Input file path
            output_path: Optional output path
            **kwargs: Pipeline parameters
            
        Returns:
            Dictionary with execution results
        """
        # Load input
        from PIL import Image
        image = np.array(Image.open(input_path))
        
        # Execute stages
        context = {"input_path": input_path}
        data = image
        
        for stage in self._stages:
            data = stage.execute(data, context)
        
        # Save output
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_processed{input_path.suffix}"
        
        Image.fromarray(data).save(output_path)
        
        return {
            "output_path": output_path,
            "stages_executed": len(self._stages),
            "success": True
        }
    
    def get_stages(self) -> List[PipelineStage]:
        return self._stages
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "MyPipeline",
            "stages": [stage.get_name() for stage in self._stages]
        }
```

---

## Testing Interface Implementations

### Unit Test Example

```python
# tests/test_my_processor.py
import pytest
import numpy as np
from transformation_portal.interfaces import ImageProcessor
from transformation_portal.processors.my_processor import MyProcessor

def test_my_processor_implements_interface():
    """Verify MyProcessor implements ImageProcessor interface."""
    processor = MyProcessor()
    assert isinstance(processor, ImageProcessor)

def test_my_processor_contract():
    """Test processor adheres to interface contract."""
    processor = MyProcessor(strength=1.5)
    
    # Test input validation
    image = np.random.rand(100, 100, 3).astype(np.float32)
    result = processor.process(image)
    
    # Contract: Same shape and dtype
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    
    # Contract: Config is serializable
    config = processor.get_config()
    assert isinstance(config, dict)
    import json
    json.dumps(config)  # Should not raise

def test_my_processor_invalid_input():
    """Test processor rejects invalid inputs."""
    processor = MyProcessor()
    
    # Invalid: wrong dimensions
    with pytest.raises(ValueError):
        processor.process(np.array([1, 2, 3]))
```

### Mock for Testing Pipelines

```python
# tests/test_my_pipeline.py
from unittest.mock import MagicMock
from transformation_portal.interfaces import PipelineStage

def test_pipeline_with_mock_stages():
    """Test pipeline orchestration with mock stages."""
    from transformation_portal.processors.my_pipeline import MyPipeline
    
    # Create mock stage
    mock_stage = MagicMock(spec=PipelineStage)
    mock_stage.get_name.return_value = "MockStage"
    mock_stage.execute.return_value = np.ones((100, 100, 3))
    
    # Build pipeline
    pipeline = MyPipeline()
    pipeline.add_stage(mock_stage)
    
    # Execute
    result = pipeline.execute(Path("input.jpg"))
    
    # Verify stage was called
    mock_stage.execute.assert_called_once()
    assert result["success"] is True
```

---

## Common Patterns

### Pattern 1: Dependency Injection

```python
class EnhancedPipeline(Pipeline):
    """Pipeline with injected dependencies."""
    
    def __init__(
        self,
        depth_estimator: DepthEstimator,
        material_segmenter: MaterialSegmenter,
        enhancer: Enhancer
    ):
        self.depth_estimator = depth_estimator
        self.material_segmenter = material_segmenter
        self.enhancer = enhancer
```

**Benefits**: Easy to swap implementations, testable with mocks.

### Pattern 2: Factory Functions

```python
def create_depth_estimator(model_type: str = "auto") -> DepthEstimator:
    """Factory for depth estimators."""
    if model_type == "auto":
        try:
            from .depth_anything_v2 import DepthAnythingV2
            return DepthAnythingV2()
        except ImportError:
            from .fallback_depth import FallbackDepth
            return FallbackDepth()
    elif model_type == "midas":
        from .midas_depth import MidasDepth
        return MidasDepth()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

### Pattern 3: Adapter for Legacy Code

```python
from transformation_portal.interfaces import ImageProcessor

class LegacyProcessorAdapter(ImageProcessor):
    """Adapter for legacy processor without interface."""
    
    def __init__(self, legacy_processor):
        self._legacy = legacy_processor
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Adapt old API to new interface
        return self._legacy.old_method(image)
    
    def get_config(self) -> Dict[str, Any]:
        return {"type": "LegacyAdapter", "wrapped": type(self._legacy).__name__}
```

---

## Troubleshooting

### Issue: Import Errors

**Problem**: `ImportError: cannot import name 'ImageProcessor'`

**Solution**:
```python
# ❌ Wrong
from transformation_portal.interfaces.processor import ImageProcessor

# ✅ Correct
from transformation_portal.interfaces import ImageProcessor
```

### Issue: Type Checker Warnings

**Problem**: mypy/pyright complains about missing methods

**Solution**: Ensure all abstract methods are implemented:
```python
class MyProcessor(ImageProcessor):
    # Must implement both methods
    def process(self, image, **kwargs): pass
    def get_config(self): pass
```

### Issue: Circular Imports

**Problem**: Module A imports Module B which imports Module A

**Solution**: Use interfaces to break the cycle:
```python
# ❌ Bad: Direct import creates cycle
from module_b import ConcreteProcessor

# ✅ Good: Import interface only
from transformation_portal.interfaces import ImageProcessor

def process(processor: ImageProcessor):
    # Accept any ImageProcessor implementation
    pass
```

---

## Migration Checklist

- [ ] Identify module type (processor, enhancer, segmenter, etc.)
- [ ] Choose appropriate interface
- [ ] Inherit from interface class
- [ ] Implement all abstract methods
- [ ] Add `get_config()` method
- [ ] Update type hints
- [ ] Add input validation (use base class methods)
- [ ] Write contract tests
- [ ] Update documentation
- [ ] Run `python scripts/validation/check_module_boundaries.py`

---

## References

- [ADR-001: Module Interface Contracts](adr/ADR-001-module-interface-contracts.md)
- [Interface Source Code](../../src/transformation_portal/interfaces/)
- [Example Tests](../../tests/test_interface_contracts.py)
- [Architecture Overview](ARCHITECTURE_REVIEW_2025.md)

---

**Questions?** See `docs/TROUBLESHOOTING.md` or open an issue.
