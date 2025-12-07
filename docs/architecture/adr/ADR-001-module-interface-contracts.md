# ADR-001: Module Interface Contracts

**Date**: December 7, 2025  
**Status**: Proposed  
**Architect**: Transformation Portal Architecture Team

## Context

The Transformation Portal has grown organically with 27+ subdirectories in `src/transformation_portal/`. While conceptually following a layered architecture (pipelines → processors → enhancers → utils), there are no enforced contracts or interfaces. This has led to:

1. **Boundary violations**: Utils importing from processors, processors from pipelines
2. **Tight coupling**: Direct implementation dependencies instead of interface contracts
3. **Testing challenges**: Cannot mock dependencies without complex setup
4. **Unclear responsibilities**: No formal specification of what each layer provides

**Evidence from analysis**:
- 98 internal imports within `transformation_portal` package
- Pipelines module has 42 external imports (highest coupling)
- No abstract base classes defining module contracts
- Module boundaries exist only in documentation, not code

## Decision

We will introduce **explicit interface contracts** for all major architectural layers using Python Abstract Base Classes (ABCs). All implementations must conform to these interfaces.

**Interfaces to create**:

1. **`interfaces/processor.py`** - Base contract for all image processors
2. **`interfaces/pipeline.py`** - Base contract for multi-stage pipelines
3. **`interfaces/enhancer.py`** - Base contract for enhancement algorithms
4. **`interfaces/segmenter.py`** - Base contract for material/semantic segmentation
5. **`interfaces/estimator.py`** - Base contract for depth/normal estimation

**Interface example**:
```python
# src/transformation_portal/interfaces/processor.py
from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

class ImageProcessor(ABC):
    """
    Base interface for all image processing operations.
    
    Contract:
    - Must accept numpy array (H, W, C) in [0, 1] float or [0, 255] uint8
    - Must return numpy array of same shape and dtype
    - Must be stateless OR clearly document state management
    - Configuration must be serializable (for reproducibility)
    """
    
    @abstractmethod
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Process input image.
        
        Args:
            image: Input image (H, W, C) numpy array
            **kwargs: Processor-specific parameters
            
        Returns:
            Processed image, same shape and dtype as input
            
        Raises:
            ValueError: If image format is invalid
            ProcessingError: If processing fails
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return current processor configuration.
        
        Returns:
            Dictionary of configuration parameters (JSON-serializable)
        """
        pass
    
    def validate_input(self, image: np.ndarray) -> None:
        """
        Validate input image meets contract requirements.
        
        Raises:
            ValueError: If image format invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(image)}")
        
        if image.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D array, got {image.ndim}D")
        
        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise ValueError(f"Expected 1, 3, or 4 channels, got {image.shape[2]}")
```

## Consequences

### Positive

1. **Enforced boundaries**: CI checks can validate no cross-layer violations
2. **Testability**: Easy to mock interfaces for unit testing
3. **Documentation**: Interface docstrings serve as API documentation
4. **Type safety**: mypy/pyright can validate interface conformance
5. **Extensibility**: Third-party plugins can implement interfaces
6. **Clarity**: Explicit contracts eliminate ambiguity
7. **Refactoring safety**: Change implementation without breaking interface

### Negative

1. **Initial overhead**: Requires retrofitting existing code (est. 1 week effort)
2. **Learning curve**: Developers must understand interface-based design
3. **Boilerplate**: More code required for interface + implementation
4. **Breaking changes**: Some existing APIs may need adjustment
5. **Migration period**: Temporary coexistence of old/new patterns

### Neutral

1. **No performance impact**: ABCs have negligible runtime overhead
2. **Tooling compatible**: Works with existing pytest, mypy, pylint setup

## Implementation Plan

### Phase 1: Interface Definition (Week 1)
1. Create `src/transformation_portal/interfaces/` package
2. Define 5 core interfaces (processor, pipeline, enhancer, segmenter, estimator)
3. Add comprehensive docstrings with contract specifications
4. Write interface contract tests

### Phase 2: Gradual Migration (Weeks 2-4)
1. **Week 2**: Migrate `processors/` to implement `ImageProcessor`
   - MaterialResponse, LuxuryVideoMasterGrader
2. **Week 3**: Migrate `enhancers/` to implement `Enhancer`
   - enhance_aerial, board_material_aerial_enhancer
3. **Week 4**: Migrate `pipelines/` to implement `Pipeline`
   - lux_render_pipeline, unified_luxury_pipeline

### Phase 3: Enforcement (Week 5)
1. Add CI check: `scripts/validation/check_interface_conformance.py`
2. Update documentation with interface examples
3. Add interface-based testing guide
4. Mark old non-interface code as deprecated

## Alternatives Considered

### Alternative 1: Protocol (PEP 544) Structural Subtyping
**Pros**: No explicit inheritance, more flexible  
**Cons**: Weaker enforcement, harder to document  
**Rejected**: Explicit ABCs provide clearer contracts

### Alternative 2: TypedDict for Configuration
**Pros**: Type-safe configuration  
**Cons**: Doesn't enforce method contracts  
**Rejected**: Complements but doesn't replace interfaces

### Alternative 3: No Interfaces (Status Quo)
**Pros**: No migration effort  
**Cons**: Continues current coupling and boundary violations  
**Rejected**: Technical debt will compound

## Validation Criteria

**Success metrics**:
- [ ] 100% of processors implement `ImageProcessor` interface
- [ ] 100% of pipelines implement `Pipeline` interface
- [ ] CI boundary check passes (no cross-layer imports)
- [ ] Interface documentation published
- [ ] At least 1 third-party plugin uses interfaces

**Acceptance test**:
```python
def test_material_response_implements_interface():
    """Verify MaterialResponse conforms to ImageProcessor interface."""
    from transformation_portal.interfaces import ImageProcessor
    from transformation_portal.processors.material_response import MaterialResponse
    
    # Structural check
    assert issubclass(MaterialResponse, ImageProcessor)
    
    # Behavioral check
    processor = MaterialResponse()
    image = np.random.rand(100, 100, 3).astype(np.float32)
    result = processor.process(image)
    
    # Contract validation
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert isinstance(processor.get_config(), dict)
```

## Related ADRs

- ADR-002: Dependency Management Strategy (uses interfaces for plugin loading)
- ADR-004: Monolithic File Refactoring (enabled by interface decomposition)
- ADR-005: Event-Driven Analytics (interfaces for event producers/consumers)

## References

- **PEP 3119**: Introducing Abstract Base Classes
- **Python ABC Documentation**: https://docs.python.org/3/library/abc.html
- **Clean Architecture** (Robert Martin): Dependency inversion principle
- **Repository Analysis**: `docs/architecture/ARCHITECTURE_REVIEW_2025.md`

---

**Approval**: Pending stakeholder review  
**Implementation**: Targeting Q1 2026  
**Review Date**: March 7, 2026
