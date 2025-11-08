"""Plugin interface definitions for extensible architecture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import inspect


class PluginType(Enum):
    """Types of plugins supported by the system."""
    DEPTH_MODEL = "depth_model"
    PROCESSOR = "processor"
    ENHANCER = "enhancer"
    LUT_PROVIDER = "lut_provider"
    RERANKER = "reranker"
    UPSCALER = "upscaler"
    TONE_MAPPER = "tone_mapper"
    CUSTOM = "custom"


@dataclass
class PluginMetadata:
    """Metadata describing a plugin."""
    name: str
    version: str
    plugin_type: PluginType
    description: str = ""
    author: str = ""
    license: str = "MIT"
    dependencies: List[str] = field(default_factory=list)
    min_portal_version: str = "0.1.0"
    max_portal_version: Optional[str] = None
    deprecated: bool = False
    replacement: Optional[str] = None
    homepage: str = ""
    tags: List[str] = field(default_factory=list)
    
    def is_compatible(self, portal_version: str) -> bool:
        """Check if plugin is compatible with current portal version."""
        from packaging import version
        portal_ver = version.parse(portal_version)
        min_ver = version.parse(self.min_portal_version)
        
        if portal_ver < min_ver:
            return False
            
        if self.max_portal_version:
            max_ver = version.parse(self.max_portal_version)
            if portal_ver > max_ver:
                return False
                
        return True


class PluginInterface(ABC):
    """Base interface that all plugins must implement.
    
    Plugins provide extensible functionality for the Transformation Portal,
    enabling hot-swappable components and community contributions.
    
    Attributes:
        metadata: Plugin metadata including name, version, and dependencies
        
    Example:
        >>> class MyDepthModel(PluginInterface):
        ...     def __init__(self):
        ...         self.metadata = PluginMetadata(
        ...             name="my_depth_model",
        ...             version="1.0.0",
        ...             plugin_type=PluginType.DEPTH_MODEL,
        ...             description="Custom depth estimation model"
        ...         )
        ...     
        ...     def initialize(self, config):
        ...         self.model = load_my_model(config)
        ...     
        ...     def execute(self, image):
        ...         return self.model.predict(image)
    """
    
    def __init__(self):
        """Initialize plugin with metadata."""
        self.metadata: PluginMetadata = self._create_metadata()
        self._initialized = False
        self._config: Dict[str, Any] = {}
    
    @abstractmethod
    def _create_metadata(self) -> PluginMetadata:
        """Create plugin metadata. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize plugin with configuration.
        
        Args:
            config: Optional configuration dictionary
            
        Raises:
            PluginInitializationError: If initialization fails
        """
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute plugin's main functionality.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Plugin execution result
            
        Raises:
            PluginExecutionError: If execution fails
        """
        pass
    
    def validate(self) -> bool:
        """Validate plugin configuration and state.
        
        Returns:
            True if plugin is valid, False otherwise
        """
        return self._initialized
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        self._initialized = False
        self._config = {}
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information.
        
        Returns:
            Dictionary with plugin details
        """
        return {
            'name': self.metadata.name,
            'version': self.metadata.version,
            'type': self.metadata.plugin_type.value,
            'description': self.metadata.description,
            'author': self.metadata.author,
            'initialized': self._initialized,
            'deprecated': self.metadata.deprecated,
            'replacement': self.metadata.replacement,
        }
    
    def __repr__(self) -> str:
        """String representation of plugin."""
        return (f"<{self.__class__.__name__} "
                f"name='{self.metadata.name}' "
                f"version='{self.metadata.version}'>")


class DepthModelPlugin(PluginInterface):
    """Specialized interface for depth estimation model plugins."""
    
    @abstractmethod
    def estimate_depth(self, image: Any, **kwargs) -> Any:
        """Estimate depth map from image.
        
        Args:
            image: Input image (PIL, numpy array, or tensor)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Depth map (format depends on implementation)
        """
        pass
    
    def execute(self, image: Any, **kwargs) -> Any:
        """Execute depth estimation (delegates to estimate_depth)."""
        return self.estimate_depth(image, **kwargs)


class ProcessorPlugin(PluginInterface):
    """Specialized interface for image/video processor plugins."""
    
    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> Any:
        """Process input data.
        
        Args:
            input_data: Input image/video data
            **kwargs: Processing parameters
            
        Returns:
            Processed output
        """
        pass
    
    def execute(self, input_data: Any, **kwargs) -> Any:
        """Execute processing (delegates to process)."""
        return self.process(input_data, **kwargs)


class EnhancerPlugin(PluginInterface):
    """Specialized interface for enhancement plugins."""
    
    @abstractmethod
    def enhance(self, image: Any, strength: float = 1.0, **kwargs) -> Any:
        """Enhance image.
        
        Args:
            image: Input image
            strength: Enhancement strength (0.0 to 1.0)
            **kwargs: Additional parameters
            
        Returns:
            Enhanced image
        """
        pass
    
    def execute(self, image: Any, **kwargs) -> Any:
        """Execute enhancement (delegates to enhance)."""
        return self.enhance(image, **kwargs)


class PluginInitializationError(Exception):
    """Raised when plugin initialization fails."""
    pass


class PluginExecutionError(Exception):
    """Raised when plugin execution fails."""
    pass


class PluginValidationError(Exception):
    """Raised when plugin validation fails."""
    pass
