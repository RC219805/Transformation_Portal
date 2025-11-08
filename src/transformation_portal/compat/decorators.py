"""Deprecation decorators for backwards-compatibility."""

import functools
import warnings
from typing import Callable, Optional, Any


def deprecated(
    replacement: Optional[str] = None,
    removal_version: Optional[str] = None,
    message: Optional[str] = None,
    category: type = DeprecationWarning
):
    """Mark a function, method, or class as deprecated.
    
    Args:
        replacement: Name of replacement function/class
        removal_version: Version when this will be removed
        message: Custom deprecation message
        category: Warning category (default: DeprecationWarning)
        
    Example:
        >>> @deprecated(
        ...     replacement="new_process_image",
        ...     removal_version="2.0.0"
        ... )
        ... def old_process_image(image):
        ...     return new_process_image(image)
    """
    def decorator(obj):
        # Handle classes
        if isinstance(obj, type):
            original_init = obj.__init__
            
            @functools.wraps(original_init)
            def new_init(self, *args, **kwargs):
                _show_deprecation_warning(obj.__name__, replacement, removal_version, message, category)
                original_init(self, *args, **kwargs)
            
            obj.__init__ = new_init
            obj.__deprecated__ = True
            return obj
        
        # Handle functions/methods
        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            _show_deprecation_warning(obj.__name__, replacement, removal_version, message, category)
            return obj(*args, **kwargs)
        
        wrapper.__deprecated__ = True
        return wrapper
    
    return decorator


def renamed_function(old_name: str, new_name: str, removal_version: Optional[str] = None):
    """Decorator for functions that have been renamed.
    
    Args:
        old_name: Original function name
        new_name: New function name
        removal_version: Version when old name will be removed
        
    Example:
        >>> @renamed_function("process_image", "enhance_image", "2.0.0")
        ... def enhance_image(image):
        ...     return image
        ...
        >>> # Legacy name still works with warning
        >>> process_image = enhance_image
    """
    message = f"Function '{old_name}' has been renamed to '{new_name}'"
    return deprecated(replacement=new_name, removal_version=removal_version, message=message)


def renamed_class(old_name: str, new_name: str, removal_version: Optional[str] = None):
    """Decorator for classes that have been renamed.
    
    Args:
        old_name: Original class name
        new_name: New class name
        removal_version: Version when old name will be removed
        
    Example:
        >>> @renamed_class("OldProcessor", "NewProcessor", "2.0.0")
        ... class NewProcessor:
        ...     pass
        ...
        >>> # Legacy name still works
        >>> OldProcessor = NewProcessor
    """
    message = f"Class '{old_name}' has been renamed to '{new_name}'"
    return deprecated(replacement=new_name, removal_version=removal_version, message=message)


def renamed_module(old_path: str, new_path: str, removal_version: Optional[str] = None):
    """Create a deprecation warning for renamed modules.
    
    Args:
        old_path: Original module path
        new_path: New module path
        removal_version: Version when old path will be removed
        
    Example:
        >>> # In old_module.py:
        >>> renamed_module(
        ...     "transformation_portal.old_module",
        ...     "transformation_portal.processors.new_module",
        ...     "2.0.0"
        ... )
        >>> from transformation_portal.processors.new_module import *
    """
    message = f"Module '{old_path}' has been moved to '{new_path}'"
    if removal_version:
        message += f" and will be removed in version {removal_version}"
    
    warnings.warn(message, DeprecationWarning, stacklevel=2)


def moved_to(new_location: str, removal_version: Optional[str] = None):
    """Decorator for functions/classes that have been moved.
    
    Args:
        new_location: New import path
        removal_version: Version when old location will be removed
        
    Example:
        >>> @moved_to("transformation_portal.processors.depth", "2.0.0")
        ... class DepthEstimator:
        ...     pass
    """
    message = f"This has been moved to '{new_location}'"
    return deprecated(replacement=new_location, removal_version=removal_version, message=message)


def _show_deprecation_warning(
    name: str,
    replacement: Optional[str],
    removal_version: Optional[str],
    custom_message: Optional[str],
    category: type
):
    """Show deprecation warning with formatted message."""
    if custom_message:
        message = custom_message
    else:
        message = f"'{name}' is deprecated"
        
        if replacement:
            message += f". Use '{replacement}' instead"
        
        if removal_version:
            message += f". It will be removed in version {removal_version}"
    
    warnings.warn(message, category, stacklevel=3)


def experimental(message: Optional[str] = None):
    """Mark a function/class as experimental (API may change).
    
    Args:
        message: Custom warning message
        
    Example:
        >>> @experimental("API may change in future versions")
        ... def new_feature():
        ...     pass
    """
    def decorator(obj):
        warning_msg = message or f"'{obj.__name__}' is experimental and its API may change"
        
        if isinstance(obj, type):
            original_init = obj.__init__
            
            @functools.wraps(original_init)
            def new_init(self, *args, **kwargs):
                warnings.warn(warning_msg, FutureWarning, stacklevel=2)
                original_init(self, *args, **kwargs)
            
            obj.__init__ = new_init
            return obj
        
        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            warnings.warn(warning_msg, FutureWarning, stacklevel=2)
            return obj(*args, **kwargs)
        
        return wrapper
    
    return decorator
