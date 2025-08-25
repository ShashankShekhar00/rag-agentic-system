"""Tool decorator for creating reusable AI tools."""

import functools
from typing import Callable, Any, Dict
from langchain.tools import StructuredTool


def tool(name: str = None, description: str = None):
    """
    Decorator to convert a function into a LangChain tool.
    
    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
    
    Returns:
        A LangChain StructuredTool instance
    """
    def decorator(func: Callable) -> StructuredTool:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"
        
        return StructuredTool.from_function(
            func=func,
            name=tool_name,
            description=tool_description
        )
    
    return decorator


def create_structured_tool(func: Callable, name: str = None, description: str = None) -> StructuredTool:
    """
    Create a structured tool from a function.
    
    Args:
        func: The function to convert
        name: Tool name
        description: Tool description
    
    Returns:
        A structured LangChain tool
    """
    return StructuredTool.from_function(
        func=func,
        name=name or func.__name__,
        description=description or func.__doc__ or f"Tool: {func.__name__}"
    )
