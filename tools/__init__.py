from .registry import get_tool, list_tools, normalize_tool_name
from utils.solve_math import solve_equation as solve_math, is_math_equation

def is_math_tool(name: str) -> bool:
    return name == "solve_math"

__all__ = ["get_tool", "list_tools", "normalize_tool_name", "is_math_tool", "solve_math", "is_math_equation"]