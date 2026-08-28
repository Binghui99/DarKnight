"""
tools/registry.py
tool registry for Dark Knight.

All tools here follow the same base64 signature:
    tool_fn(image_b64: str, **params) -> b64_str

"""
import base64
from typing import Dict, Callable

import cv2
import numpy as np

from tools.color_lines_removal import color_lines_removal

from tools.toolbox import (
    sharpen,
    remove_small_noise,
    remove_straight_line,
    binarize,
)


# Base64 ↔ numpy helpers

def _b64_to_bgr(image_b64: str):
    """Decode a data-URI base64 string to a BGR numpy array + header."""
    if "," in image_b64:
        header, encoded = image_b64.split(",", 1)
    else:
        header  = "data:image/png"
        encoded = image_b64
    img_data = base64.b64decode(encoded)
    nparr    = np.frombuffer(img_data, np.uint8)
    img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("registry: could not decode image from base64.")
    return img, header


def _arr_to_b64(arr: np.ndarray, header: str = "data:image/png") -> str:
    """Encode a numpy array (BGR or grayscale) back to a data-URI base64 string."""
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    _, buffer = cv2.imencode(".png", arr)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return header + "," + b64


def _wrap(fn: Callable) -> Callable:
    """Return a version of *fn* that accepts/returns base64 strings."""
    def wrapped(image_b64: str, **params) -> str:
        img, header = _b64_to_bgr(image_b64)
        result      = fn(img, **params)
        return _arr_to_b64(result, header)
    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    wrapped.__doc__  = getattr(fn, "__doc__",  "")
    return wrapped


#  Tool registry (all tools accept/return base64 strings)

TOOL_REGISTRY: Dict[str, Callable] = {
    # USE: coloured (non-black) wavy/diagonal lines cross the text
    "color_extract":    color_lines_removal,

    # USE: wave/blur distortion softens edges 
    "sharpen":          _wrap(sharpen),

    # USE: tiny scattered noise dots/clusters remaining after color_extract
    "remove_small_noise":         _wrap(remove_small_noise),

    # USE: single thin dark diagonal/near-horizontal line crosses all chars (solid filled)
    "remove_straight_line":       _wrap(remove_straight_line),

    # USE: char_quality="faded" , gray ink (not black) on clean white background
    "binarize":                   _wrap(binarize),

}


def normalize_tool_name(name: str) -> str:
    """Canonical tool-name normalisation"""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def get_tool(name: str) -> Callable:
    """Look up a tool by name (case-insensitive, normalised)."""
    key = normalize_tool_name(name)
    if key not in TOOL_REGISTRY:
        raise KeyError(
            f"Unknown tool '{name}'. Available: {sorted(TOOL_REGISTRY.keys())}"
        )
    return TOOL_REGISTRY[key]


def list_tools() -> list:
    """Return sorted list of registered tool names."""
    return sorted(TOOL_REGISTRY.keys())
