"""
utils/naming.py
Small helpers for building filesystem, safe names/tags.
"""
import os


def sanitize_tag(s: str) -> str:
    return s.replace(os.sep, "_").replace("/", "_")
