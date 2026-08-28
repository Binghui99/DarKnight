"""
utils/dataset_loader.py
Loads captcha datasets from JSON, PDF, or image files.

Each record is normalised to:
{
  "id":         str,
  "label":      str | None,   # ground truth (may be absent)
  "b64_string": str,           # base64-encoded PNG/JPEG with data-URI prefix
  "image_path": str | None,
  "source":     str,           # path to the source file
}
"""
import base64
import glob
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any

try:
    from PIL import Image
    import io
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def _img_to_b64(path: str) -> str:
    """Read an image file and return a data-URI base64 string."""
    with open(path, "rb") as fh:
        raw = fh.read()
    ext = Path(path).suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    b64  = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"


def _ensure_data_uri(b64_string: str, default_mime: str = "image/png") -> str:
    """Ensure b64_string has a data-URI header."""
    if b64_string.startswith("data:"):
        return b64_string
    return f"data:{default_mime};base64,{b64_string}"


def load_json_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON file whose top-level value is a list of captcha records.
    Expected fields (at least one of): b64_string, image_path.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        # Handle {"data": [...]} wrappers
        for key in ("data", "samples", "records", "captchas"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            # Single record
            data = [data]

    records = []
    for i, item in enumerate(data):
        b64 = item.get("b64_string", "")
        img_path = item.get("image_path", None)

        if not b64 and img_path and os.path.exists(img_path):
            b64 = _img_to_b64(img_path)

        if b64:
            b64 = _ensure_data_uri(b64)

        records.append({
            "id":         str(item.get("id", i)),
            "label":      item.get("label", None),
            "b64_string": b64,
            "image_path": img_path,
            "source":     path,
        })
    return records


def load_pdf_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Extract images from a PDF and treat each page image as a captcha.
    Requires pdf2image (poppler) or PyMuPDF.
    """
    records = []

    # Try PyMuPDF first
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        for page_num, page in enumerate(doc):
            mat   = fitz.Matrix(2.0, 2.0)   # 2× zoom for clarity
            pix   = page.get_pixmap(matrix=mat)
            raw   = pix.tobytes("png")
            b64   = "data:image/png;base64," + base64.b64encode(raw).decode()
            records.append({
                "id":         f"page_{page_num}",
                "label":      None,
                "b64_string": b64,
                "image_path": None,
                "source":     path,
            })
        doc.close()
        return records
    except ImportError:
        pass

    # Fallback: pdf2image
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(path, dpi=150)
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            records.append({
                "id":         f"page_{i}",
                "label":      None,
                "b64_string": b64,
                "image_path": None,
                "source":     path,
            })
        return records
    except ImportError:
        pass

    raise ImportError(
        "PDF loading requires either PyMuPDF (`pip install pymupdf`) "
        "or pdf2image (`pip install pdf2image`)."
    )


def load_image_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a single image file as a one-record dataset."""
    b64 = _img_to_b64(path)
    return [{
        "id":         Path(path).stem,
        "label":      None,
        "b64_string": b64,
        "image_path": path,
        "source":     path,
    }]


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Auto-detect format and load the dataset at *path*."""
    ext = Path(path).suffix.lower()
    if ext == ".json":
        return load_json_dataset(path)
    elif ext == ".pdf":
        return load_pdf_dataset(path)
    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".gif"):
        return load_image_dataset(path)
    else:
        raise ValueError(f"Unsupported dataset format: {ext}")


def discover_datasets(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Walk *directory* and return paths to all dataset files.
    extensions defaults to ['.json', '.pdf', '.jpg', '.jpeg', '.png'].
    """
    if extensions is None:
        extensions = [".json", ".pdf", ".jpg", ".jpeg", ".png"]

    found = []
    for ext in extensions:
        found.extend(glob.glob(os.path.join(directory, f"**/*{ext}"), recursive=True))
    return sorted(set(found))