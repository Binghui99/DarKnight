"""
tools/color_lines_removal.py
Color lines / noise removal tool.

Tool signature:
    color_lines_removal(image_b64: str, threshold: int = 200) -> str
"""
import base64
import numpy as np


def color_lines_removal(image_b64: str, threshold: int = 200, **kwargs) -> str:
    """
    Remove coloured pixels (noise, background lines) from a CAPTCHA image.

    Strategy:
    1. Find every unique colour in the image.
    2. Discard colours whose mean brightness > threshold  (near-white → keep).
    3. For remaining coloured pixels, remove colours that span more than 2/3 of the image width.
    4. Produce a white-background image with the kept pixels highlighted.

    Parameters
    ----------
    image_b64 : str
        Data-URI or raw base64 PNG/JPEG string.
    threshold : int
        Mean RGB brightness above which a colour is considered background.

    Returns
    -------
    str  – data-URI base64 PNG of the processed image.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for color_lines_removal.")

    if "," in image_b64:
        header, encoded = image_b64.split(",", 1)
    else:
        header  = "data:image/png"
        encoded = image_b64

    img_data = base64.b64decode(encoded)
    nparr    = np.frombuffer(img_data, np.uint8)
    img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("colourful_lines_removal: could not decode image.")

    h, w, _ = img.shape

    pixels        = img.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    final_mask    = np.zeros((h, w), dtype=np.uint8)

    for color in unique_colors:
        if np.mean(color) > threshold:
            continue 

        mask   = cv2.inRange(img, color, color)
        coords = cv2.findNonZero(mask)
        if coords is None:
            continue

        x, y, box_w, box_h = cv2.boundingRect(coords)

        if box_w < (2 / 3 * w):
            final_mask = cv2.bitwise_or(final_mask, mask)

    result = cv2.bitwise_not(final_mask)

    #  Encode back to base64 
    _, buffer = cv2.imencode(".png", result)
    b64_out   = base64.b64encode(buffer).decode("utf-8")
    return header + "," + b64_out