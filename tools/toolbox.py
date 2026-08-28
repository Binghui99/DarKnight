"""
tools/toolbox.py
----------------
Image preprocessing toolbox.

Each tool follows the signature:
    tool(image: np.ndarray, **params) -> np.ndarray

Base64 wrapping is handled by tools/registry.py.
color_extract is handled separately in tools/color_lines_removal.py (base64-native).
"""

import cv2
import numpy as np


def _to_gray(image: np.ndarray, copy: bool = True) -> np.ndarray:
    """
    Convert a BGR image to grayscale, or pass through an already-grayscale
    image. When *copy* is True (the default) a 2-D input is copied so the
    caller's array is never mutated through the returned reference; when
    False the original array is returned as-is (matching call sites that
    intentionally do not copy).
    """
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy() if copy else image


def sharpen(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """Sharpen character edges using unsharp masking.

    USE WHEN:
      - distortion = "wave" or "blur" — wave/blur softens edges; sharpen
        restores contrast and crispness before OCR.
      - Characters appear slightly blurry or edges are not crisp.
      - background = "gradient" with soft character strokes.

    DO NOT USE WHEN:
      - Image is already clean with sharp edges — unnecessary and may introduce
        ringing artefacts.
      - Noise is present — sharpen amplifies noise; apply denoising first.

    How it works: Gaussian blur is subtracted from the original (unsharp
    mask). The result is: output = (1+strength)*original - strength*blurred.

    Parameters
    ----------
    strength : Sharpening intensity. Default 1.5. Increase for heavier blur.
    """
    gray = _to_gray(image)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1 + strength, blurred, -strength, 0)


def remove_small_noise(image: np.ndarray, min_area: int = 15) -> np.ndarray:
    """Remove very small isolated noise pixels/clusters after color extraction.

    USE WHEN:
      - Image has tiny scattered dark pixel clusters or micro-fragments
        remaining after color_extract (e.g. residual noise dots that are
        clearly too small to be part of any character stroke).
      - Best applied right after color_extract on Chinese or complex CAPTCHAs.

    DO NOT USE WHEN:
      - Image is clean with no residual noise.
      - Characters have very thin strokes — small min_area may erase them.

    How it works: finds all connected dark components, erases any whose
    pixel area is below min_area. Characters are large (hundreds of pixels);
    noise dots are tiny (< 15 pixels). This difference cleanly separates them.

    Parameters
    ----------
    min_area : Components smaller than this (in pixels) are removed.
               Default 15. Lower = more aggressive removal.
    """
    gray = _to_gray(image)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    output = np.ones_like(gray) * 255
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            output[labels == label] = 0
    return output


def remove_straight_line(
    image: np.ndarray,
    line_intensity_threshold: int   = 55,
    fit_col_start: int              = 80,
    band_half_width: int            = 8,
    inpaint_radius: int             = 5,
    bg_kernel_size: int             = 15,
    diff_amplify: float             = 4.0,
    bilateral_d: int                = 5,
    bilateral_sigma: float          = 25.0,
    unsharp_strength: float         = 1.5,
) -> np.ndarray:
    """Remove a single diagonal/near-horizontal dark noise line, then enhance contrast.

    USE WHEN:
      - noise = "strikethrough" — exactly ONE thin dark (near-black) diagonal or
        near-horizontal line crosses ALL characters from edge to edge.
      - The line is visibly THINNER than character strokes — a narrow scratch or
        wire-like stroke, not a bold band.
      - The line pixels are very dark (near-black, intensity 0–55).
      - Works best when char_quality = "hollow": character outlines are coloured
        (not near-black), so only the line registers in the dark-pixel mask —
        clean separation with minimal character contamination.
      - May also work for char_quality = "normal" (solid chars) when there is
        enough intensity gap between line and character strokes.

    DO NOT USE WHEN:
      - noise = "arc" — thick curved band or multiple wavy lines.
      - noise = "coloured_lines" — coloured crossing lines → use color_extract instead.
      - Multiple crossing lines exist.
      - Image is clean (no crossing line) — the background-subtraction step will
        over-amplify noise and distort clean images.

    How it works:
      1. Fit a linear path through the darkest pixels in the right region to locate
         the line precisely, then build a colour-depth-aware mask around it.
      2. TELEA inpainting fills the masked region from surrounding background context.
      3. Morphological closing estimates the smooth background (image without strokes).
      4. Background subtraction reveals text strokes as bright residual signal.
      5. Amplify → bilateral filter → contrast stretch → unsharp mask → invert
         to produce dark text on a white background ready for OCR.

    Parameters
    ----------
    line_intensity_threshold : Pixels darker than this are candidate line pixels.
                               Default 55. Increase for lighter-coloured lines.
    fit_col_start            : Column from which to begin the line-path fit. Default 80.
                               Lower if the line enters the image further left.
    band_half_width          : ±px band around the predicted line centre. Default 8.
    inpaint_radius           : TELEA inpaint neighbourhood radius. Default 5.
    bg_kernel_size           : Closing kernel size for background estimation. Default 15.
    diff_amplify             : Multiplier on the background-subtracted signal. Default 4.0.
    bilateral_d              : Bilateral filter diameter. Default 5.
    bilateral_sigma          : Bilateral colour + space sigma. Default 25.
    unsharp_strength         : Sharpening weight in the unsharp mask. Default 1.5.
    """
    gray = _to_gray(image)
    h, w = gray.shape

    # Step 1: fit the line path from the clean right region
    _, binary_line = cv2.threshold(gray, line_intensity_threshold, 255, cv2.THRESH_BINARY_INV)
    line_samples = []
    for col in range(min(fit_col_start, w - 1), w):
        dark_rows = np.where(binary_line[:, col] == 255)[0]
        if len(dark_rows) > 0:
            line_samples.append((col, float(np.mean(dark_rows))))

    if len(line_samples) < 5:
        # Cannot detect line — return image unchanged (grayscale, dark-on-white)
        return gray

    cols_fit = np.array([p[0] for p in line_samples])
    rows_fit = np.array([p[1] for p in line_samples])
    slope, intercept = np.polyfit(cols_fit, rows_fit, 1)

    # Step 2: colour-depth-aware mask
    mask = np.zeros((h, w), dtype=np.uint8)
    for col in range(w):
        center  = slope * col + intercept
        r_start = max(0, int(center) - band_half_width)
        r_end   = min(h, int(center) + band_half_width + 1)
        for row in range(r_start, r_end):
            if gray[row, col] < line_intensity_threshold:
                mask[row, col] = 255
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)

    # Step 3: inpaint
    cleaned = cv2.inpaint(gray, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)

    # Step 4: background estimation via morphological closing
    bg_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_kernel_size, bg_kernel_size))
    background = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, bg_kernel)

    # Step 5: background subtraction → amplify → enhance → invert
    diff     = cv2.subtract(background, cleaned)
    diff_amp = np.clip(diff.astype(np.float32) * diff_amplify, 0, 255).astype(np.uint8)
    smooth   = cv2.bilateralFilter(diff_amp, d=bilateral_d,
                                   sigmaColor=bilateral_sigma, sigmaSpace=bilateral_sigma)
    lo, hi   = np.percentile(smooth, 1), np.percentile(smooth, 99)
    stretched = np.clip(
        (smooth.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255
    ).astype(np.uint8)
    blur      = cv2.GaussianBlur(stretched, (3, 3), 0.8)
    sharpened = cv2.addWeighted(stretched, unsharp_strength, blur, -(unsharp_strength - 1), 0)
    return cv2.bitwise_not(sharpened)


def binarize(image: np.ndarray) -> np.ndarray:
    """Convert gray-ink characters to pure black using Otsu's global threshold.

    USE WHEN:
      - char_quality = "faded" — characters appear as medium-gray ink (not pure
        black) on a clean white background. The font ink is visibly gray/faded
        rather than black; edges have an anti-aliased gradient to white.
      - Background must be clean white — this is a global threshold.

    DO NOT USE WHEN:
      - background != "clean" — uneven backgrounds cause global threshold to fail.
      - Characters are already pure black.
      - Before color_extract — destroys colour information needed by that tool.

    How it works: Otsu's method finds the optimal global threshold separating
    the bimodal gray histogram (dark character ink vs. white background).
    All character pixels (gray-82 → ~200) → pure black (0).
    All background pixels (200–255) → pure white (255).
    Eliminates anti-aliasing gradients and normalises ink colour in one step.
    """
    gray = _to_gray(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
