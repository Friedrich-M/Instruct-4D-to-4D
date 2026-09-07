"""Image-space helpers for visualisation."""

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def colorize_depth(
    depth: np.ndarray,
    minmax: Optional[Sequence[float]] = None,
    cmap: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, list]:
    """Turn a depth map into an 8-bit colour image.

    Args:
        depth: ``(H, W)`` depth values.  NaNs are treated as zero.
        minmax: ``(min, max)`` range to normalise against.  When omitted the
            range is taken from the map itself, ignoring non-positive depths so
            an empty background does not dominate the scale.
        cmap: An OpenCV colormap constant.

    Returns:
        ``(image, [min, max])`` where ``image`` is ``(H, W, 3)`` uint8 BGR.
    """
    x = np.nan_to_num(depth)
    if minmax is None:
        lo, hi = np.min(x[x > 0]), np.max(x)
    else:
        lo, hi = minmax
    # Clipping matters for a near-constant depth map, where the normalisation
    # would otherwise blow up and the cast to uint8 would be undefined.
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return cv2.applyColorMap((255 * x).astype(np.uint8), cmap), [lo, hi]
