import math
from typing import Callable, TypeAlias

import numpy as np

DitherFunction: TypeAlias = Callable[
    [np.ndarray[int, np.dtype[np.uint8]]], np.ndarray[int, np.dtype[np.bool_]]
]


def bayer_dithering(
    img: np.ndarray[int, np.dtype[np.uint8]]
) -> np.ndarray[int, np.dtype[np.bool_]]:
    """Bayer dithering.

    Args:
        img (ndarray[int, dtype[uint8]]): Input image

    Returns:
        ndarray[int, dtype[bool_]]: If pixel is white, True
    """
    # fmt: off
    threshold_tile = np.array([
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5]
    ], dtype=np.uint8)
    # fmt: on

    threshold_tile = ((threshold_tile + 1) * 15).astype(np.uint8)
    h, w = img.shape[:2]
    threshold = np.tile(threshold_tile, (math.ceil(h / 4), math.ceil(w / 4)))
    threshold = threshold[:h, :w]

    return img > threshold


DITHERING_METHODS = {"bayer": bayer_dithering}
