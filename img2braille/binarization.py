import math
from typing import Callable, TypeAlias

import cv2
import numpy as np

BinarizationFunction: TypeAlias = Callable[
    [np.ndarray[int, np.dtype[np.uint8]]], np.ndarray[int, np.dtype[np.bool_]]
]


def bayer_binarization(
    img: np.ndarray[int, np.dtype[np.uint8]]
) -> np.ndarray[int, np.dtype[np.bool_]]:
    """Bayer binarization.

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


def floyd_steinberg_dithering(
    img: np.ndarray[int, np.dtype[np.uint8]]
) -> np.ndarray[int, np.dtype[np.bool_]]:
    """Floyd-Steinberg dithering.

    Args:
        img (ndarray[int, dtype[uint8]]): Input image

    Returns:
        ndarray[int, dtype[bool_]]: If pixel is white, True
    """
    h, w = img.shape[:2]
    float_img = img.astype(np.float16)
    float_img = np.pad(float_img, pad_width=((0, 1), (1, 1)), constant_values=0)
    for y in range(0, h):
        for x in range(1, w + 1):
            old_pixel = float_img[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            float_img[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            """
            [
                0 * 7
                3 5 1
            ] * 1/16
            """
            float_img[y, x + 1] += quant_error * 7 / 16
            float_img[y + 1, x - 1] += quant_error * 3 / 16
            float_img[y + 1, x] += quant_error * 5 / 16
            float_img[y + 1, x + 1] += quant_error * 1 / 16

    return float_img[:h, 1 : w + 1] > 127


def ja_ju_ni_dithering(
    img: np.ndarray[int, np.dtype[np.uint8]]
) -> np.ndarray[int, np.dtype[np.bool_]]:
    """Jarvis, Judice, and Ninke dithering.

    Args:
        img (ndarray[int, dtype[uint8]]): Input image

    Returns:
        ndarray[int, dtype[bool_]]: If pixel is white, True
    """
    h, w = img.shape[:2]
    float_img = img.astype(np.float16)
    float_img = np.pad(float_img, pad_width=((0, 2), (2, 2)), constant_values=0)
    for y in range(0, h):
        for x in range(2, w + 2):
            old_pixel = float_img[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            float_img[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            """
            [
                0 0 * 7 5
                3 5 7 5 3
                1 3 5 3 1
            ] * 1/48
            """
            err_1 = quant_error / 48
            err_3 = quant_error * 3 / 48
            err_5 = quant_error * 5 / 48
            err_7 = quant_error * 7 / 48

            float_img[y, x + 1] += err_7
            float_img[y, x + 2] += err_5

            float_img[y + 1, x - 2] += err_3
            float_img[y + 1, x - 1] += err_5
            float_img[y + 1, x] += err_7
            float_img[y + 1, x + 1] += err_5
            float_img[y + 1, x + 2] += err_3

            float_img[y + 2, x - 2] += err_1
            float_img[y + 2, x - 1] += err_3
            float_img[y + 2, x] += err_5
            float_img[y + 2, x + 1] += err_3
            float_img[y + 2, x + 2] += err_1

    return float_img[:h, 2 : w + 2] > 127


def atkinson_dithering(
    img: np.ndarray[int, np.dtype[np.uint8]]
) -> np.ndarray[int, np.dtype[np.bool_]]:
    """Atkinson dithering.

    Args:
        img (ndarray[int, dtype[uint8]]): Input image

    Returns:
        ndarray[int, dtype[bool_]]: If pixel is white, True
    """
    h, w = img.shape[:2]
    float_img = img.astype(np.float16)
    float_img = np.pad(float_img, pad_width=((0, 2), (2, 2)), constant_values=0)
    for y in range(0, h):
        for x in range(2, w + 2):
            old_pixel = float_img[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            float_img[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            """
            [
                0 0 * 1 1
                  1 1 1
                    1
            ] * 1/8
            """
            err = quant_error / 8

            float_img[y, x + 1] += err
            float_img[y, x + 2] += err

            float_img[y + 1, x - 1] += err
            float_img[y + 1, x] += err
            float_img[y + 1, x + 1] += err

            float_img[y + 2, x] += err

    return float_img[:h, 2 : w + 2] > 127


def sierra_dithering(
    img: np.ndarray[int, np.dtype[np.uint8]]
) -> np.ndarray[int, np.dtype[np.bool_]]:
    """Sierra dithering.

    Args:
        img (ndarray[int, dtype[uint8]]): Input image

    Returns:
        ndarray[int, dtype[bool_]]: If pixel is white, True
    """
    h, w = img.shape[:2]
    float_img = img.astype(np.float16)
    float_img = np.pad(float_img, pad_width=((0, 2), (2, 2)), constant_values=0)
    for y in range(0, h):
        for x in range(2, w + 2):
            old_pixel = float_img[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            float_img[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            """
            [
                0 0 * 5 3
                2 4 5 4 2
                  2 3 2
            ] * 1/32
            """
            err_2 = quant_error * 2 / 32
            err_3 = quant_error * 3 / 32
            err_4 = quant_error * 4 / 32
            err_5 = quant_error * 5 / 32

            float_img[y, x + 1] += err_5
            float_img[y, x + 2] += err_3

            float_img[y + 1, x - 2] += err_2
            float_img[y + 1, x - 1] += err_4
            float_img[y + 1, x] += err_5
            float_img[y + 1, x + 1] += err_4
            float_img[y + 1, x + 2] += err_2

            float_img[y + 2, x - 1] += err_2
            float_img[y + 2, x] += err_3
            float_img[y + 2, x + 1] += err_2

    return float_img[:h, 2 : w + 2] > 127


def sierra_lite_dithering(
    img: np.ndarray[int, np.dtype[np.uint8]]
) -> np.ndarray[int, np.dtype[np.bool_]]:
    """Sierra dithering.

    Args:
        img (ndarray[int, dtype[uint8]]): Input image

    Returns:
        ndarray[int, dtype[bool_]]: If pixel is white, True
    """
    h, w = img.shape[:2]
    float_img = img.astype(np.float16)
    float_img = np.pad(float_img, pad_width=((0, 1), (1, 1)), constant_values=0)
    for y in range(0, h):
        for x in range(1, w + 1):
            old_pixel = float_img[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            float_img[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            """
            [
                0 * 2
                1 1
            ] * 1/4
            """
            float_img[y, x + 1] += quant_error * 2 / 4

            float_img[y + 1, x - 1] += quant_error / 4
            float_img[y + 1, x] += quant_error / 4

    return float_img[:h, 1 : w + 1] > 127


def threshold_binarization(
    img: np.ndarray[int, np.dtype[np.uint8]]
) -> np.ndarray[int, np.dtype[np.bool_]]:
    """Threshold binarization.

    Args:
        img (ndarray[int, dtype[uint8]]): Input image

    Returns:
        ndarray[int, dtype[bool_]]: If pixel is white, True
    """
    return img > 127


def otsu_binarization(
    img: np.ndarray[int, np.dtype[np.uint8]]
) -> np.ndarray[int, np.dtype[np.bool_]]:
    """Otsu binarization.

    Args:
        img (ndarray[int, dtype[uint8]]): Input image

    Returns:
        ndarray[int, dtype[bool_]]: If pixel is white, True
    """
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th > 127


BINARIZATION_METHODS = {
    "bayer": bayer_binarization,
    "threshold": threshold_binarization,
    "otsu": otsu_binarization,
    "floyd-steinberg": floyd_steinberg_dithering,
    "jajuni": ja_ju_ni_dithering,
    "atkinson": atkinson_dithering,
    "sierra": sierra_dithering,
    "sierra-lite": sierra_lite_dithering,
}
