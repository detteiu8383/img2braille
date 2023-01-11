from pathlib import PurePath

import cv2
import numpy as np

from img2braille.binarization import (
    BINARIZATION_METHODS,
    BinarizationFunction,
    bayer_binarization,
)


def load_gray_img(img_path: str):
    """Load an image as a grayscale numpy array.

    Args:
        path (str): The path to the image.

    Returns:
        ndarray: The image as a numpy array.
    """
    path_str = str(PurePath(img_path))
    return cv2.cvtColor(cv2.imread(path_str), code=cv2.COLOR_BGR2GRAY)


def scale_img(img: cv2.Mat, width: int, line_height: float = 1.0):
    """Scale an image to a given width with maintaining the aspect ratio.

    Args:
        img (Mat): Image data to resize.
        width (int): Target width (pixel).
        line_height (float, optional): Line height of output environment.
        Defaults to 1.0.

    Returns:
        np.ndarray: scaled image
    """
    h, w = img.shape[:2]
    height = max(1, int(width * h / w / line_height))
    scaled_img = cv2.resize(img, (width, height))
    return scaled_img


def binarization(img: cv2.Mat, method: BinarizationFunction):
    uint8_img = img.astype(np.uint8)
    return method(uint8_img)


def padding_img(img: np.ndarray, fill_value=False):
    """Pad an image to be divisible by 4.

    Args:
        img (ndarray): Image data to pad.

    Returns:
        ndarray: Padded image
    """
    h, w = img.shape[:2]
    pad_height = 3 - ((h - 1) % 4)
    pad = np.full((pad_height, w), fill_value=fill_value, dtype=img.dtype)
    padded_img = np.vstack((img, pad))
    return padded_img


def split_2x4(bin_img: np.ndarray):
    """Split an image into 2x4 tiles.

    Args:
        img (ndarray): Image data to split.

    Returns:
        list[list[ndarray]]: 2D list of tiles
    """
    h, w = bin_img.shape[:2]
    h_count = h // 4
    w_count = w // 2
    return [np.hsplit(row, w_count) for row in np.vsplit(bin_img, h_count)]


def convert_2x4_to_braille_int(tile: np.ndarray[int, np.dtype[np.bool_]]):
    # fmt: off
    weight = np.array(
        [
            [1 << 0, 1 << 3],
            [1 << 1, 1 << 4],
            [1 << 2, 1 << 5],
            [1 << 6, 1 << 7]
        ],
        dtype=np.uint8
    )
    # fmt: on

    return np.sum(tile * weight).astype(int)


def int_to_braille(num: int, invert: bool = False):
    """Convert an integer to a braille character.

    Args:
        num (int): Integer to convert.
        invert (bool, optional): If True, invert the braille. Defaults to False.

    Returns:
        str: Braille character
    """
    if invert:
        num = ~num & 0b11111111
    return chr(0x2800 + num)


def img_to_brailles(
    img: cv2.Mat,
    width: int,
    invert: bool = False,
    method: str = "bayer",
    line_height: float = 1.0,
):
    """Convert an image to braille characters.

    Args:
        img (cv2.Mat): Image data to convert.
        width (int): Number of characters per line.
        invert (bool, optional): If True, invert the braille. Defaults to False.
        method (str, optional): Dithering method. Defaults to "bayer".
        line_height (float, optional): Line height of output environment.
        Defaults to 1.0.

    Returns:
        list[list[str]]: 2D list of braille characters
    """
    scaled_img = scale_img(img, width * 2, line_height)  # 1 braille = 2 pixel width

    bin_img = binarization(
        scaled_img, BINARIZATION_METHODS.get(method, bayer_binarization)
    )
    if invert:
        bin_img = ~bin_img

    padded_img = padding_img(bin_img)
    tiles = split_2x4(padded_img)

    brailles = [
        [int_to_braille(convert_2x4_to_braille_int(tile)) for tile in row]
        for row in tiles
    ]

    return brailles
