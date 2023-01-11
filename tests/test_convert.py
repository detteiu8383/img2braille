import numpy as np

from img2braille.convert import convert_2x4_to_braille_int, int_to_braille


def tile_to_braille(tile: list[list[int]]):
    ndarray = np.array(tile, dtype=np.bool_)
    return int_to_braille(convert_2x4_to_braille_int(ndarray))


def test_tile_to_braille():
    # fmt: off
    tile_0 = [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0]
    ]
    # fmt: on
    assert tile_to_braille(tile_0) == "⠀"

    # fmt: off
    tile_1 = [
        [1, 0],
        [0, 0],
        [0, 0],
        [0, 0]
    ]
    # fmt: on
    assert tile_to_braille(tile_1) == "⠁"

    # fmt: off
    tile_2 = [
        [0, 1],
        [0, 0],
        [0, 0],
        [0, 0]
    ]
    # fmt: on
    assert tile_to_braille(tile_2) == "⠈"

    # fmt: off
    tile_3 = [
        [0, 0],
        [1, 0],
        [0, 0],
        [0, 0]
    ]
    # fmt: on
    assert tile_to_braille(tile_3) == "⠂"

    # fmt: off
    tile_4 = [
        [0, 0],
        [0, 1],
        [0, 0],
        [0, 0]
    ]
    # fmt: on
    assert tile_to_braille(tile_4) == "⠐"

    # fmt: off
    tile_5 = [
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 0]
    ]
    # fmt: on
    assert tile_to_braille(tile_5) == "⠄"

    # fmt: off
    tile_6 = [
        [0, 0],
        [0, 0],
        [0, 1],
        [0, 0]
    ]
    # fmt: on
    assert tile_to_braille(tile_6) == "⠠"

    # fmt: off
    tile_7 = [
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 0]
    ]
    # fmt: on
    assert tile_to_braille(tile_7) == "⡀"

    # fmt: off
    tile_8 = [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 1]
    ]
    # fmt: on
    assert tile_to_braille(tile_8) == "⢀"
