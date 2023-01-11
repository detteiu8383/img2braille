import argparse

from . import binarization, convert


def command(
    img_path: str,
    width: int,
    invert: bool = False,
    method: str = "bayer",
    line_height: float = 1.0,
):
    invert = invert or False
    method = method or "bayer"
    line_height = line_height or 1.0

    gray_img = convert.load_gray_img(img_path)
    brailles = convert.img_to_brailles(gray_img, width, invert, method, line_height)

    return "\n".join("".join(braille_row) for braille_row in brailles)


def main():
    parser = argparse.ArgumentParser(
        prog="img2braille", description="Convert an image to braille characters."
    )

    parser.add_argument("img_path", type=str, help="Path to the image file.")
    parser.add_argument("width", type=int, help="Number of characters per line.")

    parser.add_argument(
        "-i", "--invert", action="store_true", help="Invert the braille."
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=list(binarization.BINARIZATION_METHODS.keys()),
        help="Method to use for binarization.",
    )
    parser.add_argument(
        "-l", "--line-height", type=float, help="Line height of output environment."
    )

    args = parser.parse_args()
    command_result = command(
        img_path=args.img_path,
        width=args.width,
        invert=args.invert,
        method=args.method,
        line_height=args.line_height,
    )
    print(command_result)
