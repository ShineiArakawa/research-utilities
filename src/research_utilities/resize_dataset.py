

import argparse
import os
import pathlib
import re

from PIL import Image

import research_utilities.common as _common
import research_utilities.multi_processing as _mp


def resize_impl_wrapped(args) -> None:
    resize_impl(*args)


def resize_impl(
    file_path: pathlib.Path,
    save_path: pathlib.Path,
    crop_width: int,
    crop_height: int
) -> None:
    image = Image.open(file_path)

    img_width, img_height = image.size

    # Adjust the aspect ratio by cropping
    img_aspect_ratio = img_width / img_height
    target_aspect_ratio = crop_width / crop_height

    if img_aspect_ratio > target_aspect_ratio:
        new_width = int(img_height * target_aspect_ratio)
        new_height = img_height
    else:
        new_width = img_width
        new_height = int(img_width / target_aspect_ratio)

    # Center crop the image
    left = (img_width - new_width) / 2.0
    top = (img_height - new_height) / 2.0
    right = (img_width + new_width) / 2.0
    bottom = (img_height + new_height) / 2.0

    image = image.crop((left, top, right, bottom))

    # Resize the image to the target size
    image = image.resize((crop_width, crop_height), Image.Resampling.LANCZOS)

    # Save the resized image to the output directory
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize images in a directory to a target size.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "path_root_dir",
        type=pathlib.Path,
        help="Path to the root directory containing images to be resized."
    )

    parser.add_argument(
        "path_out_dir",
        type=pathlib.Path,
        help="Path to the output directory where resized images will be saved."
    )

    parser.add_argument(
        "target_size",
        type=str,
        help="Target size for resizing images. You can specify the size as 'width,height' or just specify 'size' if you want to resize the image to a square size."
    )

    parser.add_argument(
        "--n-prallels",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel processes to use for resizing images."
    )

    return parser.parse_args()


def parse_size(size_str: str) -> tuple[int, int]:
    """
    Parse the size string and return a tuple of integers.
    The size string can be in the format 'width,height' or just 'size'.

    Parameters
    ----------
    size_str : str
        The size string to be parsed. It can be in the format 'width,height' or just 'size'.

    Returns
    -------
    tuple[int, int]
        A tuple containing the width and height as integers.
    """

    size_str = size_str.strip()

    if re.match(r"^\d+,\d+$", size_str):
        tokens = size_str.split(",")
        width = int(tokens[0].strip())
        height = int(tokens[1].strip())
    else:
        width = height = int(size_str)

    return width, height


def main() -> None:
    logger = _common.get_logger()

    args = parse_args()

    # Parse the target size string to get width and height
    size = parse_size(args.target_size)

    # ------------------------------------------------------------
    # Glob the image files in the root directory recursively
    path_root_dir: pathlib.Path = args.path_root_dir.resolve()
    image_files = list(path_root_dir.rglob("*"))

    # ------------------------------------------------------------
    # Filter out non-image files

    # Find readable and writable image extensions by PIL
    ioable_exts: list[str] = []
    for ext, plugin in Image.registered_extensions().items():
        if plugin in Image.OPEN and plugin in Image.SAVE:
            ioable_exts.append(ext)

    # Filter the image files
    image_files = [
        file_path for file_path in image_files
        if file_path.is_file() and file_path.suffix.lower() in ioable_exts
    ]

    logger.info(f"Found {len(image_files)} image files in {path_root_dir}")

    # ------------------------------------------------------------
    # Arguments for resizing
    path_out_dir: pathlib.Path = args.path_out_dir.resolve()

    mp_args = []
    for file_path in image_files:
        save_path = path_out_dir / file_path.resolve().relative_to(path_root_dir)

        mp_args.append([
            file_path,
            save_path,
            size[0],
            size[1]
        ])

    # ------------------------------------------------------------
    # Run
    logger.info(f"Resizing {len(mp_args)} images to {size[0]}x{size[1]}...")

    _mp.launch_multi_process(
        func=resize_impl_wrapped,
        args=mp_args,
        n_processes=args.n_prallels,
        desc="Resizing images ... ",
        is_unordered=True,
    )

    logger.info("Resizing completed.")


if __name__ == "__main__":
    main()
