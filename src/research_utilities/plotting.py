import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import research_utilities.common as _common


def add_title(
    img: np.ndarray,
    text: str,
    text_color: list[int],
    background_color: list[int],
    text_size: int = 36,
    is_BGR: bool = True,
) -> np.ndarray:
    """Add a title to an image.
    This method uses 'Roboto-Regular' font to draw the title on the image.

    Parameters
    ----------
    img : np.ndarray
        The image to which the title will be added.
        The image must be 3D (H, W, C) and have 3 or 4 channels.
    text : str
        The title text to be added to the image.
    text_color : list[int]
        The color of the text in (R, G, B) or (R, G, B, A) format.
    background_color : list[int]
        The color of the background in (R, G, B) or (R, G, B, A) format.
    text_size : int, optional
        The size of the text, by default 36.
        This is used to calculate the height of the title.
        The width is calculated based on the text length.
    is_BGR : bool, optional
        If True, the input image is assumed to be in BGR format (OpenCV default).
        If False, the input image is assumed to be in RGB format (PIL default).
        By default True.

    Returns
    -------
    np.ndarray
        The image with the title added.
    """

    assert img.ndim == 3, f"Image must be 3D (H, W, C), got {img.ndim}D"
    assert img.shape[2] in [3, 4], f"Image must have 3 or 4 channels, got {img.shape[2]} channels"
    assert img.shape[2] == len(text_color), f"Text color must have the same number of channels as the image, got {len(text_color)} for {img.shape[2]}"
    assert img.shape[2] == len(background_color), f"Background color must have the same number of channels as the image, got {len(background_color)} for {img.shape[2]}"

    # Convert the image to PIL format
    if is_BGR:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if img.shape[2] == 3 else cv2.COLOR_BGRA2RGBA)

    img_pil = Image.fromarray(img)

    # Create a new image with the same size and a white background
    new_img = Image.new(
        "RGBA" if img.shape[2] == 4 else "RGB",
        img_pil.size,
        tuple(background_color)
    )

    # Create a draw object
    draw = ImageDraw.Draw(new_img)

    # Load a default font
    font = ImageFont.truetype(str(_common.GlobalSettings.TTF_ROBOTO_PATH), text_size)

    # Calculate text size
    text_height = text_size
    text_width = draw.textlength(text, font=font)

    padding = int(text_size * 0.2)
    title_height = text_height + padding * 2

    # Create a new image large enough for the title + original image
    total_height = img_pil.height + title_height
    canvas = Image.new(
        "RGBA" if img.shape[2] == 4 else "RGB",
        (img_pil.width, total_height),
        tuple(background_color)
    )

    # Draw title on the top
    draw_canvas = ImageDraw.Draw(canvas)
    text_position = ((img_pil.width - text_width) // 2, padding)
    draw_canvas.text(text_position, text, font=font, fill=tuple(text_color))

    # Paste the original image below the title
    canvas.paste(img_pil, (0, title_height))

    # Convert back to numpy array
    out_img = np.array(canvas)

    if is_BGR:
        # Convert back to BGR format
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR if img.shape[2] == 3 else cv2.COLOR_RGBA2BGRA)

    return out_img
