"""Video processing utilities.
"""

import pathlib
import typing

import cv2
import imageio
import numpy as np
import torch
import tqdm


def save_images_to_video(
    # autopep8: off
    images       : list[np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor                    ,
    video_path   :                                          str | pathlib.Path                    ,
    fps          :                                                  int | None =                24,
    duration     :                                                float | None =              None,
    hevc         :                                                        bool =             False,
    bitrate      :                                                         str =             '40M',
    verbose      :                                                        bool =             False,
    img_size     :                               typing.Tuple[int, int] | None =              None,
    interp       :                                                         int = cv2.INTER_NEAREST,
    channel_last :                                                        bool =             False,
    # autopep8: on
) -> None:
    """Save images to video.

    Parameters
    ----------
    images: list[np.ndarray | torch.Tensor] | torch.Tensor
        Images to save. If a list, each image must be a numpy array or torch tensor.
        If a numpy array or torch tensor, it must be 4D array/tensor.
        The pixel values must be in the range[0, 1].
    video_path: str | pathlib.Path
        Path to save the video.
    fps: int | None, optional
        Frames per second, by default 24
        If None, duration must be provided.
    duration: float | None, optional
        Duration of the video in seconds, by default None
        If None, fps must be provided.
    hevc: bool, optional
        Use High Efficiency Video Coding(HEVC)(H.265) instead of H.264, by default False
    bitrate: str, optional
        Bitrate of the video, by default '40M'
    verbose: bool, optional
        If True, print progress, by default False
    img_size: typing.Tuple[int, int] | None, optional
        Size of the image, by default None
        If None, the size of the first image will be used.
    interp: int, optional
        Interpolation method, by default cv2.INTER_NEAREST
        See 'https://docs.opencv.org/4.10.0/da/d54/group__imgproc__transform.html' for more details.
    channel_last: bool, optional
        Whether the image is in channel last format(HWC) or channel first format(CHW), by default False
    """

    assert len(images) > 0, 'No images to save.'
    assert isinstance(images, (list, np.ndarray, torch.Tensor)), 'Images must be a list or numpy array or torch tensor.'
    if isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim != 4:
        raise ValueError('Images must be 4D (N, C, H, W) if you pass a numpy array or torch tensor.')

    # Get image size
    if img_size is None:
        if isinstance(images, list):
            if channel_last:
                img_size = (images[0].shape[1], images[0].shape[0])
            else:
                img_size = (images[0].shape[2], images[0].shape[1])
        elif isinstance(images, torch.Tensor):
            if channel_last:
                img_size = (images.shape[2], images.shape[1])
            else:
                img_size = (images.shape[3], images.shape[2])

    # Resize images if necessary
    imgs_to_encode: list[np.ndarray] = []
    for i_img in range(len(images)):
        img = images[i_img]

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        assert img.ndim == 3, 'Image must be 3D'

        if not channel_last:
            # CHW to HWC
            img = img.transpose(1, 2, 0)

        if img.shape[0] != img_size[1] or img.shape[1] != img_size[0]:
            # Resize image
            img = cv2.resize(img, img_size, interpolation=interp)

        assert img.shape[2] == 3, f'Image must have 3 channels, but got {img.shape[2]} at {i_img}th image. Check the channel of the image. Are you sure the image is channel last?'
        imgs_to_encode.append(img)

    # Calculate fps
    if fps is None:
        if duration is None:
            raise ValueError('Either fps or duration must be provided.')

        fps = len(images) / float(duration)
    elif duration is not None:
        print('Warning: fps and duration are both provided. Using fps.')

    # Create directory if it doesn't exist
    video_path = pathlib.Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Get writer
    if hevc:
        codec = 'libx265'
        pixel_format = 'yuv420p'  # ['yuv420p', 'yuv422p10le', 'yuv444p'], maybe x265 supports yuv420p only
    else:
        codec = 'libx264'
        pixel_format = 'yuv420p'  # ['yuv420p', 'yuv422p', 'yuv444p', 'nv12'], maybe x264 supports yuv420p only

    writer = imageio.get_writer(
        str(video_path),
        format='ffmpeg',
        mode='I',
        fps=fps,
        codec=codec,
        bitrate=bitrate,
        pixelformat=pixel_format,
        output_params=[
            # Encoding speed (slower = better quality and smaller file)
            '-preset', 'veryslow',
            # Use full color range (0-255)
            '-color_range', '2',
            # MP4 optimizations for fast start and correct color box
            '-movflags', '+faststart+write_colr',
            # Set pixel format
            '-pix_fmt', pixel_format,
            # Suppress banner and warnings
            '-hide_banner',
            '-loglevel', 'error',
        ]
    )

    for frame in tqdm.tqdm(
        imgs_to_encode,
        desc='Creating video ... ',
        disable=not verbose,
        total=len(imgs_to_encode),
        unit='frame'
    ):
        frame = frame.clip(0.0, 1.0)
        frame = (frame * 255.0).astype(np.uint8)
        writer.append_data(frame)

    writer.close()

    if verbose:
        print(f'Video saved to {video_path}')
