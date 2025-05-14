import cv2
import torch

import research_utilities.video as video


def write_video():
    images = [
        torch.randn(3, 128, 128) for _ in range(323)
    ]

    video_path = f'./test_video.mp4'

    video.save_images_to_video(
        images,
        video_path,
        fps=30,
        hevc=False,
        bitrate='40M',
        verbose=False,
        img_size=None,
        interp=cv2.INTER_NEAREST,
        channel_last=False,
    )


if __name__ == "__main__":
    write_video()
