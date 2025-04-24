import time
import typing

import cv2
import numpy as np
import torch

import research_utilities.resampling as _resampling


def save_img(arr: torch.Tensor, file_path: str) -> None:
    arr = arr.squeeze().detach().cpu().numpy()
    arr = arr.transpose(1, 2, 0)
    arr = (arr * 255.0).astype(np.uint8)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, arr)


class Benchmarker(list):
    def __getitem__(self, n_times: int) -> typing.Callable:
        def run_benchmark(func, *args, **kwargs) -> float:
            elapsed_time = 0.0

            for _ in range(n_times):
                start_time = time.perf_counter()
                func(*args, **kwargs)
                elapsed_time += time.perf_counter() - start_time

            avg_elapsed_time = elapsed_time / n_times

            print(f'Avg. elapsed time: {avg_elapsed_time:.6f} [s] ({n_times} times)')

            return avg_elapsed_time

        return run_benchmark


benchmark = Benchmarker()

SCALE = 0.1

NUM_BENCHMARK = 1

# img_np = cv2.imread('Lenna.png')
img_np = cv2.imread('ShineiArakawa_cropped_small.png')
img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
img_np = img_np.transpose(2, 0, 1)
img_np = img_np.astype(np.float32) / 255.0

img = torch.from_numpy(img_np).unsqueeze(0).cuda()
save_img(img[0], 'img.png')

####################################################################################################
# Save
####################################################################################################
benchmark[NUM_BENCHMARK](_resampling.resample, img, SCALE, _resampling.InterpMethod.NEAREST)
out_img = _resampling.resample(img, SCALE, _resampling.InterpMethod.NEAREST)
save_img(out_img[0], 'out_img_nearest.png')

benchmark[NUM_BENCHMARK](_resampling.resample, img, SCALE, _resampling.InterpMethod.BILINEAR)
out_img = _resampling.resample(img, SCALE, _resampling.InterpMethod.BILINEAR)
save_img(out_img[0], 'out_img_bilinear.png')

benchmark[NUM_BENCHMARK](_resampling.resample, img, SCALE, _resampling.InterpMethod.BICUBIC)
out_img = _resampling.resample(img, SCALE, _resampling.InterpMethod.BICUBIC)
save_img(out_img[0], 'out_img_bicubic.png')

benchmark[NUM_BENCHMARK](_resampling.resample, img, SCALE, _resampling.InterpMethod.LANCZOS4)
out_img = _resampling.resample(img, SCALE, _resampling.InterpMethod.LANCZOS4)
save_img(out_img[0], 'out_img_lanczos4.png')
