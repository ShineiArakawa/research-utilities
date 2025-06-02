import cv2
import numpy as np
import torch

import research_utilities as rutil


def main():
    img_path = rutil.get_demo_image()
    contour = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    print(f'contour.shape: {contour.shape}')
    print(f'contour.dtype: {contour.dtype}')
    print(f'contour.min(): {contour.min()}, contour.max(): {contour.max()}')

    cmap = rutil.apply_color_map(contour, 'viridis')
    print(f'cmap.shape: {cmap.shape}')
    print(f'cmap.dtype: {cmap.dtype}')
    print(f'cmap.min(): {cmap.min()}, cmap.max(): {cmap.max()}')

    cv2.imshow('Contour', cv2.cvtColor(cmap, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
