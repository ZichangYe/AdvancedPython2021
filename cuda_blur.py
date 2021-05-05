from numba import cuda
import numpy as np


@cuda.jit
def img_blur_cuda(img, des_img, k, radius):
    '''
    numba cuda version of blurring algorithm
    '''
    i, j = cuda.grid(2)

    rows, columns, channel = img.shape
    if i >= rows or j >= columns:
        return

    ra = rows - radius
    ca = columns - radius
    if i < radius or j < radius or i >= ra or j >= ca:
        des_img[i, j, 0] = img[i, j, 0]
        des_img[i, j, 1] = img[i, j, 1]
        des_img[i, j, 2] = img[i, j, 2]
        return

    r = 0
    g = 0
    b = 0
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            i_x = i + x
            j_y = j + y
            r += img[i_x, j_y, 0] * k
            g += img[i_x, j_y, 1] * k
            b += img[i_x, j_y, 2] * k
    des_img[i, j, 0] = r
    des_img[i, j, 1] = g
    des_img[i, j, 2] = b