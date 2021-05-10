import numpy as np
from itertools import product

def img_filter(img, locations, radius=3, level=10):
    '''
    this algorithm select the most frequent intensity group in the kernel,
    and average r,b,g in the pixel group, it makes the original image like oil painting
    :param img: 3d image array height, width, channel (b,g,r)
    :param locations: list of tuple of face locations (top, right, bottom, left)
    :param radius:
    :param level:
    :return:
    '''
    des_img = np.copy(img)
    for location in locations:
        top, right, bottom, left = location
        if top+radius >= bottom-radius or left+radius>=right-radius:
            continue
        for i, j in product(range(top+radius, bottom-radius), range(left+radius, right-radius)):

            level_counter = np.zeros(level, dtype=np.uint32)
            b_level = np.zeros(level, dtype=np.uint32)
            r_level = np.zeros(level, dtype=np.uint32)
            g_level = np.zeros(level, dtype=np.uint32)

            for m, n in product(range(-radius, radius), repeat=2):
                b, g, r = img[i + m, j + n]
                avg = (b+g+r) / 3.
                pixlv = int(avg / (256 / level))
                level_counter[pixlv] += 1
                b_level[pixlv] += b
                g_level[pixlv] += g
                r_level[pixlv] += r

            most_level_Idx = np.argmax(level_counter)
            most_level_count = level_counter[most_level_Idx]

            des_img[i, j, 0] = b_level[most_level_Idx] // most_level_count
            des_img[i, j, 1] = g_level[most_level_Idx] // most_level_count
            des_img[i, j, 2] = r_level[most_level_Idx] // most_level_count

    return des_img