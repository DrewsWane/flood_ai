'''
СКРИПТ ДЛЯ ЗАПУСКА ВИЗУАЛИЗАЦИИ ИЗОБРАЖЕНИЯ И МАСКИ
УКАЖИТЕ ПУТЬ ДО ФАЙЛОВ ИЗОБРАЖЕНИЯ И МАСКИ В КОНЦЕ СТРАНИЦЫ И ЗАПУСТИТЕ СКРИПТ
'''

import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


PALLETE = [
        [0, 0, 0],
        [0, 0, 255],
        ]


def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return ((band - band_min) / ((band_max - band_min)))


def brighten(band):
    alpha = 0.13
    beta = 0
    return np.clip(alpha * band + beta, 0, 255)


def brighten(band):
    alpha = 0.13
    beta = 0
    return np.clip(alpha * band + beta, 0, 255)


def convert(im_path):
    with rasterio.open(im_path) as fin:
        red = fin.read(3)
        green = fin.read(2)
        blue = fin.read(1)

    red_b = brighten(red)
    blue_b = brighten(blue)
    green_b = brighten(green)

    red_bn = normalize(red_b)
    green_bn = normalize(green_b)
    blue_bn = normalize(blue_b)

    return np.dstack((blue_b, green_b, red_b)), np.dstack((red_bn, green_bn, blue_bn))


def plot_data(image_path, mask_path):
    plt.figure(figsize=(12, 12))
    pal = [value for color in PALLETE for value in color]

    plt.subplot(1, 2, 1)
    _, img = convert(image_path)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    with rasterio.open(mask_path) as fin:
        mask = fin.read(1)
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(pal)
    plt.imshow(mask)
    plt.show()


'''
УКАЖИТЕ ПУТЬ ДО ФАЙЛОВ ИЗОБРАЖЕНИЯ И МАСКИ
'''
plot_data('test_scoltech/images/1.tif', 'train/masks/6_2.tif')

