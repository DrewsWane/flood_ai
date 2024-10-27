'''
ФУНКЦИЯ ДЛЯ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ (ВОДНЫХ ИНДЕКСОВ) ДЛЯ МОДЕЛИ CATBOOST
'''

import os
import rasterio
import numpy as np
import pandas as pd

'''
Принимает на вход синий, зелёный, красный, nir и swir2 каналы, высоту и ширину файла
'''
def features_extractor(blue_b, green_b, red_b, nir_b, swir2_b, file_h, file_w):

    features_list = []

    for i in range(file_h):
        for j in range(file_w):

            #РАСЧЁТ ПРИЗНАКОВ ДЛЯ КЛАССИФИЦИРУЕМОГО ПИКСЕЛЯ

            ndwi = (green_b[i][j] - nir_b[i][j]) / (green_b[i][j] + nir_b[i][j]) # Расчёт NDWI
            mndwi = (green_b[i][j] + red_b[i][j]) / (nir_b[i][j] + swir2_b[i][j]) # Расчёт MNDWI
            wri = (green_b[i][j] + red_b[i][j]) / (nir_b[i][j] + swir2_b[i][j]) # Расчёт WRI
            ndvi = (nir_b[i][j] - red_b[i][j]) / (nir_b[i][j] + red_b[i][j]) # Расчёт NDVI
            bright = (green_b[i][j] + red_b[i][j] + blue_b[i][j]) # Расчёт яркости
            ndti = (red_b[i][j] - green_b[i][j]) / (red_b[i][j] + green_b[i][j]) # Расчёт NDTI

            # РАСЧЁТ ПРИЗНАКОВ ДЛЯ СОСЕДНИХ ПИКСЕЛЕЙ

            if (file_h - 1) > i > 0 and (file_w - 1) > j > 0:
                ndwi_1 = (green_b[i - 1][j - 1] - nir_b[i - 1][j - 1]) / (green_b[i - 1][j - 1] + nir_b[i - 1][j - 1])
                mndwi_1 = (green_b[i - 1][j - 1] + red_b[i - 1][j - 1]) / (nir_b[i - 1][j - 1] + swir2_b[i - 1][j - 1])
                wri_1 = (green_b[i - 1][j - 1] + red_b[i - 1][j - 1]) / (nir_b[i - 1][j - 1] + swir2_b[i - 1][j - 1])
                ndvi_1 = (nir_b[i - 1][j - 1] - red_b[i - 1][j - 1]) / (nir_b[i - 1][j - 1] + red_b[i - 1][j - 1])
                bright_1 = (green_b[i - 1][j - 1] + red_b[i - 1][j - 1] + blue_b[i - 1][j - 1])
                ndti_1 = (red_b[i - 1][j - 1] - green_b[i - 1][j - 1]) / (red_b[i - 1][j - 1] + green_b[i - 1][j - 1])

                ndwi_2 = (green_b[i - 1][j + 1] - nir_b[i - 1][j + 1]) / (green_b[i - 1][j + 1] + nir_b[i - 1][j + 1])
                mndwi_2 = (green_b[i - 1][j + 1] + red_b[i - 1][j + 1]) / (nir_b[i - 1][j + 1] + swir2_b[i - 1][j + 1])
                wri_2 = (green_b[i - 1][j + 1] + red_b[i - 1][j + 1]) / (nir_b[i - 1][j + 1] + swir2_b[i - 1][j + 1])
                ndvi_2 = (nir_b[i - 1][j + 1] - red_b[i - 1][j + 1]) / (nir_b[i - 1][j + 1] + red_b[i - 1][j + 1])
                bright_2 = (green_b[i - 1][j + 1] + red_b[i - 1][j + 1] + blue_b[i - 1][j + 1])
                ndti_2 = (red_b[i - 1][j + 1] - green_b[i - 1][j + 1]) / (red_b[i - 1][j + 1] + green_b[i - 1][j + 1])

                ndwi_3 = (green_b[i + 1][j - 1] - nir_b[i + 1][j - 1]) / (green_b[i + 1][j - 1] + nir_b[i + 1][j - 1])
                mndwi_3 = (green_b[i + 1][j - 1] + red_b[i + 1][j - 1]) / (nir_b[i + 1][j - 1] + swir2_b[i + 1][j - 1])
                wri_3 = (green_b[i + 1][j - 1] + red_b[i + 1][j - 1]) / (nir_b[i + 1][j - 1] + swir2_b[i + 1][j - 1])
                ndvi_3 = (nir_b[i + 1][j - 1] - red_b[i + 1][j - 1]) / (nir_b[i + 1][j - 1] + red_b[i + 1][j - 1])
                bright_3 = (green_b[i + 1][j - 1] + red_b[i + 1][j - 1] + blue_b[i + 1][j - 1])
                ndti_3 = (red_b[i + 1][j - 1] - green_b[i + 1][j - 1]) / (red_b[i + 1][j - 1] + green_b[i + 1][j - 1])

                ndwi_4 = (green_b[i + 1][j + 1] - nir_b[i + 1][j + 1]) / (green_b[i + 1][j + 1] + nir_b[i + 1][j + 1])
                mndwi_4 = (green_b[i + 1][j + 1] + red_b[i + 1][j + 1]) / (nir_b[i + 1][j + 1] + swir2_b[i + 1][j + 1])
                wri_4 = (green_b[i + 1][j + 1] + red_b[i + 1][j + 1]) / (nir_b[i + 1][j + 1] + swir2_b[i + 1][j + 1])
                ndvi_4 = (nir_b[i + 1][j + 1] - red_b[i + 1][j + 1]) / (nir_b[i + 1][j + 1] + red_b[i + 1][j + 1])
                bright_4 = (green_b[i + 1][j + 1] + red_b[i + 1][j + 1] + blue_b[i + 1][j + 1])
                ndti_4 = (red_b[i + 1][j + 1] - green_b[i + 1][j + 1]) / (red_b[i + 1][j + 1] + green_b[i + 1][j + 1])

                ndwi_5 = (green_b[i][j - 1] - nir_b[i][j - 1]) / (green_b[i][j - 1] + nir_b[i][j - 1])
                mndwi_5 = (green_b[i][j - 1] + red_b[i][j - 1]) / (nir_b[i][j - 1] + swir2_b[i][j - 1])
                wri_5 = (green_b[i][j - 1] + red_b[i][j - 1]) / (nir_b[i][j - 1] + swir2_b[i][j - 1])
                ndvi_5 = (nir_b[i][j - 1] - red_b[i][j - 1]) / (nir_b[i][j - 1] + red_b[i][j - 1])
                bright_5 = (green_b[i][j - 1] + red_b[i][j - 1] + blue_b[i][j - 1])
                ndti_5 = (red_b[i][j - 1] - green_b[i][j - 1]) / (red_b[i][j - 1] + green_b[i][j - 1])

                ndwi_6 = (green_b[i][j + 1] - nir_b[i][j + 1]) / (green_b[i][j + 1] + nir_b[i][j + 1])
                mndwi_6 = (green_b[i][j + 1] + red_b[i][j + 1]) / (nir_b[i][j + 1] + swir2_b[i][j + 1])
                wri_6 = (green_b[i][j + 1] + red_b[i][j + 1]) / (nir_b[i][j + 1] + swir2_b[i][j + 1])
                ndvi_6 = (nir_b[i][j + 1] - red_b[i][j + 1]) / (nir_b[i][j + 1] + red_b[i][j + 1])
                bright_6 = (green_b[i][j + 1] + red_b[i][j + 1] + blue_b[i][j + 1])
                ndti_6 = (red_b[i][j + 1] - green_b[i][j + 1]) / (red_b[i][j + 1] + green_b[i][j + 1])

                ndwi_7 = (green_b[i - 1][j] - nir_b[i - 1][j]) / (green_b[i - 1][j] + nir_b[i - 1][j])
                mndwi_7 = (green_b[i - 1][j] + red_b[i - 1][j]) / (nir_b[i - 1][j] + swir2_b[i - 1][j])
                wri_7 = (green_b[i - 1][j] + red_b[i - 1][j]) / (nir_b[i - 1][j] + swir2_b[i - 1][j])
                ndvi_7 = (nir_b[i - 1][j] - red_b[i - 1][j]) / (nir_b[i - 1][j] + red_b[i - 1][j])
                bright_7 = (green_b[i - 1][j] + red_b[i - 1][j] + blue_b[i - 1][j])
                ndti_7 = (red_b[i - 1][j] - green_b[i - 1][j]) / (red_b[i - 1][j] + green_b[i - 1][j])

                ndwi_8 = (green_b[i + 1][j] - nir_b[i + 1][j]) / (green_b[i + 1][j] + nir_b[i + 1][j])
                mndwi_8 = (green_b[i + 1][j] + red_b[i + 1][j]) / (nir_b[i + 1][j] + swir2_b[i + 1][j])
                wri_8 = (green_b[i + 1][j] + red_b[i + 1][j]) / (nir_b[i + 1][j] + swir2_b[i + 1][j])
                ndvi_8 = (nir_b[i + 1][j] - red_b[i + 1][j]) / (nir_b[i + 1][j] + red_b[i + 1][j])
                bright_8 = (green_b[i + 1][j] + red_b[i + 1][j] + blue_b[i + 1][j])
                ndti_8 = (red_b[i + 1][j] - green_b[i + 1][j]) / (red_b[i + 1][j] + green_b[i + 1][j])

            else:
                ndwi_1 = 0
                mndwi_1 = 0
                wri_1 = 0
                ndvi_1 = 0
                bright_1 = 0
                ndti_1 = 0
                ndwi_2 = 0
                mndwi_2 = 0
                wri_2 = 0
                ndvi_2 = 0
                bright_2 = 0
                ndti_2 = 0
                ndwi_3 = 0
                mndwi_3 = 0
                wri_3 = 0
                ndvi_3 = 0
                bright_3 = 0
                ndti_3 = 0
                ndwi_4 = 0
                mndwi_4 = 0
                wri_4 = 0
                ndvi_4 = 0
                bright_4 = 0
                ndti_4 = 0
                ndwi_5 = 0
                mndwi_5 = 0
                wri_5 = 0
                ndvi_5 = 0
                bright_5 = 0
                ndti_5 = 0
                ndwi_6 = 0
                mndwi_6 = 0
                wri_6 = 0
                ndvi_6 = 0
                bright_6 = 0
                ndti_6 = 0
                ndwi_7 = 0
                mndwi_7 = 0
                wri_7 = 0
                ndvi_7 = 0
                bright_7 = 0
                ndti_7 = 0
                ndwi_8 = 0
                mndwi_8 = 0
                wri_8 = 0
                ndvi_8 = 0
                bright_8 = 0
                ndti_8 = 0

            features_list.append([ndwi, mndwi, wri, ndvi, bright, ndti,
                                  ndwi_1, mndwi_1, wri_1, ndvi_1, bright_1, ndti_1,
                                  ndwi_2, mndwi_2, wri_2, ndvi_2, bright_2, ndti_2,
                                  ndwi_3, mndwi_3, wri_3, ndvi_3, bright_3, ndti_3,
                                  ndwi_4, mndwi_4, wri_4, ndvi_4, bright_4, ndti_4,
                                  ndwi_5, mndwi_5, wri_5, ndvi_5, bright_5, ndti_5,
                                  ndwi_6, mndwi_6, wri_6, ndvi_6, bright_6, ndti_6,
                                  ndwi_7, mndwi_7, wri_7, ndvi_7, bright_7, ndti_7,
                                  ndwi_8, mndwi_8, wri_8, ndvi_8, bright_8, ndti_8])

    return features_list