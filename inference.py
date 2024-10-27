'''
СКРИПТ ДЛЯ ИНФЕРЕНСА

1. Поместите tif в любой каталог и укажите путь до файла в переменной im_path
2. При необходимости укажите размер тайлов в переменной tile_size
   и название файла с предиктом в функции merge_tiles (последняя строка скрипта)
3. Запустите скрипт
   По умолчанию файл с предиктом сохранится в корневую директорию под названием merged_image_1.tif
   В директории tiles хранятся нарезанные изображения и их маски
'''

import os
import rasterio
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from split_geotiff import split_image, merge_tiles
from features_extractor import features_extractor


#УКАЖИТЕ ПУТЬ ДО ФАЙЛА
im_path = "test_scoltech/images/1.tif"

#НАРЕЗКА ФАЙЛА
with rasterio.open(im_path) as img:
    height_image, width_image = img.shape
output_folder = 'tiles'
tile_size = 500
split_image(image_path=im_path, output_folder=output_folder, tile_size=tile_size, overlap=0)

#ЗАГРУЗКА МОДЕЛИ
model = CatBoostClassifier()
model.load_model("models/flood_model_6")

directory = "tiles/images"
files_with_times = []

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        creation_time = os.path.getctime(file_path)
        files_with_times.append((filename, creation_time))

#ПОЛУЧЕНИЕ ПРИЗНАКОВ И ПРЕДИКТОВ ДЛЯ КАЖДОГО ТАЙЛА В ДИРЕКТОРИИ
sorted_files = sorted(files_with_times, key=lambda x: x[1])
for filename, _ in sorted_files:
    im_path = os.path.join(directory, filename)
    with rasterio.open(im_path) as img:
        height, width = img.shape
        meta = img.meta
        blue_b = img.read(1)
        green_b = img.read(2)
        red_b = img.read(3)
        nir_b = img.read(7)
        swir2_b = img.read(10)
        features_list = features_extractor(blue_b, green_b, red_b, nir_b, swir2_b, height, width)
        print(f"features_list: {features_list}")
        eval_data = pd.DataFrame(features_list, columns=['ndwi', 'mndwi', 'wri', 'ndvi', 'bright', 'ndti',
                                                  'ndwi_1', 'mndwi_1', 'wri_1', 'ndvi_1', 'bright_1', 'ndti_1',
                                                  'ndwi_2', 'mndwi_2', 'wri_2', 'ndvi_2', 'bright_2', 'ndti_2',
                                                  'ndwi_3', 'mndwi_3', 'wri_3', 'ndvi_3', 'bright_3', 'ndti_3',
                                                  'ndwi_4', 'mndwi_4', 'wri_4', 'ndvi_4', 'bright_4', 'ndti_4',
                                                  'ndwi_5', 'mndwi_5', 'wri_5', 'ndvi_5', 'bright_5', 'ndti_5',
                                                  'ndwi_6', 'mndwi_6', 'wri_6', 'ndvi_6', 'bright_6', 'ndti_6',
                                                  'ndwi_7', 'mndwi_7', 'wri_7', 'ndvi_7', 'bright_7', 'ndti_7',
                                                  'ndwi_8', 'mndwi_8', 'wri_8', 'ndvi_8', 'bright_8', 'ndti_8'])
        print(f"eval_data: {eval_data}")
        model = CatBoostClassifier()
        model.load_model("models/flood_model_6")
        predict_list = model.predict(eval_data)
        print(f"predict_list: {predict_list}")
        predict = np.array([list(row) for row in zip(*[iter(predict_list)] * width)])
        print(f"predict: {predict}")
        meta['count'] = 1
        print(f"mets: {meta}")
        with rasterio.open(f'tiles/preds/{filename}', 'w', **meta) as fout:
            fout.write(predict, 1)

#СОЕДИНЕНИЕ ТАЙЛОВ И СОХРАНЕНИЕ ФИНАЛЬНОГО ПРЕДИКТА
merge_tiles('tiles/preds', '1.tif', tile_size=tile_size, original_width=width_image, original_height=height_image)