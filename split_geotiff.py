'''
ФУНКЦИИ ДЛЯ НАРЕЗКИ БОЛЬШОГО ИЗОБРАЖЕНИЯ НА ТАЙТЛЫ
И СОЕДИНЕНИЯ ТАЙТЛОВ ОБРАТНО В БОЛЬШОЙ TIF
'''

from tqdm import tqdm
from typing import List, Optional
from rasterio.windows import Window
import rasterio
import os
import numpy as np


def get_tiles_with_overlap(image_width: int, image_height: int,
                           tile_size: int, overlap: int) -> List[Window]:
    step_size = tile_size - overlap
    tiles = []
    for y in range(0, image_height, step_size):
        for x in range(0, image_width, step_size):
            window = Window(x, y, tile_size, tile_size)
            window = window.intersection(Window(0, 0, image_width, image_height))
            tiles.append(window)
    return tiles


def save_tile(src_dataset: rasterio.io.DatasetReader, window: Window,
              output_folder: str, tile_index: int, image_id: int) -> None:

    transform = src_dataset.window_transform(window)
    tile_data = src_dataset.read(window=window)

    profile = src_dataset.profile
    profile.update({
        'driver': 'GTiff',
        'height': window.height,
        'width': window.width,
        'transform': transform
    })

    output_filename = os.path.join(output_folder, f"tile_{image_id}_{tile_index}.tif")
    with rasterio.open(output_filename, 'w', **profile) as dst:
        dst.write(tile_data)

'''
ФУНКЦИЯ ДЛЯ НАРЕЗКИ БОЛЬШОГО ИЗОБРАЖЕНИЯ НА ТАЙТЛЫ
'''
def split_image(image_path: str, output_folder: str, mask_path: Optional[str] = None,
                tile_size: int = 512, overlap: int = 128, image_id: int = 0) -> None:

    with rasterio.open(image_path) as src_image:
        image_width = src_image.width
        image_height = src_image.height

        images_folder = os.path.join(output_folder, 'images')
        os.makedirs(images_folder, exist_ok=True)

        if mask_path:
            masks_folder = os.path.join(output_folder, 'preds')
            os.makedirs(masks_folder, exist_ok=True)

        tiles = get_tiles_with_overlap(image_width, image_height, tile_size, overlap)

        if mask_path:
            with rasterio.open(mask_path) as src_mask:
                for idx, window in tqdm(enumerate(tiles)):
                    save_tile(src_image, window, images_folder, idx, image_id)
                    save_tile(src_mask, window, masks_folder, idx, image_id)
        else:
            for idx, window in tqdm(enumerate(tiles)):
                save_tile(src_image, window, images_folder, idx, image_id)


'''
ФУНКЦИЯ ДЛЯ СОЕДИНЕНИЯ ТАЙТЛОВ
'''
def merge_tiles(input_folder, output_path, original_width, original_height, tile_size=200, overlap=0):
    num_tiles_x = (original_width + tile_size - 1) // tile_size
    num_tiles_y = (original_height + tile_size - 1) // tile_size

    tile_files = sorted(os.listdir(input_folder))

    with rasterio.open(os.path.join(input_folder, tile_files[0])) as first_tile:
        meta = first_tile.meta.copy()

    merged_image = np.zeros((meta['count'], original_height, original_width), dtype=meta['dtype'])

    for index in range(len(tile_files)):
        tile_file = tile_files[index]
        tile_path = os.path.join(input_folder, tile_file)

        if not os.path.exists(tile_path):
            print(f"Файл не найден: {tile_path}")
            continue

        print(f"Обрабатываем файл: {tile_file}")

        with rasterio.open(tile_path) as tile:
            data = tile.read()
            print(f"Форма данных файла {tile_file}: {data.shape}")

            # Создаем новый массив для дополнения до 200x200
            padded_data = np.zeros((1, tile_size, tile_size), dtype=meta['dtype'])

            # Получаем текущие размеры
            height, width = data.shape[1], data.shape[2]

            # Заполняем новый массив данными из текущего тайла
            padded_data[0, :height, :width] = data[0]  # Заполняем только существующие данные

            # Вычисляем текущие координаты по индексу
            j = index // num_tiles_x
            i = index % num_tiles_x

            y_start = j * (tile_size - overlap)
            x_start = i * (tile_size - overlap)

            # Убедимся, что мы не выходим за пределы merged_image
            y_end = min(y_start + tile_size, merged_image.shape[1])
            x_end = min(x_start + tile_size, merged_image.shape[2])

            # Определяем области для вставки
            merged_image_slice = merged_image[:, y_start:y_end, x_start:x_end]
            padded_data_slice = padded_data[:, :y_end - y_start, :x_end - x_start]

            merged_image_slice[:] = padded_data_slice

    meta.update({
        'height': merged_image.shape[1],
        'width': merged_image.shape[2],
        'count': merged_image.shape[0]
    })

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(merged_image)