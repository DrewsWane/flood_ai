Все необходимые зависимости в файле requirements.txt

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
ДЛЯ ЗАПУСКА ИНФЕРЕНСА СЛЕДУЙТЕ ИНСТРУКЦИЯМ В ФАЙЛЕ INFERENCE
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

1. inference.py
Здесь находятся подробные инструкции по запуску инференса,
нужно поместить tif в каталог, указать до него путь и запустить скрипт,
файл с предиктом сохранится в корневой директории

2. feature_extractor.py
Здесь находится функция для расчёта водных индексов и преобразования их в признаки для модели

3. dataset_builder.py
Здесь находится скрипт, обрабатывающий тренировочные данные и создающий датасет
с нужными признаками в формате csv

4. catboost_trainer.py
Здесь находится пайплайн обучения модели catboost

5. split_geotiff.py, visualizator, calculate_metrics
Это вспомогательные файлы для нарезки изображений, визуализации данных и подсчета метрик