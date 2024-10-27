'''
ОБУЧЕНИЕ МОДЕЛИ CATBOOST
'''

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# Загрузка и подготовка данных
ds_1 = pd.read_csv('datasets/dataset_9_big.csv')
ds_2 = pd.read_csv('datasets/dataset_6_big.csv')
ds = pd.concat([ds_1, ds_2], axis=0)
y = ds['mask']
X = ds.drop('mask', axis=1)

# Разделение датасета на train и test части
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация модели
model = CatBoostClassifier(iterations=3000, learning_rate=0.1, verbose=5)
# Обучение модели
model.fit(X_train, y_train)

# Проверка скора модели
predictions = model.predict(X_test)
f = f1_score(y_test, predictions)
print(f'F1-score: {f}')

# Сохранение модели
model.save_model("flood_model_6")
