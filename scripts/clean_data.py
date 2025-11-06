import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from config import CSV_PATH_CLEAN

#Пути
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

#Загружаем данные
df = pd.read_csv(CSV_PATH_CLEAN)

#Проверяем нужные столбцы
required_columns = {'country', 'temperature', 'humidity', 'pm2_5', 'pm10', 'no2', 'o3', 'co'}
missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"В данных не хватает столбцов: {missing}")

#Убираем дубликаты
df = df.drop_duplicates()

#Убираем строки с пропущенными значениями
df = df.dropna(subset=required_columns)

#Удаление выбросов
#Функция для удаления выбросов по IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

#Применение для всех числовых столбцов
numeric_columns = ['temperature', 'humidity', 'pm2_5', 'pm10', 'no2', 'o3', 'co']
for col in numeric_columns:
    df = remove_outliers_iqr(df, col)

#Сохраняем обработанный датасет
processed_file_path = os.path.join(PROCESSED_DIR, "data_preprocessed.csv")
df.to_csv(processed_file_path, index=False)