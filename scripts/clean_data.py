import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import reverse_geocoder as rg
from config import CSV_PATH_RAW, CSV_PATH_CLEAN

# --- Пути ---
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- Загружаем сырые данные ---
df = pd.read_csv(CSV_PATH_RAW)

# Проверяем наличие самых важных столбцов
required_columns = {
    'datetime', 'lat', 'lon', 
    'pm2_5', 'pm10', 'no2', 'so2', 'o3', 'co',
    'humidity', 'wind_speed', 'temperature'
}

missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"Не хватает столбцов: {missing}")

# Убираем дубликаты
df = df.drop_duplicates()

# Убираем строки, где нет координат (иначе не определить страну)
df = df.dropna(subset=['lat', 'lon'])


#   ДОБАВЛЕНИЕ country / city ПО КООРДИНАТАМ
def reverse_geocode(row):
    # mode=1 -> отключает multiprocessing (Windows fix)
    result = rg.search((row['lat'], row['lon']), mode=1)[0]
    return pd.Series({
        'country_filled': result['cc'],
        'city_filled': result['name']
    })

# Строки где нет country или city
mask = df['country'].isna() | df['city'].isna()

df.loc[mask, ['country_filled', 'city_filled']] = df[mask].apply(reverse_geocode, axis=1)

# Заполняем пропуски
df['country'] = df['country'].fillna(df['country_filled'])
df['city'] = df['city'].fillna(df['city_filled'])

df = df.drop(columns=['country_filled', 'city_filled'])


#   УДАЛЕНИЕ ВЫБРОСОВ по IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

numeric_columns = ['temperature', 'humidity', 'pm2_5', 'pm10', 'no2', 'so2', 'o3', 'co', 'wind_speed']

for col in numeric_columns:
    df = remove_outliers_iqr(df, col)


#   СОХРАНЕНИЕ РЕЗУЛЬТАТА
os.makedirs(os.path.dirname(CSV_PATH_CLEAN), exist_ok=True)
df.to_csv(CSV_PATH_CLEAN, index=False)

print(f"Файл успешно очищен и сохранён: {CSV_PATH_CLEAN}")