import os
import pandas as pd

#Пути
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_FILE = os.path.join(BASE_DIR, "data", "processed", "data_preprocessed.csv")
STATS_DIR = os.path.join(BASE_DIR, "data", "stat")
os.makedirs(STATS_DIR, exist_ok=True)


#Загружаем данные
df = pd.read_csv(PROCESSED_FILE)

pollutants = ['pm2_5', 'pm10', 'no2', 'o3', 'co']

#статистика по всему датасету
stats_overall = df[pollutants + ['temperature', 'humidity']].describe()
stats_overall.to_csv(os.path.join(STATS_DIR, "descriptive_stats_overall.csv"))
print("статистика по всему датасету сохранена")

#статистика по странам
stats_by_country = df.groupby('country')[pollutants + ['temperature', 'humidity']].describe()
stats_by_country.to_csv(os.path.join(STATS_DIR, "descriptive_stats_by_country.csv"))
print("статистика по странам сохранена")
