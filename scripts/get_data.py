import os
import sys
import time
import requests
import pandas as pd
import random
from datetime import datetime

#путь
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#Настройки
CSV_PATH_RAW = "data/air_data_raw.csv"
API_KEY = "bf5974b41307fc5c6820a9986dada4a2"
MAX_ROWS = 10000
SLEEP = 0.4
ITERATIONS = 15  #Количество загрузок(по 1000 координат каждая)

BASE_URL_AIR = "https://api.openweathermap.org/data/2.5/air_pollution"
BASE_URL_WEATHER = "https://api.openweathermap.org/data/2.5/weather"

os.makedirs(os.path.dirname(CSV_PATH_RAW), exist_ok=True)

#рандомные координаты по всему миру
def get_random_coords(n=1000):
    return [(random.uniform(-90, 90), random.uniform(-180, 180)) for _ in range(n)]

#Данные о воздухе
def get_air_quality(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": API_KEY}
    try:
        r = requests.get(BASE_URL_AIR, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json().get("list", [])
        if not data:
            return None
        components = data[0].get("components", {})
        dt = datetime.utcfromtimestamp(data[0]["dt"]).isoformat()
        return {"datetime": dt, **components}
    except Exception:
        return None

#Данные о погоде
def get_weather(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
    try:
        r = requests.get(BASE_URL_WEATHER, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        return {
            "temperature": data.get("main", {}).get("temp"),
            "humidity": data.get("main", {}).get("humidity"),
            "wind_speed": data.get("wind", {}).get("speed"),
            "country": data.get("sys", {}).get("country"),
            "city": data.get("name"),
        }
    except Exception:
        return None

print("сбор данных")

records = []

for iteration in range(1, ITERATIONS + 1):
    coords = get_random_coords(1000)
    print(f"\n загрузка {iteration}/{ITERATIONS}: собираем данные")

    for i, (lat, lon) in enumerate(coords, start=1):
        air = get_air_quality(lat, lon)
        weather = get_weather(lat, lon)

        #Сохраняем данные
        if air and "datetime" in air:
            row = {
                "datetime": air.get("datetime"),
                "lat": lat,
                "lon": lon,
                "country": weather.get("country") if weather else None,
                "city": weather.get("city") if weather else None,
                "pm2_5": air.get("pm2_5"),
                "pm10": air.get("pm10"),
                "no2": air.get("no2"),
                "so2": air.get("so2"),
                "o3": air.get("o3"),
                "co": air.get("co"),
                "humidity": weather.get("humidity") if weather else None,
                "wind_speed": weather.get("wind_speed") if weather else None,
                "temperature": weather.get("temperature") if weather else None,
            }
            records.append(row)

        print(f"  [{i}/1000] Коорд: ({lat:.2f}, {lon:.2f}) — всего: {len(records)}")

        if len(records) >= MAX_ROWS:
            break

        time.sleep(SLEEP)

    #Промежуточное сохранение после каждой загрузки
    if records:
        df = pd.DataFrame(records)
        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime", ascending=False).reset_index(drop=True)
        df.to_csv(CSV_PATH_RAW, index=False)
        print(f"Промежуточное сохранение: {len(df)} строк в {CSV_PATH_RAW}")

    if len(records) >= MAX_ROWS:
        print("Достигнуто 10 000 строк, сбор завершён.")
        break

print(f"\nФинально сохранено {len(records)} строк в {CSV_PATH_RAW}")
