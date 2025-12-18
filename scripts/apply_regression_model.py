import os
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Пути
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

APPLY_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "data_preprocessed.csv")  # 1722 строки
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "gradient_boosting_pm25.pkl"   # имя из regression_models.py
)

#Признаки и цель
FEATURES = ['temperature', 'humidity', 'wind_speed']
TARGET = 'pm2_5'

#Загрузка данных
df = pd.read_csv(APPLY_DATA_PATH)
df = df.dropna(subset=FEATURES + [TARGET])

print(f"Apply dataset size: {len(df)} rows")

#Загрузка модели
model = joblib.load(MODEL_PATH)
print("Model loaded:", type(model).__name__)

#Применение модели
predictions = model.predict(df[FEATURES])
df['pm2_5_predicted'] = predictions

#Оценка качества
mae = mean_absolute_error(df[TARGET], predictions)
rmse = np.sqrt(mean_squared_error(df[TARGET], predictions))
r2 = r2_score(df[TARGET], predictions)

print("\nPerformance on independent dataset:")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")

#Сохранение результатов
output_path = os.path.join(MODELS_DIR, "pm25_predictions_external.csv")
df.to_csv(output_path, index=False)

print(f"\nPredictions saved to: {output_path}")
