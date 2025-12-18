import os
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, precision_score, f1_score

# Пути
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

APPLY_DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "data_preprocessed.csv"
)  # внешний датасет (1722 строки)

MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "gradient_boosting_pm10_classifier.pkl"
)

# Признаки и цель
FEATURES = ["temperature", "humidity", "wind_speed"]
TARGET = "pm10"  # нужен только для оценки, если есть

# Загрузка данных
df = pd.read_csv(APPLY_DATA_PATH)
df = df.dropna(subset=FEATURES)

print(f"Apply dataset size: {len(df)} rows")

# Загрузка модели
model = joblib.load(MODEL_PATH)
print("Model loaded:", type(model).__name__)

# Предсказание классов
df["pm10_class_predicted"] = model.predict(df[FEATURES])

# Вероятность высокого загрязнения (очень полезно!)
if hasattr(model, "predict_proba"):
    df["pm10_high_probability"] = model.predict_proba(df[FEATURES])[:, 1]

# Если в данных есть реальный pm10 → можно оценить качество
if TARGET in df.columns:
    # восстановим порог (тот же принцип, что и в обучении)
    threshold = df[TARGET].quantile(0.7)
    df["pm10_class_true"] = (df[TARGET] > threshold).astype(int)

    accuracy = accuracy_score(df["pm10_class_true"], df["pm10_class_predicted"])
    precision = precision_score(df["pm10_class_true"], df["pm10_class_predicted"])
    f1 = f1_score(df["pm10_class_true"], df["pm10_class_predicted"])

    print("\nPerformance on independent dataset:")
    print(f"Accuracy  = {accuracy:.4f}")
    print(f"Precision = {precision:.4f}")
    print(f"F1-score  = {f1:.4f}")

# Сохранение результатов
output_path = os.path.join(
    MODELS_DIR, "pm10_classification_external.csv"
)
df.to_csv(output_path, index=False)

print(f"\nClassification results saved to: {output_path}")
