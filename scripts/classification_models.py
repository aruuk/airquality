import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Пути
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "data_clean.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Загрузка данных
df = pd.read_csv(DATA_PATH)

# Создание классов pm10
# Квантильный порог
threshold = df["pm10"].quantile(0.7)

def pm10_class(pm):
    return 1 if pm > threshold else 0

df["pm10_class"] = df["pm10"].apply(pm10_class)

# Проверка распределения классов
print("\nPM10 class distribution:")
print(df["pm10_class"].value_counts())

# Признаки и цель
FEATURES = ["temperature", "humidity", "wind_speed"]
TARGET = "pm10_class"

X = df[FEATURES]
y = df[TARGET]

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Масштабирование (только для Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Модели классификации
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = []

# Обучение и оценка моделей
for name, model in models.items():

    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "F1": f1
    })

# Таблица результатов
results_df = pd.DataFrame(results).sort_values("F1", ascending=False)

results_path = os.path.join(MODELS_DIR, "classification_results_pm10.csv")
results_df.to_csv(results_path, index=False)

print("\nClassification results:")
print(results_df)
print(f"\nResults saved to: {results_path}")

# Выбор лучшей модели
best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest model based on F1-score: {best_model_name}")

# Переобучение лучшей модели на ВСЁМ датасете
if best_model_name == "Logistic Regression":
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)

    final_model = LogisticRegression(max_iter=1000)
    final_model.fit(X_scaled, y)

    joblib.dump(final_scaler, os.path.join(MODELS_DIR, "scaler_pm10.pkl"))

else:
    final_model = models[best_model_name]
    final_model.fit(X, y)

# Сохранение модели
model_filename = best_model_name.lower().replace(" ", "_") + "_pm10_classifier.pkl"
model_path = os.path.join(MODELS_DIR, model_filename)

joblib.dump(final_model, model_path)

print(f"\nFinal classification model saved to: {model_path}")
print("\nPM10 class distribution:")
print(df["pm10_class"].value_counts())