import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Пути
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "data_clean.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

#Загрузка данных
df = pd.read_csv(DATA_PATH)

#Признаки и целевая переменная
features = ['temperature', 'humidity', 'wind_speed']
target = 'pm2_5'

X = df[features]
y = df[target]

#Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Масштабирование (только для Linear Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Модели
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []

#Обучение и оценка
for name, model in models.items():

    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

#Таблица результатов
results_df = pd.DataFrame(results).sort_values("RMSE")

results_path = os.path.join(MODELS_DIR, "regression_results.csv")
results_df.to_csv(results_path, index=False)

print("\nRegression results:")
print(results_df)
print(f"\nResults saved to: {results_path}")

#Лучшая модель
best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest model based on RMSE: {best_model_name}")

#Переобучение лучшей модели на ВСЕХ данных
if best_model_name == "Linear Regression":
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)

    final_model = LinearRegression()
    final_model.fit(X_scaled, y)

    joblib.dump(final_scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

else:
    final_model = models[best_model_name]
    final_model.fit(X, y)

#Сохранение модели
model_filename = best_model_name.lower().replace(" ", "_") + "_pm25.pkl"
model_path = os.path.join(MODELS_DIR, model_filename)

joblib.dump(final_model, model_path)

print(f"\nFinal model saved to: {model_path}")
print(model)
print(model.n_estimators_)
print(model.learning_rate)
print(model.n_features_in_)
print(type(model.estimators_))
print(len(model.estimators_))
