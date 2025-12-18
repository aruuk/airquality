import os
import pandas as pd
from scipy.stats import f_oneway, pearsonr

#Пути
PROCESSED_FILE = os.path.join("data", "data_clean.csv")
RESULTS_DIR = os.path.join("data", "stats")
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(PROCESSED_FILE)

pollutants = ['pm2_5', 'pm10', 'no2', 'o3', 'co']
countries = df['country'].unique()

results = []

#ANOVA: различие загрязнителей между странами
for p in pollutants:
    groups = [df[df['country']==c][p] for c in countries]
    f_stat, p_value = f_oneway(*groups)
    results.append({
        "Тип проверки": "ANOVA",
        "Загрязнитель": p,
        "Группы / переменные": "Все страны",
        "Статистика": f_stat,
        "p-value": p_value,
        "Вывод": "Значимо" if p_value < 0.05 else "Не значимо"
    })

#Корреляция с температурой
for p in pollutants:
    corr, p_value = pearsonr(df['temperature'], df[p])
    results.append({
        "Тип проверки": "Корреляция",
        "Загрязнитель": p,
        "Группы / переменные": "Температура",
        "Статистика": corr,
        "p-value": p_value,
        "Вывод": "Значимо" if p_value < 0.05 else "Не значимо"
    })

#Корреляция с влажностью
for p in pollutants:
    corr, p_value = pearsonr(df['humidity'], df[p])
    results.append({
        "Тип проверки": "Корреляция",
        "Загрязнитель": p,
        "Группы / переменные": "Влажность",
        "Статистика": corr,
        "p-value": p_value,
        "Вывод": "Значимо" if p_value < 0.05 else "Не значимо"
    })

for p in pollutants:
    corr, p_value = pearsonr(df['wind_speed'], df[p])
    results.append({
        "Тип проверки": "Корреляция",
        "Загрязнитель": p,
        "Группы / переменные": "Скорость ветра",
        "Статистика": corr,
        "p-value": p_value,
        "Вывод": "Значимо" if p_value < 0.05 else "Не значимо"
    })

#Сохраняем результаты в CSV
results_df = pd.DataFrame(results)
results_file = os.path.join(RESULTS_DIR, "hypothesis_tests_results.csv")
results_df.to_csv(results_file, index=False)

print(f"Результаты проверки гипотез сохранены в CSV: {results_file}")
