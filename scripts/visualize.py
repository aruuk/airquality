import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt

#Пути
PROCESSED_FILE = os.path.join("data", "processed", "data_preprocessed.csv")
VISUAL_DIR = os.path.join("data", "visualizations")
os.makedirs(VISUAL_DIR, exist_ok=True)

#Загружаем предобработанные данные
df = pd.read_csv(PROCESSED_FILE)

pollutants = ['pm2_5', 'pm10', 'no2', 'o3', 'co']
countries = df['country'].unique()
colors = plt.cm.get_cmap('tab20', len(countries))

#Функция для построения графиков
def create_visualizations(df, x_col, filename_prefix):
    fig, axes = plt.subplots(len(pollutants), 3, figsize=(18, 5 * len(pollutants)))
    
    for i, p in enumerate(pollutants):

        #Scatter plot: x - температура/влажность, y - загрязнитель
        #На этом графике показано как в зависимости от температуры/влажности изменяется концентрация каждого загрязнителя
        #Позволяет выявить как меняется качество воздуха в зависимости от температуры/влажности
        for j, country in enumerate(countries):
            country_df = df[df['country'] == country]
            axes[i,0].scatter(country_df[x_col], country_df[p], label=country, alpha=0.6, color=colors(j))
        axes[i,0].set_title(f"{p} vs {x_col}")
        axes[i,0].set_xlabel(x_col)
        axes[i,0].set_ylabel(f"{p} Концентрация")
        axes[i,0].grid(True)

        #Boxplot: x - страны, y - загрязнитель
        #Медианные значения и разброс концентраций каждого загрязнителя для каждой страны
        #Позволяет сравнить уровень загрязнения воздуха между странами и определить outliers,
        #а также показать в какой стране температура/влажность оказывает большее влияние
        df.boxplot(column=p, by='country', ax=axes[i,1])
        axes[i,1].set_title(f"{p} по странам (Boxplot)")
        axes[i,1].set_xlabel("Страна")
        axes[i,1].set_ylabel(f"{p} Концентрация")

        #Histogram: x - загрязнитель, y - количество наблюдений
        #Распределение концентраций загрязнителя во всем наборе данных
        #Позволяет оценить форму смещения и частоту высоких значений концентраций загрязнений
        axes[i,2].hist(df[p], bins=30, color='skyblue', edgecolor='black')
        axes[i,2].set_title(f"{p} (Гистограмма)")
        axes[i,2].set_xlabel(f"{p} Концентрация")
        axes[i,2].set_ylabel("Количество наблюдений")
    
    plt.suptitle(f"Зависимость загрязнителей от {x_col}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(VISUAL_DIR, f"{filename_prefix}.png"))
    plt.close()

#Создаем графики
create_visualizations(df, "temperature", "pollutants_vs_temperature")
create_visualizations(df, "humidity", "pollutants_vs_humidity")

print(f"Графики сохранены в папке: {VISUAL_DIR}")
