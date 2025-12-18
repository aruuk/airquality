import os
import pandas as pd

# --- Пути ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_FILE = os.path.join(BASE_DIR, "data", "data_clean.csv")
STATS_DIR = os.path.join(BASE_DIR, "data", "stat")
os.makedirs(STATS_DIR, exist_ok=True)

# --- Загружаем данные ---
df = pd.read_csv(PROCESSED_FILE)

pollutants = ['pm2_5', 'pm10', 'no2', 'o3', 'co']
columns_to_use = pollutants + ['temperature', 'humidity', 'wind_speed']

# --- Путь к Excel ---
excel_path = os.path.join(STATS_DIR, "descriptive_stats.xlsx")

# --- Создаём Excel с форматированием ---
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    workbook = writer.book
    
    # --- Статистика по всему датасету ---
    stats_overall = df[columns_to_use].describe()
    stats_overall.to_excel(writer, sheet_name='Overall')
    
    worksheet = writer.sheets['Overall']
    format_number = workbook.add_format({'num_format': '0.00'})
    for i, col in enumerate(stats_overall.columns):
        worksheet.set_column(i+1, i+1, 12, format_number)
    
    # --- Сводная статистика по странам ---
    stats_by_country = df.groupby('country')[columns_to_use].describe()
    stats_by_country.to_excel(writer, sheet_name='By_Country')
    
    worksheet2 = writer.sheets['By_Country']
    for i, col in enumerate(stats_by_country.columns.levels[0]):  # multiindex columns
        worksheet2.set_column(i*8+1, i*8+8, 12, format_number)
    
    # --- Статистика для каждой страны отдельно ---
    countries = df['country'].unique()
    for country in countries:
        df_country = df[df['country'] == country]
        stats_country = df_country[columns_to_use].describe()
        # Excel не любит очень длинные имена листов, ограничим до 31 символа
        sheet_name = f"Country_{country}"[:31]
        stats_country.to_excel(writer, sheet_name=sheet_name)
        worksheet_country = writer.sheets[sheet_name]
        for i, col in enumerate(stats_country.columns):
            worksheet_country.set_column(i+1, i+1, 12, format_number)

print(f"Статистика успешно сохранена в Excel: {excel_path}")
