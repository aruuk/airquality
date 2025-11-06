import os

API_KEY = "bfd0aab0df8620fa3d92b5832a8deb74935b5368d2bf6267b7c9e6ed45b4c5ad"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH_RAW = os.path.join(DATA_DIR, "air_data_raw.csv")

CSV_PATH_CLEAN = os.path.join(DATA_DIR, "data", "data_clean.csv")