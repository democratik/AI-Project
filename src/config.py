"""Wspólna konfiguracja projektu: ścieżki, stałe i nazwy kolumn."""

from pathlib import Path

# katalog główny projektu 
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "diabetes.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
CLEAN_DATA_PATH = OUTPUT_DIR / "diabetes_clean.csv"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

TARGET = "Outcome"

# kolumny w których wartość 0 jest fizjologicznie niemożliwa
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# polskie etykiety cech używane na wykresach
FEATURE_LABELS = {
    "Pregnancies": "Liczba ciąż",
    "Glucose": "Glukoza [mg/dl]",
    "BloodPressure": "Ciśnienie rozkurczowe [mm Hg]",
    "SkinThickness": "Grubość fałdu skórnego [mm]",
    "Insulin": "Insulina [µU/ml]",
    "BMI": "BMI [kg/m²]",
    "DiabetesPedigreeFunction": "Obciążenie genetyczne (DPF)",
    "Age": "Wiek [lata]",
}

OUTCOME_LABELS = {0: "Brak cukrzycy", 1: "Cukrzyca"}

RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_dirs() -> None:
    """Tworzy katalogi wyjściowe, jeśli jeszcze nie istnieją."""
    for d in (OUTPUT_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
