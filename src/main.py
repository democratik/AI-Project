"""Uruchamia cały projekt: przygotowanie danych -> EDA -> modele.

Użycie (z katalogu głównego projektu):
    python src/main.py
"""

import pandas as pd

from config import FEATURES, OUTPUT_DIR, TARGET, ensure_dirs
from data_prep import prepare_data
from eda import run_eda
from model import run_models

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def section(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main() -> None:
    ensure_dirs()

    section("ETAP 1: Przygotowanie i czyszczenie danych")
    raw, with_nan, clean, report = prepare_data()
    print(f"Liczba rekordów: {len(raw)}, liczba cech: {len(FEATURES)}")
    print("\nRaport jakości danych surowych:")
    print(report)

    section("ETAP 2: Analiza eksploracyjna (EDA)")
    tables = run_eda(clean, report)
    print("Rozkład klas:\n", tables["rozklad_klas"])
    print("\nŚrednie wartości cech wg diagnozy:\n", tables["srednie_wg_diagnozy"])
    print("\nKorelacja cech z Outcome:\n", tables["korelacja_z_outcome"])
    print("\nOdsetek chorych wg grupy wiekowej:\n", tables["chorobowosc_wg_wieku"])
    print("\nOdsetek chorych wg kategorii BMI:\n", tables["chorobowosc_wg_bmi"])

    section("ETAP 3: Modele klasyfikacyjne")
    results = run_models(with_nan)
    print(results["metrics"])
    print("\nWażność cech:\n", results["importance"])
    print("\nRegresja logistyczna – analiza progu decyzyjnego:\n", results["thresholds"])

    section("Zapisane wyniki")
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if path.is_file():
            print(" -", path.relative_to(OUTPUT_DIR.parent))


if __name__ == "__main__":
    main()
