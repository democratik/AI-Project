"""Etap 1: wczytanie, kontrola jakości i czyszczenie danych"""

import numpy as np
import pandas as pd

from config import CLEAN_DATA_PATH, DATA_PATH, FEATURES, TARGET, ZERO_AS_MISSING

# słownik kategorii BMI wg klasyfikacji WHO – osobna tabela,
BMI_CATEGORIES = pd.DataFrame(
    {
        "BMI_code": [0, 1, 2, 3],
        "BMI_category": ["Niedowaga", "Norma", "Nadwaga", "Otyłość"],
        "BMI_range": ["< 18,5", "18,5 – 24,9", "25,0 – 29,9", "≥ 30,0"],
    }
)

AGE_BINS = [20, 30, 40, 50, 60, 100]
AGE_LABELS = ["21–30", "31–40", "41–50", "51–60", "60+"]


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Wczytuje surowy zbiór Pima Indians Diabetes."""
    return pd.read_csv(path)


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Zwraca tabelę z typami danych, brakami (NaN) i liczbą zer w każdej kolumnie."""
    report = pd.DataFrame(
        {
            "typ": df.dtypes.astype(str),
            "braki_NaN": df.isna().sum(),
            "liczba_zer": (df == 0).sum(),
            "procent_zer": ((df == 0).mean() * 100).round(1),
        }
    )
    return report


def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Zamienia niemożliwe wartości 0 na NaN w wybranych kolumnach."""
    df = df.copy()
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Uzupełnia braki medianą kolumny (fillna).

    Mediana jest odporna na wartości odstające (np. bardzo wysoka insulina).
    Ta wersja danych służy do analizy eksploracyjnej. W modelu uzupełnianie
    odbywa się wewnątrz Pipeline (SimpleImputer), dopasowanym tylko na zbiorze
    treningowym – dzięki temu informacja ze zbioru testowego nie "przecieka".
    """
    df = df.copy()
    medians = df[ZERO_AS_MISSING].median()
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].fillna(medians)
    return df


def add_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje grupy wiekowe oraz kategorie BMI (przez merge ze słownikiem)."""
    df = df.copy()
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS)

    df["BMI_code"] = pd.cut(
        df["BMI"], bins=[0, 18.5, 25, 30, np.inf], labels=[0, 1, 2, 3], right=False
    ).astype(int)
    df = df.merge(BMI_CATEGORIES, on="BMI_code", how="left")
    df["BMI_category"] = pd.Categorical(
        df["BMI_category"], categories=BMI_CATEGORIES["BMI_category"], ordered=True
    )
    return df


def prepare_data(save: bool = True):
    """Pełny etap przygotowania danych.

    Zwraca krotkę:
      raw      – dane surowe,
      with_nan – dane z zerami zamienionymi na NaN (wejście do modelu),
      clean    – dane po uzupełnieniu braków i z kategoriami (do EDA),
      report   – raport jakości danych surowych.
    """
    raw = load_data()

    # duplikaty i poprawność typów
    raw = raw.drop_duplicates()
    raw[TARGET] = raw[TARGET].astype(int)

    report = quality_report(raw)
    with_nan = replace_zeros_with_nan(raw)
    report["braki_po_zamianie"] = with_nan.isna().sum()

    clean = add_categories(fill_missing(with_nan))

    if save:
        CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(CLEAN_DATA_PATH, index=False)

    return raw, with_nan, clean, report


if __name__ == "__main__":
    raw, with_nan, clean, report = prepare_data()
    print("Rozmiar danych:", raw.shape)
    print(report)
    print(clean[FEATURES + [TARGET]].describe().round(2))
