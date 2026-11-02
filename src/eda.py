"""Etap 2: analiza eksploracyjna (EDA) – statystyki opisowe i wykresy."""

import matplotlib

matplotlib.use("Agg")  # zapis wykresów do plików bez okna GUI
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import (
    FEATURE_LABELS,
    FEATURES,
    FIGURES_DIR,
    OUTCOME_LABELS,
    TABLES_DIR,
    TARGET,
    ZERO_AS_MISSING,
)

PALETTE = {OUTCOME_LABELS[0]: "#4C72B0", OUTCOME_LABELS[1]: "#DD5555"}
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _with_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Diagnoza"] = out[TARGET].map(OUTCOME_LABELS)
    return out


# ---------------------------------------------------------------- tabele

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Statystyki opisowe (describe) dla wszystkich cech."""
    return df[FEATURES].describe().T.round(2)


def stats_by_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Średnie wartości cech w podziale na diagnozę (groupby)."""
    table = df.groupby(TARGET)[FEATURES].mean().T.round(2)
    table.columns = [OUTCOME_LABELS[c] for c in table.columns]
    table["Różnica"] = (table[OUTCOME_LABELS[1]] - table[OUTCOME_LABELS[0]]).round(2)
    return table


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Liczność i udział klas (value_counts)."""
    counts = df[TARGET].value_counts().sort_index()
    share = df[TARGET].value_counts(normalize=True).sort_index() * 100
    table = pd.DataFrame({"liczba": counts, "procent": share.round(1)})
    table.index = table.index.map(OUTCOME_LABELS)
    table.index.name = "Diagnoza"
    return table


def rate_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Odsetek chorych w grupach (np. wiekowych lub BMI)."""
    table = (
        df.groupby(group_col, observed=True)[TARGET]
        .agg(liczba_pacjentek="count", liczba_chorych="sum", odsetek_chorych="mean")
    )
    table["odsetek_chorych"] = (table["odsetek_chorych"] * 100).round(1)
    return table


def correlations(df: pd.DataFrame) -> pd.Series:
    """Korelacja (Pearsona) każdej cechy ze zmienną Outcome, malejąco."""
    corr = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET)
    return corr.sort_values(ascending=False).round(3)


# ---------------------------------------------------------------- wykresy

def plot_missing(report: pd.DataFrame) -> None:
    """Udział ukrytych braków (zer) w kolumnach przed czyszczeniem."""
    data = report.loc[ZERO_AS_MISSING, "procent_zer"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh([FEATURE_LABELS[c] for c in data.index], data.values, color="#8172B2")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlabel("Odsetek rekordów z wartością 0 (brak danych)")
    ax.set_title("Ukryte braki danych w surowym zbiorze")
    ax.set_xlim(0, max(data.values) * 1.15)
    _save(fig, "01_braki_danych.png")


def plot_class_balance(df: pd.DataFrame) -> None:
    counts = df[TARGET].map(OUTCOME_LABELS).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[k] for k in counts.index])
    ax.bar_label(
        bars,
        labels=[f"{v} ({v / counts.sum():.0%})" for v in counts.values],
        padding=3,
    )
    ax.set_ylabel("Liczba pacjentek")
    ax.set_title("Rozkład klas (zmienna Outcome)")
    ax.set_ylim(0, counts.max() * 1.15)
    _save(fig, "02_rozklad_klas.png")


def plot_histograms(df: pd.DataFrame) -> None:
    data = _with_labels(df)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, FEATURES):
        sns.histplot(
            data=data, x=col, hue="Diagnoza", kde=True, ax=ax,
            palette=PALETTE, stat="density", common_norm=False, alpha=0.45,
        )
        ax.set_title(FEATURE_LABELS[col])
        ax.set_xlabel("")
    fig.suptitle("Rozkłady cech w podziale na diagnozę", fontsize=15)
    _save(fig, "03_histogramy.png")


def plot_boxplots(df: pd.DataFrame) -> None:
    data = _with_labels(df)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, FEATURES):
        sns.boxplot(
            data=data, x="Diagnoza", y=col, hue="Diagnoza",
            palette=PALETTE, ax=ax, legend=False,
        )
        ax.set_title(FEATURE_LABELS[col])
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Porównanie cech: chore vs. zdrowe pacjentki (wykresy pudełkowe)", fontsize=15)
    _save(fig, "04_boxploty.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Macierz korelacji Pearsona")
    _save(fig, "05_macierz_korelacji.png")


def plot_target_correlation(df: pd.DataFrame) -> None:
    corr = correlations(df).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh([FEATURE_LABELS[c] for c in corr.index], corr.values, color="#55A868")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xlabel("Współczynnik korelacji z Outcome")
    ax.set_title("Siła związku cech z cukrzycą")
    ax.set_xlim(0, corr.max() * 1.15)
    _save(fig, "06_korelacja_z_outcome.png")


def plot_rate_by_group(df: pd.DataFrame, group_col: str, title: str, fname: str) -> None:
    table = rate_by_group(df, group_col)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(table.index.astype(str), table["odsetek_chorych"], color="#C44E52")
    ax.bar_label(
        bars,
        labels=[f"{r:.0f}%\n(n={n})" for r, n in zip(table["odsetek_chorych"], table["liczba_pacjentek"])],
        padding=3, fontsize=9,
    )
    ax.set_ylabel("Odsetek chorych [%]")
    ax.set_title(title)
    ax.set_ylim(0, max(table["odsetek_chorych"]) * 1.3)
    _save(fig, fname)


def plot_glucose_bmi_scatter(df: pd.DataFrame) -> None:
    data = _with_labels(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=data, x="Glucose", y="BMI", hue="Diagnoza",
        palette=PALETTE, alpha=0.65, ax=ax,
    )
    ax.set_xlabel(FEATURE_LABELS["Glucose"])
    ax.set_ylabel(FEATURE_LABELS["BMI"])
    ax.set_title("Glukoza a BMI – dwie najsilniejsze cechy")
    _save(fig, "09_glukoza_vs_bmi.png")


# ---------------------------------------------------------------- całość

def run_eda(clean: pd.DataFrame, report: pd.DataFrame) -> dict:
    """Generuje wszystkie tabele i wykresy EDA. Zwraca słownik tabel."""
    tables = {
        "statystyki_opisowe": descriptive_stats(clean),
        "srednie_wg_diagnozy": stats_by_outcome(clean),
        "rozklad_klas": class_balance(clean),
        "chorobowosc_wg_wieku": rate_by_group(clean, "AgeGroup"),
        "chorobowosc_wg_bmi": rate_by_group(clean, "BMI_category"),
        "korelacja_z_outcome": correlations(clean).to_frame("korelacja"),
        "raport_jakosci": report,
    }
    for name, table in tables.items():
        table.to_csv(TABLES_DIR / f"{name}.csv")

    plot_missing(report)
    plot_class_balance(clean)
    plot_histograms(clean)
    plot_boxplots(clean)
    plot_correlation_heatmap(clean)
    plot_target_correlation(clean)
    plot_rate_by_group(clean, "AgeGroup", "Odsetek chorych wg grupy wiekowej", "07_chorobowosc_wiek.png")
    plot_rate_by_group(clean, "BMI_category", "Odsetek chorych wg kategorii BMI (WHO)", "08_chorobowosc_bmi.png")
    plot_glucose_bmi_scatter(clean)
    return tables


if __name__ == "__main__":
    from config import ensure_dirs
    from data_prep import prepare_data

    ensure_dirs()
    _, _, clean, report = prepare_data()
    for name, table in run_eda(clean, report).items():
        print(f"\n=== {name} ===\n{table}")
