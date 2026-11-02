"""Etap 3: modele klasyfikacyjne i ocena wyników.

Porównujemy trzy modele:
  * Dummy (zawsze klasa większościowa) – punkt odniesienia,
  * regresja logistyczna – model główny (prosty i interpretowalny),
  * las losowy (Random Forest) – model nieliniowy dla porównania.

Każdy model działa w Pipeline: uzupełnienie braków medianą -> standaryzacja -> klasyfikator.
Imputer i skaler są dopasowywane wyłącznie na zbiorze treningowym.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_LABELS,
    FEATURES,
    FIGURES_DIR,
    METRICS_PATH,
    OUTCOME_LABELS,
    RANDOM_STATE,
    TABLES_DIR,
    TARGET,
    TEST_SIZE,
)

MODEL_NAMES = {
    "dummy": "Dummy (klasa większościowa)",
    "logreg": "Regresja logistyczna",
    "rf": "Las losowy",
}


def build_models() -> dict:
    def pipe(clf):
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

    return {
        "dummy": pipe(DummyClassifier(strategy="most_frequent")),
        "logreg": pipe(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "rf": pipe(
            RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=5,
                random_state=RANDOM_STATE, n_jobs=-1,
            )
        ),
    }


def split_data(df: pd.DataFrame):
    """Podział 80/20 ze stratyfikacją – zachowuje proporcję klas w obu zbiorach."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


# ---------------------------------------------------------------- wykresy

def plot_confusion_matrices(fitted: dict, X_test, y_test) -> None:
    keys = ["logreg", "rf"]
    fig, axes = plt.subplots(1, len(keys), figsize=(11, 4.5))
    for ax, key in zip(axes, keys):
        ConfusionMatrixDisplay.from_estimator(
            fitted[key], X_test, y_test, ax=ax, cmap="Blues", colorbar=False,
            display_labels=[OUTCOME_LABELS[0], OUTCOME_LABELS[1]],
        )
        ax.set_title(MODEL_NAMES[key])
        ax.set_xlabel("Przewidywana klasa")
        ax.set_ylabel("Rzeczywista klasa")
        ax.grid(False)
    fig.suptitle("Macierze pomyłek (zbiór testowy)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_macierze_pomylek.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(fitted: dict, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for key in ["logreg", "rf"]:
        RocCurveDisplay.from_estimator(fitted[key], X_test, y_test, ax=ax, name=MODEL_NAMES[key])
    ax.plot([0, 1], [0, 1], "k--", label="Klasyfikator losowy (AUC = 0,5)")
    ax.set_xlabel("Odsetek fałszywie pozytywnych (FPR)")
    ax.set_ylabel("Odsetek prawdziwie pozytywnych (TPR / recall)")
    ax.set_title("Krzywe ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_krzywe_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(fitted: dict) -> pd.DataFrame:
    coefs = fitted["logreg"].named_steps["clf"].coef_[0]
    importances = fitted["rf"].named_steps["clf"].feature_importances_
    table = pd.DataFrame(
        {"wspolczynnik_logreg": coefs, "waznosc_rf": importances}, index=FEATURES
    ).round(4)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lr = table["wspolczynnik_logreg"].sort_values()
    axes[0].barh(
        [FEATURE_LABELS[c] for c in lr.index], lr.values,
        color=["#DD5555" if v > 0 else "#4C72B0" for v in lr.values],
    )
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Regresja logistyczna – współczynniki\n(cechy standaryzowane)")
    axes[0].set_xlabel("Wpływ na log-szansę cukrzycy")

    rf = table["waznosc_rf"].sort_values()
    axes[1].barh([FEATURE_LABELS[c] for c in rf.index], rf.values, color="#55A868")
    axes[1].set_title("Las losowy – ważność cech\n(spadek nieczystości Gini)")
    axes[1].set_xlabel("Ważność")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "12_waznosc_cech.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table


def threshold_analysis(model, X_test, y_test) -> pd.DataFrame:
    """Jak zmieniają się precision i recall przy różnych progach decyzyjnych.

    W badaniach przesiewowych ważniejsze jest wykrycie chorych (wysoki recall)
    niż unikanie fałszywych alarmów, dlatego warto rozważyć próg niższy niż 0,5.
    """
    proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, proba)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(thresholds, precision[:-1], label="Precision", color="#4C72B0")
    ax.plot(thresholds, recall[:-1], label="Recall", color="#DD5555")
    ax.axvline(0.5, color="gray", linestyle="--", label="Próg domyślny 0,5")
    ax.axvline(0.3, color="black", linestyle=":", label="Próg 0,3")
    ax.set_xlabel("Próg prawdopodobieństwa")
    ax.set_ylabel("Wartość metryki")
    ax.set_title("Regresja logistyczna – wpływ progu decyzyjnego")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "13_analiza_progu.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for t in [0.3, 0.4, 0.5, 0.6]:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        rows.append(
            {
                "prog": t,
                "precision": round(precision_score(y_test, pred, zero_division=0), 3),
                "recall": round(recall_score(y_test, pred), 3),
                "f1": round(f1_score(y_test, pred), 3),
                "wykryte_chore (TP)": tp,
                "przeoczone_chore (FN)": fn,
                "falszywe_alarmy (FP)": fp,
            }
        )
    return pd.DataFrame(rows).set_index("prog")


# ---------------------------------------------------------------- całość

def run_models(df_with_nan: pd.DataFrame) -> dict:
    """Trenuje i ocenia modele. Zwraca słownik z wynikami."""
    X_train, X_test, y_train, y_test = split_data(df_with_nan)
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results, fitted = {}, {}
    for key, model in models.items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        model.fit(X_train, y_train)
        fitted[key] = model
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[key] = {
            "model": MODEL_NAMES[key],
            "cv_roc_auc_mean": round(cv_auc.mean(), 4),
            "cv_roc_auc_std": round(cv_auc.std(), 4),
            **{f"test_{k}": v for k, v in evaluate(y_test, y_pred, y_proba).items()},
        }

    metrics_table = pd.DataFrame(results).T.set_index("model")
    metrics_table.to_csv(TABLES_DIR / "metryki_modeli.csv")

    plot_confusion_matrices(fitted, X_test, y_test)
    plot_roc_curves(fitted, X_test, y_test)
    importance = plot_feature_importance(fitted)
    importance.to_csv(TABLES_DIR / "waznosc_cech.csv")
    thresholds = threshold_analysis(fitted["logreg"], X_test, y_test)
    thresholds.to_csv(TABLES_DIR / "analiza_progu.csv")

    summary = {
        "rozmiar_treningowy": int(len(X_train)),
        "rozmiar_testowy": int(len(X_test)),
        "modele": results,
        "analiza_progu_logreg": json.loads(thresholds.reset_index().to_json(orient="records")),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "metrics": metrics_table,
        "importance": importance,
        "thresholds": thresholds,
        "fitted": fitted,
        "split": (X_train, X_test, y_train, y_test),
    }


if __name__ == "__main__":
    from config import ensure_dirs
    from data_prep import prepare_data

    ensure_dirs()
    _, with_nan, _, _ = prepare_data()
    out = run_models(with_nan)
    print(out["metrics"])
    print(out["thresholds"])
