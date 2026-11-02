# Analiza danych medycznych – cukrzyca

Projekt zaliczeniowy z przedmiotu **Systemy Sztucznej Inteligencji – projekt** (temat nr 4).

**Autor:** Dmytro Ilchenko
**Typ problemu:** klasyfikacja binarna
**Technologie:** Python, Pandas, scikit-learn, Matplotlib, Seaborn

## Cel projektu

Celem projektu jest analiza danych pacjentek ze zbioru *Pima Indians Diabetes* oraz zbudowanie
prostego modelu predykcyjnego, który na podstawie wyników badań przewiduje, czy pacjentka choruje
na cukrzycę. Projekt obejmuje pełny proces analizy danych: czyszczenie, analizę eksploracyjną,
modelowanie i interpretację wyników.

## Zbiór danych

| Kolumna | Opis | Jednostka |
|---|---|---|
| `Pregnancies` | liczba przebytych ciąż | – |
| `Glucose` | stężenie glukozy po 2 h w doustnym teście obciążenia glukozą | mg/dl |
| `BloodPressure` | ciśnienie rozkurczowe krwi | mm Hg |
| `SkinThickness` | grubość fałdu skórnego na tricepsie | mm |
| `Insulin` | stężenie insuliny w surowicy po 2 h | µU/ml |
| `BMI` | wskaźnik masy ciała | kg/m² |
| `DiabetesPedigreeFunction` | wskaźnik obciążenia genetycznego (historia rodzinna) | – |
| `Age` | wiek | lata |
| `Outcome` | **zmienna docelowa**: 1 – cukrzyca, 0 – brak | – |

* Liczba rekordów: 768 (500 zdrowych, 268 chorych)
* Źródło: National Institute of Diabetes and Digestive and Kidney Diseases; zbiór publicznie
  dostępny (UCI Machine Learning Repository / Kaggle). Plik: `data/diabetes.csv`.

## Struktura projektu

```
AI Project/
├── data/
│   └── diabetes.csv            # surowe dane
├── src/
│   ├── config.py               # ścieżki, stałe, nazwy kolumn
│   ├── data_prep.py            # etap 1: wczytanie i czyszczenie danych
│   ├── eda.py                  # etap 2: statystyki opisowe i wykresy
│   ├── model.py                # etap 3: modele ML i ich ocena
│   └── main.py                 # uruchamia cały proces
├── notebooks/
│   └── analysis.ipynb          # cała analiza krok po kroku z komentarzami
├── outputs/                    # wyniki generowane automatycznie
│   ├── figures/                # wykresy PNG (01–13)
│   ├── tables/                 # tabele CSV
│   ├── diabetes_clean.csv      # dane po czyszczeniu
│   └── metrics.json            # metryki modeli
├── requirements.txt
└── README.md
```

## Uruchomienie

Wymagany Python 3.10 lub nowszy.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# cały proces (dane -> EDA -> modele), wyniki trafiają do outputs/
python src/main.py

# lub interaktywnie w notatniku
jupyter notebook notebooks/analysis.ipynb
```

Każdy moduł można też uruchomić osobno, np. `python src/eda.py`.

## Etapy pracy

### 1. Przygotowanie i czyszczenie danych (`data_prep.py`)

* kontrola typów danych i duplikatów (`dtypes`, `duplicated()`),
* wykrycie **ukrytych braków danych**: w kolumnach `Glucose`, `BloodPressure`, `SkinThickness`,
  `Insulin` i `BMI` wartość 0 jest fizjologicznie niemożliwa, więc zamieniamy ją na `NaN`
  (najwięcej braków: Insulin 48,7%, SkinThickness 29,6%),
* uzupełnienie braków medianą (`fillna()`),
* nowe cechy kategoryczne: grupy wiekowe (`pd.cut()`) i kategorie BMI wg WHO,
  dołączone ze słownika za pomocą `merge()`.

### 2. Analiza eksploracyjna (`eda.py`)

* statystyki opisowe (`describe()`), rozkład klas (`value_counts()`),
* średnie cech w podziale na diagnozę (`groupby()`),
* korelacje (`corr()`), mapa cieplna,
* histogramy i wykresy pudełkowe w podziale na diagnozę,
* odsetek chorych wg grup wiekowych i kategorii BMI.

### 3. Model ML (`model.py`)

Wszystkie modele działają w jednakowym `Pipeline`:
`SimpleImputer(median)` → `StandardScaler` → klasyfikator. Imputer i skaler są dopasowywane tylko
na zbiorze treningowym, co zapobiega wyciekowi informacji ze zbioru testowego.

| Model | Rola |
|---|---|
| Dummy (klasa większościowa) | punkt odniesienia |
| Regresja logistyczna | model główny, interpretowalny |
| Las losowy | model nieliniowy dla porównania |

Podział danych: 80/20 ze stratyfikacją (`random_state=42`), 5-krotna walidacja krzyżowa na zbiorze
treningowym. Metryki: accuracy, precision, recall, F1, ROC-AUC, macierz pomyłek, krzywa ROC,
ważność cech oraz analiza progu decyzyjnego.

## Wyniki

| Model | CV ROC-AUC | Test accuracy | Test precision | Test recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Dummy | 0,500 | 0,649 | 0,000 | 0,000 | 0,000 | 0,500 |
| Regresja logistyczna | 0,843 ± 0,019 | 0,708 | 0,600 | 0,500 | 0,545 | 0,813 |
| Las losowy | 0,837 ± 0,021 | 0,740 | 0,667 | 0,519 | 0,583 | 0,811 |

Analiza progu (regresja logistyczna): obniżenie progu z 0,5 do 0,3 zwiększa recall z 0,50 do 0,80
(wykrytych 43 z 54 chorych zamiast 27), przy praktycznie niezmienionej precision (0,600 → 0,597)
i wzroście liczby fałszywych alarmów (18 → 29).

## Wnioski

1. Zbiór zawiera dużo ukrytych braków zapisanych jako 0 – ich obsługa jest kluczowa dla rzetelnej analizy.
2. Najsilniejszym czynnikiem jest **poziom glukozy** (korelacja 0,49 z Outcome), następnie **BMI**,
   **wiek** i **liczba ciąż**.
3. Odsetek chorych rośnie z kategorią BMI (6,9% przy normie → 45,8% przy otyłości)
   oraz z wiekiem (21,6% w grupie 21–30 → ok. 57% w grupach 41–60).
4. Regresja logistyczna i las losowy dają porównywalne wyniki (ROC-AUC ≈ 0,81), dlatego prostszy
   i łatwiejszy do interpretacji model logistyczny jest wystarczający.
5. W zastosowaniu przesiewowym warto obniżyć próg decyzyjny, aby wykrywać więcej chorych.

**Ograniczenia:** mały zbiór (768 rekordów), jedna populacja (kobiety pochodzenia Pima, wiek ≥ 21),
duża liczba braków w kolumnie `Insulin`. Model ma charakter edukacyjny i nie zastępuje diagnozy lekarskiej.
