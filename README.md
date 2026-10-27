# Optimized Retail Inventory Management System

HCLTech Career Shaper Programme, Project No. 21 (individual project, Digital-Data-AI / AIML).

## 1. Overview

The system supports a store manager in deciding what to reorder, when to reorder it and from which supplier. It
provides demand forecasting per SKU, forecast-based reorder alerts, a supplier scorecard and a sales analytics
dashboard, exposed through a REST API and a Streamlit application. The project charter is in
`docs/project_charter.md`.

## 2. Data

| Item | Description |
|---|---|
| Source | `dataset.csv` supplied with the project brief (500 rows), extended with 4,500 generated rows |
| Result | `data/processed/dataset.csv`: 5,000 records, 500 SKUs, 30 suppliers, calendar year 2024 |
| Dictionary | `data/processed/data_dictionary.csv` |
| Assumptions and validation | `docs/data_source.md` |
| Cleaning decisions | `docs/cleaning_log.md` |

Supplier unit cost is stored as a masked Cost Index (median 100). The raw file contains real supplier prices; it is
excluded from version control (`.gitignore`) and must not be submitted.

## 3. Method

1. Data preparation and profiling (`src/prepare_data.py`, `src/eda.py`).
2. Exploratory analysis: five figures, each with an insight and a decision (`reports/eda_insights.md`).
3. Feature engineering: per-SKU history features computed from earlier records only (`src/forecasting.py`).
4. Baselines and model comparison with time-series cross-validation on the training period (`src/train.py`).
5. Alert logic and supplier scorecard (`src/alerts.py`).
6. REST API (`api/app.py`) and dashboard (`app/dashboard.py`).

The train/test split is chronological. Records before 2024-10-01 are used for training and validation; records from
2024-10-01 onward form the test set, which is evaluated once. The model is selected on cross-validated WAPE
computed on the training period only.

## 4. Results

Test period 2024-10-01 to 2024-12-31 (1,214 records).

| Model | MAE | MAPE | WAPE | CV WAPE |
|---|---|---|---|---|
| Naive (previous record) | 11.51 | 31.3% | 31.0% | 17.9% |
| SKU mean (no seasonality) | 7.68 | 20.0% | 20.7% | 21.7% |
| Seasonal naive (selected) | 5.52 | 14.2% | 14.9% | 14.4% |
| Seasonal level rule | 4.93 | 14.2% | 13.3% | 17.6% |
| Linear regression | 6.02 | 19.2% | 16.2% | 25.0% |
| Gradient boosting (tuned) | 7.29 | 24.4% | 19.6% | 20.6% |
| Random forest (tuned) | 8.32 | 30.0% | 22.4% | 19.4% |

The seasonal naive forecast was selected on cross-validated WAPE. The seasonal level rule has a lower test error, but
selecting it on that basis would use the test set for model choice, so it is reported without being adopted.

| Success criterion | Target | Result | Met |
|---|---|---|---|
| Forecast error against naive | Below naive; MAPE of 20% or less | WAPE 14.9% against 31.0%; MAPE 14.2% | Yes |
| Alerts detect simulated stock-outs | Recall of 90% or more | Recall 99.5%, precision 91.1% (store `reorder_level` rule: recall 24.1%) | Yes |
| Dashboard answers the three business questions | All three | Reorder alerts tab; stock cover by category (EDA figure 1); suppliers tab | Yes |

Full experiment results are in `reports/experiments.csv` and `reports/results.json`.

## 5. Installation and usage

Commands are shown for Windows; on macOS or Linux activate the environment with `source venv/bin/activate`.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

| Task | Command |
|---|---|
| Rebuild the processed dataset (needs `data/raw/dataset.csv`) | `python src/prepare_data.py` |
| Generate figures and insights | `python src/eda.py` |
| Run experiments and export the model | `python src/train.py` |
| Run tests | `python -m pytest tests -q` |
| Start the API (http://127.0.0.1:5000) | `python api/app.py` |
| Start the dashboard | `streamlit run app/dashboard.py` |

API endpoints:

* `GET /api/health`
* `GET /api/forecast?product_id=SKU0001&season=Festive_Diwali`
* `GET /api/reorder-alerts?category=Grocery&priority=Critical,High`
* `GET /api/suppliers?tier=High`

## 6. Deliverables

| Deliverable | Location |
|---|---|
| Working prototype | `api/app.py`, `app/dashboard.py` |
| Project report (23 pages) | `reports/Project_Report.docx`, `reports/Project_Report.pdf` (rebuilt with `python reports/build_report.py`) |
| EDA script with five figures and insights | `src/eda.py`, `reports/figures/`, `reports/eda_insights.md` |
| Cleaned dataset and data dictionary | `data/processed/` |
| Demo video (about 1 minute) | `docs/demo/demo.webm` |
| Dashboard screenshots | `reports/screenshots/` |
| Ethical Compliance Declaration (to be signed) | `docs/ethical_declaration.md` |

## 7. Ethical considerations

Supplier and product identifiers are pseudonymous codes, supplier pricing is masked, and the generated data and its
assumptions are documented. See `docs/ethical_declaration.md`.

## 8. Limitations and future work

* About 90% of the records are generated, so absolute figures illustrate the method and do not describe a real store.
* Each SKU has roughly ten records, which favours simple seasonal rules over machine-learning models. A longer daily
  history per SKU would allow a fairer comparison.
* The seasonal uplift factors are a planning assumption; they were not learned from a previous year.
* In the stock-out simulation (actual sales multiplied by lead days exceeding stock), 71% of test records are positive
  because generated stock levels are low relative to lead-time demand. Recall is therefore easier to reach than it
  would be on real data.
* The category of a SKU is inconsistent across the 500 supplied rows.
* Future work: per-SKU daily sales history, price and promotion effects, and a SQL data store.
