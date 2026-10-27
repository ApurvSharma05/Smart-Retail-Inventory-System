"""
forecasting.py - shared feature engineering, metrics and forecast helpers (Stage 7-9).

Design rules (project guide, Stage 7):
  * every feature must be known at the moment of prediction -> all history features use only
    EARLIER records of the same SKU (shift(1) before any rolling / expanding statistic);
  * split by date, never shuffle;
  * the seasonal factors are a documented planning assumption (see docs/data_source.md).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "dataset.csv"
MODEL_FILE = ROOT / "models" / "forecast.joblib"

# Retail-calendar uplift used when the data were prepared (see data_dictionary.csv / docs/data_source.md).
SEASON_MULT = {"Winter": 1.00, "Summer": 1.20, "Monsoon": 0.95, "Festive_Diwali": 1.40}
SEASONS = list(SEASON_MULT)
CATEGORIES = ["Apparel", "Electronics", "Grocery", "HomeDecor"]
DELAY_DAYS = 21          # lead time above this is a supplier delay
SAFETY_BUFFER = 1.2      # reorder point = forecast demand over lead time x 1.2 (assumption, 20% buffer)

# Chronological split: everything before the cutoff trains/validates, the last quarter is the test set.
TEST_START = pd.Timestamp("2024-10-01")

MODEL_FEATURES = ["seas_level", "level", "mult", "seas_lag", "reorder_level", "lead_days"]


def load_data(path=DATA_FILE) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Chronological, leakage-free demand features per SKU.

    level      : expanding mean of de-seasonalised sales of EARLIER records of the SKU
    seas_level : level re-seasonalised for the record's own season (the seasonal-level rule forecast)
    lag_1      : previous recorded daily_sales of the SKU (naive forecast)
    seas_lag   : previous record, de-seasonalised then re-seasonalised (seasonal naive forecast)
    prior_mean : plain mean of earlier sales of the SKU (no seasonal adjustment)
    """
    d = df.sort_values("date", kind="stable").reset_index(drop=True).copy()
    d["mult"] = d["season"].map(SEASON_MULT)
    d["des"] = d["daily_sales"] / d["mult"]
    g = d.groupby("product_id")
    d["lag_1"] = g["daily_sales"].shift(1)
    d["prior_mean"] = g["daily_sales"].transform(lambda s: s.shift(1).expanding().mean())
    d["level"] = g["des"].transform(lambda s: s.shift(1).expanding().mean())
    d["lag1_des"] = g["des"].shift(1)
    d["seas_level"] = d["level"] * d["mult"]
    d["seas_lag"] = d["lag1_des"] * d["mult"]
    # a SKU's first record has no history -> cannot be forecast, so it is dropped (logged in train.py)
    return d.dropna(subset=["level"]).reset_index(drop=True)


def chronological_split(feat: pd.DataFrame):
    return feat[feat["date"] < TEST_START].copy(), feat[feat["date"] >= TEST_START].copy()


# ---------------------------------------------------------------------------------------------
# Metrics (Quick reference: MAE / RMSE / MAPE / WAPE against a naive forecast, R2 not alone)
# ---------------------------------------------------------------------------------------------
def wape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / y.sum())


def mape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p) / y))


def all_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res, ss_tot = ((y - p) ** 2).sum(), ((y - y.mean()) ** 2).sum()
    return {"MAE": float(np.abs(y - p).mean()), "RMSE": float(np.sqrt(((y - p) ** 2).mean())),
            "MAPE": mape(y, p), "WAPE": wape(y, p), "R2": float(1 - ss_res / ss_tot)}


# ---------------------------------------------------------------------------------------------
# Deployment helpers used by the API and the dashboard
# ---------------------------------------------------------------------------------------------
def sku_state(df: pd.DataFrame) -> dict:
    """Latest per-SKU demand state from ALL history: de-seasonalised level and last de-seasonalised sale."""
    d = df.sort_values("date", kind="stable").copy()
    d["des"] = d["daily_sales"] / d["season"].map(SEASON_MULT)
    g = d.groupby("product_id")["des"]
    return {sku: {"level": float(g.get_group(sku).mean()), "last_des": float(g.get_group(sku).iloc[-1]),
                  "n_records": int(len(g.get_group(sku)))} for sku in g.groups}


def forecast_daily(bundle: dict, product_id: str, season: str, reorder_level: float = 0.0,
                   lead_days: float = 0.0) -> float:
    """Forecast units/day for a SKU in a season using the saved champion model."""
    state = bundle["sku_state"].get(product_id)
    if state is None:
        raise KeyError(f"unknown product_id {product_id!r}")
    mult = SEASON_MULT[season]
    if bundle["kind"] == "rule":
        return state["level"] * mult
    row = pd.DataFrame([{"seas_level": state["level"] * mult, "level": state["level"], "mult": mult,
                         "seas_lag": state["last_des"] * mult, "reorder_level": reorder_level,
                         "lead_days": lead_days}])[MODEL_FEATURES]
    return float(max(bundle["model"].predict(row)[0], 0.0))
