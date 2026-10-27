"""
alerts.py - reorder-alert logic and its validation (Stage 7, 9, 10).

Forecast-based rule : alert when stock_qty <= forecast_daily x lead_days x SAFETY_BUFFER
Store baseline rule : alert when stock_qty <= reorder_level (the store's own fixed trigger)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from forecasting import DELAY_DAYS, SAFETY_BUFFER


def reorder_alerts(inv: pd.DataFrame, forecast_per_day) -> pd.DataFrame:
    """Return `inv` with forecast, reorder point, cover, alert flag, order quantity and priority."""
    out = inv.copy()
    out["forecast_daily"] = np.maximum(np.asarray(forecast_per_day, float), 0.1)
    out["demand_over_lead"] = out["forecast_daily"] * out["lead_days"]
    out["reorder_point"] = out["demand_over_lead"] * SAFETY_BUFFER
    out["days_of_cover"] = out["stock_qty"] / out["forecast_daily"]
    out["alert"] = out["stock_qty"] <= out["reorder_point"]
    # order enough to cover two reorder-point cycles (guide formula), never negative
    out["suggested_order_qty"] = np.ceil((out["reorder_point"] * 2 - out["stock_qty"]).clip(lower=0)).astype(int)
    out["priority"] = np.select(
        [out["stock_qty"] <= 0, out["days_of_cover"] < out["lead_days"] / 2, out["alert"]],
        ["Critical", "High", "Medium"], default="OK")
    return out


def simulate_stockout(inv: pd.DataFrame) -> pd.Series:
    """Ground truth for validation: a stock-out happens if actual sales over the supplier lead time
    (daily_sales x lead_days, constant rate assumed) exceed the stock on hand."""
    return (inv["daily_sales"] * inv["lead_days"] > inv["stock_qty"]).astype(int)


def validate_alerts(test: pd.DataFrame, forecast_per_day) -> dict:
    """Precision/recall of the forecast-based alert vs the store's reorder_level rule."""
    truth = simulate_stockout(test)
    forecast_flag = reorder_alerts(test, forecast_per_day)["alert"].astype(int)
    store_flag = (test["stock_qty"] <= test["reorder_level"]).astype(int)
    res = {"records": int(len(test)), "simulated_stockouts": int(truth.sum()),
           "stockout_rate": float(truth.mean())}
    for name, flag in (("forecast_rule", forecast_flag), ("store_reorder_level", store_flag)):
        res[name] = {"alerts_raised": int(flag.sum()),
                     "recall": float(recall_score(truth, flag)),
                     "precision": float(precision_score(truth, flag, zero_division=0))}
    return res


def supplier_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """Supplier comparison: lead time, delay rate, stock-out rate and a 0-100 risk score."""
    sup = df.groupby("supplier_id").agg(
        records=("product_id", "size"), skus=("product_id", "nunique"),
        avg_lead_days=("lead_days", "mean"), max_lead_days=("lead_days", "max"),
        delay_rate_pct=("lead_days", lambda s: (s > DELAY_DAYS).mean() * 100),
        stockout_rate_pct=("stock_qty", lambda s: (s == 0).mean() * 100),
        avg_daily_sales=("daily_sales", "mean"), avg_cost_index=("unit_cost", "mean")).reset_index()

    def norm(s):
        return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0

    sup["risk_score"] = (100 * (0.6 * norm(sup["avg_lead_days"]) + 0.25 * norm(sup["delay_rate_pct"])
                                + 0.15 * norm(sup["stockout_rate_pct"]))).round(1)
    sup["risk_tier"] = pd.cut(sup["risk_score"].rank(pct=True), [0, 1 / 3, 2 / 3, 1.0],
                              labels=["Low", "Medium", "High"], include_lowest=True).astype(str)
    return sup.round(2).sort_values("risk_score", ascending=False).reset_index(drop=True)


def current_alerts(df: pd.DataFrame, bundle: dict, season: str | None = None) -> pd.DataFrame:
    """Alerts for the latest record of every SKU (the current stock position), forecasting demand
    for `season` (default: the season of that SKU's latest record)."""
    from forecasting import forecast_daily
    latest = df.sort_values("date", kind="stable").groupby("product_id", as_index=False).tail(1).copy()
    fc = [forecast_daily(bundle, r.product_id, season or r.season, r.reorder_level, r.lead_days)
          for r in latest.itertuples()]
    return reorder_alerts(latest, fc)
