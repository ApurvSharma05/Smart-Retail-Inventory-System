"""
api/app.py - REST API for the Retail Inventory Management System (Stage 11).

Run (repository root):  python api/app.py          -> http://127.0.0.1:5000
Endpoints:
  GET /api/health
  GET /api/forecast?product_id=SKU0001&season=Festive_Diwali
  GET /api/reorder-alerts?category=Grocery&priority=Critical,High&season=Winter
  GET /api/suppliers?tier=High
Unit costs are the masked Cost Index (median = 100); no real prices are exposed.
"""
import sys
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alerts import current_alerts, supplier_scorecard  # noqa: E402
from forecasting import CATEGORIES, MODEL_FILE, SEASONS, forecast_daily, load_data  # noqa: E402

app = Flask(__name__)
DATA = load_data()
BUNDLE = joblib.load(MODEL_FILE)
PRIORITIES = ["Critical", "High", "Medium", "OK"]


def error(message, status=400):
    return jsonify({"error": message}), status


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model": BUNDLE["name"], "records": len(DATA),
                    "skus": int(DATA["product_id"].nunique()), "test_metrics": BUNDLE["metrics"]})


@app.get("/api/forecast")
def forecast():
    sku, season = request.args.get("product_id"), request.args.get("season", "Winter")
    if not sku:
        return error("query parameter product_id is required")
    if season not in SEASONS:
        return error(f"season must be one of {SEASONS}")
    if sku not in BUNDLE["sku_state"]:
        return error(f"unknown product_id {sku!r}", 404)
    daily = forecast_daily(BUNDLE, sku, season)
    return jsonify({"product_id": sku, "season": season, "forecast_daily_units": round(daily, 2),
                    "model": BUNDLE["name"]})


@app.get("/api/reorder-alerts")
def reorder_alerts_endpoint():
    category, season = request.args.get("category"), request.args.get("season")
    priority = [p for p in request.args.get("priority", "Critical,High,Medium").split(",") if p]
    if category and category not in CATEGORIES:
        return error(f"category must be one of {CATEGORIES}")
    if season and season not in SEASONS:
        return error(f"season must be one of {SEASONS}")
    if any(p not in PRIORITIES for p in priority):
        return error(f"priority values must come from {PRIORITIES}")
    alerts = current_alerts(DATA, BUNDLE, season)
    alerts = alerts[alerts["priority"].isin(priority)]
    if category:
        alerts = alerts[alerts["category"] == category]
    cols = ["product_id", "category", "supplier_id", "stock_qty", "reorder_level", "lead_days",
            "forecast_daily", "reorder_point", "days_of_cover", "suggested_order_qty", "priority"]
    rank = {p: i for i, p in enumerate(PRIORITIES)}
    alerts = alerts.sort_values(["priority", "days_of_cover"], key=lambda s: s.map(rank) if s.name == "priority" else s)
    return jsonify({"count": len(alerts), "alerts": alerts[cols].round(1).to_dict("records")})


@app.get("/api/suppliers")
def suppliers():
    tier = request.args.get("tier")
    if tier and tier not in ("Low", "Medium", "High"):
        return error("tier must be Low, Medium or High")
    sup = supplier_scorecard(DATA)
    if tier:
        sup = sup[sup["risk_tier"] == tier]
    return jsonify({"count": len(sup), "suppliers": sup.to_dict("records")})


if __name__ == "__main__":
    app.run(debug=False)
