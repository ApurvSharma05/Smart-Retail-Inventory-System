"""Normal, boundary and invalid cases for the REST API and the alert logic (Stage 11)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))

from alerts import reorder_alerts, simulate_stockout  # noqa: E402
from forecasting import add_features, load_data  # noqa: E402
from app import app  # noqa: E402


@pytest.fixture()
def client():
    return app.test_client()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json["status"] == "ok"


def test_forecast_normal_and_invalid(client):
    ok = client.get("/api/forecast?product_id=SKU0001&season=Festive_Diwali")
    assert ok.status_code == 200 and ok.json["forecast_daily_units"] > 0
    assert client.get("/api/forecast").status_code == 400
    assert client.get("/api/forecast?product_id=SKU0001&season=Spring").status_code == 400
    assert client.get("/api/forecast?product_id=NOPE").status_code == 404


def test_festive_forecast_higher_than_winter(client):
    w = client.get("/api/forecast?product_id=SKU0010&season=Winter").json["forecast_daily_units"]
    f = client.get("/api/forecast?product_id=SKU0010&season=Festive_Diwali").json["forecast_daily_units"]
    assert f > w


def test_alerts_filters(client):
    r = client.get("/api/reorder-alerts?priority=Critical,High")
    assert r.status_code == 200 and r.json["count"] == len(r.json["alerts"])
    assert all(a["priority"] in ("Critical", "High") for a in r.json["alerts"])
    g = client.get("/api/reorder-alerts?category=Grocery").json["alerts"]
    assert all(a["category"] == "Grocery" for a in g)
    assert client.get("/api/reorder-alerts?category=Toys").status_code == 400
    assert client.get("/api/reorder-alerts?priority=Urgent").status_code == 400


def test_suppliers(client):
    r = client.get("/api/suppliers")
    assert r.json["count"] == 30
    assert client.get("/api/suppliers?tier=Bad").status_code == 400


def test_reorder_boundary():
    inv = pd.DataFrame({"stock_qty": [0, 36, 37], "lead_days": [10, 10, 10], "daily_sales": [3, 3, 3]})
    out = reorder_alerts(inv, [3.0, 3.0, 3.0])   # reorder point = 3 x 10 x 1.2 = 36
    assert list(out["alert"]) == [True, True, False]
    assert out.loc[0, "priority"] == "Critical"
    assert (out["suggested_order_qty"] >= 0).all()


def test_no_leakage_in_features():
    """Features of a record use only EARLIER records of the same SKU."""
    raw = load_data().sort_values("date", kind="stable")
    feat = add_features(raw)
    sku = "SKU0001"
    sales = raw[raw["product_id"] == sku]["daily_sales"].tolist()
    rows = feat[feat["product_id"] == sku].sort_values("date")
    assert rows["lag_1"].tolist() == sales[:-1]                      # previous record only
    assert rows["prior_mean"].iloc[0] == sales[0]                    # mean of earlier records only
    assert np.isclose(rows["prior_mean"].iloc[1], np.mean(sales[:2]))
    assert len(rows) == len(sales) - 1                               # first record has no history


def test_stockout_simulation():
    inv = pd.DataFrame({"stock_qty": [100, 100], "daily_sales": [10, 11], "lead_days": [10, 10]})
    assert list(simulate_stockout(inv)) == [0, 1]
