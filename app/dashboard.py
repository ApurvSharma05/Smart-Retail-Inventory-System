"""
app/dashboard.py - Streamlit dashboard for the Retail Inventory Management System (Stage 11).

Run (repository root):  streamlit run app/dashboard.py
Needs data/processed/dataset.csv (src/prepare_data.py) and models/forecast.joblib (src/train.py).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alerts import current_alerts, supplier_scorecard  # noqa: E402
from forecasting import (CATEGORIES, DATA_FILE, DELAY_DAYS, MODEL_FILE, SAFETY_BUFFER, SEASONS,  # noqa: E402
                         forecast_daily)

CAT_COLORS = {"Grocery": "#009E73", "Electronics": "#0072B2", "Apparel": "#E69F00", "HomeDecor": "#CC79A7"}
SEASON_COLORS = {"Winter": "#56B4E9", "Summer": "#E69F00", "Monsoon": "#009E73", "Festive_Diwali": "#D55E00"}
PRIORITY_COLORS = {"Critical": "#D55E00", "High": "#E69F00", "Medium": "#56B4E9", "OK": "#009E73"}

st.set_page_config(page_title="Retail Inventory Management", page_icon="📦", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["stock_value"] = df["stock_qty"] * df["unit_cost"]
    return df


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_FILE)


def plot(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")


try:
    data, bundle = load_data(), load_bundle()
except FileNotFoundError as exc:
    st.error(f"Missing file: {exc.filename}. Run `python src/prepare_data.py` and `python src/train.py` first.")
    st.stop()

results_file = ROOT / "reports" / "results.json"
results = json.loads(results_file.read_text()) if results_file.exists() else {}

# ------------------------------------------------------------------ sidebar filters
st.sidebar.title("📦 Filters")
sel_cat = st.sidebar.multiselect("Category", CATEGORIES, default=CATEGORIES)
sel_season = st.sidebar.multiselect("Season", SEASONS, default=SEASONS)
d_min, d_max = data["date"].min().date(), data["date"].max().date()
rng = st.sidebar.date_input("Date range", (d_min, d_max), min_value=d_min, max_value=d_max)
start, end = rng if isinstance(rng, tuple) and len(rng) == 2 else (d_min, d_max)
f = data[data["category"].isin(sel_cat) & data["season"].isin(sel_season)
         & (data["date"].dt.date >= start) & (data["date"].dt.date <= end)]
st.sidebar.caption(f"{len(f):,} of {len(data):,} records selected")
st.sidebar.info("Unit costs are a masked **Cost Index** (median = 100), not real prices.")

st.title("Optimized Retail Inventory Management System")
st.caption("Demand forecasting · reorder alerts · supplier management · sales analytics")
if f.empty:
    st.warning("No records match the current filters.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Sales analytics", "🚨 Reorder alerts", "🚚 Suppliers", "🤖 Demand forecast"])

# ------------------------------------------------------------------ tab 1: analytics
with tab1:
    c = st.columns(5)
    c[0].metric("Units sold", f"{f['daily_sales'].sum():,}")
    c[1].metric("Avg daily sales / record", f"{f['daily_sales'].mean():.1f}")
    c[2].metric("Stock value (cost index)", f"{f['stock_value'].sum() / 1e6:.2f}M")
    c[3].metric("Store-rule alert rate", f"{(f['stock_qty'] <= f['reorder_level']).mean() * 100:.1f}%")
    c[4].metric("Stock-outs", f"{(f['stock_qty'] == 0).sum():,}")
    monthly = f.groupby(["month", "category"], as_index=False)["daily_sales"].sum()
    plot(px.line(monthly, x="month", y="daily_sales", color="category", markers=True, color_discrete_map=CAT_COLORS,
                 title="Monthly units sold by category", labels={"daily_sales": "Units sold", "month": ""}))
    a, b = st.columns(2)
    with a:
        pv = f.groupby(["season", "category"], as_index=False)["daily_sales"].mean()
        plot(px.bar(pv, x="season", y="daily_sales", color="category", barmode="group", color_discrete_map=CAT_COLORS,
                    category_orders={"season": SEASONS}, title="Average daily sales: season x category",
                    labels={"daily_sales": "Avg daily sales", "season": ""}))
    with b:
        top = f.groupby("product_id")["daily_sales"].sum().nlargest(10).reset_index().sort_values("daily_sales")
        plot(px.bar(top, x="daily_sales", y="product_id", orientation="h", title="Top 10 SKUs by units sold",
                    labels={"daily_sales": "Units sold", "product_id": ""}, color_discrete_sequence=["#0072B2"]))

# ------------------------------------------------------------------ tab 2: alerts
with tab2:
    st.subheader("Forecast-based reorder alerts (current position of every SKU)")
    st.markdown(f"Alert when **stock ≤ forecast daily demand × lead days × {SAFETY_BUFFER}** "
                f"(a {int((SAFETY_BUFFER - 1) * 100)}% safety buffer, an assumption). "
                "Suggested order = 2 × reorder point − stock.")
    season_pick = st.selectbox("Forecast season", ["Season of latest record"] + SEASONS)
    al = current_alerts(data, bundle, None if season_pick.startswith("Season") else season_pick)
    al = al[al["category"].isin(sel_cat)]
    store_flag = (al["stock_qty"] <= al["reorder_level"]).sum()
    c = st.columns(4)
    c[0].metric("Items to reorder", f"{int(al['alert'].sum()):,}", f"{int(al['alert'].sum() - store_flag):+,} vs store rule")
    c[1].metric("Critical (stock-out)", f"{(al['priority'] == 'Critical').sum():,}")
    c[2].metric("High priority", f"{(al['priority'] == 'High').sum():,}")
    c[3].metric("Units to order", f"{int(al.loc[al['alert'], 'suggested_order_qty'].sum()):,}")
    pr = st.multiselect("Priority", ["Critical", "High", "Medium"], default=["Critical", "High", "Medium"])
    rank = {"Critical": 0, "High": 1, "Medium": 2, "OK": 3}
    view = al[al["priority"].isin(pr)].sort_values(
        ["priority", "days_of_cover"], key=lambda s: s.map(rank) if s.name == "priority" else s)
    cols = ["product_id", "category", "supplier_id", "stock_qty", "reorder_level", "lead_days", "forecast_daily",
            "reorder_point", "days_of_cover", "suggested_order_qty", "priority"]
    st.dataframe(view[cols].round(1), width="stretch", hide_index=True, height=380)
    st.download_button("⬇️ Download purchase list (CSV)", view[cols].round(1).to_csv(index=False).encode(),
                       "reorder_alerts.csv", "text/csv")
    if len(view):
        g = view.groupby(["category", "priority"], as_index=False).size()
        plot(px.bar(g, x="category", y="size", color="priority", color_discrete_map=PRIORITY_COLORS,
                    title="Alerts by category and priority", labels={"size": "Items", "category": ""}))
    a = results.get("alerts")
    if a:
        st.caption(f"Validated on the held-out test period against simulated stock-outs: forecast rule recall "
                   f"{a['forecast_rule']['recall']:.1%} (precision {a['forecast_rule']['precision']:.1%}) vs store "
                   f"reorder_level recall {a['store_reorder_level']['recall']:.1%}.")

# ------------------------------------------------------------------ tab 3: suppliers
with tab3:
    st.subheader("Supplier performance")
    sup = supplier_scorecard(f)
    c = st.columns(4)
    c[0].metric("Active suppliers", len(sup))
    c[1].metric("Avg lead time", f"{f['lead_days'].mean():.1f} days")
    c[2].metric(f"Delayed (> {DELAY_DAYS} d)", f"{(f['lead_days'] > DELAY_DAYS).mean() * 100:.1f}%")
    slowest = sup.sort_values("avg_lead_days").iloc[-1]
    c[3].metric("Slowest supplier", slowest["supplier_id"], f"{slowest['avg_lead_days']:.1f} days", delta_color="off")
    tiers = {"Low": "#009E73", "Medium": "#E69F00", "High": "#D55E00"}
    a, b = st.columns(2)
    with a:
        plot(px.bar(sup.sort_values("avg_lead_days"), x="supplier_id", y="avg_lead_days", color="risk_tier",
                    color_discrete_map=tiers, title="Average lead time by supplier",
                    labels={"avg_lead_days": "Avg lead days", "supplier_id": ""}))
    with b:
        plot(px.scatter(sup, x="avg_lead_days", y="avg_cost_index", size="records", color="risk_tier",
                        hover_name="supplier_id", color_discrete_map=tiers, title="Lead time vs cost index",
                        labels={"avg_lead_days": "Avg lead days", "avg_cost_index": "Avg cost index"}))
    st.dataframe(sup, width="stretch", hide_index=True, height=380)
    st.caption("Risk score (0-100) = 60% avg lead time + 25% delay rate + 15% stock-out rate (min-max normalised); "
               "tiers are terciles. Supplier IDs are pseudonymous and costs are masked.")

# ------------------------------------------------------------------ tab 4: forecast
with tab4:
    st.subheader("Demand forecast")
    m = bundle["metrics"]
    st.caption(f"Champion model: **{bundle['name']}** (chosen by time-series cross-validation, evaluated once on "
               f"records from 2024-10-01): WAPE {m['WAPE']:.1%} · MAPE {m['MAPE']:.1%} · MAE {m['MAE']:.2f} units/day"
               f" · naive forecast WAPE {results.get('naive_test_wape', float('nan')):.1%}")
    left, right = st.columns([1, 1.3])
    with left:
        sku = st.selectbox("SKU", sorted(data["product_id"].unique()))
        hist = data[data["product_id"] == sku].sort_values("date")
        last = hist.iloc[-1]
        season = st.selectbox("Season to forecast", SEASONS, index=SEASONS.index(last["season"]))
        stock = st.number_input("Current stock (units)", 0, 5000, int(last["stock_qty"]), step=10)
        lead = st.slider("Supplier lead days", 1, 35, int(last["lead_days"]))
    daily = forecast_daily(bundle, sku, season, float(last["reorder_level"]), float(lead))
    need, cover = daily * lead, stock / max(daily, 0.1)
    point = need * SAFETY_BUFFER
    with right:
        c = st.columns(3)
        c[0].metric("Forecast daily demand", f"{daily:.1f} units")
        c[1].metric("Demand over lead time", f"{need:.0f} units")
        c[2].metric("Days of cover", f"{cover:.1f} d")
        if stock <= point:
            st.error(f"🚨 Reorder now: stock {stock} is at or below the reorder point {point:.0f}. "
                     f"Suggested order: **{int(np.ceil(max(point * 2 - stock, 0)))} units**.")
        else:
            st.success(f"✅ Stock {stock} is above the reorder point {point:.0f} ({cover:.1f} days of cover "
                       f"vs {lead}-day lead time).")
        by_season = pd.DataFrame({"season": SEASONS, "units_per_day": [forecast_daily(bundle, sku, s_) for s_ in SEASONS]})
        plot(px.bar(by_season, x="season", y="units_per_day", color="season", color_discrete_map=SEASON_COLORS,
                    title=f"{sku}: forecast daily sales by season", labels={"units_per_day": "Units/day", "season": ""}
                    ).update_layout(showlegend=False))
    exp_file = ROOT / "reports" / "experiments.csv"
    if exp_file.exists():
        st.markdown("**Experiment table** (cross-validated on the training period, test opened once)")
        exp = pd.read_csv(exp_file)
        st.dataframe(exp[["model", "cv_wape", "test_mae", "test_mape", "test_wape", "test_r2", "champion"]].round(3),
                     width="stretch", hide_index=True)
