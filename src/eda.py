"""
eda.py - data profile (Stage 4) and exploratory analysis (Stage 6).

Writes 5 figures to reports/figures/, a data profile to reports/data_profile.txt and the written
insights (every figure has a takeaway tied to a decision) to reports/eda_insights.md.

Usage (repository root):  python src/eda.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecasting import DELAY_DAYS, ROOT, SEASONS, load_data  # noqa: E402

FIG = ROOT / "reports" / "figures"
PAL = {"Grocery": "#009E73", "Electronics": "#0072B2", "Apparel": "#E69F00", "HomeDecor": "#CC79A7"}
INK = "#222222"
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 200, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#DDDDDD",
                     "axes.axisbelow": True, "axes.titleweight": "bold"})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def profile(df: pd.DataFrame) -> str:
    lines = [f"shape: {df.shape}", "", "dtypes:", df.dtypes.to_string(), "",
             "missing share per column:", df.isna().mean().round(4).to_string(), "",
             f"duplicate rows: {df.duplicated().sum()}",
             f"negative stock_qty: {(df['stock_qty'] < 0).sum()}",
             f"zero stock_qty (stock-outs): {(df['stock_qty'] == 0).sum()}",
             f"zero or negative daily_sales: {(df['daily_sales'] <= 0).sum()}",
             f"lead_days <= 0: {(df['lead_days'] <= 0).sum()}",
             f"lead_days > {DELAY_DAYS} (supplier delay): {(df['lead_days'] > DELAY_DAYS).sum()}",
             f"stock already at/below reorder level: {(df['stock_qty'] <= df['reorder_level']).mean():.1%}",
             f"SKUs whose category changes between records: "
             f"{(df.groupby('product_id')['category'].nunique() > 1).sum()} of {df['product_id'].nunique()}",
             "", "summary statistics:", df.describe().T.round(2).to_string()]
    return "\n".join(lines)


def main():
    df = load_data()
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "data_profile.txt").write_text(profile(df), encoding="utf-8")
    df["days_of_cover"] = df["stock_qty"] / df["daily_sales"]
    df["alert_store_rule"] = df["stock_qty"] <= df["reorder_level"]
    ins = []

    # 1 - which categories hold the most / least stock cover?
    cover = df.groupby("category")["days_of_cover"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(cover.index, cover.values, color=[PAL[c] for c in cover.index])
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_xlabel("Average days of cover (stock_qty / daily_sales)")
    ax.set_title("Stock cover by category")
    save(fig, "fig1_cover_by_category.png")
    lo, hi = cover.index[0], cover.index[-1]
    ins.append(("Figure 1 - Stock cover by category",
                f"{hi} holds {cover[hi]:.1f} days of cover on average against {cover[lo]:.1f} for {lo} "
                f"({cover[hi] / cover[lo]:.2f}x).",
                (f"Review {hi} for overstock and prioritise replenishment checks for {lo}."
                 if cover[hi] / cover[lo] >= 1.25 else
                 "Cover is similar across categories (spread under 25%), so manage replenishment per SKU "
                 "rather than per category.")))

    # 2 - how are daily sales distributed, and what drives the spread?
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["daily_sales"], bins=40, color="#0072B2", edgecolor="white")
    axes[0].axvline(df["daily_sales"].median(), color="#D55E00", ls="--", label="median")
    axes[0].set(xlabel="Units sold per day", ylabel="Records", title="Distribution of daily sales")
    axes[0].legend(frameon=False)
    data = [df.loc[df["season"] == s, "daily_sales"] for s in SEASONS]
    axes[1].boxplot(data, tick_labels=[s.replace("_", "\n") for s in SEASONS])
    axes[1].set(ylabel="Units sold per day", title="Daily sales by season")
    save(fig, "fig2_sales_distribution.png")
    season_mean = df.groupby("season")["daily_sales"].mean()
    q99 = df["daily_sales"].quantile(0.99)
    ins.append(("Figure 2 - Daily sales distribution and seasonality",
                f"Median sales are {df['daily_sales'].median():.0f} units/day with a long right tail "
                f"(top 1% above {q99:.0f}). Festive_Diwali averages {season_mean['Festive_Diwali']:.1f} units/day "
                f"vs {season_mean['Winter']:.1f} in Winter ({season_mean['Festive_Diwali'] / season_mean['Winter'] - 1:+.0%}).",
                "Plan festive stock-ups ahead of 1 October and treat the extreme tail as demand surges, "
                "not as errors, when judging forecast accuracy."))

    # 3 - which suppliers are slowest?
    sup = df.groupby("supplier_id")["lead_days"].agg(["mean", lambda s: (s > DELAY_DAYS).mean() * 100])
    sup.columns = ["avg_lead", "delay_pct"]
    sup = sup.sort_values("avg_lead")
    fig, ax = plt.subplots(figsize=(12, 4.2))
    cut = sup["avg_lead"].quantile(0.8)
    ax.bar(sup.index, sup["avg_lead"], color=["#D55E00" if v >= cut else "#0072B2" for v in sup["avg_lead"]])
    ax.axhline(df["lead_days"].mean(), color=INK, ls="--", lw=1)
    ax.set_xticks(range(len(sup)), sup.index, rotation=90, fontsize=8)
    ax.set(ylabel="Average lead time (days)", title="Average lead time by supplier (red = slowest 20%)")
    save(fig, "fig3_supplier_lead_times.png")
    slow, fast = sup.index[-1], sup.index[0]
    ins.append(("Figure 3 - Supplier lead times",
                f"{slow} is the slowest supplier at {sup.loc[slow, 'avg_lead']:.1f} days on average; {fast} is the "
                f"fastest at {sup.loc[fast, 'avg_lead']:.1f} days. {(df['lead_days'] > DELAY_DAYS).mean():.1%} "
                f"of all records exceed {DELAY_DAYS} days.",
                f"Move urgent orders away from {slow} and open a delay review for suppliers in the red band."))

    # 4 - monthly demand trend by category
    monthly = df.groupby([df["date"].dt.month, "category"])["daily_sales"].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 4))
    for c in monthly.columns:
        ax.plot(monthly.index, monthly[c], marker="o", color=PAL[c], label=c)
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax.set(xlabel="Month (2024)", ylabel="Mean daily sales (units)", title="Monthly demand trend by category")
    ax.legend(frameon=False, ncol=4)
    save(fig, "fig4_monthly_trend.png")
    peak = monthly.mean(axis=1).idxmax()
    trough = monthly.mean(axis=1).idxmin()
    ins.append(("Figure 4 - Monthly demand trend",
                f"Demand across categories peaks in month {peak} ({monthly.mean(axis=1).max():.1f} units/day) and is "
                f"lowest in month {trough} ({monthly.mean(axis=1).min():.1f}); all categories move together.",
                "Seasonal calendar effects matter more than category differences, so forecasts must include season."))

    # 5 - how many records already trigger the store's reorder rule, and where?
    rate = df.groupby("category")["alert_store_rule"].mean().mul(100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(rate.index, rate.values, color=[PAL[c] for c in rate.index])
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set(ylabel="Records at/below reorder_level (%)", title="Store reorder-rule alert rate by category")
    save(fig, "fig5_alert_rate.png")
    overall = df["alert_store_rule"].mean() * 100
    covered_short = (df["days_of_cover"] < df["lead_days"]).mean() * 100
    ins.append(("Figure 5 - Store reorder rule vs real cover",
                f"Only {overall:.1f}% of records trigger the store's fixed reorder_level, yet {covered_short:.1f}% hold "
                f"less stock than they will sell during the supplier lead time (days of cover < lead_days).",
                "The fixed reorder_level under-warns. Use a forecast-based reorder point (demand over lead time "
                "x safety buffer); this is validated in Stage 10."))

    text = ["# EDA insights (Stage 6)", "",
            "Each figure is in `reports/figures/`; numbers are computed from `data/processed/dataset.csv` by `src/eda.py`.", ""]
    for title, finding, decision in ins:
        text += [f"## {title}", f"**Insight:** {finding}", "", f"**Decision for the store manager:** {decision}", ""]
    (ROOT / "reports" / "eda_insights.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
