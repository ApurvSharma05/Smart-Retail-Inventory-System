"""
prepare_data.py
Optimized Retail Inventory Management System - HCLTech Career Shaper Project No. 21

Loads dataset.csv (500 real rows), appends 4,500 realistic synthetic records to reach
5,000 rows across 500 SKUs (SKU0001-SKU0500) and 12 months of 2024, applies seasonal
demand multipliers, injects ~3% operational anomalies, masks supplier pricing and
writes the cleaned dataset.

Usage (from the repository root):  python src/prepare_data.py
Reads  data/raw/dataset.csv (never edited)  ->  writes data/processed/dataset.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
RNG = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "raw" / "dataset.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "dataset.csv"
COLUMNS = ["date", "product_id", "category", "season", "stock_qty", "reorder_level",
           "daily_sales", "unit_cost", "supplier_id", "lead_days"]

N_SKU = 500
N_SUPPLIERS = 30
TARGET_ROWS = 5000
CATEGORIES = ["Grocery", "Electronics", "Apparel", "HomeDecor"]
SEASON_MULT = {"Winter": 1.00, "Summer": 1.20, "Monsoon": 0.95, "Festive_Diwali": 1.40}
ANOMALY_RATE = 0.03  # share of ALL rows (150 of 5,000)


def season_for(ts: pd.Timestamp) -> str:
    """Indian retail calendar: Festive = 1 Oct - 15 Nov (Navratri -> Diwali 2024)."""
    m, d = ts.month, ts.day
    if (m == 10) or (m == 11 and d <= 15):
        return "Festive_Diwali"
    if m in (3, 4, 5, 6):
        return "Summer"
    if m in (7, 8, 9):
        return "Monsoon"
    return "Winter"  # 16 Nov - Feb


def load_original() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise SystemExit(f"dataset.csv is missing columns: {missing}")
    df = df[COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_sku_master() -> pd.DataFrame:
    """Each SKU gets a stable category, base demand, base cost and primary supplier."""
    skus = [f"SKU{i:04d}" for i in range(1, N_SKU + 1)]
    return pd.DataFrame({
        "product_id": skus,
        "category": RNG.choice(CATEGORIES, N_SKU),
        "base_demand": RNG.uniform(10, 50, N_SKU),
        "base_cost": RNG.uniform(100, 5000, N_SKU),
        "primary_supplier": [f"SUP{i:02d}" for i in RNG.integers(1, N_SUPPLIERS + 1, N_SKU)],
    })


def generate_synthetic(n_rows: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    master = build_sku_master()
    supplier_ids = [f"SUP{i:02d}" for i in range(1, N_SUPPLIERS + 1)]
    supplier_lead_mean = dict(zip(supplier_ids, RNG.uniform(6, 16, N_SUPPLIERS)))

    # every SKU appears exactly n_rows / N_SKU times (9x) in a shuffled order
    reps = n_rows // N_SKU
    sku_order = np.concatenate([RNG.permutation(master["product_id"].values) for _ in range(reps)])
    df = pd.DataFrame({"product_id": sku_order}).merge(master, on="product_id", how="left")

    # dates: spread evenly over the remaining days, sorted sequentially
    days = pd.date_range(start, end, freq="D")
    day_idx = np.sort(np.floor(np.linspace(0, len(days) - 1e-9, n_rows)).astype(int))
    df["day"] = days[day_idx]

    df["season"] = df["day"].apply(season_for)
    df["reorder_level"] = RNG.integers(20, 101, n_rows)

    # stock: ~18% of rows sit at/below the reorder point, the rest comfortably above
    low = RNG.random(n_rows) < 0.18
    stock_low = np.array([RNG.integers(1, r + 1) for r in df["reorder_level"]])
    stock_ok = np.array([RNG.integers(r + 1, 501) for r in df["reorder_level"]])
    df["stock_qty"] = np.where(low, stock_low, stock_ok)

    # demand = SKU base * season multiplier * noise
    mult = df["season"].map(SEASON_MULT)
    df["daily_sales"] = (df["base_demand"] * mult * RNG.normal(1.0, 0.10, n_rows)).round().clip(lower=1).astype(int)

    # cost: SKU base +/- 8% purchase-price variation
    df["unit_cost"] = (df["base_cost"] * RNG.normal(1.0, 0.08, n_rows)).round(2)

    # supplier: 85% primary, 15% alternate
    alt = RNG.random(n_rows) < 0.15
    df["supplier_id"] = np.where(alt, RNG.choice(supplier_ids, n_rows), df["primary_supplier"])
    lead = RNG.normal(df["supplier_id"].map(supplier_lead_mean), 2.5)
    df["lead_days"] = np.clip(np.round(lead), 3, 21).astype(int)

    df["date"] = df["day"]
    return df[COLUMNS]


def assign_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Sequential YYYY-MM-DD HH:MM:SS timestamps: 08:00 start, 30 minutes apart within a day."""
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    pos = df.groupby(df["date"].dt.normalize()).cumcount()
    df["date"] = df["date"].dt.normalize() + pd.Timedelta(hours=8) + pd.to_timedelta(pos * 30, unit="m")
    return df


def inject_anomalies(df: pd.DataFrame, protected: int) -> tuple[pd.DataFrame, dict]:
    """~3% of all rows. Only synthetic rows (index >= protected) are modified;
    your 500 original rows stay untouched."""
    n_anom = int(round(ANOMALY_RATE * len(df)))
    candidates = np.where(~df["is_original"].values)[0]
    chosen = RNG.choice(candidates, n_anom, replace=False)
    kinds = np.array_split(chosen, 3)
    stockout, surge, delay = kinds
    df.loc[stockout, "stock_qty"] = 0
    df.loc[surge, "daily_sales"] = (df.loc[surge, "daily_sales"] * RNG.uniform(2.5, 3.5, len(surge))).round().astype(int)
    df.loc[delay, "lead_days"] = RNG.integers(22, 36, len(delay))
    return df, {"stockouts": len(stockout), "demand_surges": len(surge), "supplier_delays": len(delay)}


def mask_unit_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Ethical compliance: real supplier prices must not leave the building.
    unit_cost is replaced by a relative Cost Index (median = 100) with +/-2% random
    perturbation, rounded to 1 decimal. The original currency values cannot be recovered."""
    median = df["unit_cost"].median()
    noise = RNG.normal(1.0, 0.02, len(df))
    df["unit_cost"] = (df["unit_cost"] / median * 100 * noise).round(1)
    return df


def main():
    original = load_original()
    original["is_original"] = True
    n_synth = TARGET_ROWS - len(original)

    syn_start = original["date"].max().normalize() + pd.Timedelta(days=1)
    synthetic = generate_synthetic(n_synth, syn_start, pd.Timestamp("2024-12-31"))
    synthetic["is_original"] = False

    df = pd.concat([original, synthetic], ignore_index=True)
    df["season"] = df["date"].apply(season_for)           # consistent season labels for all rows
    df = assign_timestamps(df)

    # --- cleaning -----------------------------------------------------------
    df = df.dropna(subset=COLUMNS).drop_duplicates(subset=COLUMNS).reset_index(drop=True)
    df["stock_qty"] = df["stock_qty"].clip(lower=0).astype(int)
    df["daily_sales"] = df["daily_sales"].clip(lower=0).astype(int)
    df["lead_days"] = df["lead_days"].clip(lower=1).astype(int)

    df, anomalies = inject_anomalies(df, protected=len(original))
    df = mask_unit_cost(df)

    out = df[COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(OUTPUT_FILE, index=False)

    # --- summary to paste into the report ------------------------------------
    print("=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Rows written          : {len(out):,}  -> {OUTPUT_FILE.relative_to(ROOT)}")
    print(f"Unique SKUs           : {out['product_id'].nunique()}")
    print(f"Unique suppliers      : {out['supplier_id'].nunique()}")
    print(f"Date range            : {out['date'].min()}  ->  {out['date'].max()}")
    print(f"Months covered        : {pd.to_datetime(out['date']).dt.month.nunique()}")
    print(f"Missing values        : {int(out.isna().sum().sum())}")
    print("\nRows per season:\n" + out["season"].value_counts().to_string())
    print("\nMean daily_sales per season (seasonal multipliers applied):")
    print(out.groupby("season")["daily_sales"].mean().round(2).to_string())
    print(f"\nInjected anomalies    : {sum(anomalies.values())} rows "
          f"({sum(anomalies.values()) / len(out):.1%} of data) -> {anomalies}")
    print(f"Rows with stock_qty=0 : {(out['stock_qty'] == 0).sum()} (incl. any in original data)")
    print(f"Rows lead_days > 21   : {(out['lead_days'] > 21).sum()}")
    print(f"Masked unit_cost      : Cost Index, median={out['unit_cost'].median():.1f}, "
          f"range {out['unit_cost'].min():.1f}-{out['unit_cost'].max():.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
