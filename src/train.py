"""
train.py - baselines, tuned models, honest evaluation and model export (Stages 8-10).

Process (project guide):
  1. chronological split: train/validation = records before 2024-10-01, test = 2024-10-01 onwards;
  2. every candidate is scored by TimeSeriesSplit cross-validation on the TRAIN part only;
  3. the champion is the candidate with the lowest cross-validated WAPE (the test set plays no part);
  4. the test set is opened ONCE, at the end, to report all candidates and the Stage 1 criteria.

Usage (repository root):  python src/train.py
Writes: models/forecast.joblib, reports/experiments.csv, reports/results.json
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alerts import validate_alerts  # noqa: E402
from forecasting import (MODEL_FEATURES, MODEL_FILE, ROOT, SEASON_MULT, add_features,  # noqa: E402
                         all_metrics, chronological_split, load_data, sku_state, wape)

SEED = 42
N_SPLITS = 4
RULES = {  # name -> column of the feature table that IS the forecast
    "Naive (previous record)": "lag_1",
    "SKU mean (no seasonality)": "prior_mean",
    "Seasonal naive": "seas_lag",
    "Seasonal level rule": "seas_level",
}
GRIDS = {
    "Linear regression": (LinearRegression(), {}),
    "Gradient boosting": (GradientBoostingRegressor(random_state=SEED),
                          {"n_estimators": [100, 300], "max_depth": [2, 3], "learning_rate": [0.05, 0.1]}),
    "Random forest": (RandomForestRegressor(random_state=SEED, n_jobs=-1),
                      {"n_estimators": [300], "min_samples_leaf": [3, 10], "max_depth": [None, 6]}),
}


def cv_scores(train: pd.DataFrame, predict_fold) -> tuple[float, float]:
    """Mean WAPE / MAE over expanding-window folds of the training period."""
    w, m = [], []
    for tr_idx, va_idx in TimeSeriesSplit(N_SPLITS).split(train):
        va = train.iloc[va_idx]
        p = predict_fold(train.iloc[tr_idx], va)
        w.append(wape(va["daily_sales"], p))
        m.append(float(np.abs(va["daily_sales"].to_numpy() - p).mean()))
    return float(np.mean(w)), float(np.mean(m))


def main():
    raw = load_data()
    feat = add_features(raw)
    dropped = len(raw) - len(feat)
    train, test = chronological_split(feat)
    train = train.sort_values("date", kind="stable").reset_index(drop=True)
    print(f"records {len(raw):,} -> {len(feat):,} with history ({dropped} first-seen SKU records dropped)")
    print(f"train/validation {len(train):,} rows  < 2024-10-01   |   test {len(test):,} rows  >= 2024-10-01")

    rows, fitted = [], {}

    # --- rule baselines: nothing to fit, so cross-validation just scores them on the validation folds ---
    for name, col in RULES.items():
        cv_w, cv_m = cv_scores(train, lambda tr, va, c=col: va[c].to_numpy())
        rows.append({"model": name, "settings": f"forecast = {col}", "kind": "rule", "cv_wape": cv_w, "cv_mae": cv_m})

    # --- learned models, tuned on the training period only ---
    for name, (est, grid) in GRIDS.items():
        t0 = time.perf_counter()
        search = GridSearchCV(est, grid or {"fit_intercept": [True]}, cv=TimeSeriesSplit(N_SPLITS),
                              scoring="neg_mean_absolute_error", n_jobs=1)
        search.fit(train[MODEL_FEATURES], train["daily_sales"])
        best = search.best_estimator_
        cv_w, cv_m = cv_scores(train, lambda tr, va, b=best: b.__class__(**b.get_params())
                               .fit(tr[MODEL_FEATURES], tr["daily_sales"]).predict(va[MODEL_FEATURES]))
        fitted[name] = best
        rows.append({"model": name, "settings": json.dumps(search.best_params_) if grid else "defaults",
                     "kind": "model", "cv_wape": cv_w, "cv_mae": cv_m, "fit_seconds": time.perf_counter() - t0})

    table = pd.DataFrame(rows)
    champion = table.loc[table["cv_wape"].idxmin(), "model"]
    print("\nCross-validated candidates (train period only):")
    print(table[["model", "kind", "cv_wape", "cv_mae"]].round(3).to_string(index=False))
    print(f"\nChampion by CV WAPE: {champion}")

    # --- Stage 10: the test set is opened once ---
    y = test["daily_sales"].to_numpy()
    preds = {n: test[c].to_numpy() for n, c in RULES.items()}
    for n, m in fitted.items():
        preds[n] = m.predict(test[MODEL_FEATURES])
    for n, p in preds.items():
        for k, v in all_metrics(y, p).items():
            table.loc[table["model"] == n, f"test_{k.lower()}"] = v
    table["champion"] = table["model"] == champion
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "experiments.csv", index=False)
    print("\nTest-set results (opened once):")
    print(table[["model", "test_mae", "test_rmse", "test_mape", "test_wape", "test_r2"]].round(3).to_string(index=False))

    # error analysis for the champion
    champ_pred = preds[champion]
    err = test.assign(pred=champ_pred, abs_err=np.abs(y - champ_pred))
    by_season = err.groupby("season").apply(lambda g: wape(g["daily_sales"], g["pred"]), include_groups=False)
    by_cat = err.groupby("category").apply(lambda g: wape(g["daily_sales"], g["pred"]), include_groups=False)
    surge = err["daily_sales"] > 2.2 * err["seas_level"]
    errors = {"wape_by_season": by_season.round(4).to_dict(), "wape_by_category": by_cat.round(4).to_dict(),
              "surge_rows": int(surge.sum()),
              "wape_excluding_surges": wape(err.loc[~surge, "daily_sales"], err.loc[~surge, "pred"])}

    alert_results = validate_alerts(test, champ_pred)
    naive_wape = float(table.loc[table["model"] == "Naive (previous record)", "test_wape"].iloc[0])
    champ_row = table[table["model"] == champion].iloc[0]
    criteria = {
        "forecast_beats_naive": bool(champ_row["test_wape"] < naive_wape),
        "forecast_mape_le_20pct": bool(champ_row["test_mape"] <= 0.20),
        "alert_recall_ge_90pct": bool(alert_results["forecast_rule"]["recall"] >= 0.90),
    }
    results = {"champion": champion, "n_train": int(len(train)), "n_test": int(len(test)),
               "test_start": "2024-10-01", "champion_test": {k: float(champ_row[f"test_{k.lower()}"])
                                                              for k in ("MAE", "RMSE", "MAPE", "WAPE", "R2")},
               "naive_test_wape": naive_wape, "errors": errors, "alerts": alert_results, "criteria": criteria}
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print("\nChampion test metrics:", {k: round(v, 4) for k, v in results["champion_test"].items()})
    print("Error analysis:", json.dumps(errors, indent=1))
    print("Alert validation:", json.dumps(alert_results, indent=1))
    print("Stage 1 criteria:", criteria)

    # --- export: the champion is refit on ALL history for deployment (no test peeking happened) ---
    model = None
    if table.loc[table["model"] == champion, "kind"].iloc[0] == "model":
        model = fitted[champion].__class__(**fitted[champion].get_params())
        model.fit(feat[MODEL_FEATURES], feat["daily_sales"])
    bundle = {"name": champion, "kind": table.loc[table["model"] == champion, "kind"].iloc[0], "model": model,
              "features": MODEL_FEATURES, "season_mult": SEASON_MULT, "sku_state": sku_state(raw),
              "metrics": results["champion_test"]}
    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)
    print(f"\nSaved {MODEL_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
