# Cleaning Log

The cleaned dataset is reproduced with `python src/prepare_data.py`. The output is identical on every run (seed 42).

| Step | Decision and reason | Rows affected |
|---|---|---|
| Parse `date` | Converted to datetime (`YYYY-MM-DD HH:MM:SS`); required for the chronological split | 5,000 |
| Missing values | Rows with missing values would be dropped; none were present | 0 |
| Duplicates | Exact duplicate rows would be dropped; none were present | 0 |
| Impossible values | `stock_qty` and `daily_sales` clipped at 0 and `lead_days` at 1; none required | 0 |
| Valid extremes | Stock-outs, demand surges and supplier delays retained as genuine operational events | 150 (3.0%) |
| Sensitive data | `unit_cost` converted to a Cost Index (median 100, plus or minus 2% noise) because supplier pricing is commercially sensitive | 5,000 |
| Final size | 5,000 rows, 500 SKUs, 30 suppliers | - |

During modelling, the first record of each SKU has no earlier history and cannot be forecast. These 500 records are
excluded inside the modelling step only (`src/forecasting.py`); they remain in the stored dataset.
