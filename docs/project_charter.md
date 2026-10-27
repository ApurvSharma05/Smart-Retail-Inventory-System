# Project Charter

**Project:** Optimized Retail Inventory Management System (Project No. 21)
**Domain:** Digital-Data-AI / AIML
**Format:** Individual project

## Problem statement

A retail store runs out of fast-selling products while slow-moving products tie up working capital. The store manager
must decide what to reorder, when and from which supplier. The current approach, a fixed `reorder_level` per product,
does not account for seasonal demand or supplier lead time.

**Target user:** store or inventory manager.

## Modules

| Brief requirement | Module | Location |
|---|---|---|
| Demand forecasting | Forecast of daily units per SKU and season | `src/forecasting.py`, `src/train.py` |
| Reorder alerts | Forecast-based reorder point and alert priority | `src/alerts.py` |
| Supplier management | Supplier scorecard (lead time, delay rate, risk tier) | `src/alerts.py` |
| Sales analytics dashboards | Streamlit dashboard | `app/dashboard.py` |
| REST API (listed stack) | Flask API | `api/app.py` |

Task types: forecasting (regression) and rule-based alerting.

## Success criteria

| Criterion | Target |
|---|---|
| Forecast error on the test period | WAPE below the naive forecast; MAPE of 20% or less |
| Alert detection of simulated stock-outs | Recall of 90% or more |
| Dashboard | Answers what to reorder, which category is overstocked and which supplier is slowest |

Outcomes are recorded in `reports/results.json` and in the README.

## Scope

In scope: the supplied dataset, generated history, forecasting, alert logic, REST API and dashboard.
Out of scope: payment and purchase-order approval workflow.

## Risks

* About 90% of the records are generated, so absolute metrics illustrate the method rather than real performance.
* Supplier pricing is commercially sensitive and is masked into a Cost Index.
