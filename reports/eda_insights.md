# EDA insights (Stage 6)

Each figure is in `reports/figures/`; numbers are computed from `data/processed/dataset.csv` by `src/eda.py`.

## Figure 1 - Stock cover by category
**Insight:** Apparel holds 9.5 days of cover on average against 8.5 for Grocery (1.12x).

**Decision for the store manager:** Cover is similar across categories (spread under 25%), so manage replenishment per SKU rather than per category.

## Figure 2 - Daily sales distribution and seasonality
**Insight:** Median sales are 33 units/day with a long right tail (top 1% above 73). Festive_Diwali averages 43.1 units/day vs 31.3 in Winter (+38%).

**Decision for the store manager:** Plan festive stock-ups ahead of 1 October and treat the extreme tail as demand surges, not as errors, when judging forecast accuracy.

## Figure 3 - Supplier lead times
**Insight:** SUP18 is the slowest supplier at 15.3 days on average; SUP19 is the fastest at 8.0 days. 1.0% of all records exceed 21 days.

**Decision for the store manager:** Move urgent orders away from SUP18 and open a delay review for suppliers in the red band.

## Figure 4 - Monthly demand trend
**Insight:** Demand across categories peaks in month 10 (42.5 units/day) and is lowest in month 7 (29.0); all categories move together.

**Decision for the store manager:** Seasonal calendar effects matter more than category differences, so forecasts must include season.

## Figure 5 - Store reorder rule vs real cover
**Insight:** Only 18.1% of records trigger the store's fixed reorder_level, yet 68.0% hold less stock than they will sell during the supplier lead time (days of cover < lead_days).

**Decision for the store manager:** The fixed reorder_level under-warns. Use a forecast-based reorder point (demand over lead time x safety buffer); this is validated in Stage 10.
