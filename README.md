# Grocery Demand Forecasting (4 cities, festival-aware, BigBasket catalog)

Forecasts daily demand for a grocery catalog in Mumbai, Delhi, Bengaluru and
Chennai, flags products at risk of stocking out or overstocking, and shows
it all in a small web dashboard. Indian festivals (Diwali, Holi, Ganesh
Chaturthi, ...) drive large demand spikes, so the calendar is a first-class
input.

The product catalog is real (57,013 BigBasket products). **Sales history and
stock levels are synthetic** — see [Limitations](#limitations).

## Architecture

One pipeline, run by `run_pipeline.py`:

```
BigBasket catalog (data/archive_products.csv)
  └─ sample_products.py      344 tracked products, 43 per model category (seeded)
      └─ generate_synthetic_data.py   5 years of daily POS history per product x city
          └─ build_features.py        calendar, product and leak-free lag features
              └─ train_model.py       LightGBM per segment, CV + held-out test, risk report
                  └─ forecast_regional.py   direct forecasts for tracked products,
                                            K-nearest analogs for the rest of the catalog
                      └─ build_bigbasket_regional_risk.py   stock simulation + risk rule
                          └─ dashboard.py   customer storefront + admin views
```

| Path | What it is |
|---|---|
| `src/config.py` | Paths, lead time, feature list, shelf lives, replenishment and price constants |
| `src/festival_calendar.py` | Festival dates 2021–2027, directional festival features, demand uplift curves |
| `src/sample_products.py` | Maps BigBasket categories to 8 model categories (`data/category_map.csv`), samples tracked products |
| `src/generate_synthetic_data.py` | Synthetic POS history driven by product price, brand popularity, pack size, city, weekday, festivals, promotions |
| `src/inventory.py` | Shared replenishment simulation and the stockout/overstock/reorder rule |
| `src/build_features.py` | Unconstrained demand target, features, categorical encodings |
| `src/metrics.py` | MAE, RMSE, WAPE, MASE, Bias% (MAPE kept as legacy) |
| `src/train_model.py` | Time-series CV, Optuna tuning, final-test evaluation, model risk report |
| `src/forecast_regional.py` | Catalog-wide forecasts per city; free samples, store packaging and sub-₹1 products listed in `data/bigbasket_not_forecast.csv` |
| `src/build_bigbasket_regional_risk.py` | Catalog-wide risk report per city |
| `src/compare_models.py` | Comparison study: baselines, XGBoost, CatBoost, LightGBM on the same folds |
| `src/alerts.py` | Per-city vendor email alerts from the model risk report (dry run by default) |
| `config/vendors.json` | Vendor name and email(s) per city, optional per-category overrides |
| `src/stockout_view.py` | Tables, filters and brand summary behind the admin "stocks out first" views |
| `src/product_detail.py` | Data for the admin product panel (`/api/product`): history, per-day forecast, festivals, stock, analogs |
| `dashboard.py` | Web dashboard (standard library HTTP server) |
| `tests/test_pipeline.py`, `tests/test_alerts.py`, `tests/test_forecast_regional.py`, `tests/test_compare_models.py`, `tests/test_dashboard_filters.py`, `tests/test_product_panel.py` | pytest checks |

Generated files go to `data/`, `models/` and `outputs/` (not tracked by git).

## Setup

Python 3.13 (3.12 also works).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, LightGBM needs OpenMP: `brew install libomp`.

The pipeline needs the catalog as `data/archive_products.csv`. If you only
have the original `archive.zip` of BigBasket Excel files, put it in `data/`
and `pip install openpyxl`; the first pipeline step converts it.

## Run

```bash
python run_pipeline.py   # ~5 minutes; prints per-step timings
python dashboard.py      # http://127.0.0.1:8000
python -m pytest         # after the pipeline has run
```

`run_pipeline.py` trains with the tuned parameters saved in
`models/params_<segment>.json` (`train_model.py --trials 0`). To re-tune, run
`python src/train_model.py --trials 120` (about 35 minutes).

Dashboard pages:

- `/customer` — storefront over the full catalog (search, categories, basket demo).
- `/admin` — per-city KPIs (stockout/overstock today, next festival, forecast
  WAPE), the model risk report with product names, regional tabs covering the
  whole catalog with each row's forecast source, and a browser for every
  generated dataset.

The dashboard has no authentication. Run it only on a trusted machine.

### Stockout views (admin)

Both stockout tables on `/admin` are "stocks out first" views: the model
risk report ("Stockout risk: reorder before stock runs out", the 344 tracked
products) and the regional "Stockout risk and recommended restock" (the whole
forecast catalog). Each row is one product in the selected city. Products
listed under several SKUs appear once (`unique_products()`).

- **Columns:** MRP; units sold in the last 7 and 28 days; average daily
  sales over 28 days; trend (last 28 days vs the 28 before, ▲/▼); days until
  stockout (= days of cover); next delivery; reorder qty; forecast source.
  Sales history covers the days before stock is counted: the latest
  test-window day for the model report, the first forecast day for the
  regional one. Analog catalog products have no history, so they show
  their forecast daily demand labelled "forecast (no history)" and no
  sales or trend.
- **Default order:** lowest days until stockout first, then the highest
  daily rate (sales, or the forecast without history). Cover, units sold,
  average daily sales, MRP, reorder qty and trend headers sort on click.
- **Filters (all combine):** brand (searchable, with product counts),
  category, MRP band (<₹50, ₹50–200, ₹200–500, >₹500) or custom min/max,
  top 10% / 25% sellers by units sold in 28 days (within the city), rising
  / falling trend, days until stockout ≤2 / ≤4 / ≤7, direct / analog,
  product-name search. "Clear filters" resets them. The count reads "N of M
  products".
- **By brand** tab: units sold (28d), revenue (units × MRP), products, at
  stockout risk, reorder units and average cover per brand, sorted by units
  sold. Clicking a brand filters the table to it. Products without history
  count as 28 × forecast daily and are marked `*`.
- **Download CSV** exports every filtered row in the current sort order.
- Filter state lives in the URL (`m_…` for the model report, `r_…` for the
  regional view, plus `city` and `rank`), so a view can be bookmarked.
  Switching city keeps the filters.
- "Highest demand for the upcoming occasion" can rank by units or by
  estimated revenue (forecast units × MRP, with the same occasion
  weighting); products without an MRP can't rank by revenue.

The model report's table holds every tracked product, not only the stockout
rows: `train_model.build_risk_report` is re-run on
`outputs/test_predictions.csv` for the latest day, so cover, status and
reorder qty match `outputs/stockout_risk_report.csv` exactly. All sales,
stock, cover, delivery and reorder figures come from synthetic or
simulated data, and the views say so.

The API is server-side and reads a merged table kept in memory (built in
the background at startup):

```
GET /api/stockout?scope=model|regional&city=<city or all>&brand=&category=&min_price=&max_price=
                 &top=10|25&trend=rising|falling&max_cover=&source=direct|analog&q=
                 &sort=cover|units_28d|avg_daily_sales|price|reorder_qty|trend&order=asc|desc
                 &limit=50&offset=0[&format=csv]          -> {total, rows, products_in_city, planning_day}
GET /api/stockout/brands?scope=&city=&<same filters>      -> {total, rows, categories}
```

An unknown city returns a 404 JSON error, and bad parameters return 400.

### Product detail panel (admin)

Click any product name in the risk tables (model report, regional
stockout and overstock, highest demand) to open a side panel for that
product in the selected city. Esc or × closes it. The panel is also in the
URL (`product`, `product_city`, `product_scope`), so it can be bookmarked.

The chart is inline SVG with no chart library, one y-axis in units per day.
Hovering shows a crosshair and the values for that day, and a "Data table"
toggle lists every value. "Today" is the last POS day the regional forecast
was made from.

- **Tracked products:** the last 60 days of units sold (solid) and true
  demand (dotted), stockout days shaded, then the next 7 days of forecast
  (dashed). The forecast is LightGBM per day,
  `forecast_regional.daily_forecasts`, the same values the regional forecast
  averages: their mean equals `forecast_daily`.
- **Analog products:** only the forecast. It is a 7-day average, so the line
  is flat. A table lists the 5 tracked products it was built from, with
  their normalised inverse-distance weights, price factors (0.5–2×) and
  their own forecasts. `forecast_regional.py` saves these as
  `analog_weights` / `analog_price_adj`, and the forecast values are
  unchanged.
- **On every chart:** festival dates from the festival calendar as labelled
  vertical markers, and the current opening stock as a horizontal line. If
  the stock is more than 3× the largest daily value, the line is drawn at
  the top and marked "off scale".
- **Beside the chart:** brand, category, MRP, stock, days of cover, next
  delivery, reorder qty, daily forecast and forecast source. These come from
  the table the product was clicked in: the model risk report (stock
  counted on the latest test day) or the regional report (stock simulated
  for the first forecast day).

Sales, true demand, stockout days and stock are labelled simulated, and
analog forecasts are labelled estimated. Series colours are the dataviz
reference slots 1–2 (blue `#2a78d6` for sales, orange `#eb6834` for demand
and forecast). They passed the palette validator on the white panel
(all-pairs CVD ΔE 24.7), and dash patterns tell the orange lines apart.

```
GET /api/product?city=<city>&sku_id=<sku>[&scope=regional|model]
    -> {city, sku_id, scope, today, details, stock, history, forecast, festivals, analogs, notes}
```

An unknown city or SKU returns a 404 JSON error.

## Email alerts

`src/alerts.py` emails each city's vendor the day's rows from the model risk
report (`outputs/stockout_risk_report.csv` / `overstock_risk_report.csv`) —
the same rows the admin page shows. It doesn't re-forecast or change the risk
rule.

Each email has the subject `[Stockout x | Overstock y] <City> inventory alert – <date>`,
summary counts, a STOCKOUT table sorted by days of cover (top 25), an
OVERSTOCK table (top 15), and every row of the city's list attached as CSV.

```bash
python -m src.alerts --city all                      # dry run: outputs/alerts/<date>_<city>.html/.csv
python -m src.alerts --city Bengaluru --date 2026-08-20
python -m src.alerts --city Bengaluru --send         # real email
python run_pipeline.py --alerts                      # pipeline, then a dry run for all cities
```

- **Dry run is the default.** Only `--send` (or "Email vendor" on the admin
  page, after a confirmation) sends mail. The date defaults to the report's
  latest day. Cities with no risk rows are skipped unless `--always` is given.
- **SMTP:** copy `.env.example` to `.env` and fill it in. Mail goes out over
  STARTTLS; for Gmail, use an App Password. Variables already set in the
  environment take priority over `.env`. The password is never printed or
  logged.
- **Log:** every real send, successful or failed, is appended to
  `outputs/alerts/sent_log.csv` (timestamp, city, vendor, recipients, counts,
  status, error).
- **Vendors:** `config/vendors.json` has one entry per city. `vendor_email` can
  be a string or a list. `category_overrides` sends some model categories'
  rows to another vendor in a separate email:
  `"category_overrides": {"dairy": {"vendor_name": "Dairy Co", "vendor_email": "dairy@example.com"}}`.
- **Dashboard:** on `/admin`, "Preview alert" and "Email vendor" act on the
  selected city tab (`POST /api/send-alert?city=<city>&send=0|1`). The
  endpoint only accepts requests from the dashboard page itself (custom
  header + `127.0.0.1`/`localhost` Host), so other websites can't trigger a
  send.

## Results

Synthetic data, so these show the pipeline works, not real-world accuracy.
The generator adds ±12% day-to-day noise, which alone puts the best possible
WAPE near 9.6%.

**Final test** (2026-04-01 → 2026-08-25, never used for tuning), WAPE %:

| Segment | Overall | Festival weeks | Stockout days | vs `true_demand` (overall / festival) | MASE | Bias |
|---|---|---|---|---|---|---|
| Perishable (dairy, sweets) | 10.14 | 11.29 | 18.75 | 10.08 / 11.23 | 0.24 | +0.5% |
| Non-perishable (6 categories) | 10.25 | 10.95 | 11.12 | 10.19 / 10.91 | 0.34 | −0.4% |

**Cross-validation WAPE %, baseline (original Poisson, 400 trees) → tuned.**
Fold 3 covers Navratri, Dussehra and Diwali and is the headline festival metric.

| Segment | F1 (no festivals) | F2 (Ganesh Chaturthi) | **F3 (Navratri–Diwali)** | F4 (Sankranti, Holi) | Mean |
|---|---|---|---|---|---|
| Perishable | 10.18 → 9.97 | 10.96 → 10.76 | **10.17 → 9.93** | 10.66 → 10.27 | 10.49 → 10.23 |
| Non-perishable | 10.34 → 10.17 | 10.92 → 10.69 | **10.46 → 10.34** | 10.85 → 10.55 | 10.64 → 10.44 |

- WAPE by category on the final test: 10.0–10.5% for all 8 categories.
- `true_demand` is the synthetic demand before the stock cap; it exists only
  because the data is synthetic.
- Coverage: 56,214 of 57,013 catalog products (98.6%) get a forecast in every
  city — 344 directly from the models, the rest by analogs. 762 products fit
  no model category, 6 are non-food items in food categories, and 31 are
  "not forecast", each listed with its reason in
  `data/bigbasket_not_forecast.csv`: 1 free sample (by name), 28 store
  packaging supplies ("MCIP Consumables" boxes, bags, labels, tape — by name,
  household only) and 2 products with an MRP below ₹1 (both ₹0). Cheap real
  products such as ₹3.75 biscuits and ₹3 shampoo sachets are forecast. The
  admin regional panel shows the count with a link to the list.
- One row per product in the dashboard's regional lists: the catalog lists
  some products under several SKUs (about 11% of SKUs), so "Highest demand
  for the upcoming occasion" keeps one row per normalised name + pack size —
  the directly forecast SKU if there is one, else the lowest SKU id. SKU
  counts and the catalog file are unchanged.
- Risk split: about 97.5% of tracked product-days are "normal" over the test
  window; 99.8% of catalog products are "normal" for the coming week.

`outputs/metrics.json` has every number above plus MAE, RMSE and the tuned
parameters.

## Model comparison

`src/compare_models.py` is a separate study that checks whether LightGBM is
the right choice. It never touches `models/` or the production outputs. For
each segment it scores these models on the same features, the same
`demand` target, the same 4 CV folds (7-day gap) and the same final test
window as `train_model.py`:

| Model | What it is |
|---|---|
| Seasonal naive | `lag_7`: same weekday one week back, the latest value the 7-day lead time allows |
| Moving average 28d | `roll_mean_28_lag7`: 28-day mean ending 7 days back |
| XGBoost | `reg:tweedie`, native categorical splits |
| CatBoost | `Tweedie:variance_power=p`, native categorical features |
| LightGBM | `tweedie`, native categorical features (the production search space) |

Each ML model runs twice:
- **default**: library defaults, with only the Tweedie objective set. CatBoost
  requires a variance power, so it gets 1.5, which is the LightGBM and
  XGBoost default.
- **tuned**: Optuna TPE with the same budget (`--trials`, default 40), the
  same seed, the same objective (mean CV WAPE) and the same early stopping
  (50 rounds on validation MAE, up to 3000 trees). The Tweedie power is
  searched over [1.1, 1.6] for all three models, and the other ranges match
  wherever the parameters mean the same thing. CatBoost early-stops on MAE of
  exp(raw prediction): its built-in MAE compares the log-scale value with the
  target under a Tweedie loss.

The reported CV WAPE re-scores the chosen parameters on every fold with a
fixed number of trees, so default and tuned settings are scored the same way.
The risk-level scores (accuracy, stockout recall/precision/F1, macro F1) use
the delivery-aware rule and 7-day forward window from
`risk_confusion_matrix.py`. `tests/test_compare_models.py` checks that every
model is fitted and scored on identical folds and the identical test window.

```bash
python src/compare_models.py --trials 40                  # full study
python src/compare_models.py --trials 2 --max-series 20   # ~3-minute smoke test
```

Outputs: `outputs/model_comparison.csv` / `.json` and charts in
`outputs/comparison/` (`wape_by_model.png`, `cv_wape_by_fold.png`,
`training_time.png`). A `--max-series` run writes to
`outputs/comparison_sample/` instead. The script ends by listing every metric
on which another model beats tuned LightGBM.

**Run time:** almost all of it is tuning, and CatBoost dominates. Measured on a
10-core Mac, one perishable trial (4 early-stopped folds) takes about 15–45 s
for LightGBM or XGBoost and 250–430 s for CatBoost. Non-perishable is about
3× the data. That works out to roughly 25 minutes per trial across all
models and both segments, plus about an hour of fixed work, so
`--trials 40` is about 17 hours, `--trials 10` about 5 hours and
`--trials 5` about 3 hours. Every model gets the same `--trials`, so a
smaller budget stays fair.

## Key design choices

- **7-day lead time, leak-free lags.** Orders take a week, so every lag and
  rolling feature is shifted by at least `LEAD_TIME_DAYS` = 7 (`lag_7`,
  `lag_14`, `lag_21`, `lag_364`, `roll_mean_{7,14,28}_lag7`,
  `roll_std_14_lag7`). Forecasts for the next 7 days only use sales up to
  yesterday.
- **Directional festival features.** Days until the next festival and since
  the last one (capped at 60), the next festival's weight, and whether it is
  within 7 days.
- **Unconstrained demand target.** Sales on stockout days are capped by
  stock. The target `demand` replaces them with a trailing 14-day baseline of
  festival-adjusted, non-stockout sales times today's festival uplift, so the
  model learns what customers wanted, not what was on the shelf. On stockout
  days it recovers ~98% of true demand (censored sales: ~45%).
- **Tweedie LightGBM, tuned per segment.** Perishable and non-perishable
  models are tuned separately with Optuna (TPE, 120 trials each) on mean CV
  WAPE, with early stopping on each fold. CV uses four expanding windows with
  a 7-day gap; the final test window is never seen by tuning.
- **Analog forecasting for the long tail.** A product without history borrows
  the forecasts of its 5 nearest tracked products in the same category (by
  price, pack size and brand popularity, same sub-category preferred),
  inverse-distance weighted and adjusted for price with the generator's
  elasticity.
- **Weekly, shelf-life-aware replenishment.** Each product is delivered every
  min(7, shelf life) days and ordered up to its trailing demand for that
  cycle plus 2 safety days; perishable stock is written off at its shelf
  life. A product is at stockout risk when its stock won't last until the
  next delivery, and overstocked when its cover exceeds 7 + 2 + shelf life
  days. The same code (`src/inventory.py`) generates the synthetic history
  and builds both risk reports.

## Limitations

- **Synthetic sales history.** Demand is generated from rules (price
  elasticity, brand popularity, festivals, weekday, promotions, noise). The
  models are learning the generator, so the accuracy numbers say nothing
  about real-world accuracy.
- **Simulated stock.** There is no stock feed. Opening stock for catalog
  products is simulated from recent demand with the replenishment policy
  above (`stock_source = "simulated"`).
- **Analog forecasts share their analogs' patterns.** 99% of catalog products
  are forecast from 5 tracked neighbours, so they inherit those products'
  festival and weekday shapes, scaled only for price. The price factor is
  clipped to 0.5–2×, so an analog forecast always stays within half to twice
  its neighbours' weighted mean. Before this clip a ₹0.01 free sample was
  forecast at ~20,000 units a day. About 45% of catalog products are priced
  far enough from their analogs to hit the clip.
- **Category stand-ins.** The catalog has no fresh produce: frozen vegetables
  stand in for "vegetables" and dry fruits for "fruits". "Snacks" also covers
  other packaged foods. 762 products (meat and fish, staples, oils, baby food,
  toys, sports, kitchen goods) fit no model category and are not forecast.
- **The Diwali check is in-sample.** Diwali-week forecasts (sweets ~3× a
  normal week, dairy ~2.2×) were checked on October 2025, which is inside the
  training period.
- **Festival dates** for 2021–2027 were checked against public holiday lists;
  lunar festivals can shift by a day by region. Re-check each year.
- **Promotions** are assumed off in forecasts, so weeks with planned
  promotions will be under-forecast.

## Using real data

Replace `data/pos_synthetic.csv` with a POS export that has the same columns:

| Column | Notes |
|---|---|
| `date`, `city`, `store_id`, `sku_id`, `category`, `is_perishable` | one row per city/store/product/day; `category` must be one of the model categories |
| `unit_price`, `on_promotion` | selling price, 0/1 promotion flag |
| `units_sold` | what was actually sold |
| `opening_stock` | stock at the start of the day, before sales — needed to detect stockouts |
| `stockout_flag` | 1 if the product ran out that day |
| `days_until_delivery` | optional; without it the risk rule assumes the 7-day lead time |

`data/sampled_products.csv` is the product master (`sku_id`, `brand`, `mrp`,
`pack_size_value`, `brand_catalog_count`); keep it in step with your
products. Then run `build_features.py`, `train_model.py --trials 40`,
`forecast_regional.py` and `build_bigbasket_regional_risk.py`.
