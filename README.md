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
| `src/alerts.py` | Per-city vendor email alerts from the model risk report (dry run by default) |
| `config/vendors.json` | Vendor name and email(s) per city, optional per-category overrides |
| `dashboard.py` | Web dashboard (standard library HTTP server) |
| `tests/test_pipeline.py`, `tests/test_alerts.py`, `tests/test_forecast_regional.py` | pytest checks |

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
