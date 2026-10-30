"""
generate_synthetic_data.py
----------------------------
Generates a synthetic, SKU-level, multi-city daily POS dataset shaped like a
REAL grocery POS/Shopify export, for ~300 real BigBasket products
(data/sampled_products.csv, see sample_products.py). This exists so you can build and test the
whole pipeline end-to-end before wiring in your real store data.

>>> WHEN YOU HAVE REAL DATA: replace the output of this script with your own
>>> POS export using the SAME COLUMN NAMES, and everything downstream
>>> (build_features.py, train_model.py) will work unchanged. <<<

Schema produced (data/pos_synthetic.csv):
    date, city, store_id, sku_id, category, is_perishable,
    unit_price, on_promotion, units_sold, opening_stock, stockout_flag,
    days_until_delivery (days until the next delivery arrives; 1 = tomorrow),
    true_demand   (synthetic-only: demand before the stock cap, for evaluation)

The history runs to END_DATE (= config.EVALUATION_END), the data every
evaluation result is built on. extend() then continues each series to
config.SERVING_DATE for serving only, with its own random stream
(config.EXTENSION_SEED) and the same demand formula, festival calendar,
city multipliers and replenishment policy (picking up each series' stock,
delivery cycle and recent demand), so the rows up to END_DATE stay
byte-identical. It also writes data/serving_opening_stock.csv: each series'
opening stock and days until delivery on the morning after SERVING_DATE
(the first forecast day), simulated by the same policy.

Simulated realistically with:
  - per-product baseline demand from real attributes (MRP, brand
    popularity, pack size) on top of a category level
  - weekday/weekend effect
  - festival-driven demand spikes (via festival_calendar.festival_uplift)
  - random promotions with a demand lift
  - the periodic-review inventory policy in inventory.simulate_inventory
    (delivery every min(7, shelf life) days, order-up-to with safety stock,
    shelf-life write-offs). Orders lag festival spikes, producing genuine
    stockouts (units_sold capped by opening_stock), and occasional
    over-orders produce genuine overstock.
"""

import numpy as np
import pandas as pd
from config import DATA_DIR, EVALUATION_END, EXTENSION_SEED, PRICE_ELASTICITY, RANDOM_SEED, SERVING_DATE
from festival_calendar import festival_uplift, PERISHABLE_CATEGORIES
from inventory import review_days_for, simulate_inventory
from sample_products import SAMPLED_PRODUCTS_PATH, sample as sample_products

RNG = np.random.default_rng(RANDOM_SEED)

CITIES = {
    # city: (relative demand multiplier, festival intensity multiplier)
    "Mumbai": (1.25, 1.05),
    "Delhi": (1.15, 1.10),
    "Bengaluru": (1.05, 0.90),
    "Chennai": (0.95, 0.95),
}

# category -> typical daily units for a median-priced, mid-popularity product
CATEGORY_BASE_UNITS = {
    "dairy": 28,
    "vegetables": 35,
    "fruits": 30,
    "sweets": 15,
    "beverages": 20,
    "snacks": 18,
    "household": 8,
    "personal_care": 6,
}
# Weekend demand lift; personal care is bought evenly through the week.
WEEKEND_LIFT = {"personal_care": 1.05}
DEFAULT_WEEKEND_LIFT = 1.20
PACK_ELASTICITY = 0.3    # bigger packs sell fewer units

START_DATE = "2021-08-26"
END_DATE = EVALUATION_END   # 5 years of history
POS_PATH = DATA_DIR / "pos_synthetic.csv"
SERVING_STOCK_PATH = DATA_DIR / "serving_opening_stock.csv"
TRAILING_DAYS = 14  # inventory.simulate_inventory orders up to the trailing 14-day mean


def build_sku_catalog(rng=RNG):
    """
    One row per sampled real product (data/sampled_products.csv) with a base
    daily demand driven by its attributes: cheaper -> more units, popular
    brands (by catalog count) -> more units, bigger packs -> fewer units.
    """
    if not SAMPLED_PRODUCTS_PATH.exists():
        sample_products()
    products = pd.read_csv(SAMPLED_PRODUCTS_PATH)
    by_category = products.groupby("category")

    # products without an MRP get their category's median price
    median_mrp = by_category["mrp"].transform("median")
    mrp = products["mrp"].fillna(median_mrp)
    price_factor = ((mrp / median_mrp) ** -PRICE_ELASTICITY).clip(0.3, 3.0)

    brand_rank = products["brand_catalog_count"].rank(pct=True)
    brand_factor = 0.7 + 0.8 * brand_rank

    median_pack = products.groupby(["category", "pack_size_unit"])["pack_size_value"].transform("median")
    pack_factor = ((products["pack_size_value"] / median_pack) ** -PACK_ELASTICITY).clip(0.5, 2.0).fillna(1.0)

    base_units = (products["category"].map(CATEGORY_BASE_UNITS) * price_factor * brand_factor
                  * pack_factor * rng.uniform(0.85, 1.15, len(products)))
    return pd.DataFrame({
        "sku_id": products["sku_id"],
        "category": products["category"],
        "is_perishable": products["category"].isin(PERISHABLE_CATEGORIES).astype(int),
        "base_units": base_units,
        "unit_price": mrp.round(2),
    })


def generate():
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    catalog = build_sku_catalog()

    category_uplift = {category: festival_uplift(pd.Series(dates), category) for category in catalog["category"].unique()}

    all_rows = []
    for city, (city_mult, fest_mult) in CITIES.items():
        for _, sku in catalog.iterrows():
            category = sku["category"]
            base = sku["base_units"] * city_mult

            weekday_effect = np.where(dates.day_of_week.isin([5, 6]), WEEKEND_LIFT.get(category, DEFAULT_WEEKEND_LIFT), 1.0)
            fest_effect = category_uplift[category]
            fest_effect = 1 + (fest_effect - 1) * fest_mult  # city-specific festival intensity

            on_promo = RNG.random(len(dates)) < 0.08
            promo_effect = np.where(on_promo, RNG.uniform(1.2, 1.5, len(dates)), 1.0)

            noise = RNG.normal(1.0, 0.12, len(dates))
            true_demand = base * weekday_effect * fest_effect * promo_effect * noise
            true_demand = np.clip(true_demand, 0, None)

            opening_stock, sold, days_until_delivery = (a[0] for a in simulate_inventory(true_demand, base, [category], RNG))
            units_sold = sold.round().astype(int)
            stockout_flag = (true_demand > opening_stock).astype(int)

            df = pd.DataFrame({
                "date": dates,
                "city": city,
                "store_id": f"{city[:3].upper()}-01",
                "sku_id": sku["sku_id"],
                "category": category,
                "is_perishable": sku["is_perishable"],
                "unit_price": sku["unit_price"],
                "on_promotion": on_promo.astype(int),
                "units_sold": units_sold,
                "opening_stock": opening_stock.round().astype(int),
                "stockout_flag": stockout_flag,
                "days_until_delivery": days_until_delivery,
                "true_demand": true_demand.round(2),
            })
            all_rows.append(df)

    full = pd.concat(all_rows, ignore_index=True)
    return full.sort_values(["date", "city", "sku_id"]).reset_index(drop=True)


def extend(history, end=SERVING_DATE, seed=EXTENSION_SEED):
    """
    Continue every series of `history` from the day after it ends to `end`,
    with its own random stream. Returns the new POS rows (same columns and
    dtypes) and each series' opening stock / days until delivery on the
    morning after `end`.
    """
    rng = np.random.default_rng(seed)
    catalog = build_sku_catalog(np.random.default_rng(RANDOM_SEED))  # the same base demand and prices as generate()
    last_day = history["date"].max()
    # one day past `end`: its opening stock is the serving stock (no sales are kept for it)
    dates = pd.date_range(last_day + pd.Timedelta(days=1), pd.Timestamp(end) + pd.Timedelta(days=1), freq="D")
    category_uplift = {category: festival_uplift(pd.Series(dates), category) for category in catalog["category"].unique()}
    last = history[history["date"] == last_day].set_index(["city", "sku_id"])
    recent = history[history["date"] > last_day - pd.Timedelta(days=TRAILING_DAYS)].pivot(
        index=["city", "sku_id"], columns="date", values="true_demand")

    all_rows, next_morning = [], []
    for city, (city_mult, fest_mult) in CITIES.items():
        for _, sku in catalog.iterrows():
            category = sku["category"]
            base = sku["base_units"] * city_mult

            weekday_effect = np.where(dates.day_of_week.isin([5, 6]), WEEKEND_LIFT.get(category, DEFAULT_WEEKEND_LIFT), 1.0)
            fest_effect = 1 + (category_uplift[category] - 1) * fest_mult

            on_promo = rng.random(len(dates)) < 0.08
            promo_effect = np.where(on_promo, rng.uniform(1.2, 1.5, len(dates)), 1.0)

            noise = rng.normal(1.0, 0.12, len(dates))
            true_demand = np.clip(base * weekday_effect * fest_effect * promo_effect * noise, 0, None)

            # pick up the series where the history left it
            state = last.loc[(city, sku["sku_id"])]
            review_days = review_days_for([category])[0]
            phase = (int(state["days_until_delivery"]) - 1) % review_days  # next delivery on day days_until_delivery
            closing_stock = float(state["opening_stock"] - state["units_sold"])
            opening_stock, sold, days_until_delivery = (a[0] for a in simulate_inventory(
                true_demand, base, [category], rng, initial_stock=[closing_stock], phase=[phase],
                history=recent.loc[(city, sku["sku_id"])].to_numpy()[None, :]))

            df = pd.DataFrame({
                "date": dates,
                "city": city,
                "store_id": f"{city[:3].upper()}-01",
                "sku_id": sku["sku_id"],
                "category": category,
                "is_perishable": sku["is_perishable"],
                "unit_price": sku["unit_price"],
                "on_promotion": on_promo.astype(int),
                "units_sold": sold.round().astype(int),
                "opening_stock": opening_stock.round().astype(int),
                "stockout_flag": (true_demand > opening_stock).astype(int),
                "days_until_delivery": days_until_delivery,
                "true_demand": true_demand.round(2),
            })
            all_rows.append(df.iloc[:-1])
            next_morning.append(df.iloc[-1:][["date", "city", "sku_id", "opening_stock", "days_until_delivery"]])

    rows = pd.concat(all_rows, ignore_index=True).sort_values(["date", "city", "sku_id"]).reset_index(drop=True)
    return rows, pd.concat(next_morning, ignore_index=True).sort_values(["city", "sku_id"]).reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    df.to_csv(POS_PATH, index=False)  # the evaluation history, exactly as before
    extension, serving_stock = extend(df)
    extension.to_csv(POS_PATH, mode="a", header=False, index=False)
    serving_stock.to_csv(SERVING_STOCK_PATH, index=False)
    print(f"Saved {len(df):,} rows across {df['city'].nunique()} cities, "
          f"{df['sku_id'].nunique()} products, {df['date'].min().date()} to {df['date'].max().date()} (evaluation history)")
    print(f"Appended {len(extension):,} rows, {extension['date'].min().date()} to {extension['date'].max().date()} "
          f"(serving extension, seed {EXTENSION_SEED}) to {POS_PATH}")
    print(f"Saved the opening stock for {serving_stock['date'].iloc[0].date()} to {SERVING_STOCK_PATH}")
    print("\nStockout rate: history", round(df["stockout_flag"].mean() * 100, 2), "%, extension",
          round(extension["stockout_flag"].mean() * 100, 2), "%")
