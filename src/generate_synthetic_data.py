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
from pathlib import Path
from config import PRICE_ELASTICITY, RANDOM_SEED
from festival_calendar import festival_uplift, PERISHABLE_CATEGORIES
from inventory import simulate_inventory
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
END_DATE = "2026-08-25"   # 5 years of history


def build_sku_catalog():
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
                  * pack_factor * RNG.uniform(0.85, 1.15, len(products)))
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


if __name__ == "__main__":
    df = generate()
    out_path = Path(__file__).resolve().parents[1] / "data" / "pos_synthetic.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows across {df['city'].nunique()} cities, "
          f"{df['sku_id'].nunique()} products, {df['date'].min().date()} to {df['date'].max().date()}")
    print(df.head())
    print("\nOverall stockout rate:", round(df["stockout_flag"].mean() * 100, 2), "%")
